#!/usr/bin/env python3
"""Convert converted tziakcha replay logs into official-judge initdata.

The official judge needs a full wall string. Converted tziakcha records do not
store `initdata`, but they do expose the initial deal and observed draw stream.
This script reconstructs a legal wall whose per-seat draw prefixes match the
recorded replay, then fills the unobserved tail from remaining tile counts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SEGMENT_LEN = 36
TILES: list[str] = (
    [f"{suit}{rank}" for suit in ("W", "B", "T") for rank in range(1, 10) for _ in range(4)]
    + [f"F{rank}" for rank in range(1, 5) for _ in range(4)]
    + [f"J{rank}" for rank in range(1, 4) for _ in range(4)]
    + [f"H{rank}" for rank in range(1, 9)]
)
TILE_COUNTS = Counter(TILES)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _output_at(record: dict[str, Any], index: int) -> dict[str, Any]:
    logs = record.get("logs") or []
    if not (0 <= index < len(logs)):
        return {}
    item = logs[index]
    if not isinstance(item, dict):
        return {}
    output = item.get("output") or {}
    return output if isinstance(output, dict) else {}


def _parse_quan(record: dict[str, Any]) -> int:
    output = _output_at(record, 0)
    content = output.get("content") or {}
    for request in content.values():
        parts = str(request).split()
        if len(parts) >= 3 and parts[0] == "0":
            try:
                return max(0, min(3, int(parts[2])))
            except ValueError:
                continue
    return 0


def _split_deal_request(request: str) -> tuple[list[str], list[int]]:
    parts = str(request).split()
    if len(parts) < 18 or parts[0] != "1":
        raise ValueError(f"invalid DEAL request: {request!r}")
    flower_counts = [int(value) for value in parts[1:5]]
    hand = parts[5:18]
    if len(hand) != 13:
        raise ValueError(f"DEAL request did not expose 13 standing tiles: {request!r}")
    return hand, flower_counts


def _deal_prefixes(record: dict[str, Any]) -> list[list[str]]:
    logs = record.get("logs") or []
    for index in range(0, len(logs), 2):
        output = _output_at(record, index)
        if str((output.get("display") or {}).get("action") or "").upper() != "DEAL":
            continue
        content = output.get("content") or {}
        hands: list[list[str]] = [[] for _ in range(4)]
        flower_counts_by_request: list[list[int]] = []
        all_flower_tiles: list[str] = []
        for seat in range(4):
            hand, flower_counts = _split_deal_request(str(content.get(str(seat), "")))
            hands[seat] = hand
            flower_counts_by_request.append(flower_counts)
            if seat == 0:
                all_flower_tiles = str(content.get("0", "")).split()[18:]
        flower_counts = flower_counts_by_request[0] if flower_counts_by_request else [0, 0, 0, 0]
        flower_by_seat: list[list[str]] = [[] for _ in range(4)]
        cursor = 0
        for seat, count in enumerate(flower_counts):
            flower_by_seat[seat] = all_flower_tiles[cursor : cursor + int(count)]
            cursor += int(count)
        return [flower_by_seat[seat] + hands[seat] for seat in range(4)]
    raise ValueError("record has no DEAL output")


def _draw_prefixes(record: dict[str, Any]) -> list[list[str]]:
    prefixes: list[list[str]] = [[] for _ in range(4)]
    logs = record.get("logs") or []
    for index in range(0, len(logs), 2):
        output = _output_at(record, index)
        display = output.get("display") or {}
        action = str(display.get("action") or "").upper()
        if action not in {"DRAW", "BUHUA"}:
            continue
        try:
            player = int(display.get("player"))
        except (TypeError, ValueError):
            continue
        tile = str(display.get("tile") or "").strip()
        if 0 <= player < 4 and tile:
            prefixes[player].append(tile)
    return prefixes


def _remaining_tiles(consumed: list[list[str]]) -> list[str]:
    counts = Counter(TILE_COUNTS)
    for tile in [tile for seat_tiles in consumed for tile in seat_tiles]:
        counts[tile] -= 1
        if counts[tile] < 0:
            raise ValueError(f"tile consumed more often than legal count: {tile}")
    return sorted(tile for tile, count in counts.items() for _ in range(count))


def reconstruct_walltiles(record: dict[str, Any]) -> list[str]:
    """Return a 144-tile wall whose per-seat pop prefix follows the replay."""

    deal = _deal_prefixes(record)
    draws = _draw_prefixes(record)
    consumed = [deal[seat] + draws[seat] for seat in range(4)]
    remaining = _remaining_tiles(consumed)
    segments: list[list[str]] = []
    cursor = 0
    for seat, prefix in enumerate(consumed):
        if len(prefix) > SEGMENT_LEN:
            raise ValueError(f"seat {seat} consumes {len(prefix)} tiles, exceeding {SEGMENT_LEN}")
        filler_needed = SEGMENT_LEN - len(prefix)
        filler = remaining[cursor : cursor + filler_needed]
        if len(filler) != filler_needed:
            raise ValueError("not enough remaining tiles to fill wall")
        cursor += filler_needed
        segments.append(filler + list(reversed(prefix)))
    if cursor != len(remaining):
        raise ValueError("remaining tile accounting did not end exactly")
    return [tile for segment in segments for tile in segment]


def record_to_initdata(record: dict[str, Any], *, srand: int = 0) -> dict[str, Any]:
    return {
        "quan": _parse_quan(record),
        "srand": int(srand),
        "walltiles": " ".join(reconstruct_walltiles(record)),
    }


def convert_file(args: argparse.Namespace) -> dict[str, Any]:
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "format": "tziakcha_prepared_initdata_v1",
        "source": args.prepared,
        "out_jsonl": str(out_path),
        "records_seen": 0,
        "records_written": 0,
        "errors": [],
        "min_written": int(args.min_written),
    }
    with out_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(Path(args.prepared)):
            summary["records_seen"] += 1
            if args.limit is not None and summary["records_written"] >= int(args.limit):
                break
            try:
                initdata = record_to_initdata(record, srand=int(args.srand))
            except Exception as exc:  # noqa: BLE001 - report and continue unless strict.
                error = {
                    "record": str(record.get("source_record_id") or record.get("match_id") or ""),
                    "error": str(exc),
                }
                summary["errors"].append(error)
                if args.fail_on_error:
                    raise
                continue
            out.write(
                json.dumps(
                    {
                        "match_id": record.get("match_id") or record.get("source_record_id"),
                        "source_record_id": record.get("source_record_id") or record.get("match_id"),
                        "source": "tziakcha_prepared_reconstructed_wall",
                        "initdata": initdata,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            summary["records_written"] += 1
    if int(summary["records_written"]) < int(args.min_written):
        raise ValueError(
            f"only reconstructed {summary['records_written']} records; "
            f"required at least {args.min_written}"
        )
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-written", type=int, default=0)
    parser.add_argument("--srand", type=int, default=0)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    summary = convert_file(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
