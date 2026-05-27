#!/usr/bin/env python3
"""Filter an aligned high-ELO raw/prepared corpus to matching player seats."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from re import Pattern
from typing import Iterator

from tziakcha_records import record_step


def filter_corpus_by_player(
    *,
    raw_path: Path,
    prepared_path: Path,
    raw_out: Path,
    prepared_out: Path,
    summary_out: Path,
    player_pattern: Pattern[str],
    min_elo: float,
) -> dict:
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    prepared_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    summary: Counter[str] = Counter()
    player_counts: Counter[str] = Counter()
    failures: list[dict] = []

    with raw_out.open("w", encoding="utf-8") as raw_dst, prepared_out.open("w", encoding="utf-8") as prepared_dst:
        for line_number, (raw_record, prepared_record) in enumerate(zip_aligned_jsonl(raw_path, prepared_path), start=1):
            summary["records_seen"] += 1
            raw_record_id = source_record_id(raw_record)
            prepared_record_id = source_record_id(prepared_record)
            if raw_record_id and prepared_record_id and raw_record_id != prepared_record_id:
                failures.append(
                    {
                        "line": line_number,
                        "raw_record": raw_record_id,
                        "prepared_record": prepared_record_id,
                    }
                )
                summary["source_record_mismatches"] += 1
                continue
            try:
                step = record_step(raw_record)
            except Exception as exc:
                failures.append({"line": line_number, "record": raw_record_id, "error": str(exc)})
                summary["record_step_errors"] += 1
                continue
            selected = selected_record_players(step, player_pattern=player_pattern, min_elo=min_elo)
            if not selected:
                summary["dropped_no_selected_player"] += 1
                continue

            raw_filtered = dict(raw_record)
            prepared_filtered = dict(prepared_record)
            for record in (raw_filtered, prepared_filtered):
                record["train_players"] = sorted(selected)
                record["train_player_names"] = {player: selected[player] for player in sorted(selected)}
                record["source_record_id"] = raw_record_id
            raw_dst.write(json.dumps(raw_filtered, ensure_ascii=False, separators=(",", ":")) + "\n")
            prepared_dst.write(json.dumps(prepared_filtered, ensure_ascii=False, separators=(",", ":")) + "\n")
            summary["records_written"] += 1
            summary["train_player_slots"] += len(selected)
            player_counts.update(selected.values())

    out = {
        "format": "mcr_high_elo_player_filtered_corpus_v1",
        "raw_path": str(raw_path),
        "prepared_path": str(prepared_path),
        "raw_out": str(raw_out),
        "prepared_out": str(prepared_out),
        "summary_out": str(summary_out),
        "player_regex": player_pattern.pattern,
        "min_elo": min_elo,
        "records_seen": summary["records_seen"],
        "records_written": summary["records_written"],
        "dropped_no_selected_player": summary["dropped_no_selected_player"],
        "train_player_slots": summary["train_player_slots"],
        "record_step_errors": summary["record_step_errors"],
        "source_record_mismatches": summary["source_record_mismatches"],
        "row_count_mismatches": summary["row_count_mismatches"],
        "selected_player_record_counts": dict(player_counts),
        "failures_preview": failures[:50],
    }
    summary_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def selected_record_players(step: dict, *, player_pattern: Pattern[str], min_elo: float) -> dict[str, str]:
    selected: dict[str, str] = {}
    for index, player in enumerate(step.get("p") or []):
        if not isinstance(player, dict):
            continue
        name = str(player.get("n") or "").strip()
        if not player_pattern.search(name):
            continue
        try:
            elo = float(player.get("e") if player.get("e") is not None else 0.0)
        except (TypeError, ValueError):
            continue
        if elo > min_elo:
            selected[str(index)] = name
    return selected


def source_record_id(record: dict) -> str:
    return str(record.get("source_record_id") or record.get("id") or record.get("match_id") or "")


def zip_aligned_jsonl(raw_path: Path, prepared_path: Path) -> Iterator[tuple[dict, dict]]:
    with raw_path.open("r", encoding="utf-8-sig") as raw_src, prepared_path.open("r", encoding="utf-8-sig") as prepared_src:
        while True:
            raw_line = next_nonempty(raw_src)
            prepared_line = next_nonempty(prepared_src)
            if raw_line is None and prepared_line is None:
                return
            if raw_line is None or prepared_line is None:
                raise ValueError("raw and prepared JSONL row counts differ")
            yield json.loads(raw_line), json.loads(prepared_line)


def next_nonempty(src) -> str | None:
    for line in src:
        if line.strip():
            return line
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--prepared-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--player-regex", required=True)
    parser.add_argument("--min-elo", type=float, default=2300.0)
    args = parser.parse_args()

    summary = filter_corpus_by_player(
        raw_path=Path(args.raw),
        prepared_path=Path(args.prepared),
        raw_out=Path(args.raw_out),
        prepared_out=Path(args.prepared_out),
        summary_out=Path(args.summary_out),
        player_pattern=re.compile(args.player_regex),
        min_elo=args.min_elo,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if not summary["failures_preview"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
