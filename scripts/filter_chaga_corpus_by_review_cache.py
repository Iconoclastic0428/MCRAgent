#!/usr/bin/env python3
"""Filter CHAGA training rows to seats with cached review JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from re import Pattern

from audit_chaga_review_alignment import build_session_api_seat_maps, selected_players_for_audit


def filter_by_review_cache(
    *,
    raw_path: Path,
    prepared_path: Path,
    cache_dir: Path,
    raw_out: Path,
    prepared_out: Path,
    summary_out: Path,
    player_pattern: Pattern[str],
) -> dict:
    raw_records = list(iter_jsonl(raw_path))
    prepared_records = list(iter_jsonl(prepared_path))
    if len(raw_records) != len(prepared_records):
        raise ValueError(f"raw/prepared row mismatch: {len(raw_records)} != {len(prepared_records)}")
    session_api_seats = build_session_api_seat_maps(raw_records)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    prepared_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    summary: Counter[str] = Counter()
    player_counts: Counter[str] = Counter()
    sessions: set[str] = set()
    targets: set[tuple[str, int]] = set()

    with raw_out.open("w", encoding="utf-8") as raw_dst, prepared_out.open("w", encoding="utf-8") as prepared_dst:
        for raw_record, prepared_record in zip(raw_records, prepared_records):
            summary["records_seen"] += 1
            session_id = str(raw_record.get("belongs") or "")
            selected = selected_players_for_audit(raw_record, player_pattern=player_pattern, use_train_players=True)
            cached_selected: dict[str, str] = {}
            for player_text, player_name in selected.items():
                try:
                    seat = int(player_text)
                except (TypeError, ValueError):
                    continue
                api_seat = session_api_seats.get(session_id, {}).get(player_name, seat)
                cache_path = cache_dir / f"{session_id}_seat{api_seat}.json"
                if valid_review_cache(cache_path):
                    cached_selected[str(player_text)] = str(player_name)
                    targets.add((session_id, api_seat))
                elif cache_path.exists():
                    summary["invalid_cached_review"] += 1
            if not cached_selected:
                summary["dropped_without_cached_review"] += 1
                continue
            raw_filtered = dict(raw_record)
            prepared_filtered = dict(prepared_record)
            for record in (raw_filtered, prepared_filtered):
                record["train_players"] = sorted(cached_selected)
                record["train_player_names"] = {player: cached_selected[player] for player in sorted(cached_selected)}
            raw_dst.write(json.dumps(raw_filtered, ensure_ascii=False, separators=(",", ":")) + "\n")
            prepared_dst.write(json.dumps(prepared_filtered, ensure_ascii=False, separators=(",", ":")) + "\n")
            summary["records_written"] += 1
            summary["train_player_slots"] += len(cached_selected)
            player_counts.update(cached_selected.values())
            sessions.add(session_id)

    out = {
        "format": "mcr_chaga_review_cache_filtered_corpus_v1",
        "raw_path": str(raw_path),
        "prepared_path": str(prepared_path),
        "cache_dir": str(cache_dir),
        "raw_out": str(raw_out),
        "prepared_out": str(prepared_out),
        "summary_out": str(summary_out),
        "player_regex": player_pattern.pattern,
        "records_seen": summary["records_seen"],
        "records_written": summary["records_written"],
        "dropped_without_cached_review": summary["dropped_without_cached_review"],
        "invalid_cached_review": summary["invalid_cached_review"],
        "train_player_slots": summary["train_player_slots"],
        "sessions_written": len(sessions),
        "review_targets_used": len(targets),
        "selected_player_record_counts": dict(player_counts),
    }
    summary_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def valid_review_cache(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if isinstance(data, list):
        return True
    if isinstance(data, dict):
        return data.get("code") in {None, 0}
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--prepared-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--player-regex", default=r"^CHAGA0[2-8]$")
    args = parser.parse_args()

    summary = filter_by_review_cache(
        raw_path=Path(args.raw),
        prepared_path=Path(args.prepared),
        cache_dir=Path(args.cache_dir),
        raw_out=Path(args.raw_out),
        prepared_out=Path(args.prepared_out),
        summary_out=Path(args.summary_out),
        player_pattern=re.compile(args.player_regex),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
