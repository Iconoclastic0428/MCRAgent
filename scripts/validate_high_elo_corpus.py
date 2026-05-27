#!/usr/bin/env python3
"""Validate that corpus train_players are exactly high-ELO seats for each raw record."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tziakcha_records import record_step


def validate_high_elo_corpus(raw_path: Path, *, min_elo: float) -> dict:
    summary: Counter[str] = Counter()
    failures: list[dict] = []
    for raw_record in _iter_jsonl(raw_path):
        summary["records"] += 1
        try:
            step = record_step(raw_record)
        except Exception as exc:
            summary["record_step_errors"] += 1
            failures.append({"record": raw_record.get("id"), "error": str(exc)})
            continue
        expected = expected_high_elo_players(step, min_elo=min_elo)
        actual = set(str(player) for player in raw_record.get("train_players") or [])
        if actual != expected:
            summary["train_player_mismatches"] += 1
            failures.append(
                {
                    "record": raw_record.get("id"),
                    "session": raw_record.get("belongs"),
                    "expected": sorted(expected),
                    "actual": sorted(actual),
                    "players": _player_debug(step),
                }
            )
        summary["train_player_slots"] += len(actual)
        for player in actual:
            try:
                player_data = (step.get("p") or [])[int(player)]
                elo = float(player_data.get("e") if isinstance(player_data, dict) else 0.0)
            except (TypeError, ValueError, IndexError):
                summary["invalid_train_player_seats"] += 1
                continue
            if elo <= min_elo:
                summary["below_threshold_train_player_slots"] += 1
    return {
        "raw_path": str(raw_path),
        "min_elo": min_elo,
        "records": summary["records"],
        "train_player_slots": summary["train_player_slots"],
        "record_step_errors": summary["record_step_errors"],
        "train_player_mismatches": summary["train_player_mismatches"],
        "invalid_train_player_seats": summary["invalid_train_player_seats"],
        "below_threshold_train_player_slots": summary["below_threshold_train_player_slots"],
        "failures_preview": failures[:50],
    }


def expected_high_elo_players(step: dict, *, min_elo: float) -> set[str]:
    expected: set[str] = set()
    for index, player in enumerate(step.get("p") or []):
        if not isinstance(player, dict):
            continue
        try:
            elo = float(player.get("e") if player.get("e") is not None else 0.0)
        except (TypeError, ValueError):
            continue
        if elo > min_elo:
            expected.add(str(index))
    return expected


def _player_debug(step: dict) -> list[dict]:
    out: list[dict] = []
    for index, player in enumerate(step.get("p") or []):
        if isinstance(player, dict):
            out.append({"seat": index, "name": player.get("n"), "elo": player.get("e")})
        else:
            out.append({"seat": index, "name": str(player), "elo": None})
    return out


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--min-elo", type=float, default=2300.0)
    parser.add_argument("--summary-out")
    args = parser.parse_args()

    summary = validate_high_elo_corpus(Path(args.raw), min_elo=args.min_elo)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["failures_preview"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
