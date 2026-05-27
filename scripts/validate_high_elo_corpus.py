#!/usr/bin/env python3
"""Validate that corpus train_players are exactly high-ELO seats for each raw record."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from tziakcha_records import record_step


def validate_high_elo_corpus(raw_path: Path, *, min_elo: float, prepared_path: Path | None = None) -> dict:
    summary: Counter[str] = Counter()
    failures: list[dict] = []
    prepared_iter = _iter_jsonl(prepared_path) if prepared_path else None
    for line_number, raw_record in enumerate(_iter_jsonl(raw_path), start=1):
        summary["records"] += 1
        prepared_record = _next_prepared(prepared_iter)
        if prepared_record is not None:
            summary["prepared_records"] += 1
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
        _validate_train_player_names(raw_record, step, failures, summary)
        if prepared_path and prepared_record is None:
            summary["prepared_row_count_mismatches"] += 1
            failures.append(
                {
                    "line": line_number,
                    "record": raw_record.get("id"),
                    "error": "missing prepared row",
                }
            )
        elif prepared_record is not None:
            _validate_prepared_row(
                raw_record=raw_record,
                prepared_record=prepared_record,
                expected=expected,
                line_number=line_number,
                failures=failures,
                summary=summary,
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
    if prepared_iter is not None and _next_prepared(prepared_iter) is not None:
        summary["prepared_row_count_mismatches"] += 1
        failures.append({"error": "extra prepared rows"})
    return {
        "raw_path": str(raw_path),
        "prepared_path": str(prepared_path) if prepared_path else None,
        "min_elo": min_elo,
        "records": summary["records"],
        "train_player_slots": summary["train_player_slots"],
        "record_step_errors": summary["record_step_errors"],
        "train_player_mismatches": summary["train_player_mismatches"],
        "train_player_name_mismatches": summary["train_player_name_mismatches"],
        "invalid_train_player_seats": summary["invalid_train_player_seats"],
        "below_threshold_train_player_slots": summary["below_threshold_train_player_slots"],
        "prepared_records": summary["prepared_records"],
        "prepared_row_count_mismatches": summary["prepared_row_count_mismatches"],
        "prepared_train_player_mismatches": summary["prepared_train_player_mismatches"],
        "prepared_source_record_mismatches": summary["prepared_source_record_mismatches"],
        "prepared_train_player_name_mismatches": summary["prepared_train_player_name_mismatches"],
        "failures_preview": failures[:50],
    }


def _validate_train_player_names(raw_record: dict, step: dict, failures: list[dict], summary: Counter[str]) -> None:
    names = raw_record.get("train_player_names") or {}
    if not isinstance(names, dict):
        return
    players = step.get("p") or []
    for player, actual_name in names.items():
        try:
            player_data = players[int(player)]
        except (TypeError, ValueError, IndexError):
            continue
        expected_name = str(player_data.get("n") or "").strip() if isinstance(player_data, dict) else str(player_data).strip()
        if str(actual_name) != expected_name:
            summary["train_player_name_mismatches"] += 1
            failures.append(
                {
                    "record": raw_record.get("id"),
                    "session": raw_record.get("belongs"),
                    "player": str(player),
                    "expected_name": expected_name,
                    "actual_name": str(actual_name),
                }
            )


def _validate_prepared_row(
    *,
    raw_record: dict,
    prepared_record: dict,
    expected: set[str],
    line_number: int,
    failures: list[dict],
    summary: Counter[str],
) -> None:
    prepared_actual = set(str(player) for player in prepared_record.get("train_players") or [])
    if prepared_actual != expected:
        summary["prepared_train_player_mismatches"] += 1
        failures.append(
            {
                "line": line_number,
                "record": raw_record.get("id"),
                "expected": sorted(expected),
                "prepared_actual": sorted(prepared_actual),
            }
        )
    raw_record_id = str(raw_record.get("source_record_id") or raw_record.get("id") or raw_record.get("match_id") or "")
    prepared_record_id = str(prepared_record.get("source_record_id") or prepared_record.get("id") or prepared_record.get("match_id") or "")
    if raw_record_id and prepared_record_id and raw_record_id != prepared_record_id:
        summary["prepared_source_record_mismatches"] += 1
        failures.append(
            {
                "line": line_number,
                "expected_record": raw_record_id,
                "prepared_record": prepared_record_id,
            }
        )
    raw_names = raw_record.get("train_player_names") or {}
    prepared_names = prepared_record.get("train_player_names") or {}
    if isinstance(raw_names, dict) and isinstance(prepared_names, dict) and raw_names != prepared_names:
        summary["prepared_train_player_name_mismatches"] += 1
        failures.append(
            {
                "line": line_number,
                "record": raw_record.get("id"),
                "expected_names": raw_names,
                "prepared_names": prepared_names,
            }
        )


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


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def _next_prepared(prepared_iter: Iterator[dict] | None) -> dict | None:
    if prepared_iter is None:
        return None
    try:
        return next(prepared_iter)
    except StopIteration:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--min-elo", type=float, default=2300.0)
    parser.add_argument("--summary-out")
    args = parser.parse_args()

    summary = validate_high_elo_corpus(
        Path(args.raw),
        min_elo=args.min_elo,
        prepared_path=Path(args.prepared) if args.prepared else None,
    )
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if not summary["failures_preview"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
