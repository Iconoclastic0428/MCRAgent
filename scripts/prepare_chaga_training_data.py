#!/usr/bin/env python3
"""Prepare CHAGA02-CHAGA08-only training/eval Botzone-style records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Callable

from evaluate_chaga_replay import DEFAULT_PLAYER_RE, selected_players_from_raw_record
from tziakcha_records import convert_record, record_step


def prepare_converted_record(
    raw_record: dict,
    converted_record: dict,
    *,
    player_scores: dict[str, float] | None = None,
    min_score: float | None = None,
    player_pattern: str = DEFAULT_PLAYER_RE.pattern,
) -> dict | None:
    selected = selected_players(
        raw_record,
        player_scores=player_scores,
        min_score=min_score,
        player_pattern=player_pattern,
    )
    if not selected:
        return None
    prepared = dict(converted_record)
    prepared["source_record_id"] = str(raw_record.get("id") or raw_record.get("match_id") or "")
    prepared["source_players"] = player_names_from_record(raw_record)
    prepared["train_players"] = sorted(selected)
    prepared["train_player_names"] = {player: selected[player] for player in sorted(selected)}
    return prepared


def load_player_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        for row in csv.DictReader(src):
            name = str(row.get("Player Name") or "").strip()
            if not name:
                continue
            try:
                scores[name] = float(row.get("Value") or 0.0)
            except (TypeError, ValueError):
                continue
    return scores


def selected_players(
    raw_record: dict,
    *,
    player_scores: dict[str, float] | None,
    min_score: float | None,
    player_pattern: str,
) -> dict[str, str]:
    if player_scores is None and min_score is None and player_pattern == DEFAULT_PLAYER_RE.pattern:
        return selected_players_from_raw_record(raw_record, DEFAULT_PLAYER_RE)

    pattern = re.compile(player_pattern) if player_pattern else None
    selected: dict[str, str] = {}
    for index, name in enumerate(player_names_from_record(raw_record)):
        if pattern is not None and not pattern.search(name):
            continue
        if player_scores is not None or min_score is not None:
            score = (player_scores or {}).get(name)
            if score is None:
                continue
            if min_score is not None and score <= min_score:
                continue
        selected[str(index)] = name
    return selected


def player_names_from_record(raw_record: dict) -> list[str]:
    try:
        step = record_step(raw_record)
    except Exception:
        return []
    names: list[str] = []
    for player in step.get("p") or []:
        if isinstance(player, dict):
            names.append(str(player.get("n", "")).strip())
        else:
            names.append(str(player).strip())
    return names


def build_split(
    raw_path: Path,
    out_dir: Path,
    *,
    prefix: str,
    eval_fraction: float,
    seed: int,
    player_scores: dict[str, float] | None = None,
    min_score: float | None = None,
    player_pattern: str = DEFAULT_PLAYER_RE.pattern,
    converter: Callable[[dict], dict] = convert_record,
) -> dict:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_records: list[dict] = []
    summary = {
        "raw": str(raw_path),
        "out_dir": str(out_dir),
        "prefix": prefix,
        "eval_fraction": eval_fraction,
        "seed": seed,
        "player_pattern": player_pattern,
        "min_score": min_score,
        "score_filtered": player_scores is not None or min_score is not None,
        "records_seen": 0,
        "records_with_selected_players": 0,
        "records_converted": 0,
        "train_records": 0,
        "eval_records": 0,
        "selected_player_record_counts": {},
        "convert_errors": [],
        "player_extract_errors": [],
    }
    with raw_path.open("r", encoding="utf-8-sig") as src:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            summary["records_seen"] += 1
            raw_record = json.loads(line)
            try:
                selected = selected_players(
                    raw_record,
                    player_scores=player_scores,
                    min_score=min_score,
                    player_pattern=player_pattern,
                )
            except Exception as exc:
                summary["player_extract_errors"].append({"line": line_number, "error": str(exc)})
                continue
            if not selected:
                continue
            summary["records_with_selected_players"] += 1
            try:
                converted = converter(raw_record)
            except Exception as exc:
                summary["convert_errors"].append({"line": line_number, "error": str(exc)})
                continue
            prepared = prepare_converted_record(
                raw_record,
                converted,
                player_scores=player_scores,
                min_score=min_score,
                player_pattern=player_pattern,
            )
            if prepared is None:
                continue
            prepared_records.append(prepared)
            summary["records_converted"] += 1
            for name in prepared["train_player_names"].values():
                counts = summary["selected_player_record_counts"]
                counts[name] = counts.get(name, 0) + 1

    indices = list(range(len(prepared_records)))
    random.Random(seed).shuffle(indices)
    eval_count = max(1, math.ceil(len(indices) * eval_fraction)) if indices else 0
    if len(indices) > 1:
        eval_count = min(eval_count, len(indices) - 1)
    eval_indices = set(indices[:eval_count])
    train_records = [record for index, record in enumerate(prepared_records) if index not in eval_indices]
    eval_records = [record for index, record in enumerate(prepared_records) if index in eval_indices]
    summary["train_records"] = len(train_records)
    summary["eval_records"] = len(eval_records)

    paths = {
        "train_out": out_dir / f"{prefix}_train.jsonl",
        "eval_out": out_dir / f"{prefix}_eval.jsonl",
        "all_out": out_dir / f"{prefix}_all.jsonl",
        "summary_out": out_dir / f"{prefix}_summary.json",
    }
    _write_jsonl(paths["train_out"], train_records)
    _write_jsonl(paths["eval_out"], eval_records)
    _write_jsonl(paths["all_out"], prepared_records)
    for key, path in paths.items():
        summary[key] = str(path)
    paths["summary_out"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out-dir", default="data/processed/chaga")
    parser.add_argument("--prefix", default="tziakcha_chaga0208")
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--elo-csv", default=None)
    parser.add_argument("--min-elo", type=float, default=None)
    parser.add_argument("--player-pattern", default=DEFAULT_PLAYER_RE.pattern)
    args = parser.parse_args()

    player_scores = load_player_scores(Path(args.elo_csv)) if args.elo_csv else None
    summary = build_split(
        Path(args.raw),
        Path(args.out_dir),
        prefix=args.prefix,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        player_scores=player_scores,
        min_score=args.min_elo,
        player_pattern=args.player_pattern,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
