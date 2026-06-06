"""Merge sharded Tjong self-play outputs into the PPO input raw files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .collect_selfplay import SelfplaySummaryAccumulator


def merge_shards(
    *,
    shard_dir: Path,
    shards: int,
    out_raw: Path,
    out_fan_items: Path,
    summary_out: Path | None = None,
    expected_games: int | None = None,
    min_games: int | None = None,
    min_total_hu_rate: float | None = None,
    max_huang_rate: float | None = None,
    raw_pattern: str = "tjong_selfplay_raw_shard_{index:04d}.jsonl",
    fan_pattern: str = "tjong_selfplay_fan_items_shard_{index:04d}.jsonl",
    summary_pattern: str = "tjong_selfplay_summary_shard_{index:04d}.json",
) -> dict[str, Any]:
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_fan_items.parent.mkdir(parents=True, exist_ok=True)
    accumulator = SelfplaySummaryAccumulator()
    shard_summaries = []
    raw_lines_total = 0
    fan_lines_total = 0

    with out_raw.open("w", encoding="utf-8") as raw_dst, out_fan_items.open("w", encoding="utf-8") as fan_dst:
        for index in range(int(shards)):
            raw_path = shard_dir / raw_pattern.format(index=index)
            fan_path = shard_dir / fan_pattern.format(index=index)
            shard_summary_path = shard_dir / summary_pattern.format(index=index)
            if not raw_path.exists():
                raise FileNotFoundError(f"missing raw shard {index}: {raw_path}")
            if not fan_path.exists():
                raise FileNotFoundError(f"missing fan shard {index}: {fan_path}")
            shard_summary = _read_json(shard_summary_path) if shard_summary_path.exists() else {}
            raw_lines = 0
            with raw_path.open("r", encoding="utf-8") as raw_src:
                for line in raw_src:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    accumulator.add(record)
                    raw_dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    raw_lines += 1
            fan_lines = _copy_nonempty_lines(fan_path, fan_dst)
            if shard_summary and int(shard_summary.get("games", -1)) != raw_lines:
                raise ValueError(
                    f"shard {index} summary games={shard_summary.get('games')} but raw lines={raw_lines}"
                )
            shard_summaries.append(
                {
                    "index": int(index),
                    "raw": str(raw_path),
                    "fan_items": str(fan_path),
                    "summary": str(shard_summary_path) if shard_summary_path.exists() else None,
                    "raw_lines": int(raw_lines),
                    "fan_lines": int(fan_lines),
                    "summary_games": shard_summary.get("games"),
                }
            )
            raw_lines_total += raw_lines
            fan_lines_total += fan_lines

    summary = accumulator.to_summary()
    summary.update(
        {
            "format": "tjong_selfplay_shard_merge_v1",
            "shard_dir": str(shard_dir),
            "shards": int(shards),
            "out_raw": str(out_raw),
            "out_fan_items": str(out_fan_items),
            "raw_lines": int(raw_lines_total),
            "fan_item_lines": int(fan_lines_total),
            "expected_games": expected_games,
            "min_games": min_games,
            "min_total_hu_rate": min_total_hu_rate,
            "max_huang_rate": max_huang_rate,
            "shard_summaries": shard_summaries,
        }
    )
    validate_merge_summary(summary)
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def validate_merge_summary(summary: dict[str, Any]) -> None:
    games = int(summary.get("games", -1))
    if int(summary.get("raw_lines", -1)) != games:
        raise ValueError(f"raw line count does not match games: {summary}")
    if int(summary.get("fan_item_lines", -1)) != games:
        raise ValueError(f"fan item line count does not match games: {summary}")
    expected_games = summary.get("expected_games")
    if expected_games is not None and games != int(expected_games):
        raise ValueError(f"merged games {games} != expected games {expected_games}")
    min_games = summary.get("min_games")
    if min_games is not None and games < int(min_games):
        raise ValueError(f"merged games {games} < minimum games {min_games}")
    min_total_hu_rate = summary.get("min_total_hu_rate")
    if min_total_hu_rate is not None and float(summary.get("hu_rate") or 0.0) < float(min_total_hu_rate):
        raise ValueError(f"merged HU rate is below threshold: {summary}")
    max_huang_rate = summary.get("max_huang_rate")
    if max_huang_rate is not None and float(summary.get("huang_rate") or 0.0) > float(max_huang_rate):
        raise ValueError(f"merged HUANG rate is above threshold: {summary}")
    if len(summary.get("score_table") or []) != 4:
        raise ValueError(f"merged summary missing 4 score table entries: {summary}")


def _copy_nonempty_lines(path: Path, dst) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
            count += 1
    return count


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out-raw", required=True)
    parser.add_argument("--out-fan-items", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--expected-games", type=int, default=None)
    parser.add_argument("--min-games", type=int, default=None)
    parser.add_argument("--min-total-hu-rate", type=float, default=None)
    parser.add_argument("--max-huang-rate", type=float, default=None)
    parser.add_argument("--raw-pattern", default="tjong_selfplay_raw_shard_{index:04d}.jsonl")
    parser.add_argument("--fan-pattern", default="tjong_selfplay_fan_items_shard_{index:04d}.jsonl")
    parser.add_argument("--summary-pattern", default="tjong_selfplay_summary_shard_{index:04d}.json")
    args = parser.parse_args()
    merge_shards(
        shard_dir=Path(args.shard_dir),
        shards=args.shards,
        out_raw=Path(args.out_raw),
        out_fan_items=Path(args.out_fan_items),
        summary_out=Path(args.summary_out) if args.summary_out else None,
        expected_games=args.expected_games,
        min_games=args.min_games,
        min_total_hu_rate=args.min_total_hu_rate,
        max_huang_rate=args.max_huang_rate,
        raw_pattern=args.raw_pattern,
        fan_pattern=args.fan_pattern,
        summary_pattern=args.summary_pattern,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
