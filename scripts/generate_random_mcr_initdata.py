#!/usr/bin/env python3
"""Generate deterministic random official-judge initdata JSONL records."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from convert_tziakcha_prepared_to_initdata import TILES, TILE_COUNTS


def random_walltiles(seed: int) -> list[str]:
    tiles = list(TILES)
    rng = random.Random(int(seed))
    rng.shuffle(tiles)
    if len(tiles) != 144:
        raise ValueError(f"expected 144 tiles, got {len(tiles)}")
    if Counter(tiles) != TILE_COUNTS:
        raise ValueError("random wall tile counts are not legal MCR counts")
    return tiles


def initdata_for_seed(seed: int, *, quan: int | None = None) -> dict:
    return {
        "quan": int(seed) % 4 if quan is None else int(quan) % 4,
        "srand": int(seed),
        "walltiles": " ".join(random_walltiles(seed)),
    }


def generate_file(args: argparse.Namespace) -> dict:
    games = int(args.games)
    start_seed = int(args.start_seed)
    if games <= 0:
        raise ValueError("--games must be positive")
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for index in range(games):
            seed = start_seed + index
            record = {
                "match_id": f"random-seed-{seed:010d}",
                "source": "deterministic_random_mcr_wall",
                "initdata": initdata_for_seed(seed, quan=args.quan),
            }
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "format": "random_mcr_initdata_v1",
        "out_jsonl": str(out_path),
        "games": games,
        "start_seed": start_seed,
        "end_seed": start_seed + games - 1,
        "quan": args.quan,
        "tile_count_per_game": 144,
    }
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--start-seed", type=int, default=2026060500)
    parser.add_argument("--quan", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(generate_file(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
