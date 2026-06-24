#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from benchmark_json_policies import load_initdata  # noqa: E402
from rl_collect_mcr import DEFAULT_CHECKPOINT, LoggingFeaturePolicy, run_game  # noqa: E402


def parse_seats(value: str) -> list[int]:
    if value.lower() == "all":
        return [0, 1, 2, 3]
    seats = [int(part) for part in value.split(",") if part.strip()]
    if not seats or any(seat < 0 or seat > 3 for seat in seats):
        raise ValueError(f"--seats must be all or comma-separated seats in 0..3, got {value!r}")
    return seats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--raw", default=str(WORKSPACE_ROOT / "data" / "eval" / "botzone_mcr_first64_suit_permuted_384.jsonl"))
    parser.add_argument("--judge", default=str(WORKSPACE_ROOT / "build" / "official_judge" / "mcr_judge.exe"))
    parser.add_argument("--games-per-seat", type=int, default=512)
    parser.add_argument("--initdata-offset", type=int, default=0)
    parser.add_argument("--seats", default="all")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--legal-dueling-mean", action="store_true")
    args = parser.parse_args()

    seats = parse_seats(str(args.seats))
    initdata_items = load_initdata(Path(args.raw), limit=int(args.games_per_seat), offset=int(args.initdata_offset))
    if not initdata_items:
        raise RuntimeError(f"no initdata loaded from {args.raw}")

    policies: list[LoggingFeaturePolicy] = [
        LoggingFeaturePolicy(
            args.checkpoint,
            record=False,
            policy_name="baseline",
            legal_dueling_mean=bool(args.legal_dueling_mean),
            seed=2000 + seat,
        )
        for seat in range(4)
    ]
    entries: dict[str, dict[str, Any]] = {}
    start = time.time()
    for game_index, initdata in enumerate(initdata_items):
        initdata_index = int(args.initdata_offset) + game_index
        _, result = run_game(
            initdata=initdata,
            policies=policies,
            trainee=policies[0],
            game_id=initdata_index,
            judge=Path(args.judge),
            reward_scale=64.0,
            reward_clip=4.0,
            max_turns=int(args.max_turns),
            initdata_index=initdata_index,
            initdata_offset=int(args.initdata_offset),
            baseline_score=0.0,
        )
        display = result["display"]
        terminal_action = str(display.get("action") or "UNKNOWN")
        scores = [float(value) for value in result["scores"]]
        for seat in seats:
            entries[f"{initdata_index}:{seat}"] = {
                "score": scores[seat],
                "terminal_action": terminal_action,
            }
        print(
            json.dumps(
                {
                    "event": "baseline_score_progress",
                    "game_index": game_index,
                    "initdata_index": initdata_index,
                    "terminal_action": terminal_action,
                    "elapsed_s": time.time() - start,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "raw": str(Path(args.raw).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "games_per_seat_requested": int(args.games_per_seat),
        "games_per_seat_loaded": len(initdata_items),
        "initdata_offset": int(args.initdata_offset),
        "seats": seats,
        "entries": entries,
        "elapsed_s": time.time() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "entries"}, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
