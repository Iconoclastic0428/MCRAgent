#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from rl_collect_mcr import main as collect_main  # noqa: E402


def read_version(param_dir: Path) -> dict:
    path = param_dir / "version.json"
    if not path.exists():
        return {"version": 0, "checkpoint": "current.pkl"}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--param-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--out-buffer", required=True)
    parser.add_argument("--actor-id", type=int, required=True)
    parser.add_argument("--games-per-iteration", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=64.0)
    parser.add_argument("--reward-clip", type=float, default=4.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    out_buffer = Path(args.out_buffer)
    out_buffer.mkdir(parents=True, exist_ok=True)
    iteration = 0
    while True:
        version = read_version(Path(args.param_dir))
        checkpoint = Path(args.param_dir) / str(version.get("checkpoint", "current.pkl"))
        if not checkpoint.exists():
            checkpoint = Path(args.baseline_checkpoint)
        tmp_out = out_buffer / f"iter_{iteration:06d}.tmp"
        final_out = out_buffer / f"iter_{iteration:06d}"
        if tmp_out.exists():
            shutil.rmtree(tmp_out)
        argv = [
            "rl_collect_mcr.py",
            "--checkpoint",
            str(checkpoint),
            "--baseline-checkpoint",
            str(args.baseline_checkpoint),
            "--raw",
            str(args.raw),
            "--judge",
            str(args.judge),
            "--games-per-seat",
            str(max(1, int(args.games_per_iteration) // 4)),
            "--out",
            str(tmp_out),
            "--reward-scale",
            str(args.reward_scale),
            "--reward-clip",
            str(args.reward_clip),
            "--epsilon",
            str(args.epsilon),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            collect_main()
        finally:
            sys.argv = old_argv
        if final_out.exists():
            shutil.rmtree(final_out)
        os.replace(tmp_out, final_out)
        iteration += 1
        if args.once:
            return 0
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
