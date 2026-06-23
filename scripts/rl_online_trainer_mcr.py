#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_AGENT_DIR = (
    WORKSPACE_ROOT
    / "third_party"
    / "mahjong-agent-2025-aug12-excludespecial-3pkl"
    / "feature-agent"
)


def write_version(param_dir: Path, version: int, checkpoint: str, notes: str) -> None:
    param_dir.mkdir(parents=True, exist_ok=True)
    tmp = param_dir / "version.tmp.json"
    tmp.write_text(
        json.dumps(
            {
                "version": int(version),
                "checkpoint": checkpoint,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "notes": notes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, param_dir / "version.json")


def drain_actor_buffer(workdir: Path, iteration: int) -> list[Path]:
    target = workdir / "drained" / f"iter_{iteration:06d}"
    target.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for shard in (workdir / "actor_buffer").glob("actor_*/*/shard_*.npz"):
        dst = target / f"{shard.parent.parent.name}_{shard.parent.name}_{shard.name}"
        shutil.move(str(shard), dst)
        moved.append(dst)
    return moved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--offline-replay", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha-cql", type=float, default=0.5)
    parser.add_argument("--beta-kl", type=float, default=0.5)
    parser.add_argument("--beta-bc", type=float, default=0.1)
    parser.add_argument("--updates-per-iteration", type=int, default=1000)
    parser.add_argument("--eval-every-iterations", type=int, default=5)
    parser.add_argument("--promotion-games-per-seat", type=int, default=512)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    param_dir = workdir / "params"
    param_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.init_checkpoint, param_dir / "current.pkl")
    shutil.copy2(args.init_checkpoint, param_dir / "champion.pkl")
    write_version(param_dir, 0, "current.pkl", "initialized from base16")

    iteration = 0
    while True:
        iteration += 1
        drained = drain_actor_buffer(workdir, iteration)
        replay_dirs = [Path(args.offline_replay)]
        replay_dirs.extend(path.parent for path in drained)
        out = workdir / "trainer" / "checkpoints" / f"iter_{iteration:06d}"
        cmd = [
            sys.executable,
            str(FEATURE_AGENT_DIR / "train_cql_mcr.py"),
            "--replay",
            *[str(path) for path in replay_dirs],
            "--init-checkpoint",
            str(param_dir / "current.pkl"),
            "--out",
            str(out),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--gamma",
            str(args.gamma),
            "--alpha-cql",
            str(args.alpha_cql),
            "--beta-kl",
            str(args.beta_kl),
            "--beta-bc",
            str(args.beta_bc),
            "--max-updates",
            str(args.updates_per_iteration),
            "--legal-dueling-mean",
        ]
        subprocess.run(cmd, check=True)
        shutil.copy2(out / "final.pkl", param_dir / "current.pkl")
        write_version(param_dir, iteration, "current.pkl", f"online iteration {iteration}")
        if args.once:
            return 0
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
