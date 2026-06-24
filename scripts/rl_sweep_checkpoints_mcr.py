#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = WORKSPACE_ROOT / "scripts" / "rl_evaluate_mcr.py"


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == "final.pkl":
        return (1, 0, path.name)
    match = re.search(r"update_(\d+)\.pkl$", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (0, -1, path.name)


def discover_checkpoints(path: Path) -> list[Path]:
    candidates: list[Path] = []
    for root in [path, path / "checkpoints"]:
        if root.exists():
            candidates.extend(root.glob("update_*.pkl"))
    final_path = path / "final.pkl"
    if final_path.exists():
        candidates.append(final_path)
    unique = sorted({item.resolve() for item in candidates}, key=checkpoint_sort_key)
    if not unique:
        raise RuntimeError(f"no update_*.pkl or final.pkl checkpoints found under {path}")
    return unique


def metric_row(checkpoint: Path, payload: dict[str, Any], eval_json: Path) -> dict[str, Any]:
    decision = payload.get("promotion_decision") or {}
    return {
        "checkpoint": str(checkpoint),
        "eval_json": str(eval_json),
        "average_score": payload.get("average_score"),
        "average_score_ci95": payload.get("average_score_ci95"),
        "average_score_delta": payload.get("average_score_delta"),
        "average_score_delta_ci95": payload.get("average_score_delta_ci95"),
        "hu_rate": payload.get("hu_rate"),
        "hu_rate_delta": payload.get("hu_rate_delta"),
        "deal_in_rate": payload.get("deal_in_rate"),
        "deal_in_rate_delta": payload.get("deal_in_rate_delta"),
        "first_place_rate": payload.get("first_place_rate"),
        "first_place_rate_delta": payload.get("first_place_rate_delta"),
        "blocked_hu_rate": payload.get("blocked_hu_rate"),
        "invalid_response_rate": payload.get("invalid_response_rate"),
        "top1_agreement": payload.get("top1_agreement_with_base"),
        "promotion_decision": decision.get("decision"),
        "promotion_reasons": decision.get("reasons"),
    }


def sort_for_screen(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        score_delta = row.get("average_score_delta")
        hu_delta = row.get("hu_rate_delta")
        deal_delta = row.get("deal_in_rate_delta")
        return (
            float("-inf") if score_delta is None else float(score_delta),
            float("-inf") if hu_delta is None else float(hu_delta),
            float("inf") if deal_delta is None else -float(deal_delta),
        )

    return sorted(rows, key=key, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--paired-baseline-checkpoint", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--games-per-seat", type=int, default=128)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legal-dueling-mean", action="store_true")
    parser.add_argument("--max-turns", type=int, default=500)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = discover_checkpoints(checkpoint_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    eval_dir = out.with_suffix("")
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        eval_json = eval_dir / f"{checkpoint.stem}_eval.json"
        cmd = [
            sys.executable,
            str(EVALUATOR),
            "--candidate-checkpoint",
            str(checkpoint),
            "--champion-checkpoint",
            str(args.champion_checkpoint),
            "--paired-baseline-checkpoint",
            str(args.paired_baseline_checkpoint),
            "--use-paired-delta",
            "--raw",
            str(args.raw),
            "--judge",
            str(args.judge),
            "--games-per-seat",
            str(args.games_per_seat),
            "--max-turns",
            str(args.max_turns),
            "--out",
            str(eval_json),
        ]
        if args.legal_dueling_mean:
            cmd.append("--legal-dueling-mean")
        print(json.dumps({"event": "sweep_eval_start", "checkpoint": str(checkpoint), "out": str(eval_json)}), flush=True)
        subprocess.run(cmd, check=True)
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
        rows.append(metric_row(checkpoint, payload, eval_json))

    output = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "champion_checkpoint": str(Path(args.champion_checkpoint).resolve()),
        "paired_baseline_checkpoint": str(Path(args.paired_baseline_checkpoint).resolve()),
        "raw": str(Path(args.raw).resolve()),
        "games_per_seat": int(args.games_per_seat),
        "checkpoints": [str(path) for path in checkpoints],
        "results": rows,
        "sorted_by_average_score_delta": sort_for_screen(rows),
    }
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
