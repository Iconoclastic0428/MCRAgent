#!/usr/bin/env python3
"""Run a fixed-league official-judge sweep for QADV reranker lambdas."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evaluate_fixed_league_feasibility import summarize_feasibility
from official_judge_match import DEFAULT_ALEO, DEFAULT_JUDGE, DEFAULT_SAMPLE, run_match_set


def parse_lambdas(value: str) -> list[float]:
    lambdas = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not lambdas:
        raise ValueError("at least one lambda is required")
    return lambdas


def lambda_key(value: float) -> str:
    return f"{float(value):.2f}"


def _safe_metric(metrics: dict[str, Any]) -> bool:
    return (
        int(metrics.get("low_fan_hu_count") or 0) == 0
        and int(metrics.get("all_players_low_fan_hu_count") or 0) == 0
        and int(metrics.get("action_outside_legal_mask_count") or 0) == 0
        and int(metrics.get("illegal_prediction_count") or 0) == 0
    )


def _is_promotable(metrics: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace) -> bool:
    if not metrics.get("safe"):
        return False
    point_delta = float(metrics.get("average_point_delta") or 0.0)
    baseline_delta = float(baseline.get("average_point_delta") or 0.0)
    if point_delta <= baseline_delta + float(args.min_point_delta):
        return False
    deal_in = metrics.get("target_deal_in_rate")
    baseline_deal_in = baseline.get("target_deal_in_rate")
    if deal_in is not None and baseline_deal_in is not None:
        if float(deal_in) > float(baseline_deal_in) + float(args.max_deal_in_regression):
            return False
    return True


def _run_one_lambda(args: argparse.Namespace, lam: float) -> tuple[dict[str, Any], dict[str, Any]]:
    run_args = copy.copy(args)
    run_args.qadv_lambda = float(lam)
    run_args.qadv_model = None if float(lam) == 0.0 else args.qadv_model
    official = run_match_set(run_args)
    feasibility = summarize_feasibility(official, target_player=int(args.target_player))
    return official, feasibility


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lambdas = parse_lambdas(args.lambdas)
    lambda_metrics: dict[str, dict[str, Any]] = {}
    baseline_key = lambda_key(lambdas[0])

    for lam in lambdas:
        key = lambda_key(lam)
        official, feasibility = _run_one_lambda(args, lam)
        feasibility["safe"] = _safe_metric(feasibility)
        official_path = out_dir / f"fixed_league_lambda{key.replace('.', '')}_official.json"
        feasibility_path = out_dir / f"fixed_league_lambda{key.replace('.', '')}_feasibility.json"
        official_path.write_text(json.dumps(official, ensure_ascii=False, indent=2), encoding="utf-8")
        feasibility_path.write_text(json.dumps(feasibility, ensure_ascii=False, indent=2), encoding="utf-8")
        lambda_metrics[key] = {
            **feasibility,
            "official_path": str(official_path),
            "feasibility_path": str(feasibility_path),
        }

    baseline = lambda_metrics[baseline_key]
    promotable = {
        key: metrics
        for key, metrics in lambda_metrics.items()
        if key != baseline_key and _is_promotable(metrics, baseline, args)
    }
    best_key = None
    if promotable:
        best_key = max(
            promotable,
            key=lambda key: (
                float(promotable[key].get("average_point_delta") or 0.0),
                -float(key),
            ),
        )

    summary = {
        "format": "mcr_qadv_fixed_league_sweep_v1",
        "model": args.model,
        "qadv_model": args.qadv_model,
        "raw": args.raw,
        "games": int(args.games),
        "offset": int(args.offset),
        "target_player": int(args.target_player),
        "baseline_lambda": baseline_key,
        "best_promotable_lambda": best_key,
        "lambda_metrics": lambda_metrics,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--qadv-model", required=True)
    parser.add_argument("--lambdas", default="0.00,0.05,0.10,0.20,0.35,0.50")
    parser.add_argument("--policy", default="transformer")
    parser.add_argument("--opponent", default="sample")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--opponent-qadv-model", default=None)
    parser.add_argument("--opponent-qadv-lambda", type=float, default=0.0)
    parser.add_argument("--raw", default="data/raw/botzone_mcr_1000.jsonl")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--target-player", type=int, default=0)
    parser.add_argument("--judge", default=str(DEFAULT_JUDGE))
    parser.add_argument("--aleo-exe", default=str(DEFAULT_ALEO))
    parser.add_argument("--sample-exe", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--lawlorentz-levels", type=int, default=1)
    parser.add_argument("--out-dir", default="runs/qadv/fixed_league_sweep")
    parser.add_argument("--summary-out", default="runs/qadv/fixed_league_sweep/summary.json")
    parser.add_argument("--min-point-delta", type=float, default=0.0)
    parser.add_argument("--max-deal-in-regression", type=float, default=0.02)
    args = parser.parse_args()
    summary = run_sweep(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
