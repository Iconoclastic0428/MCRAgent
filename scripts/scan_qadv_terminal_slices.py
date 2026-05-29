#!/usr/bin/env python3
"""Scan exact official-judge offsets for terminal-rich QADV evaluation slices."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from official_judge_match import DEFAULT_ALEO, DEFAULT_JUDGE, DEFAULT_SAMPLE
from sweep_qadv_fixed_league import run_sweep


def _int_metric(metrics: dict[str, Any], key: str) -> int:
    try:
        return int(metrics.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _float_metric(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize_slice(sweep_summary: dict[str, Any]) -> dict[str, Any]:
    lambda_metrics = dict(sweep_summary.get("lambda_metrics") or {})
    terminal_signal_count = 0
    target_terminal_signal_count = 0
    unsafe_lambdas: list[str] = []
    max_average_point_delta = None
    for key, metrics in lambda_metrics.items():
        terminal_signal_count += _int_metric(metrics, "hu_count")
        terminal_signal_count += _int_metric(metrics, "end_wait_count")
        target_terminal_signal_count += _int_metric(metrics, "target_hu_count")
        target_terminal_signal_count += _int_metric(metrics, "target_end_wait_count")
        point_delta = _float_metric(metrics, "average_point_delta")
        max_average_point_delta = point_delta if max_average_point_delta is None else max(max_average_point_delta, point_delta)
        if not bool(metrics.get("safe")):
            unsafe_lambdas.append(str(key))
    has_point_signal = max_average_point_delta is not None and abs(float(max_average_point_delta)) > 1e-9
    return {
        "offset": sweep_summary.get("offset"),
        "games": sweep_summary.get("games"),
        "terminal_signal_count": terminal_signal_count,
        "target_terminal_signal_count": target_terminal_signal_count,
        "max_average_point_delta": float(max_average_point_delta or 0.0),
        "has_signal": terminal_signal_count > 0 or target_terminal_signal_count > 0 or has_point_signal,
        "unsafe_lambdas": unsafe_lambdas,
        "best_promotable_lambda": sweep_summary.get("best_promotable_lambda"),
    }


def scan_offsets(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    offset_results: list[dict[str, Any]] = []
    terminal_offsets: list[int] = []
    best_promotable_lambda = None
    best_promotable_offset = None

    offsets = [
        int(args.offset_start) + index * int(args.offset_step)
        for index in range(int(args.offset_count))
    ]
    for offset in offsets:
        slice_dir = out_dir / f"offset_{offset:05d}"
        sweep_args = copy.copy(args)
        sweep_args.offset = offset
        sweep_args.games = int(args.games_per_slice)
        sweep_args.out_dir = str(slice_dir)
        sweep_args.summary_out = str(slice_dir / "summary.json")
        sweep_summary = run_sweep(sweep_args)
        sweep_summary["offset"] = offset
        slice_summary = summarize_slice(sweep_summary)
        slice_summary["summary_path"] = str(slice_dir / "summary.json")
        offset_results.append(slice_summary)
        if slice_summary["has_signal"]:
            terminal_offsets.append(offset)
        if slice_summary.get("best_promotable_lambda") is not None:
            best_promotable_lambda = slice_summary["best_promotable_lambda"]
            best_promotable_offset = offset
        if (
            not bool(args.scan_all)
            and int(args.target_signal_slices) > 0
            and len(terminal_offsets) >= int(args.target_signal_slices)
        ):
            break

    summary = {
        "format": "mcr_qadv_terminal_slice_scan_v1",
        "model": args.model,
        "qadv_model": args.qadv_model,
        "lambdas": args.lambdas,
        "raw": args.raw,
        "games_per_slice": int(args.games_per_slice),
        "offset_start": int(args.offset_start),
        "offset_step": int(args.offset_step),
        "requested_offset_count": int(args.offset_count),
        "scanned_offset_count": len(offset_results),
        "terminal_slice_count": len(terminal_offsets),
        "terminal_offsets": terminal_offsets,
        "best_promotable_lambda": best_promotable_lambda,
        "best_promotable_offset": best_promotable_offset,
        "offset_results": offset_results,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--qadv-model", required=True)
    parser.add_argument("--lambdas", default="0.00,0.05,0.50")
    parser.add_argument("--policy", default="transformer")
    parser.add_argument("--opponent", default="sample")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--opponent-qadv-model", default=None)
    parser.add_argument("--opponent-qadv-lambda", type=float, default=0.0)
    parser.add_argument("--raw", default="data/raw/botzone_mcr_1000.jsonl")
    parser.add_argument("--games-per-slice", type=int, default=4)
    parser.add_argument("--offset-start", type=int, default=0)
    parser.add_argument("--offset-count", type=int, default=16)
    parser.add_argument("--offset-step", type=int, default=4)
    parser.add_argument("--target-signal-slices", type=int, default=2)
    parser.add_argument("--scan-all", action="store_true")
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--target-player", type=int, default=0)
    parser.add_argument("--judge", default=str(DEFAULT_JUDGE))
    parser.add_argument("--aleo-exe", default=str(DEFAULT_ALEO))
    parser.add_argument("--sample-exe", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--lawlorentz-levels", type=int, default=1)
    parser.add_argument("--out-dir", default="runs/qadv/terminal_slice_scan")
    parser.add_argument("--summary-out", default="runs/qadv/terminal_slice_scan/summary.json")
    parser.add_argument("--min-point-delta", type=float, default=0.0)
    parser.add_argument("--max-deal-in-regression", type=float, default=0.02)
    args = parser.parse_args()
    summary = scan_offsets(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
