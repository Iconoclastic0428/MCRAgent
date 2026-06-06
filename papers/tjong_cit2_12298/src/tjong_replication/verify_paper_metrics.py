"""Fail-fast comparison of supervised metrics against the Tjong paper."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

from .paper_metrics import (
    DEFAULT_PAPER_METRIC_TOLERANCE,
    PAPER_NAME,
    PAPER_REPORTED_SUPERVISED_METRICS,
)


def _metric_float(value: Any) -> float:
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError("metric is not finite")
    return observed


def compare_supervised_metrics(
    metrics: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_PAPER_METRIC_TOLERANCE,
    targets: Mapping[str, float] = PAPER_REPORTED_SUPERVISED_METRICS,
) -> dict:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    comparisons = {}
    passed = True
    for name, target in targets.items():
        entry = {
            "observed": metrics.get(name),
            "target": target,
            "delta": None,
            "absolute_delta": None,
            "within_tolerance": False,
            "passed": False,
        }
        if name not in metrics:
            entry["reason"] = "missing_metric"
            comparisons[name] = entry
            passed = False
            continue
        try:
            observed = _metric_float(metrics[name])
        except (TypeError, ValueError) as exc:
            entry["reason"] = f"invalid_metric: {exc}"
            comparisons[name] = entry
            passed = False
            continue

        delta = observed - target
        absolute_delta = abs(delta)
        within_tolerance = absolute_delta <= tolerance
        entry.update(
            {
                "observed": observed,
                "delta": delta,
                "absolute_delta": absolute_delta,
                "within_tolerance": within_tolerance,
                "passed": within_tolerance,
            }
        )
        if not within_tolerance:
            passed = False
        comparisons[name] = entry

    target_integrity = {
        "checked": False,
        "passed": True,
        "comparisons": {},
    }
    embedded_targets = metrics.get("paper_reported")
    if isinstance(embedded_targets, Mapping):
        target_integrity["checked"] = True
        for name, target in targets.items():
            embedded_entry = {
                "observed": embedded_targets.get(name),
                "target": target,
                "passed": False,
            }
            try:
                embedded_value = _metric_float(embedded_targets[name])
            except (KeyError, TypeError, ValueError) as exc:
                embedded_entry["reason"] = f"invalid_or_missing_target: {exc}"
                target_integrity["passed"] = False
            else:
                embedded_entry["observed"] = embedded_value
                embedded_entry["passed"] = embedded_value == target
                if embedded_value != target:
                    target_integrity["passed"] = False
            target_integrity["comparisons"][name] = embedded_entry
    elif embedded_targets is not None:
        target_integrity["checked"] = True
        target_integrity["passed"] = False
        target_integrity["reason"] = "paper_reported is not an object"

    return {
        "paper": PAPER_NAME,
        "tolerance": tolerance,
        "targets": dict(targets),
        "passed": bool(passed and target_integrity["passed"]),
        "comparisons": comparisons,
        "paper_reported_target_integrity": target_integrity,
    }


def verify_metrics_file(
    metrics_path: Path,
    *,
    tolerance: float = DEFAULT_PAPER_METRIC_TOLERANCE,
    summary_out: Path | None = None,
    require_strict_deterministic: bool = False,
) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = compare_supervised_metrics(metrics, tolerance=tolerance)
    summary["metrics_path"] = str(metrics_path)
    deterministic_eval = metrics.get("deterministic_eval")
    strict_deterministic = metrics.get("strict_deterministic")
    summary["metrics_determinism"] = {
        "required": bool(require_strict_deterministic),
        "deterministic_eval": deterministic_eval,
        "strict_deterministic": strict_deterministic,
        "passed": (not require_strict_deterministic)
        or (deterministic_eval is True and strict_deterministic is True),
    }
    if not summary["metrics_determinism"]["passed"]:
        summary["passed"] = False
    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not summary["passed"]:
        raise ValueError(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_PAPER_METRIC_TOLERANCE)
    parser.add_argument("--require-strict-deterministic", action="store_true")
    args = parser.parse_args()

    try:
        summary = verify_metrics_file(
            Path(args.metrics),
            tolerance=args.tolerance,
            summary_out=Path(args.summary_out) if args.summary_out else None,
            require_strict_deterministic=args.require_strict_deterministic,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
