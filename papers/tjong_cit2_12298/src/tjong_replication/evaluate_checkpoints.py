"""Evaluate multiple supervised checkpoints on the same deterministic dataset."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import torch

from .evaluate_supervised import configure_deterministic_eval, load_model
from .paper_metrics import PAPER_NAME, PAPER_REPORTED_SUPERVISED_METRICS
from .train_supervised import evaluate_model, load_tensor_dataset

ACCURACY_METRICS = ("action_accuracy", "claim_accuracy", "discard_accuracy")
LOSS_METRICS = ("decision_loss", "action_loss", "claim_loss", "discard_loss")


def discover_checkpoints(paths: list[str], patterns: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for item in paths:
        discovered.append(Path(item))
    for pattern in patterns:
        discovered.extend(Path(match) for match in glob.glob(pattern))
    unique = {str(path): path for path in discovered}
    checkpoints = sorted(unique.values(), key=lambda path: str(path))
    if not checkpoints:
        raise ValueError("no checkpoints were provided or discovered")
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    return checkpoints


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and payload.get("epoch") is not None:
        completed_epoch = payload.get("completed_epoch")
        if completed_epoch is None and payload.get("batch_in_epoch") is None:
            completed_epoch = True
        return {
            "checkpoint_epoch": int(payload["epoch"]),
            "checkpoint_batch_in_epoch": payload.get("batch_in_epoch"),
            "checkpoint_global_batch": payload.get("global_batch"),
            "checkpoint_completed_epoch": completed_epoch,
        }
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    epochs = metrics.get("epochs") if isinstance(metrics, dict) else None
    if epochs:
        return {
            "checkpoint_epoch": int(epochs[-1].get("epoch")),
            "checkpoint_batch_in_epoch": None,
            "checkpoint_global_batch": epochs[-1].get("global_batch"),
            "checkpoint_completed_epoch": True,
        }
    return {
        "checkpoint_epoch": None,
        "checkpoint_batch_in_epoch": None,
        "checkpoint_global_batch": None,
        "checkpoint_completed_epoch": None,
    }


def evaluate_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    configure_deterministic_eval(args.seed, strict=args.strict_deterministic)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = discover_checkpoints(args.checkpoint, args.checkpoint_glob)
    dataset = load_tensor_dataset(Path(args.eval_pt), expected_encoding_version=args.require_encoding_version)
    results: list[dict[str, Any]] = []
    results_jsonl = Path(args.results_jsonl) if getattr(args, "results_jsonl", None) else None
    if results_jsonl is not None:
        results_jsonl.parent.mkdir(parents=True, exist_ok=True)
        results_jsonl.write_text("", encoding="utf-8")
    for checkpoint in checkpoints:
        model = load_model(
            checkpoint,
            device,
            expected_encoding_version=args.require_encoding_version,
            require_paper_config=args.require_paper_config,
        )
        metrics = evaluate_model(
            model,
            dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        result = {
            "checkpoint": str(checkpoint),
            "checkpoint_name": checkpoint.name,
            **checkpoint_metadata(checkpoint),
            **metrics,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if results_jsonl is not None:
            with results_jsonl.open("a", encoding="utf-8") as dst:
                dst.write(json.dumps(result, sort_keys=True) + "\n")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "paper": PAPER_NAME,
        "eval_pt": args.eval_pt,
        "deterministic_eval": True,
        "deterministic_seed": args.seed,
        "strict_deterministic": args.strict_deterministic,
        "device": str(device),
        "required_encoding_version": args.require_encoding_version,
        "require_paper_config": args.require_paper_config,
        "results_jsonl": str(results_jsonl) if results_jsonl is not None else None,
        "checkpoint_count": len(results),
        "results": results,
        "best_by_metric": best_by_metric(results),
        "paper_reported": dict(PAPER_REPORTED_SUPERVISED_METRICS),
    }
    if args.metrics_out:
        metrics_out = Path(args.metrics_out)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, sort_keys=True), flush=True)
    return summary


def best_by_metric(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for metric in ACCURACY_METRICS:
        candidates = [result for result in results if result.get(metric) is not None]
        if candidates:
            result = max(candidates, key=lambda item: float(item[metric]))
            best[metric] = {
                "checkpoint": result["checkpoint"],
                "checkpoint_epoch": result.get("checkpoint_epoch"),
                "value": result[metric],
            }
    for metric in LOSS_METRICS:
        candidates = [result for result in results if result.get(metric) is not None]
        if candidates:
            result = min(candidates, key=lambda item: float(item[metric]))
            best[metric] = {
                "checkpoint": result["checkpoint"],
                "checkpoint_epoch": result.get("checkpoint_epoch"),
                "value": result[metric],
            }
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    parser.add_argument("--eval-pt", required=True)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--results-jsonl", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--require-paper-config", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict-deterministic", action="store_true")
    args = parser.parse_args()
    evaluate_checkpoints(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
