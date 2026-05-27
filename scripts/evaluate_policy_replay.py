#!/usr/bin/env python3
"""Evaluate a Botzone-style policy wrapper against scraped replay logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from policy_bot import BotzonePolicy, ShantenHeuristicPredictor, SklearnPredictor
from train_bc_policy import derive_action_type


def actual_response(response_log: dict, player: str) -> str | None:
    item = response_log.get(str(player)) or {}
    response = item.get("raw") or item.get("response")
    return None if response is None else str(response).strip()


def evaluate_records(records: Iterable[dict], predictor=None) -> dict:
    metrics = {
        "matches": 0,
        "examples": 0,
        "exact_matches": 0,
        "action_type_matches": 0,
        "active_draw_examples": 0,
        "active_draw_exact_matches": 0,
        "active_draw_action_type_matches": 0,
        "reaction_examples": 0,
        "reaction_exact_matches": 0,
        "reaction_action_type_matches": 0,
        "reaction_nonpass_actual": 0,
        "reaction_nonpass_predicted": 0,
        "reaction_nonpass_exact_matches": 0,
        "fallback_predictions": 0,
        "illegal_predictions": 0,
    }

    for record in records:
        metrics["matches"] += 1
        policies = {str(player): BotzonePolicy(predictor) for player in range(4)}
        logs = record.get("logs") or []
        for turn in range(0, len(logs) - 1, 2):
            requests = (logs[turn].get("output") or {}).get("content") or {}
            responses = logs[turn + 1]
            for player, request in requests.items():
                actual = actual_response(responses, str(player))
                if actual is None:
                    continue
                predicted = policies[str(player)].respond(str(request))
                metrics["examples"] += 1
                if policies[str(player)].last_fallback_used:
                    metrics["fallback_predictions"] += 1
                if policies[str(player)].last_illegal_prediction:
                    metrics["illegal_predictions"] += 1
                if predicted == actual:
                    metrics["exact_matches"] += 1
                if derive_action_type(predicted) == derive_action_type(actual):
                    metrics["action_type_matches"] += 1
                if str(request).startswith("2 "):
                    metrics["active_draw_examples"] += 1
                    if predicted == actual:
                        metrics["active_draw_exact_matches"] += 1
                    if derive_action_type(predicted) == derive_action_type(actual):
                        metrics["active_draw_action_type_matches"] += 1
                if str(request).startswith("3 "):
                    metrics["reaction_examples"] += 1
                    if predicted == actual:
                        metrics["reaction_exact_matches"] += 1
                    if derive_action_type(predicted) == derive_action_type(actual):
                        metrics["reaction_action_type_matches"] += 1
                    if derive_action_type(actual) != "PASS":
                        metrics["reaction_nonpass_actual"] += 1
                    if derive_action_type(predicted) != "PASS":
                        metrics["reaction_nonpass_predicted"] += 1
                    if derive_action_type(actual) != "PASS" and predicted == actual:
                        metrics["reaction_nonpass_exact_matches"] += 1

    for numerator, denominator, key in [
        ("exact_matches", "examples", "exact_accuracy"),
        ("action_type_matches", "examples", "action_type_accuracy"),
        ("active_draw_exact_matches", "active_draw_examples", "active_draw_exact_accuracy"),
        (
            "active_draw_action_type_matches",
            "active_draw_examples",
            "active_draw_action_type_accuracy",
        ),
        ("illegal_predictions", "examples", "illegal_prediction_rate"),
        ("fallback_predictions", "examples", "fallback_prediction_rate"),
        ("reaction_exact_matches", "reaction_examples", "reaction_exact_accuracy"),
        (
            "reaction_action_type_matches",
            "reaction_examples",
            "reaction_action_type_accuracy",
        ),
        (
            "reaction_nonpass_exact_matches",
            "reaction_nonpass_predicted",
            "reaction_nonpass_precision",
        ),
        (
            "reaction_nonpass_exact_matches",
            "reaction_nonpass_actual",
            "reaction_nonpass_recall",
        ),
    ]:
        metrics[key] = (
            metrics[numerator] / metrics[denominator] if metrics[denominator] else None
        )
    return metrics


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/botzone_mcr_80.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--heuristic", choices=["shanten"], default=None)
    parser.add_argument("--out", default="runs/policy_replay_eval.json")
    args = parser.parse_args()

    predictor = None
    if args.heuristic == "shanten":
        predictor = ShantenHeuristicPredictor()
    elif args.model:
        predictor = SklearnPredictor(Path(args.model))
    metrics = evaluate_records(load_records(Path(args.raw)), predictor=predictor)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
