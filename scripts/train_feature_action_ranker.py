#!/usr/bin/env python3
"""Train a numeric feature ranker for legal MCR Botzone responses."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from fan_feature_ranker import fan_feature_names, featurize_fan_candidates
from feature_ranker import feature_names, featurize_candidates
from train_legal_action_ranker import (
    candidate_weight,
    group_accuracy,
    load_candidates,
    request_kind_matches,
    split_by_match_id,
)


def maybe_auc(y_true: list[int], scores: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def train(args: argparse.Namespace) -> dict:
    raw_path = Path(args.raw)
    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        item
        for item in load_candidates(
            raw_path,
            min_player_score=getattr(args, "min_player_score", None),
            winner_only=bool(getattr(args, "winner_only", False)),
        )
        if request_kind_matches(item["request"], args.request_kind)
    ]
    train_items, test_items = split_by_match_id(candidates, args.test_fraction)
    if not train_items:
        raise ValueError("no feature-ranker candidates available")

    feature_mode = getattr(args, "feature_mode", "numeric_v1")
    if feature_mode == "numeric_fan_v1":
        x_train = featurize_fan_candidates(train_items)
        names = fan_feature_names()
    elif feature_mode == "numeric_v1":
        x_train = featurize_candidates(train_items)
        names = feature_names()
    else:
        raise ValueError(f"unknown feature mode: {feature_mode}")
    y_train = [int(item["label"]) for item in train_items]
    weights = [candidate_weight(item, args.nonpass_decision_weight) for item in train_items]

    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=1e-3,
        early_stopping=False if len(train_items) < 1000 else True,
        random_state=37,
    )
    model.fit(x_train, y_train, sample_weight=weights)

    train_scores = model.predict_proba(x_train)[:, 1]
    metrics = {
        "raw": str(raw_path),
        "candidates": len(candidates),
        "train_candidates": len(train_items),
        "test_candidates": len(test_items),
        "train_decisions": sum(y_train),
        "test_decisions": sum(int(item["label"]) for item in test_items),
        "train_match_count": len({item["match_id"] for item in train_items}),
        "test_match_count": len({item["match_id"] for item in test_items}),
        "model_out": str(model_path),
        "metrics_out": str(metrics_path),
        "request_kind": args.request_kind,
        "nonpass_decision_weight": args.nonpass_decision_weight,
        "min_player_score": getattr(args, "min_player_score", None),
        "winner_only": bool(getattr(args, "winner_only", False)),
        "feature_mode": feature_mode,
        "feature_count": len(names),
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "min_samples_leaf": args.min_samples_leaf,
        "train_group_accuracy": group_accuracy(train_items, list(train_scores)),
    }

    if test_items:
        x_test = (
            featurize_fan_candidates(test_items)
            if feature_mode == "numeric_fan_v1"
            else featurize_candidates(test_items)
        )
        y_test = [int(item["label"]) for item in test_items]
        test_scores = model.predict_proba(x_test)[:, 1]
        test_pred = [1 if score >= 0.5 else 0 for score in test_scores]
        metrics["test_binary_accuracy"] = float(accuracy_score(y_test, test_pred))
        metrics["test_auc"] = maybe_auc(y_test, list(test_scores))
        metrics["test_group_accuracy"] = group_accuracy(test_items, list(test_scores))

    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "feature_action_ranker",
                "feature_mode": feature_mode,
                "model": model,
                "feature_names": names,
                "metrics": metrics,
            },
            out,
        )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/botzone_mcr_1000.jsonl")
    parser.add_argument("--model-out", default="models/feature_action_ranker.pkl")
    parser.add_argument("--metrics-out", default="runs/feature_action_ranker_metrics.json")
    parser.add_argument("--request-kind", choices=["all", "draw", "reaction"], default="all")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--nonpass-decision-weight", type=float, default=1.0)
    parser.add_argument("--min-player-score", type=float, default=None)
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--feature-mode", choices=["numeric_v1", "numeric_fan_v1"], default="numeric_v1")
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
