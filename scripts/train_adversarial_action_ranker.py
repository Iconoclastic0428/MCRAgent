#!/usr/bin/env python3
"""Train an adversarially weighted legal-action ranker from outcome-labeled records."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from legal_actions import response_candidate_text
from train_legal_action_ranker import (
    candidate_text,
    candidate_weight,
    group_accuracy,
    load_candidates,
    request_kind_matches,
    split_by_match_id,
)


def build_pipeline(max_features: int, random_state: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=max_features,
                    lowercase=False,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-5,
                    max_iter=80,
                    tol=1e-4,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def successful_outcome(item: dict) -> int:
    score = item.get("player_score")
    return 1 if score is not None and float(score) > 0 else 0


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
        for item in load_candidates(raw_path)
        if request_kind_matches(item["request"], args.request_kind)
    ]
    if not candidates:
        raise ValueError("no adversarial candidates available")

    actual_items = [item for item in candidates if item["label"] == 1]
    if len({successful_outcome(item) for item in actual_items}) < 2:
        raise ValueError("adversarial discriminator needs both positive and nonpositive outcomes")

    discriminator = build_pipeline(args.max_features, random_state=41)
    x_disc = [candidate_text(item) for item in actual_items]
    y_disc = [successful_outcome(item) for item in actual_items]
    discriminator.fit(x_disc, y_disc)
    disc_scores = discriminator.predict_proba(x_disc)[:, 1]

    train_items, test_items = split_by_match_id(candidates, args.test_fraction)
    if not train_items:
        raise ValueError("no generator candidates available")

    generator = build_pipeline(args.max_features, random_state=43)
    x_train = [candidate_text(item) for item in train_items]
    y_train = [item["label"] for item in train_items]
    adv_scores = discriminator.predict_proba(x_train)[:, 1]
    weights = [
        candidate_weight(item, args.nonpass_decision_weight)
        * (1.0 + float(args.adversarial_strength) * float(score))
        for item, score in zip(train_items, adv_scores)
    ]
    generator.fit(x_train, y_train, clf__sample_weight=weights)

    train_scores = generator.predict_proba(x_train)[:, 1]
    metrics = {
        "raw": str(raw_path),
        "feature_mode": "adversarial_outcome_weighted_v1",
        "discriminator_items": len(actual_items),
        "discriminator_positive_items": sum(y_disc),
        "discriminator_accuracy": float(
            accuracy_score(y_disc, [1 if score >= 0.5 else 0 for score in disc_scores])
        ),
        "discriminator_auc": maybe_auc(y_disc, list(disc_scores)),
        "generator_candidates": len(candidates),
        "train_candidates": len(train_items),
        "test_candidates": len(test_items),
        "train_decisions": sum(y_train),
        "test_decisions": sum(item["label"] for item in test_items),
        "train_match_count": len({item["match_id"] for item in train_items}),
        "test_match_count": len({item["match_id"] for item in test_items}),
        "model_out": str(model_path),
        "metrics_out": str(metrics_path),
        "request_kind": args.request_kind,
        "adversarial_strength": args.adversarial_strength,
        "nonpass_decision_weight": args.nonpass_decision_weight,
        "mean_adversarial_weight": float(sum(weights) / len(weights)),
        "train_group_accuracy": group_accuracy(train_items, list(train_scores)),
    }

    if test_items:
        x_test = [candidate_text(item) for item in test_items]
        y_test = [item["label"] for item in test_items]
        test_scores = generator.predict_proba(x_test)[:, 1]
        metrics["test_binary_accuracy"] = float(
            accuracy_score(y_test, [1 if score >= 0.5 else 0 for score in test_scores])
        )
        metrics["test_auc"] = maybe_auc(y_test, list(test_scores))
        metrics["test_group_accuracy"] = group_accuracy(test_items, list(test_scores))

    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "legal_action_ranker",
                "feature_mode": "adversarial_outcome_weighted_v1",
                "pipeline": generator,
                "discriminator": discriminator,
                "metrics": metrics,
            },
            out,
        )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--request-kind", choices=["all", "draw", "reaction"], default="all")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=400_000)
    parser.add_argument("--adversarial-strength", type=float, default=2.0)
    parser.add_argument("--nonpass-decision-weight", type=float, default=1.0)
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
