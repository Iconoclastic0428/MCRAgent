#!/usr/bin/env python3
"""Train a first behavior-cloning policy from Botzone MCR examples."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def derive_action_type(response: Any) -> str:
    if response is None:
        return "MISSING"
    response = str(response).strip()
    if not response:
        return "EMPTY"
    return response.split()[0].upper()


def load_examples(path: Path, target: str, request_prefix: str | None = None) -> list[dict]:
    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            if request_prefix and not str(item.get("request") or "").startswith(request_prefix):
                continue
            label = item.get(target)
            if label is None or derive_action_type(label) in {"MISSING", "EMPTY"}:
                continue
            text = item.get("input_text") or item.get("request") or ""
            if not text.strip():
                continue
            examples.append(
                {
                    "match_id": str(item["match_id"]),
                    "input_text": text,
                    "label": str(label).strip(),
                    "response": item.get("response"),
                }
            )
    return examples


def split_by_match_id(examples: list[dict], test_fraction: float = 0.2) -> tuple[list[dict], list[dict]]:
    if not examples:
        return [], []
    match_ids = sorted({item["match_id"] for item in examples})
    if len(match_ids) == 1:
        return examples, []

    test_count = max(1, math.ceil(len(match_ids) * test_fraction))
    test_ids = set(match_ids[-test_count:])
    train = [item for item in examples if item["match_id"] not in test_ids]
    test = [item for item in examples if item["match_id"] in test_ids]
    return train, test


def build_pipeline(max_features: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=max_features,
                    lowercase=False,
                    min_df=1,
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
                    random_state=7,
                ),
            ),
        ]
    )


def train(args: argparse.Namespace) -> dict:
    in_path = Path(args.infile)
    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_examples(in_path, args.target, args.request_prefix)
    train_examples, test_examples = split_by_match_id(examples, args.test_fraction)
    if not train_examples:
        raise ValueError("no training examples available")

    x_train = [item["input_text"] for item in train_examples]
    y_train = [item["label"] for item in train_examples]
    pipeline = build_pipeline(args.max_features)
    pipeline.fit(x_train, y_train)

    metrics = {
        "input": str(in_path),
        "target": args.target,
        "examples": len(examples),
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "train_match_count": len({item["match_id"] for item in train_examples}),
        "test_match_count": len({item["match_id"] for item in test_examples}),
        "model_out": str(model_path),
        "metrics_out": str(metrics_path),
        "request_prefix": args.request_prefix,
    }

    majority_label = max(set(y_train), key=y_train.count)
    train_predictions = pipeline.predict(x_train)
    metrics["train_accuracy"] = float(accuracy_score(y_train, train_predictions))
    metrics["majority_label"] = majority_label
    metrics["majority_train_accuracy"] = float(
        accuracy_score(y_train, [majority_label] * len(y_train))
    )

    if test_examples:
        x_test = [item["input_text"] for item in test_examples]
        y_test = [item["label"] for item in test_examples]
        test_predictions = pipeline.predict(x_test)
        metrics["test_accuracy"] = float(accuracy_score(y_test, test_predictions))
        metrics["majority_test_accuracy"] = float(
            accuracy_score(y_test, [majority_label] * len(y_test))
        )
        metrics["classification_report"] = classification_report(
            y_test,
            test_predictions,
            zero_division=0,
            output_dict=True,
        )

        if args.target == "response":
            pred_action = [derive_action_type(pred) for pred in test_predictions]
            true_action = [derive_action_type(label) for label in y_test]
            metrics["test_action_type_accuracy"] = float(accuracy_score(true_action, pred_action))

    with model_path.open("wb") as out:
        pickle.dump({"pipeline": pipeline, "target": args.target, "metrics": metrics}, out)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="data/processed/botzone_mcr_bc_40.jsonl")
    parser.add_argument("--target", choices=["response", "action_type"], default="response")
    parser.add_argument("--model-out", default="models/bc_policy_response.pkl")
    parser.add_argument("--metrics-out", default="runs/bc_policy_response_metrics.json")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--request-prefix", default=None)
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
