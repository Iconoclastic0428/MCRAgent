#!/usr/bin/env python3
"""Train a legal discard ranker from Botzone MCR replay logs."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from evaluate_policy_replay import actual_response
from hand_features import candidate_feature_text
from policy_bot import split_tiles


def response_tile(response: str | None) -> str | None:
    if response is None:
        return None
    parts = response.strip().split()
    if len(parts) == 2 and parts[0].upper() == "PLAY":
        return parts[1]
    return None


def update_hand_from_actual_response(hand: Counter[str], response: str | None) -> None:
    tile = response_tile(response)
    if tile and hand[tile] > 0:
        hand[tile] -= 1
        if hand[tile] <= 0:
            del hand[tile]


def iter_discard_candidates(record: dict) -> Iterable[dict]:
    histories: dict[str, list[str]] = {str(player): [] for player in range(4)}
    hands: dict[str, Counter[str]] = {str(player): Counter() for player in range(4)}

    logs = record.get("logs") or []
    for turn in range(0, len(logs) - 1, 2):
        requests = (logs[turn].get("output") or {}).get("content") or {}
        responses = logs[turn + 1]

        for player, request in requests.items():
            player = str(player)
            request = str(request)
            tokens = request.split()
            if tokens and tokens[0] == "1":
                hands[player].clear()
                hands[player].update(split_tiles(tokens[5:18]))
            elif tokens and tokens[0] == "2" and len(tokens) >= 2:
                hands[player][tokens[1]] += 1
                actual = actual_response(responses, player)
                target_tile = response_tile(actual)
                if target_tile:
                    input_text = "\n".join([*histories[player], f"REQ {request}"])
                    for tile in sorted(tile for tile, count in hands[player].items() if count > 0):
                        yield {
                            "match_id": record["match_id"],
                            "player": int(player),
                            "turn_index": turn // 2,
                            "input_text": input_text,
                            "candidate_tile": tile,
                            "hand_tiles": list(hands[player].elements()),
                            "label": 1 if tile == target_tile else 0,
                        }

        for player, request in requests.items():
            player = str(player)
            request = str(request)
            actual = actual_response(responses, player)
            histories[player].append(f"REQ {request}")
            if actual is not None:
                histories[player].append(f"RES {actual}")
            update_hand_from_actual_response(hands[player], actual)


def load_candidates(path: Path) -> list[dict]:
    candidates: list[dict] = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                candidates.extend(iter_discard_candidates(json.loads(line)))
    return candidates


def split_by_match_id(items: list[dict], test_fraction: float) -> tuple[list[dict], list[dict]]:
    match_ids = sorted({item["match_id"] for item in items})
    if len(match_ids) <= 1:
        return items, []
    test_count = max(1, math.ceil(len(match_ids) * test_fraction))
    test_ids = set(match_ids[-test_count:])
    return (
        [item for item in items if item["match_id"] not in test_ids],
        [item for item in items if item["match_id"] in test_ids],
    )


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
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-5,
                    max_iter=60,
                    tol=1e-4,
                    class_weight="balanced",
                    random_state=11,
                ),
            ),
        ]
    )


def candidate_text(item: dict) -> str:
    feature_text = candidate_feature_text(
        item.get("hand_tiles") or [],
        item["candidate_tile"],
        drawn_tile=last_drawn_tile(item["input_text"]),
    )
    return f"{item['input_text']}\n{feature_text}"


def last_drawn_tile(input_text: str) -> str | None:
    for line in reversed(input_text.splitlines()):
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == "REQ" and parts[1] == "2":
            return parts[2]
    return None


def group_accuracy(items: list[dict], scores: list[float]) -> float | None:
    groups: dict[tuple[str, int, int], list[tuple[dict, float]]] = {}
    for item, score in zip(items, scores):
        key = (item["match_id"], item["player"], item["turn_index"])
        groups.setdefault(key, []).append((item, score))
    if not groups:
        return None
    correct = 0
    for group in groups.values():
        best_item = max(group, key=lambda pair: pair[1])[0]
        correct += 1 if best_item["label"] == 1 else 0
    return correct / len(groups)


def train(args: argparse.Namespace) -> dict:
    raw_path = Path(args.raw)
    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(raw_path)
    train_items, test_items = split_by_match_id(candidates, args.test_fraction)
    if not train_items:
        raise ValueError("no discard candidates available")

    pipeline = build_pipeline(args.max_features)
    x_train = [candidate_text(item) for item in train_items]
    y_train = [item["label"] for item in train_items]
    pipeline.fit(x_train, y_train)

    metrics = {
        "raw": str(raw_path),
        "candidates": len(candidates),
        "train_candidates": len(train_items),
        "test_candidates": len(test_items),
        "train_draw_decisions": sum(y_train),
        "test_draw_decisions": sum(item["label"] for item in test_items),
        "train_match_count": len({item["match_id"] for item in train_items}),
        "test_match_count": len({item["match_id"] for item in test_items}),
        "model_out": str(model_path),
        "metrics_out": str(metrics_path),
    }

    train_scores = pipeline.predict_proba(x_train)[:, 1]
    metrics["train_group_accuracy"] = group_accuracy(train_items, list(train_scores))
    if test_items:
        x_test = [candidate_text(item) for item in test_items]
        y_test = [item["label"] for item in test_items]
        test_scores = pipeline.predict_proba(x_test)[:, 1]
        test_pred = [1 if score >= 0.5 else 0 for score in test_scores]
        metrics["test_binary_accuracy"] = float(accuracy_score(y_test, test_pred))
        metrics["test_auc"] = float(roc_auc_score(y_test, test_scores))
        metrics["test_group_accuracy"] = group_accuracy(test_items, list(test_scores))

    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "discard_ranker",
                "feature_mode": "shanten",
                "pipeline": pipeline,
                "metrics": metrics,
            },
            out,
        )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/botzone_mcr_160.jsonl")
    parser.add_argument("--model-out", default="models/discard_ranker_160.pkl")
    parser.add_argument("--metrics-out", default="runs/discard_ranker_160_metrics.json")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=300_000)
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
