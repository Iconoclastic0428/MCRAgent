#!/usr/bin/env python3
"""Reward-weighted policy training from lightweight self-play trajectories."""

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

from legal_actions import apply_response, generate_legal_responses, response_candidate_text
from policy_bot import split_tiles
from train_legal_action_ranker import request_kind_matches
from train_legal_action_ranker import group_accuracy


def iter_reward_weighted_candidates(
    game: dict, player_filter: int | None = None, request_kind: str = "all"
) -> Iterable[dict]:
    histories: dict[int, list[str]] = {player: [] for player in range(4)}
    hands: dict[int, Counter[str]] = {player: Counter() for player in range(4)}
    player_ids: dict[int, int | None] = {player: None for player in range(4)}
    rewards = game.get("rewards") or [0.0, 0.0, 0.0, 0.0]

    for decision_index, item in enumerate(game.get("trajectory") or []):
        player = int(item["player"])
        request = str(item["request"])
        response = str(item["response"])
        tokens = request.split()
        if tokens and tokens[0] == "0" and len(tokens) >= 2:
            player_ids[player] = int(tokens[1])
        elif tokens and tokens[0] == "1":
            hands[player].clear()
            hands[player].update(split_tiles(tokens[5:18]))
        elif tokens and tokens[0] == "2" and len(tokens) >= 2:
            hands[player][tokens[1]] += 1

        candidates = generate_legal_responses(player_ids[player], request, hands[player])
        should_yield = (
            (player_filter is None or player == player_filter)
            and request_kind_matches(request, request_kind)
        )
        if should_yield and response in candidates and request.split()[0] in {"2", "3"}:
            input_text = "\n".join([*histories[player], f"REQ {request}"])
            weight = max(0.05, 1.0 + float(rewards[player]))
            for candidate in candidates:
                yield {
                    "match_id": str(game.get("seed", game.get("match_id", "game"))),
                    "decision_index": decision_index,
                    "turn_index": decision_index,
                    "player": player,
                    "request": request,
                    "input_text": input_text,
                    "candidate_response": candidate,
                    "label": 1 if candidate == response else 0,
                    "hand": Counter(hands[player]),
                    "sample_weight": weight,
                }

        histories[player].append(f"REQ {request}")
        histories[player].append(f"RES {response}")
        apply_response(hands[player], request, response)


def load_games(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results") or []


def split_by_game(items: list[dict], test_fraction: float) -> tuple[list[dict], list[dict]]:
    game_ids = sorted({item["match_id"] for item in items})
    if len(game_ids) <= 1:
        return items, []
    test_count = max(1, math.ceil(len(game_ids) * test_fraction))
    test_ids = set(game_ids[-test_count:])
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
                    max_iter=80,
                    tol=1e-4,
                    class_weight="balanced",
                    random_state=23,
                ),
            ),
        ]
    )


def candidate_text(item: dict) -> str:
    return response_candidate_text(
        item["input_text"],
        item["candidate_response"],
        item["hand"],
        item["request"],
    )


def train(args: argparse.Namespace) -> dict:
    trajectory_path = Path(args.trajectories)
    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        candidate
        for game in load_games(trajectory_path)
        for candidate in iter_reward_weighted_candidates(
            game, player_filter=args.player_filter, request_kind=args.request_kind
        )
    ]
    train_items, test_items = split_by_game(candidates, args.test_fraction)
    if not train_items:
        raise ValueError("no reward-weighted candidates available")

    pipeline = build_pipeline(args.max_features)
    x_train = [candidate_text(item) for item in train_items]
    y_train = [item["label"] for item in train_items]
    weights = [item["sample_weight"] for item in train_items]
    pipeline.fit(x_train, y_train, clf__sample_weight=weights)

    metrics = {
        "trajectories": str(trajectory_path),
        "candidates": len(candidates),
        "train_candidates": len(train_items),
        "test_candidates": len(test_items),
        "train_decisions": sum(y_train),
        "test_decisions": sum(item["label"] for item in test_items),
        "train_game_count": len({item["match_id"] for item in train_items}),
        "test_game_count": len({item["match_id"] for item in test_items}),
        "model_out": str(model_path),
        "metrics_out": str(metrics_path),
        "player_filter": args.player_filter,
        "request_kind": args.request_kind,
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
                "kind": "legal_action_ranker",
                "feature_mode": "reward_weighted_selfplay_v1",
                "pipeline": pipeline,
                "metrics": metrics,
            },
            out,
        )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--model-out", default="models/reward_weighted_policy.pkl")
    parser.add_argument("--metrics-out", default="runs/reward_weighted_policy_metrics.json")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=400_000)
    parser.add_argument("--player-filter", type=int, default=None)
    parser.add_argument("--request-kind", choices=["all", "draw", "reaction"], default="all")
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
