#!/usr/bin/env python3
"""Train a fan-aware numeric feature ranker from official trajectories."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from fan_feature_ranker import fan_feature_names, featurize_fan_candidates
from legal_actions import apply_response, generate_legal_responses
from policy_bot import split_tiles
from train_legal_action_ranker import (
    candidate_weight,
    group_accuracy,
    request_kind_matches,
    split_by_match_id,
)


def fan_aware_player_reward(game: dict, player: int) -> float:
    scores = game.get("scores") or [0, 0, 0, 0]
    score = float(scores[player]) if player < len(scores) else 0.0
    finish = game.get("finish") or {}
    action = str(finish.get("action") or "")
    winner = finish.get("winner")
    fan_count = float(finish.get("fan_count") or 0.0)

    if action in {"WH", "WA"} or score <= -30:
        return -1.0
    if action == "HUANG":
        return 0.0
    if winner == player and fan_count >= 8:
        fan_bonus = min(max(fan_count - 8.0, 0.0), 40.0) / 40.0
        score_bonus = min(max(score, 0.0), 96.0) / 192.0
        return 1.0 + fan_bonus + score_bonus
    if score > 0:
        return 0.5 + min(score, 96.0) / 192.0
    if score < 0:
        return max(-0.8, score / 32.0)
    return 0.0


def sample_weight_from_reward(
    reward: float,
    positive_reward_scale: float,
    negative_reward_floor: float,
) -> float:
    if reward >= 0.0:
        return 1.0 + positive_reward_scale * reward
    return max(float(negative_reward_floor), 1.0 + reward)


def iter_fan_rewarded_candidates(
    game: dict,
    player_filter: int | None = None,
    request_kind: str = "all",
    positive_reward_scale: float = 1.0,
    negative_reward_floor: float = 0.1,
    nonpass_decision_weight: float = 1.0,
    min_player_reward: float | None = None,
) -> Iterable[dict]:
    histories: dict[int, list[str]] = {player: [] for player in range(4)}
    hands: dict[int, Counter[str]] = {player: Counter() for player in range(4)}
    player_ids: dict[int, int | None] = {player: None for player in range(4)}
    rewards = {player: fan_aware_player_reward(game, player) for player in range(4)}
    fan_count = int((game.get("finish") or {}).get("fan_count") or 0)

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
            and (
                min_player_reward is None
                or rewards[player] >= float(min_player_reward)
            )
        )
        if should_yield and response in candidates and tokens and tokens[0] in {"2", "3"}:
            input_text = "\n".join([*histories[player], f"REQ {request}"])
            reward = rewards[player]
            reward_weight = sample_weight_from_reward(
                reward,
                positive_reward_scale=positive_reward_scale,
                negative_reward_floor=negative_reward_floor,
            )
            actual_action = response.split()[0] if response else "PASS"
            base_weight = 1.0 if actual_action == "PASS" else float(nonpass_decision_weight)
            for candidate in candidates:
                yield {
                    "match_id": str(game.get("match_id", game.get("seed", "game"))),
                    "decision_index": decision_index,
                    "turn_index": decision_index,
                    "player": player,
                    "request": request,
                    "input_text": input_text,
                    "candidate_response": candidate,
                    "actual_response": response,
                    "label": 1 if candidate == response else 0,
                    "hand": Counter(hands[player]),
                    "sample_weight": base_weight * reward_weight,
                    "fan_reward": reward,
                    "fan_count": fan_count,
                }

        histories[player].append(f"REQ {request}")
        histories[player].append(f"RES {response}")
        apply_response(hands[player], request, response)


def load_games(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload.get("results") or []


def maybe_auc(y_true: list[int], scores: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def train(args: argparse.Namespace) -> dict:
    trajectory_path = Path(args.trajectories)
    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        candidate
        for game in load_games(trajectory_path)
        for candidate in iter_fan_rewarded_candidates(
            game,
            player_filter=args.player_filter,
            request_kind=args.request_kind,
            positive_reward_scale=args.positive_reward_scale,
            negative_reward_floor=args.negative_reward_floor,
            nonpass_decision_weight=args.nonpass_decision_weight,
            min_player_reward=getattr(args, "min_player_reward", None),
        )
    ]
    train_items, test_items = split_by_match_id(candidates, args.test_fraction)
    if not train_items:
        raise ValueError("no fan-rewarded candidates available")

    x_train = featurize_fan_candidates(train_items)
    y_train = [int(item["label"]) for item in train_items]
    weights = [float(item["sample_weight"]) for item in train_items]

    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=1e-3,
        early_stopping=False if len(train_items) < 1000 else True,
        random_state=71,
    )
    model.fit(x_train, y_train, sample_weight=weights)

    train_scores = model.predict_proba(x_train)[:, 1]
    metrics = {
        "trajectories": str(trajectory_path),
        "feature_mode": "numeric_fan_v1",
        "candidates": len(candidates),
        "train_candidates": len(train_items),
        "test_candidates": len(test_items),
        "train_decisions": sum(y_train),
        "test_decisions": sum(int(item["label"]) for item in test_items),
        "train_match_count": len({item["match_id"] for item in train_items}),
        "test_match_count": len({item["match_id"] for item in test_items}),
        "model_out": str(model_path),
        "metrics_out": str(metrics_path),
        "player_filter": args.player_filter,
        "request_kind": args.request_kind,
        "feature_count": len(fan_feature_names()),
        "mean_sample_weight": float(sum(weights) / len(weights)),
        "positive_reward_scale": args.positive_reward_scale,
        "negative_reward_floor": args.negative_reward_floor,
        "nonpass_decision_weight": args.nonpass_decision_weight,
        "min_player_reward": getattr(args, "min_player_reward", None),
        "train_group_accuracy": group_accuracy(train_items, list(train_scores)),
    }

    if test_items:
        x_test = featurize_fan_candidates(test_items)
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
                "feature_mode": "numeric_fan_v1",
                "model": model,
                "feature_names": fan_feature_names(),
                "metrics": metrics,
            },
            out,
        )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--model-out", default="models/fan_rewarded_feature_ranker.pkl")
    parser.add_argument("--metrics-out", default="runs/fan_rewarded_feature_ranker_metrics.json")
    parser.add_argument("--request-kind", choices=["all", "draw", "reaction"], default="draw")
    parser.add_argument("--player-filter", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--positive-reward-scale", type=float, default=1.0)
    parser.add_argument("--negative-reward-floor", type=float, default=0.1)
    parser.add_argument("--nonpass-decision-weight", type=float, default=1.0)
    parser.add_argument("--min-player-reward", type=float, default=None)
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
