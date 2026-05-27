#!/usr/bin/env python3
"""Train a legal response ranker for MCR Botzone policy decisions."""

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
from legal_actions import apply_response, generate_legal_responses, response_candidate_text
from policy_bot import split_tiles


def player_score(record: dict, player: str) -> int | float | None:
    scores = record.get("scores") or {}
    return scores.get(player)


def is_winning_player(record: dict, player: str) -> bool:
    scores = record.get("scores") or {}
    if not scores or player not in scores:
        return False
    best_score = max(scores.values())
    return scores[player] == best_score


def player_passes_outcome_filter(
    record: dict,
    player: str,
    min_player_score: float | None = None,
    winner_only: bool = False,
) -> bool:
    train_players = record.get("train_players")
    if train_players is not None and str(player) not in {str(item) for item in train_players}:
        return False
    score = player_score(record, player)
    if min_player_score is not None and (score is None or score < min_player_score):
        return False
    if winner_only and not is_winning_player(record, player):
        return False
    return True


def iter_legal_action_candidates(
    record: dict,
    min_player_score: float | None = None,
    winner_only: bool = False,
) -> Iterable[dict]:
    histories: dict[str, list[str]] = {str(player): [] for player in range(4)}
    hands: dict[str, Counter[str]] = {str(player): Counter() for player in range(4)}
    player_ids: dict[str, int | None] = {str(player): None for player in range(4)}
    meld_counts: dict[str, int] = {str(player): 0 for player in range(4)}

    logs = record.get("logs") or []
    for turn in range(0, len(logs) - 1, 2):
        requests = (logs[turn].get("output") or {}).get("content") or {}
        responses = logs[turn + 1]

        for player, request in requests.items():
            player = str(player)
            request = str(request)
            tokens = request.split()
            if tokens and tokens[0] == "0" and len(tokens) >= 2:
                player_ids[player] = int(tokens[1])
            elif tokens and tokens[0] == "1":
                hands[player].clear()
                hands[player].update(split_tiles(tokens[5:18]))
                meld_counts[player] = 0
            elif tokens and tokens[0] == "2" and len(tokens) >= 2:
                hands[player][tokens[1]] += 1

            actual = actual_response(responses, player)
            if actual is None:
                continue
            if not player_passes_outcome_filter(record, player, min_player_score, winner_only):
                continue
            candidates = generate_legal_responses(
                player_ids[player], request, hands[player], meld_count=meld_counts[player]
            )
            actual_forced_into_candidates = False
            if actual not in candidates:
                if actual == "HU":
                    candidates = [*candidates, "HU"]
                    actual_forced_into_candidates = True
                else:
                    continue
            score = player_score(record, player)
            input_text = "\n".join([*histories[player], f"REQ {request}"])
            for candidate in candidates:
                yield {
                    "match_id": record["match_id"],
                    "player": int(player),
                    "turn_index": turn // 2,
                    "request": request,
                    "input_text": input_text,
                    "candidate_response": candidate,
                    "actual_response": actual,
                    "label": 1 if candidate == actual else 0,
                    "hand": Counter(hands[player]),
                    "meld_count": meld_counts[player],
                    "actual_forced_into_candidates": (
                        actual_forced_into_candidates and candidate == actual
                    ),
                    "player_score": score,
                    "player_is_winner": is_winning_player(record, player),
                }

        for player, request in requests.items():
            player = str(player)
            request = str(request)
            actual = actual_response(responses, player)
            histories[player].append(f"REQ {request}")
            if actual is not None:
                histories[player].append(f"RES {actual}")
                apply_response(hands[player], request, actual)
                meld_counts[player] += meld_delta(request, actual)


def meld_delta(request: str, response: str) -> int:
    tokens = request.split()
    parts = response.strip().split()
    if not tokens or not parts:
        return 0
    action = parts[0].upper()
    if action in {"PENG", "CHI", "GANG"}:
        return 1
    if action == "BUGANG":
        return 0
    return 0


def load_candidates(
    path: Path,
    min_player_score: float | None = None,
    winner_only: bool = False,
) -> list[dict]:
    candidates: list[dict] = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                candidates.extend(
                    iter_legal_action_candidates(
                        json.loads(line),
                        min_player_score=min_player_score,
                        winner_only=winner_only,
                    )
                )
    return candidates


def request_kind_matches(request: str, request_kind: str) -> bool:
    tokens = request.split()
    if request_kind == "all":
        return True
    if request_kind == "draw":
        return bool(tokens and tokens[0] == "2")
    if request_kind == "reaction":
        return bool(tokens and tokens[0] == "3")
    raise ValueError(f"unknown request kind: {request_kind}")


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
                    max_iter=80,
                    tol=1e-4,
                    class_weight="balanced",
                    random_state=17,
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


def candidate_weight(item: dict, nonpass_decision_weight: float) -> float:
    actual_action = item["actual_response"].split()[0] if item.get("actual_response") else "PASS"
    if actual_action == "PASS":
        return 1.0
    return nonpass_decision_weight


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

    candidates = [
        item
        for item in load_candidates(
            raw_path,
            min_player_score=args.min_player_score,
            winner_only=args.winner_only,
        )
        if request_kind_matches(item["request"], args.request_kind)
    ]
    train_items, test_items = split_by_match_id(candidates, args.test_fraction)
    if not train_items:
        raise ValueError("no legal action candidates available")

    pipeline = build_pipeline(args.max_features)
    x_train = [candidate_text(item) for item in train_items]
    y_train = [item["label"] for item in train_items]
    sample_weight = [candidate_weight(item, args.nonpass_decision_weight) for item in train_items]
    pipeline.fit(x_train, y_train, clf__sample_weight=sample_weight)

    metrics = {
        "raw": str(raw_path),
        "candidates": len(candidates),
        "train_candidates": len(train_items),
        "test_candidates": len(test_items),
        "train_decisions": sum(y_train),
        "test_decisions": sum(item["label"] for item in test_items),
        "train_match_count": len({item["match_id"] for item in train_items}),
        "test_match_count": len({item["match_id"] for item in test_items}),
        "model_out": str(model_path),
        "metrics_out": str(metrics_path),
        "request_kind": args.request_kind,
        "nonpass_decision_weight": args.nonpass_decision_weight,
        "min_player_score": args.min_player_score,
        "winner_only": args.winner_only,
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
                "feature_mode": "legal_response_candidates_v1",
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
    parser.add_argument("--model-out", default="models/legal_action_ranker_160.pkl")
    parser.add_argument("--metrics-out", default="runs/legal_action_ranker_160_metrics.json")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=400_000)
    parser.add_argument("--request-kind", choices=["all", "draw", "reaction"], default="all")
    parser.add_argument("--nonpass-decision-weight", type=float, default=1.0)
    parser.add_argument("--min-player-score", type=float, default=None)
    parser.add_argument("--winner-only", action="store_true")
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
