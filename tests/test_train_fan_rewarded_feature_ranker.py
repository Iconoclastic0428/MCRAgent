import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_fan_rewarded_feature_ranker import (
    fan_aware_player_reward,
    iter_fan_rewarded_candidates,
    load_games,
    train,
)


def test_fan_aware_player_reward_prefers_high_fan_wins_and_penalizes_loss():
    high_fan_win = {
        "scores": [64, -16, -16, -32],
        "finish": {"action": "HU", "winner": 0, "fan_count": 24},
    }
    low_fan_win = {
        "scores": [32, -8, -8, -16],
        "finish": {"action": "HU", "winner": 0, "fan_count": 8},
    }
    opponent_win = {
        "scores": [-16, 40, -8, -16],
        "finish": {"action": "HU", "winner": 1, "fan_count": 16},
    }
    draw = {"scores": [0, 0, 0, 0], "finish": {"action": "HUANG", "winner": None}}

    assert fan_aware_player_reward(high_fan_win, 0) > fan_aware_player_reward(low_fan_win, 0)
    assert fan_aware_player_reward(low_fan_win, 0) > fan_aware_player_reward(draw, 0)
    assert fan_aware_player_reward(opponent_win, 0) < 0.0


def test_iter_fan_rewarded_candidates_uses_fan_metadata_for_sample_weight():
    game = {
        "match_id": "g1",
        "scores": [48, -16, -16, -16],
        "finish": {"action": "HU", "winner": 0, "fan_count": 16},
        "trajectory": [
            {"player": 0, "request": "0 0 1", "response": "PASS"},
            {
                "player": 0,
                "request": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
                "response": "PASS",
            },
            {"player": 0, "request": "2 W4", "response": "PLAY W4"},
        ],
    }

    candidates = list(iter_fan_rewarded_candidates(game, player_filter=0, request_kind="draw"))
    positive = [item for item in candidates if item["label"] == 1]

    assert positive
    assert positive[0]["candidate_response"] == "PLAY W4"
    assert positive[0]["sample_weight"] > 1.0
    assert all(item["fan_count"] == 16 for item in candidates)


def test_iter_fan_rewarded_candidates_can_filter_low_reward_games():
    losing_game = {
        "match_id": "loss",
        "scores": [-16, 48, -16, -16],
        "finish": {"action": "HU", "winner": 1, "fan_count": 16},
        "trajectory": [
            {"player": 0, "request": "0 0 1", "response": "PASS"},
            {
                "player": 0,
                "request": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
                "response": "PASS",
            },
            {"player": 0, "request": "2 W4", "response": "PLAY W4"},
        ],
    }

    candidates = list(
        iter_fan_rewarded_candidates(
            losing_game,
            player_filter=0,
            request_kind="draw",
            min_player_reward=0.5,
        )
    )

    assert candidates == []


def test_load_games_accepts_utf8_bom(tmp_path):
    path = tmp_path / "trajectories.json"
    path.write_text('\ufeff{"results": [{"match_id": "g1"}]}', encoding="utf-8")

    assert load_games(path) == [{"match_id": "g1"}]


def test_train_fan_rewarded_feature_ranker_writes_numeric_fan_payload(tmp_path):
    trajectories_path = tmp_path / "trajectories.json"
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"
    payload = {
        "results": [
            {
                "match_id": "win",
                "scores": [48, -16, -16, -16],
                "finish": {"action": "HU", "winner": 0, "fan_count": 16},
                "trajectory": [
                    {"player": 0, "request": "0 0 1", "response": "PASS"},
                    {
                        "player": 0,
                        "request": (
                            "1 0 0 0 0 "
                            "W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
                        ),
                        "response": "PASS",
                    },
                    {"player": 0, "request": "2 W4", "response": "PLAY W4"},
                ],
            },
            {
                "match_id": "draw",
                "scores": [0, 0, 0, 0],
                "finish": {"action": "HUANG", "winner": None, "fan_count": 0},
                "trajectory": [
                    {"player": 0, "request": "0 0 1", "response": "PASS"},
                    {
                        "player": 0,
                        "request": (
                            "1 0 0 0 0 "
                            "W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
                        ),
                        "response": "PASS",
                    },
                    {"player": 0, "request": "2 F1", "response": "PLAY F1"},
                ],
            },
        ]
    }
    trajectories_path.write_text(json.dumps(payload), encoding="utf-8")

    class Args:
        trajectories = str(trajectories_path)
        model_out = str(model_path)
        metrics_out = str(metrics_path)
        request_kind = "draw"
        player_filter = 0
        test_fraction = 0.5
        max_iter = 2
        learning_rate = 0.1
        max_leaf_nodes = 3
        min_samples_leaf = 1
        nonpass_decision_weight = 1.0
        positive_reward_scale = 1.0
        negative_reward_floor = 0.1

    metrics = train(Args())

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["feature_mode"] == "numeric_fan_v1"
    assert metrics["mean_sample_weight"] > 0.0
    assert np.isfinite(metrics["train_group_accuracy"])
    with model_path.open("rb") as src:
        model_payload = pickle.load(src)
    assert model_payload["kind"] == "feature_action_ranker"
    assert model_payload["feature_mode"] == "numeric_fan_v1"
