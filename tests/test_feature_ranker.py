import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fan_feature_ranker import fan_feature_names
from feature_ranker import feature_names, featurize_candidate
from train_feature_action_ranker import train


def test_featurize_candidate_marks_action_tiles_and_shanten():
    item = {
        "request": "2 W4",
        "input_text": "REQ 2 W4",
        "candidate_response": "PLAY W2",
        "hand": {
            tile: 1
            for tile in "W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 W4".split()
        },
    }

    names = feature_names()
    values = featurize_candidate(item)

    assert len(values) == len(names)
    assert values[names.index("request_draw")] == 1.0
    assert values[names.index("action_PLAY")] == 1.0
    assert values[names.index("discard_W2")] == 1.0
    assert values[names.index("drawn_W4")] == 1.0
    assert values[names.index("candidate_is_drawn")] == 0.0
    assert values[names.index("delta_min_shanten")] == (
        values[names.index("after_min_shanten")] - values[names.index("current_min_shanten")]
    )


def test_train_feature_action_ranker_writes_payload(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"
    records = [
        {
            "match_id": "m1",
            "logs": [
                {"output": {"content": {"0": "0 0 1"}}},
                {"0": {"response": "PASS"}},
                {
                    "output": {
                        "content": {
                            "0": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
                        }
                    }
                },
                {"0": {"response": "PASS"}},
                {"output": {"content": {"0": "2 W4"}}},
                {"0": {"response": "PLAY W4"}},
            ],
        },
        {
            "match_id": "m2",
            "logs": [
                {"output": {"content": {"0": "0 0 1"}}},
                {"0": {"response": "PASS"}},
                {
                    "output": {
                        "content": {
                            "0": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
                        }
                    }
                },
                {"0": {"response": "PASS"}},
                {"output": {"content": {"0": "2 F1"}}},
                {"0": {"response": "PLAY F1"}},
            ],
        },
    ]
    raw_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    class Args:
        raw = str(raw_path)
        model_out = str(model_path)
        metrics_out = str(metrics_path)
        request_kind = "draw"
        test_fraction = 0.5
        nonpass_decision_weight = 1.0
        max_iter = 2
        learning_rate = 0.1
        max_leaf_nodes = 3
        min_samples_leaf = 1

    metrics = train(Args())

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["feature_count"] == len(feature_names())
    assert metrics["candidates"] > 0
    assert np.isfinite(metrics["train_group_accuracy"])


def test_train_feature_action_ranker_can_filter_to_winning_players(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"
    records = [
        {
            "match_id": "m1",
            "scores": {"0": -8, "1": 8},
            "logs": [
                {"output": {"content": {"0": "0 0 1", "1": "0 1 1"}}},
                {"0": {"response": "PASS"}, "1": {"response": "PASS"}},
                {
                    "output": {
                        "content": {
                            "0": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
                            "1": "1 0 0 0 0 W4 W5 W6 B4 B5 B6 T4 T5 T6 F3 F4 J3 J1",
                        }
                    }
                },
                {"0": {"response": "PASS"}, "1": {"response": "PASS"}},
                {"output": {"content": {"0": "2 W4", "1": "2 B4"}}},
                {"0": {"response": "PLAY W4"}, "1": {"response": "PLAY B4"}},
            ],
        }
    ]
    raw_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    class Args:
        raw = str(raw_path)
        model_out = str(model_path)
        metrics_out = str(metrics_path)
        request_kind = "draw"
        test_fraction = 0.5
        nonpass_decision_weight = 1.0
        min_player_score = None
        winner_only = True
        max_iter = 2
        learning_rate = 0.1
        max_leaf_nodes = 3
        min_samples_leaf = 1

    metrics = train(Args())

    assert metrics["winner_only"] is True
    assert metrics["candidates"] > 0
    assert metrics["train_decisions"] == 1


def test_train_feature_action_ranker_can_write_numeric_fan_payload(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"
    records = [
        {
            "match_id": "m1",
            "logs": [
                {"output": {"content": {"0": "0 0 1"}}},
                {"0": {"response": "PASS"}},
                {
                    "output": {
                        "content": {
                            "0": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 F1 F2 J1 J2"
                        }
                    }
                },
                {"0": {"response": "PASS"}},
                {"output": {"content": {"0": "2 W9"}}},
                {"0": {"response": "PLAY F1"}},
            ],
        },
        {
            "match_id": "m2",
            "logs": [
                {"output": {"content": {"0": "0 0 1"}}},
                {"0": {"response": "PASS"}},
                {
                    "output": {
                        "content": {
                            "0": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
                        }
                    }
                },
                {"0": {"response": "PASS"}},
                {"output": {"content": {"0": "2 F1"}}},
                {"0": {"response": "PLAY F1"}},
            ],
        },
    ]
    raw_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    class Args:
        raw = str(raw_path)
        model_out = str(model_path)
        metrics_out = str(metrics_path)
        request_kind = "draw"
        test_fraction = 0.5
        nonpass_decision_weight = 1.0
        min_player_score = None
        winner_only = False
        feature_mode = "numeric_fan_v1"
        max_iter = 2
        learning_rate = 0.1
        max_leaf_nodes = 3
        min_samples_leaf = 1

    metrics = train(Args())

    assert metrics["feature_mode"] == "numeric_fan_v1"
    assert metrics["feature_count"] == len(fan_feature_names())
