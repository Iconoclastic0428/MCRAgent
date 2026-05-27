import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_adversarial_action_ranker import train


def test_train_adversarial_action_ranker_writes_generator_and_discriminator_metrics(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"
    records = [
        {
            "match_id": "m1",
            "scores": {"0": 16, "1": -16},
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
                {"0": {"response": "PLAY W1"}, "1": {"response": "PLAY B4"}},
            ],
        },
        {
            "match_id": "m2",
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
                {"0": {"response": "PLAY W4"}, "1": {"response": "PLAY W4"}},
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
        max_features = 2000
        adversarial_strength = 2.0
        nonpass_decision_weight = 1.0

    metrics = train(Args())

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["feature_mode"] == "adversarial_outcome_weighted_v1"
    assert metrics["discriminator_items"] > 0
    assert metrics["generator_candidates"] > 0
    assert 0.0 <= metrics["mean_adversarial_weight"] <= 3.0
    with model_path.open("rb") as src:
        payload = pickle.load(src)
    assert payload["kind"] == "legal_action_ranker"
    assert payload["metrics"]["adversarial_strength"] == 2.0
