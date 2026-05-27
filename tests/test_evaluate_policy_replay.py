import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_policy_replay import evaluate_records


class DrawnTilePredictor:
    def predict_response(self, input_text):
        return "PLAY W4"


def test_evaluate_records_counts_exact_action_and_legal_draw_predictions():
    record = {
        "match_id": "m1",
        "logs": [
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
    }

    metrics = evaluate_records([record], predictor=DrawnTilePredictor())

    assert metrics["examples"] == 2
    assert metrics["exact_matches"] == 2
    assert metrics["action_type_matches"] == 2
    assert metrics["active_draw_examples"] == 1
    assert metrics["active_draw_exact_matches"] == 1
    assert metrics["illegal_predictions"] == 0


def test_evaluate_records_counts_reaction_metrics():
    record = {
        "match_id": "m1",
        "logs": [
            {"output": {"content": {"0": "0 0 3"}}},
            {"0": {"response": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3"
                    }
                }
            },
            {"0": {"response": "PASS"}},
            {"output": {"content": {"0": "3 1 PLAY W1"}}},
            {"0": {"response": "PENG B1"}},
        ],
    }

    class PengPredictor:
        def predict_response(self, input_text):
            return "PENG B1"

    metrics = evaluate_records([record], predictor=PengPredictor())

    assert metrics["reaction_examples"] == 1
    assert metrics["reaction_nonpass_actual"] == 1
    assert metrics["reaction_nonpass_predicted"] == 1
    assert metrics["reaction_nonpass_exact_matches"] == 1
    assert metrics["reaction_nonpass_precision"] == 1
    assert metrics["reaction_nonpass_recall"] == 1
    assert metrics["reaction_exact_matches"] == 1
