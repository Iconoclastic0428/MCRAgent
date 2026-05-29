import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_fixed_league_feasibility import (  # noqa: E402
    infer_deal_in_player,
    summarize_feasibility,
)


def test_infer_deal_in_player_uses_unique_loser_penalty():
    result = {
        "scores": [-17, 33, -8, -8],
        "final_output": {"display": {"action": "HU", "player": 1}},
    }

    assert infer_deal_in_player(result) == 0


def test_infer_deal_in_player_returns_none_for_self_draw_like_equal_losers():
    result = {
        "scores": [-8, 24, -8, -8],
        "final_output": {"display": {"action": "HU", "player": 1}},
    }

    assert infer_deal_in_player(result) is None


def test_summarize_feasibility_reports_target_metrics_and_hu_safety():
    official_summary = {
        "policy": "transformer",
        "model": "models/baseline.pt",
        "qadv_model": "models/qadv.pt",
        "qadv_lambda": 0.05,
        "opponent": "sample",
        "average_scores": [5.0, -2.0, -2.0, -1.0],
        "policy_diagnostics_totals": [
            {
                "illegal_predictions": 2,
                "fan_check_rejects": 3,
                "fan_check_accepts": 1,
                "fan_check_errors": 0,
            },
            {"illegal_predictions": 1},
        ],
        "results": [
            {
                "turns": 10,
                "scores": [33, -17, -8, -8],
                "policy_diagnostics": [
                    {"illegal_predictions": 1, "fan_check_rejects": 2},
                    {},
                    {},
                    {},
                ],
                "final_output": {
                    "display": {
                        "action": "HU",
                        "player": 0,
                        "fanCnt": 8,
                    }
                },
            },
            {
                "turns": 22,
                "scores": [-17, 33, -8, -8],
                "policy_diagnostics": [
                    {"illegal_predictions": 1, "fan_check_accepts": 1},
                    {},
                    {},
                    {},
                ],
                "final_output": {
                    "display": {
                        "action": "HU",
                        "player": 1,
                        "fanCnt": 12,
                    }
                },
            },
            {
                "turns": 30,
                "scores": [0, 0, 0, 0],
                "policy_diagnostics": [
                    {"fan_check_rejects": 1},
                    {"illegal_predictions": 1},
                    {},
                    {},
                ],
                "final_output": {
                    "display": {
                        "action": "HUANG",
                        "canHu": [1, -4, 0, -4],
                    }
                },
            },
            {
                "turns": 18,
                "scores": [16, -8, -4, -4],
                "final_output": {
                    "display": {
                        "action": "HU",
                        "player": 0,
                        "fanCnt": 6,
                    }
                },
            },
        ],
    }

    summary = summarize_feasibility(official_summary, target_player=0)

    assert summary["format"] == "mcr_fixed_league_feasibility_v1"
    assert summary["source_qadv_model"] == "models/qadv.pt"
    assert summary["source_qadv_lambda"] == 0.05
    assert summary["games"] == 4
    assert summary["average_point_delta"] == 5.0
    assert summary["hu_rate"] == 3 / 4
    assert summary["target_hu_rate"] == 2 / 4
    assert summary["average_hu_turn"] == (10 + 22 + 18) / 3
    assert summary["target_average_hu_turn"] == 14.0
    assert summary["target_deal_in_count"] == 1
    assert summary["target_deal_in_rate"] == 1 / 4
    assert summary["end_wait_count"] == 2
    assert summary["target_end_wait_count"] == 1
    assert summary["target_end_wait_rate"] == 1 / 4
    assert summary["wait_when_deal_in_count"] == 0
    assert summary["illegal_prediction_count"] == 2
    assert summary["action_outside_legal_mask_count"] == 2
    assert summary["fan_check_reject_count"] == 3
    assert summary["low_fan_hu_count"] == 1
    assert summary["min_target_hu_fan"] == 6
    assert "inferred" in " ".join(summary["notes"]).lower()
