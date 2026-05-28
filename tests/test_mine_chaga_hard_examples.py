import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mine_chaga_hard_examples import classify_prediction_row, hard_example_weight  # noqa: E402


def test_classify_prediction_row_accepts_relaxed_match():
    row = {
        "predicted_normalized": "PLAY W2",
        "chaga_top1_action": "PLAY W1",
        "chaga_top3_actions": "PLAY W1|PLAY W2|PLAY W3",
        "teacher_accept_top3": True,
        "top1_match": False,
        "top3_match": True,
        "relaxed_match": True,
    }

    assert classify_prediction_row(row) == "accepted"


def test_classify_prediction_row_separates_wrong_family_and_same_family_not_top5():
    wrong_family = {
        "predicted_normalized": "PENG",
        "chaga_top1_action": "PLAY W1",
        "chaga_top3_actions": "PLAY W1|PLAY W2|PLAY W3",
        "relaxed_match": False,
    }
    same_family = {
        "predicted_normalized": "PLAY B9",
        "chaga_top1_action": "PLAY W1",
        "chaga_top3_actions": "PLAY W1|PLAY W2|PLAY W3",
        "relaxed_match": False,
    }

    assert classify_prediction_row(wrong_family) == "wrong_family"
    assert classify_prediction_row(same_family) == "same_family_not_top5"
    assert hard_example_weight(same_family) > hard_example_weight(wrong_family)
