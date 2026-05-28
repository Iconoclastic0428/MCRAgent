import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_algorithmic_chaga_review import (  # noqa: E402
    aggregate_algorithmic_candidate_metrics,
    choose_algorithmic_action,
    response_family,
)


def test_choose_algorithmic_action_prefers_discard_with_more_fan_valid_waits():
    action_mask = np.array([1, 1, 1], dtype=np.int8)
    features = np.zeros((3, 7), dtype=np.float32)
    features[1, 1] = 1.0
    features[1, 3] = 0.10
    features[2, 1] = 1.0
    features[2, 3] = 0.40

    action, response = choose_algorithmic_action(
        action_mask,
        features,
        allow_hu=False,
        action_to_response=lambda action: ["Pass", "Play W1", "Play W2"][action],
    )

    assert action == 2
    assert response == "Play W2"


def test_choose_algorithmic_action_never_chooses_hu_when_fan_gate_disallows_it():
    action_mask = np.array([1, 1, 1], dtype=np.int8)
    features = np.zeros((3, 7), dtype=np.float32)
    features[2, 1] = 1.0

    blocked_action, blocked_response = choose_algorithmic_action(
        action_mask,
        features,
        allow_hu=False,
        action_to_response=lambda action: ["Pass", "Hu", "Play W1"][action],
    )
    allowed_action, allowed_response = choose_algorithmic_action(
        action_mask,
        features,
        allow_hu=True,
        action_to_response=lambda action: ["Pass", "Hu", "Play W1"][action],
    )

    assert (blocked_action, blocked_response) == (2, "Play W1")
    assert (allowed_action, allowed_response) == (1, "Hu")


def test_response_family_normalizes_chi_peng_gang_hu_and_play():
    assert response_family("Play W1") == "PLAY"
    assert response_family("Chi W3") == "CHI"
    assert response_family("Peng") == "PENG"
    assert response_family("Gang") == "GANG"
    assert response_family("Hu") == "HU"
    assert response_family("Pass") == "PASS"


def test_aggregate_algorithmic_candidate_metrics_counts_family_slices():
    rows = [
        {"relaxed_match": True, "top1_match": False, "top3_match": True, "chaga_top1_action": "PLAY W1"},
        {"relaxed_match": False, "top1_match": False, "top3_match": False, "chaga_top1_action": "CHI W3"},
        {"relaxed_match": True, "top1_match": True, "top3_match": True, "chaga_top1_action": "HU"},
    ]

    metrics = aggregate_algorithmic_candidate_metrics(rows)

    assert metrics["original_samples"] == 3
    assert metrics["original_relaxed_accuracy"] == 2 / 3
    assert metrics["by_family"]["PLAY"]["samples"] == 1
    assert metrics["by_family"]["PLAY"]["relaxed_accuracy"] == 1.0
    assert metrics["by_family"]["CHI"]["relaxed_accuracy"] == 0.0
    assert metrics["by_family"]["HU"]["top1_accuracy"] == 1.0
