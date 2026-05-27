import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fan_feature_ranker import (
    fan_feature_names,
    featurize_fan_candidate,
    featurize_fan_legal_responses,
)


def test_featurize_fan_candidate_adds_fan_potential_signals():
    item = {
        "request": "2 W9",
        "input_text": "REQ 2 W9",
        "candidate_response": "PLAY J1",
        "hand": {
            "W1": 2,
            "W2": 1,
            "W3": 1,
            "W4": 1,
            "W5": 1,
            "W6": 1,
            "W7": 1,
            "W8": 1,
            "W9": 1,
            "F1": 2,
            "J1": 1,
            "J2": 1,
        },
    }

    names = fan_feature_names()
    values = featurize_fan_candidate(item)

    assert len(values) == len(names)
    assert values[names.index("fan_after_max_suit_fraction")] > 0.6
    assert values[names.index("fan_after_honor_fraction")] > 0.0
    assert values[names.index("fan_after_terminal_honor_fraction")] > 0.0
    assert values[names.index("fan_after_pair_count")] >= 2 / 7
    assert values[names.index("fan_after_seven_pair_progress")] > 0.0


def test_featurize_fan_legal_responses_returns_numeric_matrix():
    hand = {
        tile: 1
        for tile in "W1 W2 W3 W4 W5 W6 W7 W8 W9 B1 B2 B3 F1 F1".split()
    }

    features = featurize_fan_legal_responses(
        input_text="REQ 2 W9",
        request="2 W9",
        hand=hand,
        responses=["PLAY F1", "PLAY W1"],
    )

    assert isinstance(features, np.ndarray)
    assert features.shape == (2, len(fan_feature_names()))
    max_suit_index = fan_feature_names().index("fan_after_max_suit_fraction")
    assert features[0, max_suit_index] > features[1, max_suit_index]
