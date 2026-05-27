import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hand_features import (
    candidate_feature_text,
    hand_feature_text,
    min_shanten,
    regular_shanten,
    seven_pairs_shanten,
    thirteen_orphans_shanten,
)


def test_regular_shanten_detects_complete_and_tenpai_hands():
    complete = "W1 W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 F1 F1".split()
    tenpai = complete[:-1]

    assert regular_shanten(complete) == -1
    assert regular_shanten(tenpai) == 0


def test_seven_pairs_shanten_detects_complete_and_tenpai_hands():
    complete = "W1 W1 W2 W2 B1 B1 B2 B2 T1 T1 T2 T2 F1 F1".split()
    tenpai = complete[:-1]

    assert seven_pairs_shanten(complete) == -1
    assert seven_pairs_shanten(tenpai) == 0


def test_thirteen_orphans_shanten_detects_complete_and_tenpai_hands():
    complete = "W1 W9 B1 B9 T1 T9 F1 F2 F3 F4 J1 J2 J3 W1".split()
    tenpai = complete[:-1]

    assert thirteen_orphans_shanten(complete) == -1
    assert thirteen_orphans_shanten(tenpai) == 0


def test_hand_feature_text_scores_candidate_after_discard():
    hand = "W1 W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 F1 F2".split()

    features = hand_feature_text(hand, candidate_tile="F2")

    assert "CAND F2" in features
    assert "REG_SHANTEN 0" in features
    assert "MIN_SHANTEN 0" in features


def test_candidate_feature_text_marks_drawn_tile_and_delta():
    hand = "W1 W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 F1 F2".split()

    features = candidate_feature_text(hand, candidate_tile="F2", drawn_tile="F2")

    assert "DRAWN_TILE F2" in features
    assert "CAND_IS_DRAWN 1" in features
    assert "SHANTEN_DELTA_VS_DRAWN 0" in features
