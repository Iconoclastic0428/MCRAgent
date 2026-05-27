import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_chaga_review_alignment import (  # noqa: E402
    actual_action_from_review_row,
    build_session_api_seat_maps,
    normalized_action,
    relaxed_candidate_match,
)


def test_normalized_action_compares_discard_by_tile_type():
    assert normalized_action("Play F1") == normalized_action("PLAY F1")
    assert normalized_action("Peng J2") == "PENG"
    assert normalized_action("Peng") == "PENG"


def test_actual_action_from_review_row_uses_tile_type_for_play_and_pass_for_zero_claim():
    assert actual_action_from_review_row({"r": 2, "v": 0}) == "Play W1"
    assert actual_action_from_review_row({"r": 2, "v": 111}) == "Play F1"
    assert actual_action_from_review_row({"r": 4, "v": 0}) == "Pass"
    assert actual_action_from_review_row({"r": 4, "v": 129}) == "Peng"
    assert actual_action_from_review_row({"r": 6, "v": 135}) == "Hu"


def test_relaxed_candidate_match_counts_top3_only_for_first_six_discards():
    candidates = [
        [9.0, "Play W1"],
        [8.0, "Play W2"],
        [7.0, "Play W3"],
        [6.0, "Play W4"],
    ]

    assert relaxed_candidate_match("Play W3", candidates, play_ordinal=6)
    assert not relaxed_candidate_match("Play W3", candidates, play_ordinal=7)
    assert not relaxed_candidate_match("Play W4", candidates, play_ordinal=3)
    assert relaxed_candidate_match("Peng", [[1.0, "Peng"]], play_ordinal=None)


def test_build_session_api_seat_maps_uses_round_zero_order():
    records = [
        {
            "belongs": "s1",
            "step": {"i": 1, "p": [{"n": "Rotated"}]},
        },
        {
            "belongs": "s1",
            "step": {"i": 0, "p": [{"n": "A"}, {"n": "B"}]},
        },
    ]

    assert build_session_api_seat_maps(records) == {"s1": {"A": 0, "B": 1}}
