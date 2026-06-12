import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from legal_actions import apply_response, can_complete_hand, generate_legal_responses, response_candidate_text


def test_generate_reaction_candidates_for_peng_and_gang():
    hand = Counter("W1 W1 W1 B1".split())

    responses = generate_legal_responses(player_id=2, request="3 1 PLAY W1", hand=hand)

    assert "PASS" in responses
    assert "GANG" in responses
    assert "PENG B1" in responses


def test_generate_chi_candidates_only_for_next_player():
    hand = Counter("W1 W3 B1".split())

    next_player = generate_legal_responses(player_id=1, request="3 0 PLAY W2", hand=hand)
    non_next_player = generate_legal_responses(player_id=2, request="3 0 PLAY W2", hand=hand)

    assert "CHI W2 B1" in next_player
    assert all(not response.startswith("CHI") for response in non_next_player)


def test_generate_reaction_candidates_for_actor_own_discard_only_passes():
    hand = Counter("W1 W1 W1 B1".split())

    responses = generate_legal_responses(player_id=0, request="3 0 PLAY W1", hand=hand)

    assert responses == ["PASS"]


def test_apply_peng_response_removes_claim_tiles_and_discard():
    hand = Counter("W1 W1 B1 B2".split())

    apply_response(hand, request="3 0 PLAY W1", response="PENG B2")

    assert hand == Counter({"B1": 1})


def test_response_candidate_text_includes_action_and_hand_features():
    hand = Counter("W1 W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 F1 F2".split())

    text = response_candidate_text("REQ 2 F2", "PLAY F2", hand, request="2 F2")

    assert "RESP PLAY F2" in text
    assert "ACTION PLAY" in text
    assert "MIN_SHANTEN 0" in text


def test_can_complete_hand_accounts_for_exposed_melds():
    hand = Counter("B1 B2 B3 T2 T3 T4 F1".split())

    assert can_complete_hand(hand, "F1", meld_count=2)
    assert not can_complete_hand(hand, "F1", meld_count=0)


def test_generate_hu_candidate_for_greater_honors_and_knitted_tiles():
    hand = Counter("W1 W4 W7 B2 B5 B8 T3 F1 F2 F3 F4 J1 J2 J3".split())

    responses = generate_legal_responses(player_id=0, request="2 J3", hand=hand)

    assert "HU" in responses


def test_generate_hu_candidate_for_lesser_honors_and_knitted_tiles():
    hand = Counter("W1 W4 W7 B2 B5 B8 T3 T6 F1 F2 F3 F4 J1 J2".split())

    responses = generate_legal_responses(player_id=0, request="2 J2", hand=hand)

    assert "HU" in responses


def test_generate_hu_candidate_for_knitted_straight():
    hand = Counter("W1 W4 W7 B2 B5 B8 T3 T6 T9 W2 W3 W4 J1 J1".split())

    responses = generate_legal_responses(player_id=0, request="2 J1", hand=hand)

    assert "HU" in responses


def test_can_complete_knitted_straight_with_one_exposed_meld():
    hand = Counter("W1 W4 W7 B2 B5 B8 T3 T6 T9 J1".split())

    assert can_complete_hand(hand, "J1", meld_count=1)
