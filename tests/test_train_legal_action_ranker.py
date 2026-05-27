import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_legal_action_ranker import iter_legal_action_candidates, request_kind_matches


def test_iter_legal_action_candidates_labels_peng_candidate():
    record = {
        "match_id": "m1",
        "logs": [
            {"output": {"content": {"2": "0 2 3"}}},
            {"2": {"response": "PASS"}},
            {
                "output": {
                    "content": {
                        "2": "1 0 0 0 0 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3"
                    }
                }
            },
            {"2": {"response": "PASS"}},
            {"output": {"content": {"2": "3 1 PLAY W1"}}},
            {"2": {"response": "PENG B1"}},
        ],
    }

    candidates = list(iter_legal_action_candidates(record))
    positive = [item for item in candidates if item["label"] == 1]

    assert any(item["candidate_response"] == "PENG B1" for item in positive)
    assert any(item["actual_response"] == "PENG B1" for item in candidates)
    assert any(item["candidate_response"] == "PASS" for item in candidates)
    assert all(item["match_id"] == "m1" for item in candidates)


def test_request_kind_matches_draw_and_reaction_modes():
    assert request_kind_matches("2 W1", "draw")
    assert not request_kind_matches("3 0 PLAY W1", "draw")
    assert request_kind_matches("3 0 PLAY W1", "reaction")
    assert not request_kind_matches("2 W1", "reaction")
    assert request_kind_matches("0 0 3", "all")


def test_iter_legal_action_candidates_can_filter_by_positive_final_score():
    record = {
        "match_id": "m2",
        "scores": {"0": 24, "1": -24},
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
            {"output": {"content": {"0": "2 W1", "1": "2 B4"}}},
            {"0": {"response": "PLAY W1"}, "1": {"response": "PLAY B4"}},
        ],
    }

    candidates = list(iter_legal_action_candidates(record, min_player_score=1))

    assert candidates
    assert {item["player"] for item in candidates} == {0}
    assert all(item["player_score"] == 24 for item in candidates)


def test_iter_legal_action_candidates_can_filter_to_winners():
    record = {
        "match_id": "m3",
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
            {"output": {"content": {"0": "2 W1", "1": "2 B4"}}},
            {"0": {"response": "PLAY W1"}, "1": {"response": "PLAY B4"}},
        ],
    }

    candidates = list(iter_legal_action_candidates(record, winner_only=True))

    assert candidates
    assert {item["player"] for item in candidates} == {1}
    assert all(item["player_score"] == 8 for item in candidates)


def test_iter_legal_action_candidates_keeps_hu_after_exposed_melds():
    record = {
        "match_id": "melded-hu",
        "scores": {"0": -8, "1": 24, "2": -8, "3": -8},
        "logs": [
            {"output": {"content": {"1": "0 1 0"}}},
            {"1": {"response": "PASS"}},
            {
                "output": {
                    "content": {
                        "1": "1 0 0 0 0 W1 W1 J1 W4 W5 T1 B1 B2 B3 T2 T3 T4 F1"
                    }
                }
            },
            {"1": {"response": "PASS"}},
            {"output": {"content": {"1": "3 0 PLAY W1"}}},
            {"1": {"response": "PENG J1"}},
            {"output": {"content": {"1": "3 1 PENG J1"}}},
            {"1": {"response": "PASS"}},
            {"output": {"content": {"1": "3 0 PLAY W3"}}},
            {"1": {"response": "CHI W4 T1"}},
            {"output": {"content": {"1": "3 1 CHI W4 T1"}}},
            {"1": {"response": "PASS"}},
            {"output": {"content": {"1": "3 0 PLAY F1"}}},
            {"1": {"response": "HU"}},
        ],
    }

    candidates = list(iter_legal_action_candidates(record))
    hu_candidates = [
        item
        for item in candidates
        if item["turn_index"] == 6 and item["candidate_response"] == "HU"
    ]

    assert hu_candidates
    assert hu_candidates[0]["label"] == 1
    assert hu_candidates[0]["meld_count"] == 2


def test_iter_legal_action_candidates_keeps_recorded_hu_when_shape_checker_misses_it():
    record = {
        "match_id": "special-hu",
        "logs": [
            {"output": {"content": {"0": "0 0 0"}}},
            {"0": {"response": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W3 W6 W9 T2 T5 T8 F1 F2 F3 F4 J1 J2 J3"
                    }
                }
            },
            {"0": {"response": "PASS"}},
            {"output": {"content": {"0": "3 3 PLAY B7"}}},
            {"0": {"response": "HU"}},
        ],
    }

    candidates = list(iter_legal_action_candidates(record))
    hu_candidates = [item for item in candidates if item["candidate_response"] == "HU"]

    assert hu_candidates
    assert hu_candidates[0]["label"] == 1
    assert hu_candidates[0]["actual_forced_into_candidates"]
