import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_reward_weighted_policy import iter_reward_weighted_candidates


def test_iter_reward_weighted_candidates_uses_terminal_player_reward():
    game = {
        "match_id": "g1",
        "rewards": [1.0, -1.0, -1.0, -1.0],
        "trajectory": [
            {"player": 0, "request": "0 0 3", "response": "PASS"},
            {
                "player": 0,
                "request": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
                "response": "PASS",
            },
            {"player": 0, "request": "2 W4", "response": "PLAY W4"},
        ],
    }

    candidates = list(iter_reward_weighted_candidates(game))
    positive = [item for item in candidates if item["label"] == 1]

    assert any(item["candidate_response"] == "PLAY W4" for item in positive)
    assert all(item["sample_weight"] == 2.0 for item in candidates)
    assert all(item["turn_index"] == item["decision_index"] for item in candidates)


def test_iter_reward_weighted_candidates_can_filter_to_controlled_player():
    game = {
        "match_id": "g2",
        "rewards": [1.0, -1.0, -1.0, -1.0],
        "trajectory": [
            {"player": 0, "request": "0 0 3", "response": "PASS"},
            {"player": 1, "request": "0 1 3", "response": "PASS"},
            {
                "player": 0,
                "request": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
                "response": "PASS",
            },
            {
                "player": 1,
                "request": "1 0 0 0 0 W4 W5 W6 B4 B5 B6 T4 T5 T6 F3 F4 J3 J1",
                "response": "PASS",
            },
            {"player": 0, "request": "2 W4", "response": "PLAY W4"},
            {"player": 1, "request": "2 W7", "response": "PLAY W7"},
        ],
    }

    candidates = list(iter_reward_weighted_candidates(game, player_filter=0))

    assert candidates
    assert {item["player"] for item in candidates} == {0}


def test_iter_reward_weighted_candidates_can_filter_to_reactions():
    game = {
        "match_id": "g3",
        "rewards": [1.0, 0.0, 0.0, 0.0],
        "trajectory": [
            {"player": 0, "request": "0 0 3", "response": "PASS"},
            {
                "player": 0,
                "request": "1 0 0 0 0 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3",
                "response": "PASS",
            },
            {"player": 0, "request": "2 W4", "response": "PLAY W4"},
            {"player": 0, "request": "3 1 PLAY W1", "response": "PENG B1"},
        ],
    }

    candidates = list(
        iter_reward_weighted_candidates(game, player_filter=0, request_kind="reaction")
    )

    assert candidates
    assert {item["request"] for item in candidates} == {"3 1 PLAY W1"}
    assert any(item["candidate_response"] == "PENG B1" for item in candidates)
