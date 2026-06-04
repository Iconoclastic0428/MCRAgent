import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_qadv_terminal_trajectories import (  # noqa: E402
    QADV_TERMINAL_SCHEMA,
    build_qadv_terminal_row,
    summarize_terminal_rows,
    validate_qadv_terminal_row,
)
from mine_chaga_hard_examples import qadv_candidate_digest  # noqa: E402


def _prediction_row(**overrides):
    row = {
        "example_key": "rec-1|0|12|2 W1",
        "record_id": "rec-1",
        "session_id": "sess-1",
        "turn": 12,
        "player": 0,
        "request": "2 W1",
        "response": "PLAY W1",
        "play_ordinal": 5,
        "allow_hu": False,
        "candidate_actions": [10, 11, 12],
        "candidate_norms": ["PLAY W1", "PLAY W2", "PASS"],
        "candidate_rule_features": [[0.0] * 7 for _ in range(3)],
        "base_logits": [0.4, 1.2, -0.2],
        "base_ranks": [1.0, 0.0, 2.0],
        "scalar_features": [0.0, 3.0 / 235.0, 0.0],
        "teacher_candidate_norms": ["PLAY W1", "PLAY W2"],
        "teacher_target_dist": [0.9, 0.1, 0.0],
        "teacher_accept_top3": False,
        "accepted_norms": ["PLAY W1"],
        "accepted_action_ids": [10],
        "has_teacher_distribution": True,
        "candidate_truncated": False,
    }
    row.update(overrides)
    return row


def _terminal(**overrides):
    terminal = {
        "result_type": "HU",
        "winner": 0,
        "discarder": None,
        "is_self_draw": True,
        "final_turn": 80,
        "hu_turn": 80,
        "player_won": True,
        "player_dealt_in": False,
        "score_delta": 16,
        "placement_delta": 1,
        "end_wait": False,
        "wait_when_deal_in": False,
        "fan": 16,
        "base_fan": 16,
        "low_fan_hu": False,
    }
    terminal.update(overrides)
    return terminal


def test_terminal_row_schema_round_trips_and_digest_matches():
    row = build_qadv_terminal_row(_prediction_row(), _terminal())
    loaded = json.loads(json.dumps(row))

    assert loaded["schema_version"] == QADV_TERMINAL_SCHEMA
    assert loaded["candidate_digest"] == qadv_candidate_digest(
        loaded["legal"]["candidate_action_ids"],
        loaded["legal"]["candidate_norms"],
    )
    assert loaded["chosen"]["action_id"] == 10
    assert loaded["chosen"]["is_legal"] is True
    validate_qadv_terminal_row(loaded)


def test_terminal_return_is_nonzero_for_nonzero_score_delta():
    win_row = build_qadv_terminal_row(_prediction_row(), _terminal(score_delta=32, player_won=True))
    loss_row = build_qadv_terminal_row(
        _prediction_row(player=2, response="PLAY W2"),
        _terminal(winner=0, discarder=2, player_won=False, player_dealt_in=True, score_delta=-16),
    )

    assert win_row["return"]["point_delta_norm"] > 0
    assert win_row["return"]["terminal_return"] > 0
    assert loss_row["return"]["point_delta_norm"] < 0
    assert loss_row["return"]["terminal_return"] < 0


def test_low_fan_hu_terminal_fails_gate():
    row = build_qadv_terminal_row(
        _prediction_row(response="HU", allow_hu=True, candidate_actions=[1], candidate_norms=["HU"], base_logits=[5.0]),
        _terminal(result_type="HU", fan=4, base_fan=4, low_fan_hu=True),
    )

    with pytest.raises(ValueError, match="low-fan Hu"):
        validate_qadv_terminal_row(row)


def test_action_outside_mask_fails_gate():
    row = build_qadv_terminal_row(_prediction_row(response="PLAY W9"), _terminal())

    assert row["safety"]["action_outside_mask"] is True
    with pytest.raises(ValueError, match="outside legal mask"):
        validate_qadv_terminal_row(row)


def test_return_std_gate_rejects_all_zero_returns():
    rows = [
        build_qadv_terminal_row(
            _prediction_row(example_key=f"rec-{index}|0|2|2 W1", record_id=f"rec-{index}"),
            _terminal(result_type="HUANG", winner=None, hu_turn=None, final_turn=80, score_delta=0, player_won=False),
        )
        for index in range(4)
    ]

    summary = summarize_terminal_rows(rows, min_rows=4, min_games=4, min_return_std=0.03)

    assert summary["rows"] == 4
    assert summary["return_std"] == 0.0
    assert summary["gate_failures"]["return_std"] == 0.0
