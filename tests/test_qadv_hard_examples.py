import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mine_chaga_hard_examples import (  # noqa: E402
    assert_qadv_candidate_digest,
    build_qadv_hard_example,
    qadv_accepted_norms,
    qadv_candidate_digest,
    validate_qadv_hard_example,
)


def _source_row(**overrides):
    row = {
        "example_key": "rec|0|12|2 W1",
        "record_id": "rec",
        "session_id": "sess",
        "turn": 12,
        "player": 0,
        "request": "2 W1",
        "response": "PLAY W4",
        "candidate_actions": [10, 11, 12, 13, 14],
        "candidate_norms": ["PLAY W1", "PLAY W2", "PLAY W3", "PLAY W4", "PASS"],
        "candidate_rule_features": [[0.0] * 7 for _ in range(5)],
        "base_logits": [0.1, 2.0, 1.4, 0.3, -0.5],
        "scalar_features": [0.0, 5.0 / 235.0, 0.0],
        "teacher_candidate_norms": ["PLAY W1", "PLAY W2", "PLAY W3", "PLAY W5", "PLAY W6"],
        "teacher_target_dist": [0.70, 0.20, 0.10, 0.0, 0.0],
        "teacher_accept_top3": True,
        "allow_hu": False,
        "has_teacher_distribution": True,
    }
    row.update(overrides)
    return row


def test_qadv_first_four_play_accepts_top3_and_fifth_play_requires_top1():
    candidates = ("PLAY W1", "PLAY W2", "PLAY W3", "PLAY W4")

    assert qadv_accepted_norms(candidates, play_ordinal=4, relaxed_play_top3_count=4) == candidates[:3]
    assert qadv_accepted_norms(candidates, play_ordinal=5, relaxed_play_top3_count=4) == candidates[:1]


def test_qadv_non_play_always_requires_top1():
    candidates = ("PENG", "PASS", "HU")

    assert qadv_accepted_norms(candidates, play_ordinal=1, relaxed_play_top3_count=4) == ("PENG",)


def test_qadv_baseline_wrong_top1_becomes_hard_negative_and_accepted_never_does():
    row = build_qadv_hard_example(_source_row())

    assert row["accepted_norms"] == ["PLAY W1", "PLAY W2", "PLAY W3"]
    assert 11 in row["accepted_action_ids"]
    assert 11 not in row["hard_negative_action_ids"]
    assert 13 in row["hard_negative_action_ids"]
    assert row["hard_negative_reasons"][str(13)] == ["base_top3_wrong"]
    validate_qadv_hard_example(row)


def test_qadv_late_top3_but_not_top1_is_hard_negative():
    row = build_qadv_hard_example(
        _source_row(
            teacher_accept_top3=False,
            base_logits=[0.1, 2.0, 1.4, 0.3, -0.5],
        )
    )

    assert row["accepted_norms"] == ["PLAY W1"]
    assert 11 in row["hard_negative_action_ids"]
    assert "chaga_rank_2_3_not_strict_accepted" in row["hard_negative_reasons"][str(11)]
    assert row["error_type"] == "top3_but_not_accepted_under_strict_rule"
    validate_qadv_hard_example(row)


def test_qadv_rejects_low_fan_hu_from_cached_candidates():
    row = build_qadv_hard_example(
        _source_row(
            candidate_actions=[1, 2, 3],
            candidate_norms=["HU", "PLAY W1", "PASS"],
            candidate_rule_features=[[0.0] * 7 for _ in range(3)],
            base_logits=[10.0, 0.2, 0.1],
            teacher_candidate_norms=["PLAY W1", "PASS"],
            teacher_target_dist=[0.0, 1.0, 0.0],
            teacher_accept_top3=False,
            allow_hu=False,
        )
    )

    assert "HU" not in row["candidate_norms"]
    assert row["candidate_actions"] == [2, 3]
    validate_qadv_hard_example(row)


def test_qadv_schema_round_trips_and_candidate_digest_is_checked():
    row = build_qadv_hard_example(_source_row())
    loaded = json.loads(json.dumps(row))

    assert loaded["schema"] == "qadv_hard_v1"
    assert loaded["candidate_digest"] == qadv_candidate_digest(
        loaded["candidate_actions"],
        loaded["candidate_norms"],
    )
    assert_qadv_candidate_digest(loaded)


def test_qadv_candidate_digest_mismatch_fails_training_join():
    row = build_qadv_hard_example(_source_row())
    row["candidate_actions"][0] = 99

    with pytest.raises(ValueError, match="candidate digest mismatch"):
        assert_qadv_candidate_digest(row)


def test_qadv_hard_example_requires_nonempty_accepted_set():
    row = build_qadv_hard_example(
        _source_row(
            teacher_candidate_norms=["PLAY T9"],
            teacher_target_dist=[0.0] * 5,
            teacher_accept_top3=False,
        )
    )

    with pytest.raises(ValueError, match="empty accepted"):
        validate_qadv_hard_example(row)
