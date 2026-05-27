import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_chaga_training_corpus import (  # noqa: E402
    assert_session_disjoint,
    assert_train_val_audit_excludes_test,
    assert_target_attachment_clean,
    validate_reviewed_examples,
)
from train_transformer_candidate import TransformerExample, action_response  # noqa: E402


def action_id_for_response(response: str) -> int:
    normalized = response.upper()
    for action in range(235):
        if action_response(action).upper() == normalized:
            return action
    raise AssertionError(f"response not found: {response}")


def minimal_example(
    candidate_responses: list[str],
    *,
    target_response: str,
    teacher_candidate_norms: tuple[str, ...],
    teacher_accept_top3: bool = False,
    allow_hu: bool = False,
) -> TransformerExample:
    candidate_actions = [action_id_for_response(response) for response in candidate_responses]
    action_mask = np.zeros(235, dtype=np.int8)
    action_mask[candidate_actions] = 1
    target_action = action_id_for_response(target_response)
    action_mask[target_action] = 1
    return TransformerExample(
        obs=np.zeros((71, 4, 9), dtype=np.float32),
        action_mask=action_mask,
        act=target_action,
        player=0,
        turn=1,
        response=target_response,
        history_tokens=np.zeros((1,), dtype=np.int64),
        value_target=0.0,
        allow_hu=allow_hu,
        candidate_rule_features=np.zeros((235, 7), dtype=np.float32),
        teacher_action_distribution=np.ones((235,), dtype=np.float32) / 235.0,
        teacher_accept_top3=teacher_accept_top3,
        teacher_candidate_norms=teacher_candidate_norms,
    )


def test_corpus_gate_rejects_session_overlap():
    with pytest.raises(ValueError, match="session overlap"):
        assert_session_disjoint(
            {
                "train": {"s1", "s2"},
                "val": {"s2"},
                "test": {"s3"},
            }
        )


def test_corpus_gate_rejects_test_session_in_train_val_audit():
    with pytest.raises(ValueError, match="test sessions"):
        assert_train_val_audit_excludes_test({"s1", "s3"}, {"s3"})


def test_corpus_gate_rejects_unattached_audit_targets():
    with pytest.raises(ValueError, match="unattached_audit_targets"):
        assert_target_attachment_clean(
            "train",
            {
                "audit_target_entries": 2,
                "teacher_original_targets": 1,
                "teacher_targets": 1,
                "unattached_audit_targets": 1,
            },
        )


def test_corpus_gate_rejects_teacher_target_count_mismatch():
    with pytest.raises(ValueError, match="teacher_original_targets"):
        assert_target_attachment_clean(
            "train",
            {
                "audit_target_entries": 2,
                "teacher_original_targets": 2,
                "teacher_targets": 1,
                "unattached_audit_targets": 0,
            },
        )


def test_corpus_gate_rejects_original_target_without_teacher_distribution():
    example = minimal_example(
        ["Play W1"],
        target_response="Play W1",
        teacher_candidate_norms=("PLAY W1",),
    )
    example.teacher_action_distribution = None

    with pytest.raises(ValueError, match="without mapped teacher distribution"):
        validate_reviewed_examples("train", [example], max_candidates=235)


def test_corpus_gate_rejects_empty_accept_mask():
    example = minimal_example(
        ["Play W1"],
        target_response="Play W1",
        teacher_candidate_norms=("PLAY T9",),
    )

    with pytest.raises(ValueError, match="accepted candidate mask"):
        validate_reviewed_examples("train", [example], max_candidates=235)


def test_corpus_gate_rejects_top3_relaxation_for_non_play_top1():
    example = minimal_example(
        ["Pass"],
        target_response="Pass",
        teacher_candidate_norms=("PENG W1", "PENG W2", "PENG W3"),
        teacher_accept_top3=True,
    )

    with pytest.raises(ValueError, match="top-3 relaxation"):
        validate_reviewed_examples("train", [example], max_candidates=235)


def test_corpus_gate_rejects_hu_when_allow_hu_false():
    example = minimal_example(
        ["Pass", "Hu"],
        target_response="Pass",
        teacher_candidate_norms=("HU",),
        allow_hu=False,
    )

    with pytest.raises(ValueError, match="HU"):
        validate_reviewed_examples("train", [example], max_candidates=235)


def test_corpus_gate_accepts_valid_relaxed_play_row():
    example = minimal_example(
        ["Play W1", "Play W2", "Play W3"],
        target_response="Play W1",
        teacher_candidate_norms=("PLAY W1", "PLAY W2", "PLAY W3"),
        teacher_accept_top3=True,
    )

    summary = validate_reviewed_examples("train", [example], max_candidates=235)

    assert summary["reviewed_examples"] == 1
    assert summary["empty_accept_mask"] == 0
    assert summary.get("top3_relaxation_non_play", 0) == 0
