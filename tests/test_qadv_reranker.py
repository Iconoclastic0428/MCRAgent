import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qadv_reranker import (  # noqa: E402
    QAdvReranker,
    accepted_set_loss,
    conservative_q_loss,
    hard_negative_pair_loss,
    qadv_final_scores,
    select_qadv_action,
    soft_chaga_loss,
)


def test_accepted_set_loss_is_lower_when_accepted_score_is_high():
    mask = torch.tensor([[True, True, True]])
    accepted = torch.tensor([[True, False, False]])
    good = accepted_set_loss(torch.tensor([[3.0, 0.0, -1.0]]), accepted, mask)
    bad = accepted_set_loss(torch.tensor([[-1.0, 3.0, 2.0]]), accepted, mask)

    assert float(good.item()) < float(bad.item())


def test_hard_negative_pair_loss_penalizes_negative_above_positive():
    accepted = torch.tensor([[True, False, False]])
    hard_negative = torch.tensor([[False, True, False]])
    zero = hard_negative_pair_loss(torch.tensor([[2.0, 0.0, 0.0]]), accepted, hard_negative, margin=0.5)
    positive = hard_negative_pair_loss(torch.tensor([[0.0, 2.0, 0.0]]), accepted, hard_negative, margin=0.5)

    assert float(zero.item()) == 0.0
    assert float(positive.item()) > 0.0


def test_conservative_q_loss_increases_for_large_unsupported_q():
    mask = torch.tensor([[True, True, True]])
    accepted = torch.tensor([[True, False, False]])
    modest = conservative_q_loss(torch.tensor([[1.0, 0.0, 0.0]]), accepted, mask)
    unsupported_large = conservative_q_loss(torch.tensor([[1.0, 8.0, 0.0]]), accepted, mask)

    assert float(unsupported_large.item()) > float(modest.item())


def test_lambda_zero_reproduces_base_prediction_exactly():
    base_logits = torch.tensor([[0.1, 3.0, 0.2]])
    q_scores = torch.tensor([[9.0, -10.0, 9.0]])
    mask = torch.tensor([[True, True, True]])

    assert select_qadv_action(base_logits, q_scores, mask, lambda_q=0.0).tolist() == [1]


def test_qadv_final_scores_never_select_masked_action():
    base_logits = torch.tensor([[0.1, 3.0, 0.2]])
    q_scores = torch.tensor([[0.0, 10.0, 0.0]])
    mask = torch.tensor([[True, False, True]])

    assert select_qadv_action(base_logits, q_scores, mask, lambda_q=1.0).tolist() == [2]
    assert torch.isneginf(qadv_final_scores(base_logits, q_scores, mask, lambda_q=1.0)[0, 1])


def test_qadv_final_scores_masks_low_fan_hu_even_if_candidate_is_present():
    base_logits = torch.tensor([[10.0, 0.0, 0.0]])
    q_scores = torch.tensor([[10.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True]])
    candidate_is_hu = torch.tensor([[True, False, False]])
    allow_hu = torch.tensor([False])

    assert select_qadv_action(
        base_logits,
        q_scores,
        mask,
        lambda_q=1.0,
        candidate_is_hu=candidate_is_hu,
        allow_hu=allow_hu,
    ).tolist() != [0]


def test_soft_chaga_loss_ignores_zero_weight_masked_candidates_without_nan():
    scores = torch.tensor([[1.0, 0.0, float("-inf")]])
    teacher = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False]])

    loss = soft_chaga_loss(scores, teacher, mask)

    assert torch.isfinite(loss)


def test_qadv_reranker_outputs_candidate_scores():
    model = QAdvReranker(action_vocab_size=32, rule_feature_size=7, scalar_feature_size=3, hidden_size=32)
    batch = {
        "candidate_actions": torch.tensor([[1, 2, 3]]),
        "candidate_family_ids": torch.tensor([[2, 2, 0]]),
        "candidate_rule_features": torch.zeros((1, 3, 7)),
        "base_logits": torch.tensor([[0.1, 0.2, 0.3]]),
        "base_ranks": torch.tensor([[2.0, 1.0, 0.0]]),
        "scalar_features": torch.tensor([[0.0, 3.0 / 235.0, 0.0]]),
        "candidate_mask": torch.tensor([[True, True, False]]),
    }

    scores = model(batch)

    assert scores.shape == (1, 3)
    assert torch.isneginf(scores[0, 2])
