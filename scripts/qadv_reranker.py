#!/usr/bin/env python3
"""Small Q-adversarial reranker for frozen Transformer candidate logits."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


QADV_FAMILY_IDS = {
    "PASS": 0,
    "HU": 1,
    "PLAY": 2,
    "CHI": 3,
    "PENG": 4,
    "GANG": 5,
    "BUGANG": 6,
    "ABANDON": 7,
}


def qadv_family_id(action_norm: str | None) -> int:
    if not action_norm:
        return 0
    return int(QADV_FAMILY_IDS.get(str(action_norm).strip().split()[0].upper(), 0))


class QAdvReranker(nn.Module):
    """Candidate-local Q scorer layered on top of frozen base logits."""

    def __init__(
        self,
        *,
        action_vocab_size: int,
        rule_feature_size: int = 7,
        scalar_feature_size: int = 3,
        family_vocab_size: int = 8,
        action_embedding_dim: int = 48,
        family_embedding_dim: int = 8,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(int(action_vocab_size), int(action_embedding_dim))
        self.family_embedding = nn.Embedding(int(family_vocab_size), int(family_embedding_dim))
        input_size = (
            int(action_embedding_dim)
            + int(family_embedding_dim)
            + int(rule_feature_size)
            + int(scalar_feature_size)
            + 2
        )
        layers: list[nn.Module] = []
        current = input_size
        for _ in range(max(1, int(num_layers) - 1)):
            layers.extend(
                [
                    nn.Linear(current, int(hidden_size)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current = int(hidden_size)
        layers.append(nn.Linear(current, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_actions = batch["candidate_actions"].long()
        candidate_family_ids = batch["candidate_family_ids"].long().clamp(0, self.family_embedding.num_embeddings - 1)
        candidate_mask = batch["candidate_mask"].bool()
        base_logits = batch["base_logits"].float()
        base_ranks = batch["base_ranks"].float()
        candidate_rule_features = batch["candidate_rule_features"].float()
        scalar_features = batch["scalar_features"].float()

        action_state = self.action_embedding(candidate_actions.clamp(0, self.action_embedding.num_embeddings - 1))
        family_state = self.family_embedding(candidate_family_ids)
        scalar_state = scalar_features[:, None, :].expand(-1, candidate_actions.shape[1], -1)
        base_rank_norm = base_ranks / max(1.0, float(candidate_actions.shape[1] - 1))
        features = torch.cat(
            [
                action_state,
                family_state,
                candidate_rule_features,
                scalar_state,
                base_logits.unsqueeze(-1),
                base_rank_norm.unsqueeze(-1),
            ],
            dim=-1,
        )
        q_scores = self.scorer(features).squeeze(-1)
        return q_scores.masked_fill(~candidate_mask, float("-inf"))


def _safe_candidate_mask(
    candidate_mask: torch.Tensor,
    *,
    candidate_is_hu: torch.Tensor | None = None,
    allow_hu: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = candidate_mask.bool().clone()
    if candidate_is_hu is not None and allow_hu is not None:
        hu_mask = candidate_is_hu.bool()
        allow = allow_hu.bool()
        while allow.dim() < hu_mask.dim():
            allow = allow.unsqueeze(-1)
        mask = mask & ~(hu_mask & ~allow)
    return mask


def qadv_final_scores(
    base_logits: torch.Tensor,
    q_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    lambda_q: float,
    candidate_is_hu: torch.Tensor | None = None,
    allow_hu: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = _safe_candidate_mask(candidate_mask, candidate_is_hu=candidate_is_hu, allow_hu=allow_hu)
    finite_q = torch.where(torch.isfinite(q_scores), q_scores, torch.zeros_like(q_scores))
    final = base_logits.float() + float(lambda_q) * finite_q.float()
    return final.masked_fill(~mask, float("-inf"))


def select_qadv_action(
    base_logits: torch.Tensor,
    q_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    lambda_q: float,
    candidate_is_hu: torch.Tensor | None = None,
    allow_hu: torch.Tensor | None = None,
) -> torch.Tensor:
    final = qadv_final_scores(
        base_logits,
        q_scores,
        candidate_mask,
        lambda_q=lambda_q,
        candidate_is_hu=candidate_is_hu,
        allow_hu=allow_hu,
    )
    return torch.argmax(final, dim=1)


def accepted_set_loss(
    final_scores: torch.Tensor,
    accepted_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    legal = candidate_mask.bool()
    accepted = accepted_mask.bool() & legal
    row_has = accepted.any(dim=1)
    if not bool(row_has.any().item()):
        return final_scores.new_tensor(0.0)
    legal_lse = torch.logsumexp(final_scores.masked_fill(~legal, float("-inf")), dim=1)
    accepted_lse = torch.logsumexp(final_scores.masked_fill(~accepted, float("-inf")), dim=1)
    return (legal_lse[row_has] - accepted_lse[row_has]).mean()


def hard_negative_pair_loss(
    q_scores: torch.Tensor,
    accepted_mask: torch.Tensor,
    hard_negative_mask: torch.Tensor,
    *,
    margin: float = 0.5,
) -> torch.Tensor:
    accepted = accepted_mask.bool()
    hard_negative = hard_negative_mask.bool()
    row_has = accepted.any(dim=1) & hard_negative.any(dim=1)
    if not bool(row_has.any().item()):
        return q_scores.new_tensor(0.0)
    positive = q_scores.masked_fill(~accepted, float("-inf")).amax(dim=1)
    negative = q_scores.masked_fill(~hard_negative, float("-inf")).amax(dim=1)
    return F.relu(float(margin) - positive[row_has] + negative[row_has]).mean()


def conservative_q_loss(
    q_scores: torch.Tensor,
    accepted_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    legal = candidate_mask.bool()
    accepted = accepted_mask.bool() & legal
    row_has = accepted.any(dim=1)
    if not bool(row_has.any().item()):
        return q_scores.new_tensor(0.0)
    finite_q = torch.where(torch.isfinite(q_scores), q_scores, torch.zeros_like(q_scores))
    legal_lse = torch.logsumexp(finite_q.masked_fill(~legal, float("-inf")), dim=1)
    accepted_mean = (finite_q * accepted.float()).sum(dim=1) / accepted.float().sum(dim=1).clamp_min(1.0)
    return (legal_lse[row_has] - accepted_mean[row_has]).mean()


def soft_chaga_loss(
    final_scores: torch.Tensor,
    teacher_dist: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    legal = candidate_mask.bool()
    teacher = teacher_dist.float().masked_fill(~legal, 0.0)
    row_total = teacher.sum(dim=1)
    row_has = row_total > 0.0
    if not bool(row_has.any().item()):
        return final_scores.new_tensor(0.0)
    teacher = teacher / row_total.clamp_min(1e-8)[:, None]
    log_probs = F.log_softmax(final_scores.masked_fill(~legal, float("-inf")), dim=1)
    safe_log_probs = torch.where(teacher > 0.0, log_probs, torch.zeros_like(log_probs))
    return -(teacher[row_has] * safe_log_probs[row_has]).sum(dim=1).mean()


def q_l2_loss(q_scores: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    legal = candidate_mask.bool()
    if not bool(legal.any().item()):
        return q_scores.new_tensor(0.0)
    finite_q = torch.where(torch.isfinite(q_scores), q_scores, torch.zeros_like(q_scores))
    return (finite_q[legal] ** 2).mean()


def qadv_total_loss(
    *,
    base_logits: torch.Tensor,
    q_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    accepted_mask: torch.Tensor,
    hard_negative_mask: torch.Tensor,
    teacher_dist: torch.Tensor,
    train_lambda: float = 1.0,
    margin: float = 0.5,
    accept_weight: float = 1.0,
    pair_weight: float = 0.5,
    cql_weight: float = 0.03,
    soft_weight: float = 0.05,
    q_l2_weight: float = 0.0001,
) -> tuple[torch.Tensor, dict[str, float]]:
    final_scores = qadv_final_scores(base_logits, q_scores, candidate_mask, lambda_q=train_lambda)
    losses = {
        "accepted_set_loss": accepted_set_loss(final_scores, accepted_mask, candidate_mask),
        "hard_negative_pair_loss": hard_negative_pair_loss(
            q_scores,
            accepted_mask,
            hard_negative_mask,
            margin=margin,
        ),
        "conservative_q_loss": conservative_q_loss(q_scores, accepted_mask, candidate_mask),
        "soft_chaga_loss": soft_chaga_loss(final_scores, teacher_dist, candidate_mask),
        "q_l2_loss": q_l2_loss(q_scores, candidate_mask),
    }
    total = (
        float(accept_weight) * losses["accepted_set_loss"]
        + float(pair_weight) * losses["hard_negative_pair_loss"]
        + float(cql_weight) * losses["conservative_q_loss"]
        + float(soft_weight) * losses["soft_chaga_loss"]
        + float(q_l2_weight) * losses["q_l2_loss"]
    )
    return total, {key: float(value.detach().cpu().item()) for key, value in losses.items()}
