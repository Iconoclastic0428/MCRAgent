"""PPO primitives with the hyperparameters stated in the Tjong paper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PPOConfig:
    policy_clip: float = 0.2
    value_clip: float = 0.3
    grad_clip: float = 0.5
    entropy_coef: float = 0.0
    value_coef: float = 0.5


def clipped_policy_loss(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - float(clip), 1.0 + float(clip)) * advantage
    return -torch.min(unclipped, clipped).mean()


def clipped_value_loss(
    *,
    value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    value_clipped = old_value + torch.clamp(value - old_value, -float(clip), float(clip))
    loss_unclipped = (value - returns).pow(2)
    loss_clipped = (value_clipped - returns).pow(2)
    return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()


def ppo_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor | None = None,
    config: PPOConfig | None = None,
) -> dict[str, float]:
    cfg = config or PPOConfig()
    policy = clipped_policy_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantage=advantage,
        clip=cfg.policy_clip,
    )
    value_loss = clipped_value_loss(value=value, old_value=old_value, returns=returns, clip=cfg.value_clip)
    entropy_term = torch.zeros((), device=value.device)
    if entropy is not None:
        entropy_term = entropy.mean()
    loss = policy + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_term
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_term.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
    }
