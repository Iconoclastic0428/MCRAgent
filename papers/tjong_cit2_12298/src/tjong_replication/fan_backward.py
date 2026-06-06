"""Fan backward reward shaping from Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from .encoding import TILE_TYPES, tile_multiset


@dataclass(frozen=True)
class FanItem:
    score: float
    tiles: tuple[int, ...]


@dataclass
class PlayerStep:
    turn_id: int
    hand_private: torch.Tensor
    hand_claim: torch.Tensor
    action: str = ""
    tile: int | None = None


@dataclass
class PlayerTrajectory:
    steps: list[PlayerStep] = field(default_factory=list)
    tiles: list[int] = field(default_factory=list)
    claim_events: list[tuple[int, tuple[int, ...], int]] = field(default_factory=list)


def calculate_score(hand_private: torch.Tensor, hand_claim: torch.Tensor, fans: Iterable[FanItem]) -> torch.Tensor:
    score_type = torch.zeros(TILE_TYPES, dtype=torch.float32, device=hand_private.device)
    for fan in fans:
        if not fan.tiles:
            continue
        share = float(fan.score) / float(len(fan.tiles))
        for tile in fan.tiles:
            score_type[int(tile)] += share
    counts = hand_private.float() + hand_claim.float()
    return torch.where(counts > 0, score_type / counts.clamp_min(1.0), torch.zeros_like(score_type))


def winning_reward(hand_hu: torch.Tensor, hand_now: torch.Tensor, score: torch.Tensor) -> float:
    diff = torch.abs(hand_hu.float() - hand_now.float())
    return float(((hand_hu.float() - diff) * score.float()).sum().item())


def fan_backward(
    *,
    trajectories: dict[int, PlayerTrajectory],
    winner: int,
    loser: int | None,
    tjong: int,
    fans: Iterable[FanItem],
) -> list[float]:
    winner_traj = trajectories[int(winner)]
    tjong_traj = trajectories[int(tjong)]
    if not winner_traj.steps:
        return [0.0 for _ in tjong_traj.steps]
    tlast = winner_traj.tiles[-1] if winner_traj.tiles else None
    final_state = winner_traj.steps[-1]
    hand_private = final_state.hand_private.float()
    hand_claim = final_state.hand_claim.float()
    hand_hu = hand_private + hand_claim
    if tlast is not None:
        hand_hu = hand_hu + tile_multiset([tlast], device=hand_hu.device)
    score = calculate_score((hand_hu - hand_claim).clamp_min(0), hand_claim, fans)
    rewards = [0.0 for _ in tjong_traj.steps]

    if int(tjong) == int(winner):
        for index, step in enumerate(tjong_traj.steps):
            current = step.hand_private.float() + step.hand_claim.float()
            if step.tile is not None:
                current = current - tile_multiset([step.tile], device=current.device)
            rewards[index] = winning_reward(hand_hu, current, score)
        return rewards

    for turn_id, tiles, from_whom in winner_traj.claim_events:
        if int(from_whom) != int(tjong):
            continue
        penalty = -float((tile_multiset(tiles, device=score.device) * score).sum().item())
        for index, step in enumerate(tjong_traj.steps):
            if int(step.turn_id) == int(turn_id):
                rewards[index] += penalty
                break

    if loser is not None and int(tjong) == int(loser) and rewards and tlast is not None:
        rewards[-1] += -float((tile_multiset([tlast], device=score.device) * score).sum().item())
    return rewards
