"""Build PPO rollout tensors from Tjong decision tensors.

This is the bridge between rollout collection and PPO optimisation. It freezes
the old policy log-probability and old value estimate from a checkpoint, then
computes returns and advantages from fan-backward rewards when present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .evaluate_supervised import load_model
from .tensorize_botzone import metadata_column
from .train_ppo import hierarchical_log_prob_and_entropy
from .train_supervised import load_tensor_dataset, unpack_batch, validate_tensor_encoding


TENSOR_KEYS = [
    "visible_tiles",
    "game_features",
    "rewards",
    "previous_actions",
    "sub_visible_tiles",
    "sub_game_features",
    "sub_rewards",
    "sub_previous_actions",
    "hidden_tiles",
    "action_label",
    "claim_label",
    "discard_label",
    "value_target",
]


def build_rollout_tensors(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source = torch.load(args.tensor_pt, map_location="cpu")
    validate_tensor_encoding(source, expected_version=args.require_encoding_version, path=Path(args.tensor_pt))
    dataset = load_tensor_dataset(Path(args.tensor_pt), expected_encoding_version=args.require_encoding_version)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_model(
        Path(args.checkpoint),
        device,
        expected_encoding_version=args.require_encoding_version,
        require_paper_config=args.require_paper_config,
    )
    model.eval()

    old_log_probs: list[torch.Tensor] = []
    old_values: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = unpack_batch(batch, device)
            outputs = model(**inputs)
            log_prob, _ = hierarchical_log_prob_and_entropy(
                outputs,
                action_label=labels[0],
                claim_label=labels[1],
                discard_label=labels[2],
            )
            old_log_probs.append(log_prob.detach().cpu())
            old_values.append(outputs["value"].detach().cpu())

    rewards, actual_reward_source = rollout_rewards(
        source,
        reward_key=args.reward_key,
        terminal_scale=args.terminal_scale,
        require_reward_key=args.require_reward_key,
    )
    returns = reward_to_go(
        rewards,
        metadata=source.get("metadata") if isinstance(source, dict) else None,
        gamma=args.gamma,
    )
    old_value = torch.cat(old_values, dim=0).float()
    advantage = returns - old_value
    if args.normalize_advantage and advantage.numel() > 1:
        std = advantage.std(unbiased=False)
        if float(std) > 1e-8:
            advantage = (advantage - advantage.mean()) / std

    payload = {key: source[key] for key in TENSOR_KEYS if key in source}
    payload.update(
        {
            "old_log_prob": torch.cat(old_log_probs, dim=0).float(),
            "old_value": old_value,
            "returns": returns.float(),
            "advantage": advantage.float(),
            "rollout_reward": rewards.float(),
        }
    )
    if "metadata" in source:
        payload["metadata"] = source["metadata"]
    if "encoding_schema" in source:
        payload["encoding_schema"] = source["encoding_schema"]
    payload["rollout_summary"] = {
        "source_tensor": args.tensor_pt,
        "checkpoint": args.checkpoint,
        "reward_key": args.reward_key,
        "actual_reward_source": actual_reward_source,
        "terminal_scale": args.terminal_scale,
        "gamma": args.gamma,
        "normalize_advantage": args.normalize_advantage,
        "required_encoding_version": args.require_encoding_version,
        "require_paper_config": args.require_paper_config,
        "examples": int(payload["action_label"].shape[0]),
        "reward_mean": float(rewards.mean().item()) if rewards.numel() else None,
        "reward_std": float(rewards.std(unbiased=False).item()) if rewards.numel() > 1 else 0.0,
        "return_mean": float(returns.mean().item()) if returns.numel() else None,
        "advantage_mean": float(advantage.mean().item()) if advantage.numel() else None,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    summary = {**payload["rollout_summary"], "output": str(out_path)}
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def rollout_rewards(
    source: dict[str, Any],
    *,
    reward_key: str,
    terminal_scale: float,
    require_reward_key: bool = False,
) -> tuple[torch.Tensor, str]:
    if reward_key in source:
        reward = source[reward_key]
        if reward.ndim > 1:
            reward = reward[:, -1]
        return reward.float(), reward_key
    if require_reward_key:
        raise ValueError(f"required reward key {reward_key!r} is absent from rollout source")
    if "fan_backward_reward" in source:
        reward = source["fan_backward_reward"]
        if reward.ndim > 1:
            reward = reward[:, -1]
        return reward.float(), "fan_backward_reward"
    if "value_target" not in source:
        raise ValueError(
            f"reward key {reward_key!r} is absent and no fan_backward_reward/value_target fallback is available"
        )
    return source["value_target"].float() * float(terminal_scale), "value_target_terminal_fallback"


def reward_to_go(
    rewards: torch.Tensor,
    *,
    metadata: dict[str, Any] | None,
    gamma: float,
) -> torch.Tensor:
    rewards = rewards.float().reshape(-1)
    returns = torch.zeros_like(rewards)
    if not metadata:
        running = 0.0
        for index in range(rewards.numel() - 1, -1, -1):
            running = float(rewards[index]) + float(gamma) * running
            returns[index] = running
        return returns

    match_ids = metadata_column(metadata, "match_id")
    players = metadata_column(metadata, "player")
    if len(match_ids) != rewards.numel() or len(players) != rewards.numel():
        running = 0.0
        for index in range(rewards.numel() - 1, -1, -1):
            running = float(rewards[index]) + float(gamma) * running
            returns[index] = running
        return returns

    running_by_key: dict[tuple[str, int], float] = {}
    for index in range(rewards.numel() - 1, -1, -1):
        key = (str(match_ids[index]), int(players[index]))
        running_by_key[key] = float(rewards[index]) + float(gamma) * running_by_key.get(key, 0.0)
        returns[index] = running_by_key[key]
    return returns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-pt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--reward-key", default="fan_backward_reward")
    parser.add_argument("--require-reward-key", action="store_true")
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--require-paper-config", action="store_true")
    parser.add_argument("--terminal-scale", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--normalize-advantage", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    build_rollout_tensors(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
