"""PPO training phase for the Tjong paper replication.

This entry point consumes rollout tensors produced by self-play. The paper does
not publish its rollout serialization, so this file defines the minimal tensor
schema needed to reproduce the stated PPO update exactly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .actions import ACTION_TO_INDEX
from .evaluate_supervised import validate_checkpoint_provenance
from .model import TjongConfig, TjongNetwork
from .ppo import PPOConfig, clipped_policy_loss, clipped_value_loss
from .train_supervised import (
    CLAIM_ACTION_INDICES,
    validate_tensor_encoding,
)


ROLLOUT_REQUIRED = [
    "visible_tiles",
    "game_features",
    "action_label",
    "claim_label",
    "discard_label",
    "old_log_prob",
    "advantage",
    "returns",
    "old_value",
]
PAPER_PPO_BATCH_SIZE = 1024
PAPER_PPO_LR = 1e-4
PAPER_POLICY_CLIP = 0.2
PAPER_VALUE_CLIP = 0.3
PAPER_GRAD_CLIP = 0.5


def write_metrics_file(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temp_path.replace(path)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def validate_resume_config(resume_payload: dict, config: TjongConfig) -> None:
    observed = resume_payload.get("config") or {}
    expected = config.__dict__
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if key not in observed or observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume checkpoint config mismatch: {json.dumps(mismatches, sort_keys=True)}")


def validate_paper_ppo_args(args: argparse.Namespace, ppo_config: PPOConfig) -> None:
    if not bool(getattr(args, "require_paper_config", False)):
        return
    expected = {
        "batch_size": PAPER_PPO_BATCH_SIZE,
        "lr": PAPER_PPO_LR,
        "policy_clip": PAPER_POLICY_CLIP,
        "value_clip": PAPER_VALUE_CLIP,
        "grad_clip": PAPER_GRAD_CLIP,
    }
    observed = {
        "batch_size": getattr(args, "batch_size", None),
        "lr": getattr(args, "lr", None),
        "policy_clip": ppo_config.policy_clip,
        "value_clip": ppo_config.value_clip,
        "grad_clip": ppo_config.grad_clip,
    }
    mismatches = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        raise ValueError(f"paper PPO training config mismatch: {json.dumps(mismatches, sort_keys=True)}")


def checkpoint_directory(args: argparse.Namespace) -> Path:
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if checkpoint_dir:
        return Path(checkpoint_dir)
    checkpoint_out = getattr(args, "checkpoint_out", None)
    if not checkpoint_out:
        raise ValueError("--checkpoint-dir is required when periodic checkpoints are enabled without --checkpoint-out")
    checkpoint_path = Path(checkpoint_out)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_checkpoints")


def load_rollout_payload(path: Path, *, expected_encoding_version: str | None = None) -> dict:
    data = torch.load(path, map_location="cpu")
    validate_tensor_encoding(data, expected_version=expected_encoding_version, path=path)
    return data


def validate_rollout_reward_source(
    payload: dict,
    *,
    required_source: str | None,
    path: Path | None = None,
) -> None:
    if not required_source:
        return
    summary = payload.get("rollout_summary") or {}
    observed = summary.get("actual_reward_source")
    if observed != required_source:
        location = f" in {path}" if path is not None else ""
        raise ValueError(
            f"rollout reward source mismatch{location}: expected {required_source!r}, observed {observed!r}"
        )


def rollout_dataset_from_payload(data: dict) -> TensorDataset:
    missing = [key for key in ROLLOUT_REQUIRED if key not in data]
    if missing:
        raise ValueError(f"rollout dataset is missing keys: {missing}")
    rewards = data.get("rewards", torch.zeros(data["visible_tiles"].shape[:2]))
    previous_actions = data.get("previous_actions", torch.zeros(data["visible_tiles"].shape[:2], dtype=torch.long))
    sub_visible_tiles = data.get("sub_visible_tiles", data["visible_tiles"])
    sub_game_features = data.get("sub_game_features", data["game_features"])
    sub_rewards = data.get("sub_rewards", rewards)
    sub_previous_actions = data.get("sub_previous_actions", previous_actions)
    hidden_tiles = data.get(
        "hidden_tiles",
        torch.zeros(data["visible_tiles"].shape[0], TjongConfig.hidden_tile_rows, TjongConfig.tile_types),
    )
    return TensorDataset(
        data["visible_tiles"].float(),
        data["game_features"].float(),
        rewards.float(),
        previous_actions.long(),
        sub_visible_tiles.float(),
        sub_game_features.float(),
        sub_rewards.float(),
        sub_previous_actions.long(),
        hidden_tiles.float(),
        data["action_label"].long(),
        data["claim_label"].long(),
        data["discard_label"].long(),
        data["old_log_prob"].float(),
        data["advantage"].float(),
        data["returns"].float(),
        data["old_value"].float(),
    )


def load_rollout_dataset(path: Path, *, expected_encoding_version: str | None = None) -> TensorDataset:
    return rollout_dataset_from_payload(load_rollout_payload(path, expected_encoding_version=expected_encoding_version))


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    expected_encoding_version: str | None = None,
    require_paper_config: bool = False,
) -> tuple[TjongNetwork, TjongConfig, dict]:
    payload = torch.load(path, map_location="cpu")
    validate_checkpoint_provenance(
        payload,
        expected_encoding_version=expected_encoding_version,
        require_paper_config=require_paper_config,
        path=path,
    )
    allowed = {field.name for field in fields(TjongConfig)}
    raw_config = payload.get("config", {}) if isinstance(payload, dict) else {}
    config = TjongConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    model = TjongNetwork(config)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    return model, config, payload if isinstance(payload, dict) else {}


def hierarchical_log_prob_and_entropy(
    outputs: dict[str, torch.Tensor],
    *,
    action_label: torch.Tensor,
    claim_label: torch.Tensor,
    discard_label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_dist = torch.distributions.Categorical(logits=outputs["action_logits"])
    log_prob = action_dist.log_prob(action_label)
    entropy = action_dist.entropy()

    claim_actions = torch.tensor(CLAIM_ACTION_INDICES, device=action_label.device)
    claim_mask = (action_label.unsqueeze(-1) == claim_actions).any(dim=-1)
    if claim_mask.any():
        claim_dist = torch.distributions.Categorical(logits=outputs["claim_logits"][claim_mask])
        log_prob = log_prob.clone()
        entropy = entropy.clone()
        log_prob[claim_mask] = log_prob[claim_mask] + claim_dist.log_prob(claim_label[claim_mask])
        entropy[claim_mask] = entropy[claim_mask] + claim_dist.entropy()

    discard_mask = action_label == ACTION_TO_INDEX["DISCARD"]
    if discard_mask.any():
        discard_dist = torch.distributions.Categorical(logits=outputs["discard_logits"][discard_mask])
        log_prob = log_prob.clone()
        entropy = entropy.clone()
        log_prob[discard_mask] = log_prob[discard_mask] + discard_dist.log_prob(discard_label[discard_mask])
        entropy[discard_mask] = entropy[discard_mask] + discard_dist.entropy()

    return log_prob, entropy


def unpack_rollout_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    (
        visible_tiles,
        game_features,
        rewards,
        previous_actions,
        sub_visible_tiles,
        sub_game_features,
        sub_rewards,
        sub_previous_actions,
        hidden_tiles,
        action_label,
        claim_label,
        discard_label,
        old_log_prob,
        advantage,
        returns,
        old_value,
    ) = [tensor.to(device) for tensor in batch]
    inputs = {
        "visible_tiles": visible_tiles,
        "game_features": game_features,
        "rewards": rewards,
        "previous_actions": previous_actions,
        "sub_visible_tiles": sub_visible_tiles,
        "sub_game_features": sub_game_features,
        "sub_rewards": sub_rewards,
        "sub_previous_actions": sub_previous_actions,
        "hidden_tiles": hidden_tiles,
    }
    targets = {
        "action_label": action_label,
        "claim_label": claim_label,
        "discard_label": discard_label,
        "old_log_prob": old_log_prob,
        "advantage": advantage,
        "returns": returns,
        "old_value": old_value,
    }
    return inputs, targets


def ppo_update_batch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    config: PPOConfig,
) -> dict[str, float]:
    outputs = model(**inputs)
    new_log_prob, entropy = hierarchical_log_prob_and_entropy(
        outputs,
        action_label=targets["action_label"],
        claim_label=targets["claim_label"],
        discard_label=targets["discard_label"],
    )
    value = outputs["value"]
    policy = clipped_policy_loss(
        new_log_prob=new_log_prob,
        old_log_prob=targets["old_log_prob"],
        advantage=targets["advantage"],
        clip=config.policy_clip,
    )
    value_loss = clipped_value_loss(
        value=value,
        old_value=targets["old_value"],
        returns=targets["returns"],
        clip=config.value_clip,
    )
    entropy_term = entropy.mean()
    loss = policy + config.value_coef * value_loss - config.entropy_coef * entropy_term
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    approx_kl = (targets["old_log_prob"] - new_log_prob.detach()).mean()
    clip_fraction = (
        (torch.exp(new_log_prob.detach() - targets["old_log_prob"]) - 1.0).abs() > config.policy_clip
    ).float().mean()
    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_term.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
    }


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_model, config, checkpoint_payload = load_checkpoint(
        Path(args.init_checkpoint),
        device,
        expected_encoding_version=args.require_encoding_version,
        require_paper_config=args.require_paper_config,
    )
    model: nn.Module
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model).to(device)
    else:
        model = base_model
    rollout_path = Path(args.rollout_pt)
    rollout_payload = load_rollout_payload(rollout_path, expected_encoding_version=args.require_encoding_version)
    validate_rollout_reward_source(
        rollout_payload,
        required_source=getattr(args, "require_rollout_reward_source", None),
        path=rollout_path,
    )
    encoding_schema = rollout_payload.get("encoding_schema") or checkpoint_payload.get("encoding_schema") or {}
    checkpoint_encoding_version = (
        encoding_schema.get("version")
        or checkpoint_payload.get("tensor_encoding_version")
        or args.require_encoding_version
    )
    dataset = rollout_dataset_from_payload(rollout_payload)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ppo_config = PPOConfig(
        policy_clip=args.policy_clip,
        value_clip=args.value_clip,
        grad_clip=args.grad_clip,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
    )
    validate_paper_ppo_args(args, ppo_config)
    start_epoch = 1
    metrics = {
        "paper": "Tjong CIT2.12298",
        "phase": "ppo",
        "init_checkpoint": str(args.init_checkpoint),
        "rollout_pt": str(args.rollout_pt),
        "optimizer": "Adam",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "required_encoding_version": args.require_encoding_version,
        "require_paper_config": args.require_paper_config,
        "required_rollout_reward_source": getattr(args, "require_rollout_reward_source", None),
        "rollout_summary": rollout_payload.get("rollout_summary") or {},
        "checkpoint_encoding_version": checkpoint_encoding_version,
        "encoding_schema": encoding_schema,
        "model_parameters": base_model.parameter_count(),
        "data_parallel": isinstance(model, nn.DataParallel),
        "device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "config": config.__dict__,
        "ppo": ppo_config.__dict__,
        "epochs": [],
    }
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if resume_checkpoint:
        resume_payload = torch.load(resume_checkpoint, map_location="cpu")
        validate_checkpoint_provenance(
            resume_payload,
            expected_encoding_version=args.require_encoding_version,
            require_paper_config=args.require_paper_config,
            path=Path(resume_checkpoint),
        )
        validate_resume_config(resume_payload, config)
        base_model.load_state_dict(resume_payload["model"])
        if resume_payload.get("optimizer"):
            optimizer.load_state_dict(resume_payload["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        metrics = resume_payload.get("metrics") or metrics
        metrics["resumed_from"] = str(resume_checkpoint)
        start_epoch = int(resume_payload.get("epoch", len(metrics.get("epochs", [])))) + 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        for batch in loader:
            inputs, targets = unpack_rollout_batch(batch, device)
            step_metrics = ppo_update_batch(model=model, optimizer=optimizer, inputs=inputs, targets=targets, config=ppo_config)
            batches += 1
            for key, value in step_metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
        epoch_metrics = {"epoch": epoch, "batches": batches}
        epoch_metrics.update({key: value / max(1, batches) for key, value in sums.items()})
        metrics["epochs"].append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
        metrics_jsonl = getattr(args, "metrics_jsonl", None)
        if metrics_jsonl:
            metrics_jsonl_path = Path(metrics_jsonl)
            metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_jsonl_path.open("a", encoding="utf-8") as dst:
                dst.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
        if args.metrics_out:
            write_metrics_file(Path(args.metrics_out), metrics)
        checkpoint_every_epochs = int(getattr(args, "checkpoint_every_epochs", 0) or 0)
        if checkpoint_every_epochs and epoch % checkpoint_every_epochs == 0:
            checkpoint_dir = checkpoint_directory(args)
            save_ppo_checkpoint(
                model=model,
                config=config,
                optimizer=optimizer,
                metrics=metrics,
                encoding_schema=encoding_schema,
                checkpoint_encoding_version=checkpoint_encoding_version,
                epoch=epoch,
                path=checkpoint_dir / f"epoch_{epoch:04d}.pt",
            )
            save_ppo_checkpoint(
                model=model,
                config=config,
                optimizer=optimizer,
                metrics=metrics,
                encoding_schema=encoding_schema,
                checkpoint_encoding_version=checkpoint_encoding_version,
                epoch=epoch,
                path=checkpoint_dir / "latest.pt",
            )
    if args.checkpoint_out:
        save_ppo_checkpoint(
            model=model,
            config=config,
            optimizer=optimizer,
            metrics=metrics,
            encoding_schema=encoding_schema,
            checkpoint_encoding_version=checkpoint_encoding_version,
            epoch=args.epochs,
            path=Path(args.checkpoint_out),
        )
    if args.metrics_out:
        write_metrics_file(Path(args.metrics_out), metrics)
    return metrics


def save_ppo_checkpoint(
    *,
    model: nn.Module,
    config: TjongConfig,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
    encoding_schema: dict,
    checkpoint_encoding_version: str | None,
    epoch: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "config": config.__dict__,
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "encoding_schema": encoding_schema,
            "tensor_encoding_version": checkpoint_encoding_version,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--rollout-pt", required=True)
    parser.add_argument("--checkpoint-out", default="models/tjong_ppo.pt")
    parser.add_argument("--metrics-out", default="runs/tjong_ppo_metrics.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--policy-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--require-paper-config", action="store_true")
    parser.add_argument("--require-rollout-reward-source", default=None)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--metrics-jsonl", default=None)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
