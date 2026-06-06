"""Supervised training entry point using the paper hyperparameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .actions import ACTION_TO_INDEX
from .model import TjongConfig, TjongNetwork
from .tensorize_botzone import TENSOR_ENCODING_VERSION, tensor_encoding_schema

CLAIM_ACTION_INDICES = tuple(ACTION_TO_INDEX[name] for name in ("CHOW", "PONG", "MINGKONG", "BUKONG", "ANKONG"))


def validate_tensor_encoding(data: dict, *, expected_version: str | None = None, path: Path | None = None) -> None:
    if expected_version is None:
        return
    schema = data.get("encoding_schema") or {}
    observed = schema.get("version")
    if observed != expected_version:
        location = f" in {path}" if path is not None else ""
        raise ValueError(
            f"tensor encoding version mismatch{location}: expected {expected_version!r}, observed {observed!r}"
        )


def load_tensor_payload(path: Path, *, expected_encoding_version: str | None = None) -> dict:
    data = torch.load(path, map_location="cpu")
    validate_tensor_encoding(data, expected_version=expected_encoding_version, path=path)
    return data


def tensor_dataset_from_payload(data: dict) -> TensorDataset:
    required = ["visible_tiles", "game_features", "action_label", "claim_label", "discard_label"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"dataset is missing keys: {missing}")
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
    )


def load_tensor_dataset(path: Path, *, expected_encoding_version: str | None = None) -> TensorDataset:
    return tensor_dataset_from_payload(load_tensor_payload(path, expected_encoding_version=expected_encoding_version))


def unpack_batch(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
    return inputs, (action_label, claim_label, discard_label)


def decision_masks(action_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    claim_actions = torch.tensor(CLAIM_ACTION_INDICES, device=action_label.device)
    claim_mask = (action_label.unsqueeze(-1) == claim_actions).any(dim=-1)
    discard_mask = action_label == ACTION_TO_INDEX["DISCARD"]
    return claim_mask, discard_mask


def supervised_loss(outputs: dict[str, torch.Tensor], labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    action_label, claim_label, discard_label = labels
    ce = nn.CrossEntropyLoss()
    loss = ce(outputs["action_logits"], action_label)
    claim_mask, discard_mask = decision_masks(action_label)
    if claim_mask.any():
        loss = loss + ce(outputs["claim_logits"][claim_mask], claim_label[claim_mask])
    if discard_mask.any():
        loss = loss + ce(outputs["discard_logits"][discard_mask], discard_label[discard_mask])
    return loss


def batch_metric_sums(
    outputs: dict[str, torch.Tensor],
    labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    action_label, claim_label, discard_label = labels
    claim_mask, discard_mask = decision_masks(action_label)
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    metrics = {
        "action_loss_sum": float(ce_sum(outputs["action_logits"], action_label).item()),
        "action_count": float(action_label.numel()),
        "action_correct": float((outputs["action_logits"].argmax(dim=-1) == action_label).sum().item()),
        "claim_loss_sum": 0.0,
        "claim_count": float(claim_mask.sum().item()),
        "claim_correct": 0.0,
        "discard_loss_sum": 0.0,
        "discard_count": float(discard_mask.sum().item()),
        "discard_correct": 0.0,
    }
    if claim_mask.any():
        metrics["claim_loss_sum"] = float(ce_sum(outputs["claim_logits"][claim_mask], claim_label[claim_mask]).item())
        metrics["claim_correct"] = float(
            (outputs["claim_logits"][claim_mask].argmax(dim=-1) == claim_label[claim_mask]).sum().item()
        )
    if discard_mask.any():
        metrics["discard_loss_sum"] = float(
            ce_sum(outputs["discard_logits"][discard_mask], discard_label[discard_mask]).item()
        )
        metrics["discard_correct"] = float(
            (outputs["discard_logits"][discard_mask].argmax(dim=-1) == discard_label[discard_mask]).sum().item()
        )
    return metrics


def merge_metric_sums(total: dict[str, float], batch: dict[str, float]) -> None:
    for key, value in batch.items():
        total[key] = total.get(key, 0.0) + float(value)


def finalize_metric_sums(metric_sums: dict[str, float]) -> dict[str, float | None]:
    action_count = metric_sums.get("action_count", 0.0)
    claim_count = metric_sums.get("claim_count", 0.0)
    discard_count = metric_sums.get("discard_count", 0.0)
    decision_count = action_count + claim_count + discard_count
    decision_loss_sum = (
        metric_sums.get("action_loss_sum", 0.0)
        + metric_sums.get("claim_loss_sum", 0.0)
        + metric_sums.get("discard_loss_sum", 0.0)
    )
    return {
        "decision_loss": decision_loss_sum / decision_count if decision_count else None,
        "action_loss": metric_sums.get("action_loss_sum", 0.0) / action_count if action_count else None,
        "action_accuracy": metric_sums.get("action_correct", 0.0) / action_count if action_count else None,
        "action_count": int(action_count),
        "claim_loss": metric_sums.get("claim_loss_sum", 0.0) / claim_count if claim_count else None,
        "claim_accuracy": metric_sums.get("claim_correct", 0.0) / claim_count if claim_count else None,
        "claim_count": int(claim_count),
        "discard_loss": metric_sums.get("discard_loss_sum", 0.0) / discard_count if discard_count else None,
        "discard_accuracy": metric_sums.get("discard_correct", 0.0) / discard_count if discard_count else None,
        "discard_count": int(discard_count),
    }


def evaluate_model(
    model: nn.Module,
    dataset: TensorDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, float | None]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    metric_sums: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            inputs, labels = unpack_batch(batch, device)
            outputs = model(**inputs)
            merge_metric_sums(metric_sums, batch_metric_sums(outputs, labels))
    return finalize_metric_sums(metric_sums)


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = TjongConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    base_model = TjongNetwork(config)
    model: nn.Module
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model).to(device)
    else:
        model = base_model.to(device)
    train_payload = load_tensor_payload(Path(args.train_pt), expected_encoding_version=args.require_encoding_version)
    encoding_schema = train_payload.get("encoding_schema") or tensor_encoding_schema()
    checkpoint_encoding_version = encoding_schema.get("version") or args.require_encoding_version
    dataset = tensor_dataset_from_payload(train_payload)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics = {
        "paper": "Tjong CIT2.12298",
        "optimizer": "Adam",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "required_encoding_version": args.require_encoding_version,
        "paper_tensor_encoding_version": TENSOR_ENCODING_VERSION,
        "checkpoint_encoding_version": checkpoint_encoding_version,
        "encoding_schema": encoding_schema,
        "model_parameters": base_model.parameter_count(),
        "data_parallel": isinstance(model, nn.DataParallel),
        "device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "epochs": [],
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        metric_sums: dict[str, float] = {}
        for batch in loader:
            inputs, labels = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            loss = supervised_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels[0].numel())
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            merge_metric_sums(metric_sums, batch_metric_sums(outputs, labels))
        epoch_metrics = {
            "epoch": epoch,
            "optimization_loss": total_loss / max(1, total),
            **finalize_metric_sums(metric_sums),
        }
        metrics["epochs"].append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
    if args.checkpoint_out:
        Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(
            {
                "model": state_dict,
                "config": config.__dict__,
                "metrics": metrics,
                "encoding_schema": encoding_schema,
                "tensor_encoding_version": checkpoint_encoding_version,
            },
            args.checkpoint_out,
        )
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pt", required=True)
    parser.add_argument("--checkpoint-out", default="models/tjong_supervised.pt")
    parser.add_argument("--metrics-out", default="runs/tjong_supervised_metrics.json")
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--require-encoding-version", default=None)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
