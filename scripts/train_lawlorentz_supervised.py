#!/usr/bin/env python3
"""Train the Lawlorentz CNNModel on local cooked `.npz` shards."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


LAWLORENTZ_DIR = Path(__file__).resolve().parents[1] / "external" / "Chinese-Standard-Mahjong-DRL"
if str(LAWLORENTZ_DIR) not in sys.path:
    sys.path.insert(0, str(LAWLORENTZ_DIR))

from model import CNNModel  # noqa: E402


class LawlorentzShardDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.cooked_dir = self.data_dir / "cooked_data_without0"
        if not self.cooked_dir.exists():
            self.cooked_dir = self.data_dir
        count_path = self.data_dir / "count.json"
        if not count_path.exists():
            raise FileNotFoundError(f"count.json not found under {self.data_dir}")
        self.counts = [int(value) for value in json.loads(count_path.read_text(encoding="utf-8"))]
        self.offsets: list[int] = []
        total = 0
        for count in self.counts:
            self.offsets.append(total)
            total += count
        self.total = total
        self.shards = [
            np.load(self.cooked_dir / f"{index}.npz", mmap_mode="r")
            for index in range(len(self.counts))
        ]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        shard_id = bisect_right(self.offsets, index) - 1
        sample_id = index - self.offsets[shard_id]
        shard = self.shards[shard_id]
        return shard["obs"][sample_id], shard["mask"][sample_id], int(shard["act"][sample_id])


def split_indices(
    size: int,
    *,
    val_ratio: float,
    seed: int,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[list[int], list[int]]:
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    val_size = int(size * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]
    return train_indices, val_indices


def evaluate(model: CNNModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    with torch.no_grad():
        for obs, mask, action in loader:
            obs = obs.to(device)
            mask = mask.to(device)
            action = action.long().to(device)
            logits, _ = model({"observation": obs, "action_mask": mask})
            loss = F.cross_entropy(logits, action)
            pred = torch.argmax(logits, dim=1)
            total += int(action.numel())
            correct += int((pred == action).sum().item())
            loss_total += float(loss.item()) * int(action.numel())
    return {
        "loss": loss_total / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "samples": total,
    }


def train(args: argparse.Namespace) -> dict:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = LawlorentzShardDataset(Path(args.data_dir))
    train_indices, val_indices = split_indices(
        len(dataset),
        val_ratio=args.val_ratio,
        seed=seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNNModel().to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics = {
        "data_dir": args.data_dir,
        "dataset_samples": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "started_at": int(time.time()),
        "history": [],
    }

    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        loss_total = 0.0
        for obs, mask, action in train_loader:
            obs = obs.to(device)
            mask = mask.to(device)
            action = action.long().to(device)
            logits, _ = model({"observation": obs, "action_mask": mask})
            loss = F.cross_entropy(logits, action)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            pred = torch.argmax(logits, dim=1)
            total += int(action.numel())
            correct += int((pred == action).sum().item())
            loss_total += float(loss.item()) * int(action.numel())
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": loss_total / total if total else 0.0,
            "train_accuracy": correct / total if total else 0.0,
            "train_samples": total,
        }
        if val_indices:
            epoch_metrics.update({f"val_{key}": value for key, value in evaluate(model, val_loader, device).items()})
        metrics["history"].append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    metrics["checkpoint"] = str(out_path)
    metrics["finished_at"] = int(time.time())
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", default=None)
    parser.add_argument("--load-model", default=None)
    args = parser.parse_args()

    metrics = train(args)
    printable = {key: value for key, value in metrics.items() if key != "history"}
    printable["last_epoch"] = metrics["history"][-1] if metrics["history"] else None
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
