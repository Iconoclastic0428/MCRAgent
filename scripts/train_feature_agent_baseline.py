#!/usr/bin/env python3
"""Train the feature-agent baseline model on compact converted shards."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, get_worker_info


def add_feature_agent_path(feature_agent_dir: Path):
    resolved = str(feature_agent_dir.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from feature import FeatureAgent  # noqa: PLC0415
    from model import SelfVecModel  # noqa: PLC0415

    return FeatureAgent, SelfVecModel


class FeatureAgentVecDataset(Dataset):
    def __init__(self, index_path: Path, *, max_examples: int | None = None) -> None:
        self.index_path = index_path
        self.root = index_path.parent
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        obs_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        vec_parts: list[np.ndarray] = []
        act_parts: list[np.ndarray] = []
        remaining = max_examples
        for shard in self.index.get("shards") or []:
            if remaining is not None and remaining <= 0:
                break
            data = np.load(self.root / shard["path"])
            take = int(data["act"].shape[0])
            if remaining is not None:
                take = min(take, remaining)
            obs_parts.append(data["obs"][:take])
            mask_parts.append(data["mask"][:take])
            vec_parts.append(data["vec"][:take])
            act_parts.append(data["act"][:take])
            if remaining is not None:
                remaining -= take
        if not act_parts:
            raise ValueError(f"no examples found in {index_path}")
        self.obs = np.concatenate(obs_parts, axis=0)
        self.mask = np.concatenate(mask_parts, axis=0)
        self.vec = np.concatenate(vec_parts, axis=0)
        self.act = np.concatenate(act_parts, axis=0).astype(np.int64)

    def __len__(self) -> int:
        return int(self.act.shape[0])

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.obs[index]),
            torch.from_numpy(self.mask[index]),
            torch.from_numpy(self.vec[index]),
            torch.tensor(int(self.act[index]), dtype=torch.long),
        )


class StreamingFeatureAgentVecDataset(IterableDataset):
    """Load one compact shard at a time so full-data training does not pin RAM."""

    def __init__(
        self,
        index_path: Path,
        *,
        start: int,
        end: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.index_path = index_path
        self.root = index_path.parent
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.shards: list[dict[str, int | str]] = []

        cursor = 0
        for shard in self.index.get("shards") or []:
            count = int(shard["examples"])
            shard_start = cursor
            shard_end = cursor + count
            local_start = max(int(start), shard_start) - shard_start
            local_end = min(int(end), shard_end) - shard_start
            if local_start < local_end:
                self.shards.append(
                    {
                        "path": str(shard["path"]),
                        "start": int(local_start),
                        "end": int(local_end),
                    }
                )
            cursor = shard_end
        self.length = sum(int(shard["end"]) - int(shard["start"]) for shard in self.shards)

    def __len__(self) -> int:
        return int(self.length)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker = get_worker_info()
        shard_indices = list(range(len(self.shards)))
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(shard_indices)
        if worker is not None:
            shard_indices = shard_indices[worker.id :: worker.num_workers]

        for shard_index in shard_indices:
            shard = self.shards[shard_index]
            with np.load(self.root / str(shard["path"])) as data:
                obs = data["obs"]
                mask = data["mask"]
                vec = data["vec"]
                act = data["act"]
            local_indices = np.arange(int(shard["start"]), int(shard["end"]))
            if self.shuffle:
                local_rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + shard_index)
                local_rng.shuffle(local_indices)
            for index in local_indices:
                yield (
                    torch.from_numpy(obs[index]),
                    torch.from_numpy(mask[index]),
                    torch.from_numpy(vec[index]),
                    torch.tensor(int(act[index]), dtype=torch.long),
                )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def action_family_map(FeatureAgent) -> list[str]:
    agent = FeatureAgent(0)
    families = []
    for action in range(agent.ACT_SIZE):
        response = agent.action2response(action)
        head = response.split()[0] if response else "Pass"
        if head == "Pass":
            families.append("PASS")
        elif head == "Hu":
            families.append("HU")
        elif head == "Play":
            families.append("PLAY")
        elif head == "Gang":
            families.append("GANG")
        elif head == "AnGang":
            families.append("ANGANG")
        elif head == "BuGang":
            families.append("BUGANG")
        elif head == "Peng":
            families.append("PENG")
        elif head == "Chi":
            families.append("CHI")
        else:
            families.append(head.upper())
    return families


def split_dataset(dataset: Dataset, split_ratio: float) -> tuple[Subset, Subset]:
    split = int(len(dataset) * float(split_ratio))
    split = max(1, min(split, len(dataset) - 1))
    return Subset(dataset, range(0, split)), Subset(dataset, range(split, len(dataset)))


def index_example_count(index_path: Path, max_examples: int | None) -> int:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    total = int(index.get("examples") or sum(int(item["examples"]) for item in index.get("shards") or []))
    if max_examples is not None:
        total = min(total, int(max_examples))
    if total < 2:
        raise ValueError(f"need at least two examples in {index_path}, got {total}")
    return total


def build_datasets(args: argparse.Namespace):
    index_path = Path(args.index)
    total = index_example_count(index_path, args.max_examples)
    if args.load_mode == "memory":
        dataset = FeatureAgentVecDataset(index_path, max_examples=args.max_examples)
        train_dataset, val_dataset = split_dataset(dataset, args.split_ratio)
        return dataset, train_dataset, val_dataset

    split = int(total * float(args.split_ratio))
    split = max(1, min(split, total - 1))
    train_dataset = StreamingFeatureAgentVecDataset(
        index_path,
        start=0,
        end=split,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset = StreamingFeatureAgentVecDataset(
        index_path,
        start=split,
        end=total,
        shuffle=False,
        seed=args.seed,
    )
    return None, train_dataset, val_dataset


def run_epoch(
    *,
    model,
    loader: DataLoader,
    device: torch.device,
    optimizer=None,
    max_batches: int | None = None,
    families: list[str] | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    correct = 0
    total = 0
    family_counts: dict[str, int] = {}
    family_correct: dict[str, int] = {}
    for batch_index, batch in enumerate(loader, start=1):
        obs, mask, vec, target = batch
        input_dict = {
            "is_training": training,
            "obs": {
                "observation": obs.to(device),
                "action_mask": mask.to(device),
                "vec": vec.to(device),
            },
        }
        target = target.to(device)
        with torch.set_grad_enabled(training):
            logits = model(input_dict)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss).all():
                phase = "train" if training else "val"
                raise FloatingPointError(f"non-finite {phase} loss at batch {batch_index}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        pred = logits.argmax(dim=1)
        batch_total = int(target.numel())
        total_loss += float(loss.item()) * batch_total
        correct_mask = pred.eq(target)
        correct += int(correct_mask.sum().item())
        total += batch_total
        if families is not None:
            target_cpu = target.detach().cpu().tolist()
            correct_cpu = correct_mask.detach().cpu().tolist()
            for action, is_correct in zip(target_cpu, correct_cpu):
                family = families[int(action)]
                family_counts[family] = family_counts.get(family, 0) + 1
                if bool(is_correct):
                    family_correct[family] = family_correct.get(family, 0) + 1
        if max_batches is not None and batch_index >= max_batches:
            break
    breakdown = {
        family: {
            "count": int(count),
            "correct": int(family_correct.get(family, 0)),
            "accuracy": float(family_correct.get(family, 0) / count) if count else None,
        }
        for family, count in sorted(family_counts.items())
    }
    return {
        "loss": float(total_loss / total) if total else None,
        "accuracy": float(correct / total) if total else None,
        "examples": total,
        "action_breakdown": breakdown,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    FeatureAgent, SelfVecModel = add_feature_agent_path(Path(args.feature_agent_dir))
    dataset, train_dataset, val_dataset = build_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.load_mode == "memory",
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SelfVecModel(obs_dim=FeatureAgent.OBS_SIZE, vec_dim=FeatureAgent.VEC_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = out_dir / "metrics.jsonl"
    families = action_family_map(FeatureAgent)
    metrics: dict[str, Any] = {
        "format": "feature_agent_baseline_training_v1",
        "index": str(args.index),
        "feature_agent_dir": str(args.feature_agent_dir),
        "dataset_examples": len(dataset) if dataset is not None else len(train_dataset) + len(val_dataset),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "AdamW",
        "loss": "cross_entropy_235_way_model_masked_logits",
        "load_mode": args.load_mode,
        "epochs": [],
        "device": str(device),
        "seed": args.seed,
    }
    for epoch in range(args.epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if hasattr(val_dataset, "set_epoch"):
            val_dataset.set_epoch(epoch)
        checkpoint_path = out_dir / f"epoch_{epoch:03d}.pkl"
        torch.save(model.state_dict(), checkpoint_path)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=args.max_val_batches,
            families=families,
        )
        epoch_payload = {
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "train": train_metrics,
            "val": val_metrics,
        }
        metrics["epochs"].append(epoch_payload)
        with metrics_jsonl.open("a", encoding="utf-8") as dst:
            dst.write(json.dumps(epoch_payload, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps(epoch_payload, ensure_ascii=False, sort_keys=True), flush=True)
    final_path = out_dir / args.final_name
    torch.save(model.state_dict(), final_path)
    shutil.copy2(final_path, out_dir / "latest.pkl")
    metrics["final_checkpoint"] = str(final_path)
    metrics["latest_checkpoint"] = str(out_dir / "latest.pkl")
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--feature-agent-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--final-name", default="vec-fix-128-no-decay.pkl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=6088)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--load-mode", choices=["stream", "memory"], default="stream")
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
