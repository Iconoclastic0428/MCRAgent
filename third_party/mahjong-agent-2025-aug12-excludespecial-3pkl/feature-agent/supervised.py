#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from dataset import MahjongGBDataset
from feature import FeatureAgent
from model import SelfVecModel, SlideStyleModel
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


ACTION_CATEGORIES = (
    ("pass", FeatureAgent.OFFSET_ACT.Pass, FeatureAgent.OFFSET_ACT.Hu),
    ("hu", FeatureAgent.OFFSET_ACT.Hu, FeatureAgent.OFFSET_ACT.Play),
    ("discard", FeatureAgent.OFFSET_ACT.Play, FeatureAgent.OFFSET_ACT.Gang),
    ("kong", FeatureAgent.OFFSET_ACT.Gang, FeatureAgent.OFFSET_ACT.Peng),
    ("pong", FeatureAgent.OFFSET_ACT.Peng, FeatureAgent.OFFSET_ACT.Chi),
    ("chow", FeatureAgent.OFFSET_ACT.Chi, FeatureAgent.ACT_SIZE),
)

EXPECTED_OBS_SIZE = 85
EXPECTED_ACT_SIZE = 235
EXPECTED_BASE_VEC_SIZE = 117


class NoopWriter:
    def add_scalars(self, *args, **kwargs):
        pass

    def close(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train the feature-agent policy.")
    parser.add_argument("--data-folder", default="./data-vec")
    parser.add_argument("--version", default="vec-fix-act-128")
    parser.add_argument("--log-root", default="./log")
    parser.add_argument("--runs-root", default="./runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=6088)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--augment-mode", choices=("none", "all12"), default="none")
    parser.add_argument("--special-matches", default="")
    parser.add_argument("--exclude-special-matches", action="store_true")
    parser.add_argument("--fan-features-folder", default="")
    parser.add_argument("--model-kind", choices=("selfvec", "slide"), default="selfvec")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--slide-out-planes", type=int, default=8)
    parser.add_argument("--slide-vec-dim", type=int, default=78)
    parser.add_argument("--fc-hidden", type=int, default=256)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def empty_category_stats():
    return {name: {"correct": 0, "total": 0} for name, _, _ in ACTION_CATEGORIES}


def update_category_stats(stats, pred, target):
    correct_mask = pred.eq(target)
    for name, start, end in ACTION_CATEGORIES:
        category_mask = (target >= start) & (target < end)
        total = int(category_mask.sum().item())
        if total:
            stats[name]["total"] += total
            stats[name]["correct"] += int((correct_mask & category_mask).sum().item())


def category_accuracy(stats):
    out = {}
    for name, values in stats.items():
        total = values["total"]
        out[name] = None if total == 0 else values["correct"] / total
    return out


def category_totals(stats):
    return {name: values["total"] for name, values in stats.items()}


def assert_feature_layout(dataset, name):
    if FeatureAgent.OBS_SIZE != EXPECTED_OBS_SIZE:
        raise RuntimeError(
            f"FeatureAgent.OBS_SIZE={FeatureAgent.OBS_SIZE}, expected {EXPECTED_OBS_SIZE}. "
            "This run would not use the vec-fix 85-plane feature layout."
        )
    if FeatureAgent.ACT_SIZE != EXPECTED_ACT_SIZE:
        raise RuntimeError(
            f"FeatureAgent.ACT_SIZE={FeatureAgent.ACT_SIZE}, expected {EXPECTED_ACT_SIZE}."
        )
    if FeatureAgent.VEC_SIZE != EXPECTED_BASE_VEC_SIZE:
        raise RuntimeError(
            f"FeatureAgent.VEC_SIZE={FeatureAgent.VEC_SIZE}, expected {EXPECTED_BASE_VEC_SIZE}."
        )
    if len(dataset) == 0:
        raise RuntimeError(f"{name} dataset is empty")

    obs, mask, vec, _ = dataset[0]
    expected_obs_shape = (EXPECTED_OBS_SIZE, 4, 9)
    if tuple(obs.shape) != expected_obs_shape:
        raise RuntimeError(
            f"{name} observation shape is {tuple(obs.shape)}, expected {expected_obs_shape}. "
            "This looks like a stale/baseline vector dataset."
        )
    if int(mask.shape[0]) != EXPECTED_ACT_SIZE:
        raise RuntimeError(
            f"{name} action mask dim is {mask.shape[0]}, expected {EXPECTED_ACT_SIZE}."
        )
    if int(vec.shape[0]) != dataset.vec_size:
        raise RuntimeError(
            f"{name} vec dim is {vec.shape[0]}, expected dataset vec_size {dataset.vec_size}."
        )
    if float(obs[60:85].sum()) <= 0:
        raise RuntimeError(
            f"{name} observation has empty vec-fix tile attribute planes 60:85."
        )


def log_epoch_metrics(writer, phase, epoch, loss, accuracy, stats):
    category_acc = category_accuracy(stats)
    writer.add_scalars("Loss", {phase: loss}, epoch)
    writer.add_scalars("Accuracy", {phase: accuracy}, epoch)
    writer.add_scalars(
        f"Accuracy/{phase}_by_action",
        {name: value for name, value in category_acc.items() if value is not None},
        epoch,
    )
    print(
        "metrics "
        + json.dumps(
            {
                "phase": phase.lower(),
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "category_accuracy": category_acc,
                "category_total": category_totals(stats),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def evaluate_model(model, loader, device, desc, max_batches=0):
    model.eval()
    correct = 0
    total_loss = 0.0
    total = 0
    stats = empty_category_stats()
    pbar = tqdm(
        loader,
        desc=desc.ljust(20),
        bar_format="{l_bar}{bar:40}{r_bar}",
    )
    for batch_index, batch in enumerate(pbar):
        input_dict = {
            "is_training": False,
            "obs": {
                "observation": batch[0].to(device),
                "action_mask": batch[1].to(device),
                "vec": batch[2].to(device),
            },
        }
        target = batch[3].long().to(device)
        with torch.no_grad():
            logits = model(input_dict)
            loss = F.cross_entropy(logits, target)
            total_loss += loss.item() * batch[0].size(0)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, target).sum().item()
            total += batch[0].size(0)
            update_category_stats(stats, pred, target)
            pbar.set_postfix(acc=f"{correct / total:.4f}", loss=f"{total_loss / total:.4f}")

        if max_batches and batch_index + 1 >= max_batches:
            break

    return total_loss / total, correct / total, stats


def main():
    args = parse_args()
    set_seed(args.seed)

    validation_end = 1.0 - args.test_ratio
    if not 0 < args.split_ratio < validation_end <= 1.0:
        raise RuntimeError(
            f"Invalid split: split_ratio={args.split_ratio}, test_ratio={args.test_ratio}. "
            "Require 0 < split_ratio < 1 - test_ratio <= 1."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    logdir = os.path.join(args.log_root, args.version)
    os.makedirs(logdir, exist_ok=True)
    writer = (
        SummaryWriter(log_dir=os.path.join(args.runs_root, args.version))
        if SummaryWriter is not None
        else NoopWriter()
    )

    print("[Loading Train dataset]")
    train_dataset = MahjongGBDataset(
        args.data_folder,
        0,
        args.split_ratio,
        0,
        augment=not args.no_augment,
        lazy=args.lazy,
        augment_mode=args.augment_mode,
        special_matches_path=args.special_matches,
        exclude_special_matches=args.exclude_special_matches,
        fan_features_folder=args.fan_features_folder,
    )
    print("[Loading Validation dataset]")
    validate_dataset = MahjongGBDataset(
        args.data_folder,
        args.split_ratio,
        validation_end,
        0,
        augment=False,
        lazy=args.lazy,
        augment_mode="none",
        special_matches_path=args.special_matches,
        exclude_special_matches=args.exclude_special_matches,
        fan_features_folder=args.fan_features_folder,
    )
    test_dataset = None
    if args.test_ratio:
        print("[Loading Test dataset]")
        test_dataset = MahjongGBDataset(
            args.data_folder,
            validation_end,
            1,
            0,
            augment=False,
            lazy=args.lazy,
            augment_mode="none",
            special_matches_path=args.special_matches,
            exclude_special_matches=args.exclude_special_matches,
            fan_features_folder=args.fan_features_folder,
        )
    assert_feature_layout(train_dataset, "train")
    assert_feature_layout(validate_dataset, "validation")
    if test_dataset is not None:
        assert_feature_layout(test_dataset, "test")
    print(
        "dataset_summary "
        + json.dumps(
            {
                "act_size": FeatureAgent.ACT_SIZE,
                "train_base_matches": train_dataset.base_matches,
                "train_base_samples": train_dataset.base_samples,
                "train_effective_matches": train_dataset.matches,
                "train_effective_samples": len(train_dataset),
                "validation_matches": validate_dataset.base_matches,
                "validation_samples": len(validate_dataset),
                "test_matches": 0 if test_dataset is None else test_dataset.base_matches,
                "test_samples": 0 if test_dataset is None else len(test_dataset),
                "augment_mode": args.augment_mode,
                "base_vec_size": FeatureAgent.VEC_SIZE,
                "special_match_count": len(train_dataset.special_match_indices),
                "exclude_special_matches": args.exclude_special_matches,
                "train_excluded_special_matches": train_dataset.excluded_special_matches,
                "train_excluded_special_samples": train_dataset.excluded_special_samples,
                "validation_excluded_special_matches": validate_dataset.excluded_special_matches,
                "validation_excluded_special_samples": validate_dataset.excluded_special_samples,
                "test_excluded_special_matches": 0
                if test_dataset is None
                else test_dataset.excluded_special_matches,
                "test_excluded_special_samples": 0
                if test_dataset is None
                else test_dataset.excluded_special_samples,
                "fan_feature_dim": train_dataset.fan_feature_dim,
                "fc_hidden": args.fc_hidden,
                "hidden": args.hidden,
                "model_kind": args.model_kind,
                "num_blocks": args.num_blocks,
                "obs_size": FeatureAgent.OBS_SIZE,
                "slide_out_planes": args.slide_out_planes,
                "slide_vec_dim": args.slide_vec_dim,
                "vec_size": train_dataset.vec_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    vloader = DataLoader(
        dataset=validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    if args.model_kind == "slide":
        model = SlideStyleModel(
            obs_dim=FeatureAgent.OBS_SIZE,
            vec_dim=train_dataset.vec_size,
            hidden=args.hidden,
            num_blocks=args.num_blocks,
            out_planes=args.slide_out_planes,
            slide_vec_dim=args.slide_vec_dim,
            fc_hidden=args.fc_hidden,
        ).to(device)
    else:
        model = SelfVecModel(
            obs_dim=FeatureAgent.OBS_SIZE,
            vec_dim=train_dataset.vec_size,
            hidden=args.hidden,
            num_blocks=args.num_blocks,
        ).to(device)
    if args.data_parallel:
        if args.device != "cuda":
            raise RuntimeError("--data-parallel requires --device cuda")
        if torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two CUDA devices")
        model = torch.nn.DataParallel(model)
        print(f"[DataParallel] devices={torch.cuda.device_count()}", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Total number of parameters] {total_params}\n", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    training_start = time.time()
    epoch_durations = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"[Epoch {epoch}]", flush=True)
        state_dict = (
            model.module.state_dict()
            if isinstance(model, torch.nn.DataParallel)
            else model.state_dict()
        )
        torch.save(state_dict, os.path.join(logdir, f"{epoch}.pkl"))

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        train_category_stats = empty_category_stats()
        pbar = tqdm(
            loader,
            desc=f"Training Epoch {epoch}".ljust(20),
            bar_format="{l_bar:20}{bar:40}{r_bar}",
        )
        for batch_index, batch in enumerate(pbar):
            input_dict = {
                "is_training": True,
                "obs": {
                    "observation": batch[0].to(device),
                    "action_mask": batch[1].to(device),
                    "vec": batch[2].to(device),
                },
            }
            target = batch[3].long().to(device)
            logits = model(input_dict)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch[0].size(0)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, target).sum().item()
            total += batch[0].size(0)
            update_category_stats(train_category_stats, pred, target)
            pbar.set_postfix(acc=f"{correct / total:.4f}", loss=f"{total_loss / total:.4f}")

            if args.max_train_batches and batch_index + 1 >= args.max_train_batches:
                break

        avg_train_loss = total_loss / total
        train_acc = correct / total
        log_epoch_metrics(writer, "Train", epoch, avg_train_loss, train_acc, train_category_stats)

        avg_val_loss, val_acc, validation_category_stats = evaluate_model(
            model,
            vloader,
            device,
            f"Validation Epoch {epoch}",
            args.max_val_batches,
        )
        log_epoch_metrics(
            writer,
            "Validate",
            epoch,
            avg_val_loss,
            val_acc,
            validation_category_stats,
        )

        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        average_epoch_seconds = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = args.epochs - epoch - 1
        print(
            "timing "
            + json.dumps(
                {
                    "epoch": epoch,
                    "epoch_seconds": epoch_duration,
                    "average_epoch_seconds": average_epoch_seconds,
                    "elapsed_seconds": time.time() - training_start,
                    "eta_seconds": remaining_epochs * average_epoch_seconds,
                    "remaining_epochs": remaining_epochs,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    state_dict = (
        model.module.state_dict()
        if isinstance(model, torch.nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state_dict, os.path.join(logdir, "final.pkl"))
    if test_loader is not None:
        test_loss, test_acc, test_category_stats = evaluate_model(
            model,
            test_loader,
            device,
            "Test Final",
            args.max_test_batches,
        )
        log_epoch_metrics(writer, "Test", args.epochs - 1, test_loss, test_acc, test_category_stats)
    writer.close()


if __name__ == "__main__":
    main()
