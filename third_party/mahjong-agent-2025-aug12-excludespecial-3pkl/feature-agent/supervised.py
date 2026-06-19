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
from dataset import MahjongGBDataset, PackedMahjongGBDataset, build_packed_dataset
from feature import FeatureAgent
from model import SelfVecModel, SlideFPNModel, SlideStyleModel
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
    parser.add_argument("--split-mode", choices=("contiguous", "random"), default="contiguous")
    parser.add_argument("--seed", type=int, default=6088)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--augment-mode", choices=("none", "all12"), default="none")
    parser.add_argument("--special-matches", default="")
    parser.add_argument("--exclude-special-matches", action="store_true")
    parser.add_argument("--fan-features-folder", default="")
    parser.add_argument("--fan-shanten-replace-folder", default="")
    parser.add_argument("--packed-cache-dir", default="")
    parser.add_argument("--force-rebuild-packed-cache", action="store_true")
    parser.add_argument(
        "--model-kind",
        choices=("selfvec", "slide", "slide-fpn"),
        default="selfvec",
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--slide-out-planes", type=int, default=8)
    parser.add_argument("--slide-vec-dim", type=int, default=78)
    parser.add_argument("--slide-fpn-obs-planes", type=int, default=60)
    parser.add_argument("--slide-fpn-blocks", type=int, default=1)
    parser.add_argument(
        "--slide-fpn-residual",
        choices=("merged", "input"),
        default="merged",
        help="Deprecated compatibility flag. SlideFPN now uses the fixed slide-faithful residual graph.",
    )
    parser.add_argument("--slide-fpn-use-vec", action="store_true")
    parser.add_argument("--slide-fpn-vec-hidden", type=int, default=0)
    parser.add_argument(
        "--slide-fpn-stem-mode",
        choices=("preserve", "valid_width"),
        default="preserve",
    )
    parser.add_argument("--fc-hidden", type=int, default=256)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--nonblocking-transfer", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--start-epoch", type=int, default=0)
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


def configure_runtime(args):
    if args.cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append(
                {
                    "cuda_device": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gib": props.total_memory / 1024**3,
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        print("cuda_devices " + json.dumps(devices, sort_keys=True), flush=True)
    print(
        "runtime_config "
        + json.dumps(
            {
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "nonblocking_transfer": args.nonblocking_transfer,
                "pin_memory": args.pin_memory,
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": args.persistent_workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def make_loader(dataset, batch_size, shuffle, num_workers, args):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": args.pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def prepare_batch(batch, args, device, is_training):
    non_blocking = bool(args.nonblocking_transfer)
    obs = batch[0]
    if args.model_kind == "slide-fpn":
        obs = obs[:, : args.slide_fpn_obs_planes].contiguous()
    obs_dict = {
        "observation": obs.to(device, non_blocking=non_blocking),
        "action_mask": batch[1].to(device, non_blocking=non_blocking),
    }
    if args.model_kind != "slide-fpn" or args.slide_fpn_use_vec:
        obs_dict["vec"] = batch[2].to(device, non_blocking=non_blocking)
    target = batch[3].long().to(device, non_blocking=non_blocking)
    return {"is_training": is_training, "obs": obs_dict}, target


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


def build_match_splits(data_folder, split_ratio, validation_end, split_mode, seed):
    with open(os.path.join(data_folder, "count.json"), "r", encoding="utf8") as f:
        counts = json.load(f)
    match_ids = np.arange(len(counts), dtype=np.int64)
    if split_mode == "random":
        rng = np.random.default_rng(seed)
        match_ids = rng.permutation(match_ids)

    train_end = int(split_ratio * len(match_ids))
    validation_stop = int(validation_end * len(match_ids))
    train_ids = match_ids[:train_end].tolist()
    validation_ids = match_ids[train_end:validation_stop].tolist()
    test_ids = match_ids[validation_stop:].tolist()
    return train_ids, validation_ids, test_ids


def build_or_open_dataset(
    args,
    name,
    begin,
    end,
    match_indices,
    augment,
    augment_mode,
):
    if args.packed_cache_dir:
        if augment or augment_mode != "none":
            raise RuntimeError("--packed-cache-dir currently supports only non-augmented datasets")
        if args.fan_features_folder:
            raise RuntimeError("--packed-cache-dir does not pack appended fan feature vectors")
        split_cache_dir = os.path.join(args.packed_cache_dir, name)
        build_packed_dataset(
            args.data_folder,
            split_cache_dir,
            match_indices,
            name,
            special_matches_path=args.special_matches,
            exclude_special_matches=args.exclude_special_matches,
            fan_shanten_replace_folder=args.fan_shanten_replace_folder,
            force=args.force_rebuild_packed_cache,
        )
        return PackedMahjongGBDataset(split_cache_dir)

    return MahjongGBDataset(
        args.data_folder,
        begin,
        end,
        0,
        augment=augment,
        lazy=args.lazy,
        augment_mode=augment_mode,
        special_matches_path=args.special_matches,
        exclude_special_matches=args.exclude_special_matches,
        fan_features_folder=args.fan_features_folder,
        fan_shanten_replace_folder=args.fan_shanten_replace_folder,
        match_indices=None if args.split_mode == "contiguous" else match_indices,
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


def evaluate_model(model, loader, device, desc, args, max_batches=0):
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
        input_dict, target = prepare_batch(batch, args, device, is_training=False)
        with torch.no_grad():
            logits = model(input_dict)
            loss = F.cross_entropy(logits, target)
            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, target).sum().item()
            total += batch_size
            update_category_stats(stats, pred, target)
            pbar.set_postfix(acc=f"{correct / total:.4f}", loss=f"{total_loss / total:.4f}")

        if max_batches and batch_index + 1 >= max_batches:
            break

    return total_loss / total, correct / total, stats


def main():
    args = parse_args()
    set_seed(args.seed)
    configure_runtime(args)

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
    train_match_ids, validation_match_ids, test_match_ids = build_match_splits(
        args.data_folder,
        args.split_ratio,
        validation_end,
        args.split_mode,
        args.seed,
    )

    print("[Loading Train dataset]")
    train_dataset = build_or_open_dataset(
        args,
        "train",
        0,
        args.split_ratio if args.split_mode == "contiguous" else 1,
        train_match_ids,
        augment=not args.no_augment,
        augment_mode=args.augment_mode,
    )
    print("[Loading Validation dataset]")
    validate_dataset = build_or_open_dataset(
        args,
        "validation",
        args.split_ratio if args.split_mode == "contiguous" else 0,
        validation_end if args.split_mode == "contiguous" else 1,
        validation_match_ids,
        augment=False,
        augment_mode="none",
    )
    test_dataset = None
    if args.test_ratio:
        print("[Loading Test dataset]")
        test_dataset = build_or_open_dataset(
            args,
            "test",
            validation_end if args.split_mode == "contiguous" else 0,
            1,
            test_match_ids,
            augment=False,
            augment_mode="none",
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
                "fan_shanten_replace_folder": args.fan_shanten_replace_folder,
                "fc_hidden": args.fc_hidden,
                "hidden": args.hidden,
                "init_checkpoint": args.init_checkpoint,
                "model_kind": args.model_kind,
                "num_blocks": args.num_blocks,
                "obs_size": FeatureAgent.OBS_SIZE,
                "packed_cache_dir": args.packed_cache_dir,
                "slide_fpn_blocks": args.slide_fpn_blocks,
                "slide_fpn_obs_planes": args.slide_fpn_obs_planes,
                "slide_fpn_residual": args.slide_fpn_residual,
                "slide_fpn_stem_mode": args.slide_fpn_stem_mode,
                "slide_fpn_use_vec": args.slide_fpn_use_vec,
                "slide_fpn_vec_hidden": args.slide_fpn_vec_hidden,
                "slide_out_planes": args.slide_out_planes,
                "slide_vec_dim": args.slide_vec_dim,
                "split_mode": args.split_mode,
                "split_seed": args.seed,
                "start_epoch": args.start_epoch,
                "train_match_id_head": train_match_ids[:5],
                "validation_match_id_head": validation_match_ids[:5],
                "vec_size": train_dataset.vec_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, args)
    vloader = make_loader(validate_dataset, args.batch_size, False, args.num_workers, args)
    test_loader = None
    if test_dataset is not None:
        test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, args)

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
    elif args.model_kind == "slide-fpn":
        model = SlideFPNModel(
            obs_dim=args.slide_fpn_obs_planes,
            vec_dim=train_dataset.vec_size,
            hidden=args.hidden,
            num_fpn_blocks=args.slide_fpn_blocks,
            fc_hidden=args.fc_hidden,
            residual_style=args.slide_fpn_residual,
            use_vec=args.slide_fpn_use_vec,
            vec_hidden=args.slide_fpn_vec_hidden,
            stem_mode=args.slide_fpn_stem_mode,
        ).to(device)
    else:
        model = SelfVecModel(
            obs_dim=FeatureAgent.OBS_SIZE,
            vec_dim=train_dataset.vec_size,
            hidden=args.hidden,
            num_blocks=args.num_blocks,
        ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        if any(key.startswith("module.") for key in checkpoint):
            checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
        print(
            "init_checkpoint "
            + json.dumps(
                {
                    "path": args.init_checkpoint,
                    "start_epoch": args.start_epoch,
                    "loaded_keys": len(checkpoint),
                },
                sort_keys=True,
            ),
            flush=True,
        )
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

    for epoch in range(args.start_epoch, args.epochs):
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
            input_dict, target = prepare_batch(batch, args, device, is_training=True)
            logits = model(input_dict)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, target).sum().item()
            total += batch_size
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
            args,
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
            args,
            args.max_test_batches,
        )
        log_epoch_metrics(writer, "Test", args.epochs - 1, test_loss, test_acc, test_category_stats)
    writer.close()


if __name__ == "__main__":
    main()
