"""Supervised side trainer for the slide-style ResNet34 dueling model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .actions import ACTION_TO_INDEX
from .slide_resnet import (
    SLIDE_SYMMETRY_TRANSFORMS,
    SlideMahjongResNetDueling,
    SlideResNetConfig,
    transform_claim_labels,
    transform_discard_labels,
    transform_tile_tensor,
)
from .slide_v2_features import build_batch_search_features_from_tensors, load_default_official_fan_checker
from .tiles import SUIT_ORDER
from .tensorize_botzone import TENSOR_ENCODING_VERSION
from .train_supervised import (
    ShardedTensorDataset,
    batch_metric_sums,
    configure_seed,
    dataset_data_format,
    finalize_metric_sums,
    initialize_distributed,
    iter_optimized_sharded_batches,
    iter_supervised_batches,
    is_rank0,
    load_tensor_payload,
    merge_metric_sums,
    reduce_metric_sums,
    supervised_loss_components,
    tensor_dataset_from_payload,
    unpack_batch,
)

IDENTITY_TRANSFORM = (SUIT_ORDER, False)
CLAIM_ACTION_INDICES = tuple(ACTION_TO_INDEX[name] for name in ("CHOW", "PONG", "MINGKONG", "BUKONG", "ANKONG"))
DISCARD_ACTION_INDEX = ACTION_TO_INDEX["DISCARD"]


def elastic_net_penalty(
    model: torch.nn.Module,
    *,
    l1_lambda: float,
    l2_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        zero = torch.zeros(())
        return zero, zero
    reference = parameters[0]
    l1 = reference.new_zeros(())
    l2 = reference.new_zeros(())
    for parameter in parameters:
        l1 = l1 + parameter.abs().sum()
        l2 = l2 + parameter.square().sum()
    return float(l1_lambda) * l1, float(l2_lambda) * l2


def fast_local_supervised_loss_components(
    outputs: dict[str, torch.Tensor],
    labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    action_label, claim_label, discard_label = labels
    claim_mask = (
        (action_label == CLAIM_ACTION_INDICES[0])
        | (action_label == CLAIM_ACTION_INDICES[1])
        | (action_label == CLAIM_ACTION_INDICES[2])
        | (action_label == CLAIM_ACTION_INDICES[3])
        | (action_label == CLAIM_ACTION_INDICES[4])
    )
    discard_mask = action_label == DISCARD_ACTION_INDEX
    action_loss = F.cross_entropy(outputs["action_logits"], action_label)
    claim_sum = F.cross_entropy(outputs["claim_logits"][claim_mask], claim_label[claim_mask], reduction="sum")
    discard_sum = F.cross_entropy(
        outputs["discard_logits"][discard_mask],
        discard_label[discard_mask],
        reduction="sum",
    )
    claim_count = claim_mask.sum().clamp_min(1).to(dtype=claim_sum.dtype)
    discard_count = discard_mask.sum().clamp_min(1).to(dtype=discard_sum.dtype)
    claim_loss = claim_sum / claim_count
    discard_loss = discard_sum / discard_count
    return {
        "action": action_loss,
        "claim": claim_loss,
        "discard": discard_loss,
        "total": action_loss + claim_loss + discard_loss,
    }


def apply_slide_symmetry(
    inputs: dict[str, torch.Tensor],
    labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    transform: tuple[tuple[str, str, str], bool],
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if transform == IDENTITY_TRANSFORM:
        return inputs, labels
    suit_permutation, mirror = transform
    augmented_inputs = dict(inputs)
    for key in ("visible_tiles", "sub_visible_tiles", "hidden_tiles"):
        value = augmented_inputs.get(key)
        if value is not None:
            augmented_inputs[key] = transform_tile_tensor(value, suit_permutation, mirror=mirror)
    action_label, claim_label, discard_label = labels
    augmented_labels = (
        action_label,
        transform_claim_labels(claim_label, suit_permutation, mirror=mirror),
        transform_discard_labels(discard_label, suit_permutation, mirror=mirror),
    )
    return augmented_inputs, augmented_labels


def augmentation_transforms_for_batch(
    args: argparse.Namespace,
    *,
    generator: torch.Generator,
) -> tuple[tuple[tuple[str, str, str], bool], ...]:
    mode = str(args.augmentation_mode).lower()
    if mode == "none":
        return (IDENTITY_TRANSFORM,)
    if mode == "all":
        return SLIDE_SYMMETRY_TRANSFORMS
    if mode == "random":
        index = int(torch.randint(len(SLIDE_SYMMETRY_TRANSFORMS), (1,), generator=generator).item())
        return (SLIDE_SYMMETRY_TRANSFORMS[index],)
    raise ValueError(f"unknown augmentation mode: {args.augmentation_mode!r}")


def attach_v2_search_features(
    inputs: dict[str, torch.Tensor],
    *,
    fan_checker,
    levels: int,
) -> dict[str, torch.Tensor]:
    enriched = dict(inputs)
    enriched["search_features"] = build_batch_search_features_from_tensors(
        enriched["visible_tiles"],
        enriched["game_features"],
        fan_checker=fan_checker,
        levels=levels,
    )
    enriched["sub_search_features"] = build_batch_search_features_from_tensors(
        enriched["sub_visible_tiles"],
        enriched["sub_game_features"],
        fan_checker=fan_checker,
        levels=levels,
    )
    return enriched


def iter_training_batches(
    dataset,
    args: argparse.Namespace,
    *,
    epoch: int,
    rank: int = 0,
    world_size: int = 1,
):
    if bool(args.optimized_sharded_loader):
        if not isinstance(dataset, ShardedTensorDataset):
            raise ValueError("--optimized-sharded-loader requires a sharded tensor dataset")
        yield from iter_optimized_sharded_batches(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=int(args.seed) + int(epoch),
            rank=rank,
            world_size=world_size,
            prefetch_shards=args.shard_prefetch,
            pin_memory=args.pin_memory,
        )
        return
    yield from iter_supervised_batches(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=int(args.seed) + int(epoch),
    )


def train(args: argparse.Namespace) -> dict:
    distributed, local_rank, rank, world_size = initialize_distributed(args)
    configure_seed(int(args.seed))
    if distributed:
        requested_device = args.device
        if torch.cuda.is_available() and (requested_device is None or str(requested_device).startswith("cuda")):
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(requested_device or "cpu")
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        if bool(args.allow_tf32):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    train_payload = load_tensor_payload(Path(args.train_pt), expected_encoding_version=args.require_encoding_version)
    encoding_schema = train_payload.get("encoding_schema") or {}
    dataset = tensor_dataset_from_payload(train_payload, drop_shard_file_cache=bool(args.drop_shard_file_cache))
    if distributed and not bool(args.optimized_sharded_loader):
        raise ValueError("--distributed requires --optimized-sharded-loader")
    config = SlideResNetConfig(
        feature_version=args.feature_version,
        base_channels=args.base_channels,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        use_hidden_tiles=bool(args.use_hidden_tiles),
        require_search_features=bool(args.require_search_features or args.feature_version == "v2"),
        fuse_sub_encode=bool(args.fuse_sub_encode),
        channels_last=bool(args.channels_last),
    )
    raw_model = SlideMahjongResNetDueling(config).to(device)
    if bool(args.channels_last):
        raw_model = raw_model.to(memory_format=torch.channels_last)
    cuda_device_count = torch.cuda.device_count() if device.type == "cuda" else 0
    data_parallel_enabled = bool(
        not distributed and args.data_parallel and device.type == "cuda" and cuda_device_count > 1
    )
    if distributed:
        ddp_kwargs = {
            "device_ids": [local_rank] if device.type == "cuda" else None,
            "output_device": local_rank if device.type == "cuda" else None,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
        }
        if bool(args.ddp_static_graph):
            ddp_kwargs["static_graph"] = True
        try:
            model: torch.nn.Module = DistributedDataParallel(raw_model, **ddp_kwargs)
        except TypeError:
            if "static_graph" not in ddp_kwargs:
                raise
            ddp_kwargs.pop("static_graph")
            model = DistributedDataParallel(raw_model, **ddp_kwargs)
    elif data_parallel_enabled:
        model = torch.nn.DataParallel(raw_model)
    else:
        model = raw_model
    fused_adamw = bool(args.fused_adamw and device.type == "cuda")
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=True,
        ) if fused_adamw else torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    except (RuntimeError, TypeError):
        fused_adamw = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    amp_mode = str(args.amp).lower()
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    amp_enabled = device.type == "cuda" and amp_mode in {"fp16", "bf16"}
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_mode == "fp16")
    metric_sample_every_batches = int(args.metric_sample_every_batches or 1)
    finite_check_every_batches = int(args.finite_check_every_batches or 1)
    use_local_loss_normalization = bool(args.local_loss_normalization)
    profile_timing = bool(args.profile_timing)
    fan_checker = None
    if args.feature_version == "v2" and (args.v2_use_official_fan or args.v2_require_official_fan):
        fan_checker = load_default_official_fan_checker()
        if fan_checker is None and args.v2_require_official_fan:
            raise ValueError("--v2-require-official-fan was set, but the Botzone official fan checker is unavailable")
    augmentation_generator = torch.Generator()
    augmentation_generator.manual_seed(int(args.seed) + 1729)
    transforms_per_batch = len(SLIDE_SYMMETRY_TRANSFORMS) if args.augmentation_mode == "all" else 1

    metrics: dict[str, object] = {
        "paper": "slide-side-resnet34-dueling",
        "feature_version": args.feature_version,
        "feature_channels": config.resolved_in_channels(),
        "feature_mapping": "visible_history_88 + game_history_96 + hidden_slots_5 + bias_1 (+ search_30 for v2)",
        "uses_hidden_tiles": bool(args.use_hidden_tiles),
        "search_features_required": bool(config.require_search_features),
        "v2_search_levels": args.v2_search_levels,
        "v2_use_official_fan": bool(args.v2_use_official_fan),
        "v2_require_official_fan": bool(args.v2_require_official_fan),
        "v2_official_fan_available": fan_checker is not None,
        "augmentation_mode": args.augmentation_mode,
        "augmentation_transforms_per_batch": transforms_per_batch,
        "metric_sample_every_batches": metric_sample_every_batches,
        "finite_check_every_batches": finite_check_every_batches,
        "loss_normalization": "local" if use_local_loss_normalization else ("distributed" if distributed else "local"),
        "profile_timing": profile_timing,
        "amp": amp_mode if amp_enabled else "off",
        "allow_tf32": bool(args.allow_tf32),
        "cudnn_benchmark": bool(args.cudnn_benchmark),
        "fuse_sub_encode": bool(args.fuse_sub_encode),
        "channels_last": bool(args.channels_last),
        "ddp_static_graph": bool(args.ddp_static_graph),
        "fused_adamw": fused_adamw,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "elastic_net_l1_lambda": args.l1_lambda,
        "elastic_net_l2_lambda": args.l2_lambda,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "cuda_device_count": cuda_device_count,
        "data_parallel": data_parallel_enabled,
        "distributed": distributed,
        "distributed_rank": rank,
        "distributed_world_size": world_size,
        "local_rank": local_rank,
        "effective_gpu_count": world_size if distributed else (cuda_device_count if data_parallel_enabled else (1 if device.type == "cuda" else 0)),
        "train_examples": len(dataset),
        "train_data_format": dataset_data_format(dataset),
        "train_shard_count": dataset.shard_count if isinstance(dataset, ShardedTensorDataset) else 0,
        "required_encoding_version": args.require_encoding_version,
        "paper_tensor_encoding_version": TENSOR_ENCODING_VERSION,
        "checkpoint_encoding_version": encoding_schema.get("version") or args.require_encoding_version,
        "model_config": config.to_dict(),
        "model_parameters": raw_model.parameter_count(),
        "epochs": [],
    }

    start = time.time()
    global_batch = 0
    completed_epochs = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        metric_sums: dict[str, float] = {}
        total_loss = 0.0
        total_examples = 0
        timing_sums = {
            "data_wait": 0.0,
            "unpack": 0.0,
            "augment": 0.0,
            "forward_loss": 0.0,
            "backward_step": 0.0,
            "metrics": 0.0,
        }
        timing_last_log = dict(timing_sums)
        timing_last_global_batch = global_batch
        max_grad_norm_before_clip = torch.zeros((), dtype=torch.float32, device=device)
        epoch_start = time.time()
        batch_fetch_start = time.time() if profile_timing else 0.0
        for batch_index, batch in enumerate(
            iter_training_batches(dataset, args, epoch=epoch, rank=rank, world_size=world_size),
            start=1,
        ):
            if profile_timing:
                now = time.time()
                timing_sums["data_wait"] += now - batch_fetch_start
                phase_start = now
            base_inputs, base_labels = unpack_batch(batch, device, include_hidden_tiles=True)
            if profile_timing:
                now = time.time()
                timing_sums["unpack"] += now - phase_start
            transforms = augmentation_transforms_for_batch(args, generator=augmentation_generator)
            for transform_index, transform in enumerate(transforms, start=1):
                if profile_timing:
                    phase_start = time.time()
                inputs, labels = apply_slide_symmetry(base_inputs, base_labels, transform)
                if profile_timing:
                    now = time.time()
                    timing_sums["augment"] += now - phase_start
                    phase_start = now
                autocast_context = (
                    torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                optimizer.zero_grad(set_to_none=True)
                with autocast_context:
                    if args.feature_version == "v2":
                        inputs = attach_v2_search_features(
                            inputs,
                            fan_checker=fan_checker,
                            levels=int(args.v2_search_levels),
                        )
                    outputs = model(**inputs)
                    loss_components = (
                        fast_local_supervised_loss_components(outputs, labels)
                        if use_local_loss_normalization
                        else supervised_loss_components(outputs, labels, distributed=distributed)
                    )
                    ce_loss = loss_components["total"]
                    assert ce_loss is not None
                    l1_penalty, l2_penalty = elastic_net_penalty(
                        model,
                        l1_lambda=args.l1_lambda,
                        l2_lambda=args.l2_lambda,
                    )
                    loss = ce_loss + l1_penalty + l2_penalty
                if profile_timing:
                    now = time.time()
                    timing_sums["forward_loss"] += now - phase_start
                    phase_start = now
                next_global_batch = global_batch + 1
                if finite_check_every_batches > 0 and next_global_batch % finite_check_every_batches == 0:
                    if not torch.isfinite(loss.detach()).item():
                        raise ValueError(
                            f"non-finite slide supervised loss at epoch={epoch} "
                            f"batch={batch_index} transform={transform_index}: {loss}"
                        )
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if args.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        args.grad_clip,
                        error_if_nonfinite=False,
                    )
                    grad_norm_tensor = torch.as_tensor(grad_norm, dtype=torch.float32, device=device)
                    max_grad_norm_before_clip = torch.maximum(max_grad_norm_before_clip, grad_norm_tensor.detach())
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if profile_timing:
                    timing_sums["backward_step"] += time.time() - phase_start

                batch_size = int(labels[0].shape[0])
                global_batch += 1
                log_this_batch = bool(args.log_every_batches and global_batch % int(args.log_every_batches) == 0)
                sample_this_batch = (
                    metric_sample_every_batches > 0
                    and (global_batch % metric_sample_every_batches == 0 or log_this_batch)
                )
                metric_start = time.time() if profile_timing and (sample_this_batch or log_this_batch) else 0.0
                if sample_this_batch:
                    total_examples += batch_size
                    total_loss += float(loss.detach().item()) * batch_size
                    metric_outputs = {name: tensor.detach() for name, tensor in outputs.items()}
                    merge_metric_sums(metric_sums, batch_metric_sums(metric_outputs, labels))
                if profile_timing and sample_this_batch and not log_this_batch:
                    timing_sums["metrics"] += time.time() - metric_start
                if args.log_every_batches and global_batch % int(args.log_every_batches) == 0:
                    display_metric_sums = reduce_metric_sums(metric_sums, device) if distributed else metric_sums
                    display_loss_sum = total_loss
                    display_examples = total_examples
                    display_grad_norm = max_grad_norm_before_clip.detach().clone()
                    if distributed:
                        loss_tensor = torch.tensor([display_loss_sum, display_examples], dtype=torch.float64, device=device)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        display_loss_sum = float(loss_tensor[0].item())
                        display_examples = int(loss_tensor[1].item())
                        dist.all_reduce(display_grad_norm, op=dist.ReduceOp.MAX)
                    display_grad_norm_value = float(display_grad_norm.item())
                    partial = finalize_metric_sums(display_metric_sums)
                    partial.update(
                        {
                            "event": "batch",
                            "epoch": epoch,
                            "batch": batch_index,
                            "batch_transform": transform_index,
                            "global_batch": global_batch,
                            "optimization_loss": display_loss_sum / max(1, display_examples),
                            "lr": optimizer.param_groups[0]["lr"],
                            "max_grad_norm_before_clip": display_grad_norm_value,
                            "sampled_examples": display_examples,
                            "elapsed_seconds": time.time() - epoch_start,
                        }
                    )
                    if profile_timing:
                        timing_sums["metrics"] += time.time() - metric_start
                        recent_timing = {
                            name: timing_sums[name] - timing_last_log[name]
                            for name in timing_sums
                        }
                        recent_batches = global_batch - timing_last_global_batch
                        partial["timing_seconds"] = dict(timing_sums)
                        partial["recent_timing_seconds"] = recent_timing
                        partial["recent_timed_batches"] = recent_batches
                        timing_last_log = dict(timing_sums)
                        timing_last_global_batch = global_batch
                    if is_rank0():
                        print(json.dumps(partial, sort_keys=True), flush=True)
                if args.max_steps and global_batch >= int(args.max_steps):
                    break
            if profile_timing:
                batch_fetch_start = time.time()
            if args.max_steps and global_batch >= int(args.max_steps):
                break

        display_metric_sums = reduce_metric_sums(metric_sums, device) if distributed else metric_sums
        display_loss_sum = total_loss
        display_examples = total_examples
        display_grad_norm = max_grad_norm_before_clip.detach().clone()
        if distributed:
            loss_tensor = torch.tensor([display_loss_sum, display_examples], dtype=torch.float64, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            display_loss_sum = float(loss_tensor[0].item())
            display_examples = int(loss_tensor[1].item())
            dist.all_reduce(display_grad_norm, op=dist.ReduceOp.MAX)
        display_grad_norm_value = float(display_grad_norm.item())
        finalized = finalize_metric_sums(display_metric_sums)
        epoch_metrics = dict(finalized)
        epoch_metrics.update(
            {
                "event": "epoch",
                "epoch": epoch,
                "global_batch": global_batch,
                "optimization_loss": display_loss_sum / max(1, display_examples),
                "lr": optimizer.param_groups[0]["lr"],
                "max_grad_norm_before_clip": display_grad_norm_value,
                "sampled_examples": display_examples,
                "elapsed_seconds": time.time() - epoch_start,
            }
        )
        if profile_timing:
            epoch_metrics["timing_seconds"] = dict(timing_sums)
        if is_rank0():
            metrics["epochs"].append(epoch_metrics)
        completed_epochs = epoch
        if args.metrics_jsonl and is_rank0():
            Path(args.metrics_jsonl).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.metrics_jsonl).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
        if is_rank0():
            print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
        scheduler.step()
        if args.max_steps and global_batch >= int(args.max_steps):
            break

    metrics["epochs_completed"] = completed_epochs
    metrics["global_batch"] = global_batch
    metrics["elapsed_seconds"] = time.time() - start

    if args.checkpoint_out and is_rank0():
        checkpoint_path = Path(args.checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "parallel_model_state_dict": model.state_dict() if data_parallel_enabled else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config.to_dict(),
                "metrics": metrics,
                "epoch": completed_epochs,
                "tensor_encoding_version": metrics["checkpoint_encoding_version"],
            },
            checkpoint_path,
        )
    if args.metrics_out and is_rank0():
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pt", required=True)
    parser.add_argument("--checkpoint-out", default="models/tjong_slide_resnet_supervised.pt")
    parser.add_argument("--metrics-out", default="runs/tjong_slide_resnet_supervised_metrics.json")
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--feature-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--use-hidden-tiles", action="store_true")
    parser.add_argument("--require-search-features", action="store_true")
    parser.add_argument("--v2-search-levels", type=int, default=3)
    parser.add_argument("--v2-use-official-fan", action="store_true")
    parser.add_argument("--v2-require-official-fan", action="store_true")
    parser.add_argument("--augmentation-mode", choices=("none", "random", "all"), default="none")
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--l1-lambda", type=float, default=0.0)
    parser.add_argument("--l2-lambda", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="off")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--fuse-sub-encode", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--ddp-static-graph", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--metric-sample-every-batches", type=int, default=1)
    parser.add_argument("--finite-check-every-batches", type=int, default=1)
    parser.add_argument("--local-loss-normalization", action="store_true")
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--optimized-sharded-loader", action="store_true")
    parser.add_argument("--shard-prefetch", type=int, default=1)
    parser.add_argument("--drop-shard-file-cache", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
