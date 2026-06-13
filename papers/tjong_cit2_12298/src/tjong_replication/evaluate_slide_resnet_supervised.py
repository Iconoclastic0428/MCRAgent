"""Deterministic full-pass evaluation for the slide ResNet supervised model."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
import time
from typing import Any

import torch

from .evaluate_supervised import checkpoint_encoding_version, configure_deterministic_eval
from .slide_resnet import SlideMahjongResNetDueling, SlideResNetConfig
from .slide_v2_features import load_default_official_fan_checker
from .train_slide_resnet_supervised import (
    attach_v2_search_features,
    elastic_net_penalty,
    fast_local_supervised_loss_components,
)
from .train_supervised import (
    ShardedTensorDataset,
    batch_metric_sums,
    dataset_data_format,
    finalize_metric_sums,
    iter_optimized_sharded_batches,
    iter_supervised_batches,
    load_tensor_payload,
    merge_metric_sums,
    tensor_dataset_from_payload,
    unpack_batch,
)


def _config_from_checkpoint(payload: dict[str, Any]) -> SlideResNetConfig:
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("slide checkpoint is missing a structured config")
    allowed = {field.name for field in fields(SlideResNetConfig)}
    return SlideResNetConfig(**{key: value for key, value in raw_config.items() if key in allowed})


def load_slide_model(
    checkpoint: Path,
    device: torch.device,
    *,
    expected_encoding_version: str | None = None,
) -> tuple[SlideMahjongResNetDueling, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"slide checkpoint must be a structured dict: {checkpoint}")
    observed_encoding = checkpoint_encoding_version(payload)
    if expected_encoding_version is not None and observed_encoding != expected_encoding_version:
        raise ValueError(
            "checkpoint tensor encoding version mismatch: "
            f"expected {expected_encoding_version!r}, observed {observed_encoding!r}"
        )
    config = _config_from_checkpoint(payload)
    model = SlideMahjongResNetDueling(config)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"slide checkpoint is missing model_state_dict: {checkpoint}")
    model.load_state_dict(state_dict)
    model.to(device)
    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, payload


def _iter_eval_batches(dataset, args: argparse.Namespace):
    if bool(args.optimized_sharded_loader) and isinstance(dataset, ShardedTensorDataset):
        yield from iter_optimized_sharded_batches(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            rank=0,
            world_size=1,
            prefetch_shards=args.shard_prefetch,
            pin_memory=args.pin_memory,
            shuffle_within_shards=False,
        )
        return
    yield from iter_supervised_batches(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    configure_deterministic_eval(getattr(args, "seed", 0), strict=getattr(args, "strict_deterministic", False))
    if torch.cuda.is_available() and bool(args.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint_payload = load_slide_model(
        Path(args.checkpoint),
        device,
        expected_encoding_version=args.require_encoding_version,
    )
    config = model.config
    payload = load_tensor_payload(Path(args.eval_pt), expected_encoding_version=args.require_encoding_version)
    dataset = tensor_dataset_from_payload(
        payload,
        drop_shard_file_cache=bool(args.drop_shard_file_cache),
        mmap_shards=bool(args.mmap_shards),
    )
    fan_checker = None
    if config.feature_version == "v2" and (args.v2_use_official_fan or args.v2_require_official_fan):
        fan_checker = load_default_official_fan_checker()
        if fan_checker is None and args.v2_require_official_fan:
            raise ValueError("--v2-require-official-fan was set, but the official fan checker is unavailable")

    amp_mode = str(args.amp).lower()
    if amp_mode not in {"off", "fp16", "bf16"}:
        raise ValueError(f"unsupported --amp mode: {amp_mode}")
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    amp_enabled = device.type == "cuda" and amp_mode in {"fp16", "bf16"}
    autocast_context = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)

    checkpoint_metrics = checkpoint_payload.get("metrics") if isinstance(checkpoint_payload.get("metrics"), dict) else {}
    l1_lambda = float(args.l1_lambda if args.l1_lambda is not None else checkpoint_metrics.get("l1_lambda", 0.0) or 0.0)
    l2_lambda = float(args.l2_lambda if args.l2_lambda is not None else checkpoint_metrics.get("l2_lambda", 0.0) or 0.0)
    l1_penalty, l2_penalty = elastic_net_penalty(model, l1_lambda=l1_lambda, l2_lambda=l2_lambda)
    regularization_loss = float((l1_penalty + l2_penalty).detach().cpu().item())

    metric_sums: dict[str, float] = {}
    cross_entropy_sum = 0.0
    evaluated_examples = 0
    batch_count = 0
    started = time.time()
    with torch.no_grad():
        for batch in _iter_eval_batches(dataset, args):
            inputs, labels = unpack_batch(batch, device, include_hidden_tiles=bool(config.use_hidden_tiles))
            if config.feature_version == "v2":
                inputs = attach_v2_search_features(inputs, fan_checker=fan_checker, levels=args.v2_search_levels)
            with autocast_context:
                outputs = model(**inputs)
                loss_components = fast_local_supervised_loss_components(outputs, labels)
            batch_examples = int(labels[0].shape[0])
            cross_entropy_sum += float(loss_components["total"].detach().cpu().item()) * batch_examples
            merge_metric_sums(metric_sums, batch_metric_sums({key: value.detach() for key, value in outputs.items()}, labels))
            evaluated_examples += batch_examples
            batch_count += 1
            if args.max_batches and batch_count >= int(args.max_batches):
                break
            if args.progress_every_batches and batch_count % int(args.progress_every_batches) == 0:
                progress = {
                    "event": "eval_progress",
                    "batches": batch_count,
                    "evaluated_examples": evaluated_examples,
                    "elapsed_seconds": time.time() - started,
                }
                print(json.dumps(progress, sort_keys=True), flush=True)

    finalized = finalize_metric_sums(metric_sums)
    full_pass = not bool(args.max_batches)
    cross_entropy_loss = cross_entropy_sum / max(1, evaluated_examples)
    output: dict[str, Any] = {
        "paper": "slide-side-resnet34-dueling",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", 0) or 0),
        "checkpoint_global_batch": int(checkpoint_metrics.get("global_batch", 0) or 0),
        "eval_pt": str(args.eval_pt),
        "eval_data_format": dataset_data_format(dataset),
        "deterministic_eval": True,
        "deterministic_seed": getattr(args, "seed", 0),
        "strict_deterministic": getattr(args, "strict_deterministic", False),
        "device": str(device),
        "amp": amp_mode if amp_enabled else "off",
        "required_encoding_version": args.require_encoding_version,
        "model_config": config.to_dict(),
        "batch_size": int(args.batch_size),
        "full_pass": full_pass,
        "batches": batch_count,
        "evaluated_examples": evaluated_examples,
        "l1_lambda": l1_lambda,
        "l2_lambda": l2_lambda,
        "elastic_net_l1_penalty": float(l1_penalty.detach().cpu().item()),
        "elastic_net_l2_penalty": float(l2_penalty.detach().cpu().item()),
        "regularization_loss": regularization_loss,
        "cross_entropy_loss": cross_entropy_loss,
        "optimization_loss": cross_entropy_loss + regularization_loss,
        "elapsed_seconds": time.time() - started,
        **finalized,
    }
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, sort_keys=True), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-pt", required=True)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", default="off")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--optimized-sharded-loader", action="store_true")
    parser.add_argument("--shard-prefetch", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--drop-shard-file-cache", action="store_true")
    parser.add_argument("--mmap-shards", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--progress-every-batches", type=int, default=0)
    parser.add_argument("--l1-lambda", type=float, default=None)
    parser.add_argument("--l2-lambda", type=float, default=None)
    parser.add_argument("--v2-search-levels", type=int, default=1)
    parser.add_argument("--v2-use-official-fan", action="store_true")
    parser.add_argument("--v2-require-official-fan", action="store_true")
    args = parser.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
