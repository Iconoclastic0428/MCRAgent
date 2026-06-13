"""Evaluate slide ResNet behavior on rows where HU is illegal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from .actions import ACTION_NAMES, ACTION_TO_INDEX, DISCARD_SIZE
from .evaluate_supervised import configure_deterministic_eval
from .evaluate_slide_resnet_supervised import load_slide_model
from .slide_v2_features import load_default_official_fan_checker
from .train_slide_resnet_supervised import attach_v2_search_features
from .train_supervised import (
    ACTION_MASK_OFFSET,
    HAND_SELF_VISIBLE_ROW,
    ShardedTensorDataset,
    dataset_data_format,
    iter_optimized_sharded_batches,
    iter_supervised_batches,
    load_tensor_payload,
    tensor_dataset_from_payload,
    unpack_batch,
)

HU_ACTION = ACTION_TO_INDEX["HU"]
DISCARD_ACTION = ACTION_TO_INDEX["DISCARD"]


def _empty_action_counts() -> dict[str, int]:
    return {name: 0 for name in ACTION_NAMES}


def _add_action_counts(total: dict[str, int], values: torch.Tensor) -> None:
    values = values.detach().cpu()
    for index, name in enumerate(ACTION_NAMES):
        total[name] += int((values == index).sum().item())


def _iter_eval_batches(dataset, args: argparse.Namespace):
    if bool(args.optimized_sharded_loader) and isinstance(dataset, ShardedTensorDataset):
        yield from iter_optimized_sharded_batches(
            dataset,
            batch_size=args.batch_size,
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
            rank=0,
            world_size=1,
            prefetch_shards=args.shard_prefetch,
            pin_memory=args.pin_memory,
            shuffle_within_shards=True,
        )
        return
    yield from iter_supervised_batches(
        dataset,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        num_workers=args.num_workers,
        seed=int(args.seed),
    )


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
    return masked.argmax(dim=-1)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    configure_deterministic_eval(int(args.seed), strict=False)
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

    totals: dict[str, Any] = {
        "evaluated_examples": 0,
        "batches": 0,
        "hu_illegal_rows": 0,
        "hu_legal_rows": 0,
        "hu_illegal_label_outside_action_mask": 0,
        "hu_illegal_discard_legal_rows": 0,
        "hu_illegal_no_legal_action_rows": 0,
        "actual_action_counts_hu_illegal": _empty_action_counts(),
        "raw_pred_action_counts_hu_illegal": _empty_action_counts(),
        "legal_pred_action_counts_hu_illegal": _empty_action_counts(),
        "raw_action_correct_hu_illegal": 0,
        "legal_action_correct_hu_illegal": 0,
        "actual_discard_rows_hu_illegal": 0,
        "raw_pred_discard_rows_hu_illegal": 0,
        "legal_pred_discard_rows_hu_illegal": 0,
        "conditional_raw_discard_tile_correct_hu_illegal": 0,
        "conditional_legal_discard_tile_correct_hu_illegal": 0,
        "end_to_end_raw_discard_tile_correct_hu_illegal": 0,
        "end_to_end_legal_discard_tile_correct_hu_illegal": 0,
        "actual_discard_rows_hu_illegal_discard_legal": 0,
        "legal_pred_discard_rows_hu_illegal_discard_legal": 0,
        "end_to_end_legal_discard_tile_correct_hu_illegal_discard_legal": 0,
        "discard_label_outside_hand_mask_hu_illegal": 0,
    }

    started = time.time()
    with torch.no_grad():
        for batch in _iter_eval_batches(dataset, args):
            inputs, labels = unpack_batch(batch, device, include_hidden_tiles=bool(config.use_hidden_tiles))
            if config.feature_version == "v2":
                inputs = attach_v2_search_features(inputs, fan_checker=fan_checker, levels=args.v2_search_levels)
            with autocast_context:
                outputs = model(**inputs)
            action_label, _claim_label, discard_label = labels
            action_mask = inputs["game_features"][:, -1, ACTION_MASK_OFFSET : ACTION_MASK_OFFSET + len(ACTION_NAMES)] > 0
            legal_action_available = action_mask.any(dim=-1)
            hu_illegal = ~action_mask[:, HU_ACTION]
            hu_legal = action_mask[:, HU_ACTION]
            selected = hu_illegal
            raw_action_pred = outputs["action_logits"].argmax(dim=-1)
            legal_action_pred = _masked_argmax(outputs["action_logits"], action_mask)
            legal_discards = inputs["sub_visible_tiles"][:, -1, HAND_SELF_VISIBLE_ROW, :DISCARD_SIZE] > 0
            raw_discard_pred = outputs["discard_logits"].argmax(dim=-1)
            legal_discard_pred = _masked_argmax(outputs["discard_logits"], legal_discards)
            safe_discard_label = discard_label.clamp(0, DISCARD_SIZE - 1)

            totals["evaluated_examples"] += int(action_label.numel())
            totals["batches"] += 1
            totals["hu_illegal_rows"] += int(hu_illegal.sum().item())
            totals["hu_legal_rows"] += int(hu_legal.sum().item())
            totals["hu_illegal_no_legal_action_rows"] += int((selected & ~legal_action_available).sum().item())
            if selected.any():
                _add_action_counts(totals["actual_action_counts_hu_illegal"], action_label[selected])
                _add_action_counts(totals["raw_pred_action_counts_hu_illegal"], raw_action_pred[selected])
                _add_action_counts(totals["legal_pred_action_counts_hu_illegal"], legal_action_pred[selected])
                totals["raw_action_correct_hu_illegal"] += int((raw_action_pred[selected] == action_label[selected]).sum().item())
                totals["legal_action_correct_hu_illegal"] += int((legal_action_pred[selected] == action_label[selected]).sum().item())
                target_action_legal = action_mask.gather(1, action_label.clamp(0, len(ACTION_NAMES) - 1)[:, None]).squeeze(1)
                totals["hu_illegal_label_outside_action_mask"] += int((selected & ~target_action_legal).sum().item())

            actual_discard = selected & (action_label == DISCARD_ACTION)
            discard_legal = selected & action_mask[:, DISCARD_ACTION]
            actual_discard_and_discard_legal = actual_discard & action_mask[:, DISCARD_ACTION]
            totals["hu_illegal_discard_legal_rows"] += int(discard_legal.sum().item())
            totals["actual_discard_rows_hu_illegal"] += int(actual_discard.sum().item())
            totals["raw_pred_discard_rows_hu_illegal"] += int((selected & (raw_action_pred == DISCARD_ACTION)).sum().item())
            totals["legal_pred_discard_rows_hu_illegal"] += int((selected & (legal_action_pred == DISCARD_ACTION)).sum().item())
            if actual_discard.any():
                raw_tile_correct = raw_discard_pred[actual_discard] == discard_label[actual_discard]
                legal_tile_correct = legal_discard_pred[actual_discard] == discard_label[actual_discard]
                raw_e2e = (raw_action_pred[actual_discard] == DISCARD_ACTION) & raw_tile_correct
                legal_e2e = (legal_action_pred[actual_discard] == DISCARD_ACTION) & legal_tile_correct
                totals["conditional_raw_discard_tile_correct_hu_illegal"] += int(raw_tile_correct.sum().item())
                totals["conditional_legal_discard_tile_correct_hu_illegal"] += int(legal_tile_correct.sum().item())
                totals["end_to_end_raw_discard_tile_correct_hu_illegal"] += int(raw_e2e.sum().item())
                totals["end_to_end_legal_discard_tile_correct_hu_illegal"] += int(legal_e2e.sum().item())
                discard_label_in_hand = legal_discards.gather(1, safe_discard_label[:, None]).squeeze(1)
                totals["discard_label_outside_hand_mask_hu_illegal"] += int((actual_discard & ~discard_label_in_hand).sum().item())
            totals["actual_discard_rows_hu_illegal_discard_legal"] += int(actual_discard_and_discard_legal.sum().item())
            totals["legal_pred_discard_rows_hu_illegal_discard_legal"] += int(
                (discard_legal & (legal_action_pred == DISCARD_ACTION)).sum().item()
            )
            if actual_discard_and_discard_legal.any():
                legal_tile_correct_subset = legal_discard_pred[actual_discard_and_discard_legal] == discard_label[
                    actual_discard_and_discard_legal
                ]
                legal_e2e_subset = (legal_action_pred[actual_discard_and_discard_legal] == DISCARD_ACTION) & legal_tile_correct_subset
                totals["end_to_end_legal_discard_tile_correct_hu_illegal_discard_legal"] += int(legal_e2e_subset.sum().item())

            if args.max_batches and totals["batches"] >= int(args.max_batches):
                break

    hu_illegal_rows = max(1, int(totals["hu_illegal_rows"]))
    actual_discard_rows = max(1, int(totals["actual_discard_rows_hu_illegal"]))
    actual_discard_legal_rows = max(1, int(totals["actual_discard_rows_hu_illegal_discard_legal"]))
    checkpoint_metrics = checkpoint_payload.get("metrics") if isinstance(checkpoint_payload.get("metrics"), dict) else {}
    output: dict[str, Any] = {
        "event": "hu_illegal_eval",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", 0) or 0),
        "checkpoint_global_batch": int(checkpoint_metrics.get("global_batch", 0) or 0),
        "eval_pt": str(args.eval_pt),
        "eval_data_format": dataset_data_format(dataset),
        "model_config": config.to_dict(),
        "device": str(device),
        "amp": amp_mode if amp_enabled else "off",
        "batch_size": int(args.batch_size),
        "max_batches": int(args.max_batches or 0),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "elapsed_seconds": time.time() - started,
        **totals,
        "raw_action_accuracy_hu_illegal": int(totals["raw_action_correct_hu_illegal"]) / hu_illegal_rows,
        "legal_action_accuracy_hu_illegal": int(totals["legal_action_correct_hu_illegal"]) / hu_illegal_rows,
        "conditional_raw_discard_tile_accuracy_hu_illegal": int(totals["conditional_raw_discard_tile_correct_hu_illegal"])
        / actual_discard_rows,
        "conditional_legal_discard_tile_accuracy_hu_illegal": int(totals["conditional_legal_discard_tile_correct_hu_illegal"])
        / actual_discard_rows,
        "end_to_end_raw_discard_tile_accuracy_hu_illegal": int(totals["end_to_end_raw_discard_tile_correct_hu_illegal"])
        / actual_discard_rows,
        "end_to_end_legal_discard_tile_accuracy_hu_illegal": int(totals["end_to_end_legal_discard_tile_correct_hu_illegal"])
        / actual_discard_rows,
        "end_to_end_legal_discard_tile_accuracy_hu_illegal_discard_legal": int(
            totals["end_to_end_legal_discard_tile_correct_hu_illegal_discard_legal"]
        )
        / actual_discard_legal_rows,
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
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--optimized-sharded-loader", action="store_true")
    parser.add_argument("--shard-prefetch", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--drop-shard-file-cache", action="store_true")
    parser.add_argument("--mmap-shards", action="store_true")
    parser.add_argument("--v2-search-levels", type=int, default=3)
    parser.add_argument("--v2-use-official-fan", action="store_true")
    parser.add_argument("--v2-require-official-fan", action="store_true")
    args = parser.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
