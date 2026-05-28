#!/usr/bin/env python3
"""Evaluate a Transformer checkpoint against CHAGA review candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from train_transformer_candidate import (
    FeatureAgent,
    TransformerCandidateModel,
    TransformerExample,
    TransformerRawDataset,
    action_response,
    collate_transformer_examples,
    evaluate_model,
    hu_gated_candidate_mask,
    load_examples,
    load_review_target_lookup,
    normalize_teacher_action,
)


def select_reviewed_examples(examples: Iterable[TransformerExample]) -> list[TransformerExample]:
    return [example for example in examples if getattr(example, "teacher_candidate_norms", ())]


def resolve_eval_max_candidates(config: dict, args_max_candidates: int) -> int:
    """Evaluation must use the requested candidate width, not checkpoint training width."""

    return int(args_max_candidates)


def score_original_chaga_match(
    predicted_action: str,
    *,
    teacher_top1_norm: str,
    teacher_top3_norms: Iterable[str],
    accept_top3: bool,
) -> dict[str, bool]:
    predicted_norm = normalize_teacher_action(predicted_action)
    top1_norm = normalize_teacher_action(teacher_top1_norm)
    top3_norms = tuple(normalize_teacher_action(action) for action in teacher_top3_norms if action)
    top1_match = bool(predicted_norm and predicted_norm == top1_norm)
    top3_match = bool(predicted_norm and predicted_norm in top3_norms)
    return {
        "top1_match": top1_match,
        "top3_match": top3_match,
        "relaxed_match": top1_match or (bool(accept_top3) and top3_match),
    }


def collect_original_prediction_rows(
    model: TransformerCandidateModel,
    reviewed: list[TransformerExample],
    *,
    max_candidates: int,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    loader = DataLoader(
        TransformerRawDataset(reviewed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_transformer_examples(items, max_candidates=max_candidates),
    )
    rows: list[dict] = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            current_examples = reviewed[offset : offset + int(batch["target_index"].numel())]
            offset += len(current_examples)
            device_batch = {key: value.to(device) for key, value in batch.items()}
            logits, _ = model(device_batch)
            pred_slots = torch.argmax(logits, dim=1).cpu()
            for index, example in enumerate(current_examples):
                pred_slot = int(pred_slots[index].item())
                pred_action = action_response(int(batch["candidate_actions"][index, pred_slot].item()))
                candidate_norms = tuple(getattr(example, "teacher_candidate_norms", ()))
                top1_norm = candidate_norms[0] if candidate_norms else ""
                top3_norms = candidate_norms[:3]
                scores = score_original_chaga_match(
                    pred_action,
                    teacher_top1_norm=top1_norm,
                    teacher_top3_norms=top3_norms,
                    accept_top3=bool(getattr(example, "teacher_accept_top3", False)),
                )
                rows.append(
                    {
                        "turn": int(example.turn),
                        "player": int(example.player),
                        "response": example.response,
                        "predicted_action": pred_action,
                        "predicted_normalized": normalize_teacher_action(pred_action),
                        "chaga_top1_action": top1_norm,
                        "chaga_top3_actions": "|".join(top3_norms),
                        "teacher_accept_top3": bool(getattr(example, "teacher_accept_top3", False)),
                        "has_teacher_distribution": example.teacher_action_distribution is not None,
                        **scores,
                    }
                )
    return rows


def aggregate_original_candidate_metrics(rows: list[dict]) -> dict[str, float | int | None]:
    total = len(rows)
    top1 = sum(1 for row in rows if row.get("top1_match"))
    top3 = sum(1 for row in rows if row.get("top3_match"))
    relaxed = sum(1 for row in rows if row.get("relaxed_match"))
    play_rows = [row for row in rows if str(row.get("chaga_top1_action") or "").startswith("PLAY ")]
    play_total = len(play_rows)
    play_top1 = sum(1 for row in play_rows if row.get("top1_match"))
    play_top3 = sum(1 for row in play_rows if row.get("top3_match"))
    play_relaxed = sum(1 for row in play_rows if row.get("relaxed_match"))
    without_distribution = sum(1 for row in rows if not row.get("has_teacher_distribution"))
    return {
        "original_top1_accuracy": top1 / total if total else None,
        "original_top3_inclusion": top3 / total if total else None,
        "original_relaxed_accuracy": relaxed / total if total else None,
        "original_samples": total,
        "original_play_top1_accuracy": play_top1 / play_total if play_total else None,
        "original_play_top3_inclusion": play_top3 / play_total if play_total else None,
        "original_play_relaxed_accuracy": play_relaxed / play_total if play_total else None,
        "original_play_samples": play_total,
        "reviewed_without_teacher_distribution": without_distribution,
    }


def count_candidate_truncations(examples: Iterable[TransformerExample], *, max_candidates: int) -> int:
    truncated = 0
    for example in examples:
        gated = hu_gated_candidate_mask(example.action_mask, allow_hu=example.allow_hu)
        if int(gated.sum()) > int(max_candidates):
            truncated += 1
    return truncated


def load_checkpoint(path: Path, device: torch.device) -> tuple[TransformerCandidateModel, dict]:
    checkpoint = torch.load(path, map_location=device)
    config = dict(checkpoint.get("config") or {})
    model = TransformerCandidateModel(
        history_vocab_size=config.get("history_vocab_size", 1024),
        d_model=config.get("d_model", 256),
        nhead=config.get("nhead", 8),
        num_layers=config.get("num_layers", 4),
        dim_feedforward=config.get("dim_feedforward", 512),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, config


def load_evaluation_examples(args: argparse.Namespace, config: dict) -> tuple[list[TransformerExample], dict]:
    lookup = load_review_target_lookup(Path(args.review_audit_jsonl))
    return load_examples(
        [Path(path) for path in args.raw],
        history_len=int(config.get("history_len", args.history_len)),
        max_records_per_source=args.max_records_per_source,
        max_examples=args.max_examples,
        teacher_lookup=lookup,
        teacher_temperature=args.teacher_temperature,
        compute_rule_features=not bool(getattr(args, "no_rule_features", False)),
    )


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_checkpoint(Path(args.checkpoint), device)
    examples, load_summary = load_evaluation_examples(args, config)
    reviewed = select_reviewed_examples(examples)
    if not reviewed:
        raise ValueError("no reviewed examples matched the checkpoint evaluation inputs")
    max_candidates = resolve_eval_max_candidates(config, args.max_candidates)
    loader = DataLoader(
        TransformerRawDataset(reviewed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_transformer_examples(items, max_candidates=max_candidates),
    )
    metrics = evaluate_model(model, loader, device)
    original_rows = collect_original_prediction_rows(
        model,
        reviewed,
        max_candidates=max_candidates,
        batch_size=args.batch_size,
        device=device,
    )
    original_metrics = aggregate_original_candidate_metrics(original_rows)
    return {
        "format": "mcr_transformer_chaga_review_eval_v1",
        "checkpoint": args.checkpoint,
        "raw": args.raw,
        "review_audit_jsonl": args.review_audit_jsonl,
        "examples": len(examples),
        "reviewed_examples": len(reviewed),
        "evaluated_candidate_width": max_candidates,
        "compute_rule_features": not bool(getattr(args, "no_rule_features", False)),
        "candidate_truncation_count": count_candidate_truncations(reviewed, max_candidates=max_candidates),
        "load_summary": load_summary,
        **metrics,
        **original_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw", action="append", required=True)
    parser.add_argument("--review-audit-jsonl", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-len", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=FeatureAgent.ACT_SIZE)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--no-rule-features", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(args)
    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
