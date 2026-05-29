#!/usr/bin/env python3
"""Sweep Q-adversarial reranker lambda values on qadv_hard_v1 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qadv_reranker import QAdvReranker, qadv_final_scores, select_qadv_action
from train_qadv_reranker import QAdvDataset, collate_qadv_rows, load_qadv_rows
from train_transformer_candidate import FeatureAgent


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def load_qadv_checkpoint(path: Path, device: torch.device) -> QAdvReranker:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("kind") != "qadv_reranker":
        raise ValueError(f"not a qadv_reranker checkpoint: {path}")
    config = dict(checkpoint.get("config") or {})
    model = QAdvReranker(
        action_vocab_size=int(config.get("action_vocab_size", FeatureAgent.ACT_SIZE)),
        rule_feature_size=int(config.get("rule_feature_size", 7)),
        scalar_feature_size=int(config.get("scalar_feature_size", 3)),
        hidden_size=int(config.get("hidden_size", 256)),
        num_layers=int(config.get("num_layers", 3)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _row_top_norms(row: dict) -> tuple[str, tuple[str, ...], bool]:
    teacher = tuple(str(norm) for norm in (row.get("teacher_candidate_norms") or []))
    top1 = teacher[0] if teacher else ""
    return top1, teacher[:3], top1.startswith("PLAY ")


def _metric_denominator(value: int, total: int) -> float | None:
    return value / total if total else None


def evaluate_lambda_values(
    model: QAdvReranker,
    loader: DataLoader,
    device: torch.device,
    *,
    lambdas: list[float],
) -> dict[str, dict]:
    accum: dict[float, dict[str, int]] = {
        lam: {
            "samples": 0,
            "accepted": 0,
            "top1": 0,
            "top3": 0,
            "play_samples": 0,
            "play_accepted": 0,
            "play_top1": 0,
            "play_top3": 0,
            "changed": 0,
            "changed_to_accepted": 0,
            "changed_from_accepted_to_wrong": 0,
            "action_outside_mask": 0,
            "low_fan_hu": 0,
            "lambda_zero_reproduction_mismatches": 0,
        }
        for lam in lambdas
    }
    base_mismatch_checked = False
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            q_scores = model(batch)
            base_pred = select_qadv_action(
                batch["base_logits"],
                q_scores,
                batch["candidate_mask"],
                lambda_q=0.0,
                candidate_is_hu=batch["candidate_is_hu"],
                allow_hu=batch["allow_hu"],
            )
            raw_base_pred = torch.argmax(
                qadv_final_scores(
                    batch["base_logits"],
                    torch.zeros_like(q_scores),
                    batch["candidate_mask"],
                    lambda_q=0.0,
                    candidate_is_hu=batch["candidate_is_hu"],
                    allow_hu=batch["allow_hu"],
                ),
                dim=1,
            )
            for lam in lambdas:
                pred = select_qadv_action(
                    batch["base_logits"],
                    q_scores,
                    batch["candidate_mask"],
                    lambda_q=lam,
                    candidate_is_hu=batch["candidate_is_hu"],
                    allow_hu=batch["allow_hu"],
                )
                stats = accum[lam]
                for row_index, slot in enumerate(pred.detach().cpu().tolist()):
                    slot = int(slot)
                    base_slot = int(base_pred[row_index].detach().cpu().item())
                    stats["samples"] += 1
                    row = raw_batch["rows"][row_index]
                    norms = [str(norm) for norm in row.get("candidate_norms") or []]
                    predicted_norm = norms[slot] if 0 <= slot < len(norms) else ""
                    top1_norm, top3_norms, top1_is_play = _row_top_norms(row)
                    accepted = bool(raw_batch["accepted_mask"][row_index, slot].item())
                    base_accepted = bool(raw_batch["accepted_mask"][row_index, base_slot].item())
                    if accepted:
                        stats["accepted"] += 1
                    if predicted_norm == top1_norm:
                        stats["top1"] += 1
                    if predicted_norm in set(top3_norms):
                        stats["top3"] += 1
                    if top1_is_play:
                        stats["play_samples"] += 1
                        if accepted:
                            stats["play_accepted"] += 1
                        if predicted_norm == top1_norm:
                            stats["play_top1"] += 1
                        if predicted_norm in set(top3_norms):
                            stats["play_top3"] += 1
                    if slot != base_slot:
                        stats["changed"] += 1
                        if accepted and not base_accepted:
                            stats["changed_to_accepted"] += 1
                        if base_accepted and not accepted:
                            stats["changed_from_accepted_to_wrong"] += 1
                    if not bool(raw_batch["candidate_mask"][row_index, slot].item()):
                        stats["action_outside_mask"] += 1
                    if bool(raw_batch["candidate_is_hu"][row_index, slot].item()) and not bool(raw_batch["allow_hu"][row_index].item()):
                        stats["low_fan_hu"] += 1
                    if lam == 0.0 and int(raw_base_pred[row_index].detach().cpu().item()) != slot:
                        stats["lambda_zero_reproduction_mismatches"] += 1
            base_mismatch_checked = True
    if not base_mismatch_checked:
        return {}
    metrics: dict[str, dict] = {}
    for lam in lambdas:
        stats = accum[lam]
        total = stats["samples"]
        play_total = stats["play_samples"]
        metrics[f"{lam:.2f}"] = {
            "samples": total,
            "original_relaxed_accuracy": _metric_denominator(stats["accepted"], total),
            "original_top1_accuracy": _metric_denominator(stats["top1"], total),
            "original_top3_inclusion": _metric_denominator(stats["top3"], total),
            "original_play_relaxed_accuracy": _metric_denominator(stats["play_accepted"], play_total),
            "original_play_top1_accuracy": _metric_denominator(stats["play_top1"], play_total),
            "original_play_top3_inclusion": _metric_denominator(stats["play_top3"], play_total),
            "original_play_samples": play_total,
            "changed_from_base_rate": _metric_denominator(stats["changed"], total),
            "changed_to_accepted_rate": _metric_denominator(stats["changed_to_accepted"], total),
            "changed_from_accepted_to_wrong_rate": _metric_denominator(stats["changed_from_accepted_to_wrong"], total),
            "action_outside_mask": stats["action_outside_mask"],
            "low_fan_hu_count": stats["low_fan_hu"],
            "lambda_zero_reproduction_mismatches": stats["lambda_zero_reproduction_mismatches"],
        }
    return metrics


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = load_qadv_rows([Path(path) for path in args.hard_jsonl], strict=True, max_rows=args.max_rows)
    model = load_qadv_checkpoint(Path(args.q_checkpoint), device)
    loader = DataLoader(
        QAdvDataset(rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_qadv_rows,
    )
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]
    lambda_metrics = evaluate_lambda_values(model, loader, device, lambdas=lambdas)
    summary = {
        "format": "mcr_qadv_reranker_eval_v1",
        "hard_jsonl": args.hard_jsonl,
        "q_checkpoint": args.q_checkpoint,
        "rows": len(rows),
        "candidate_truncation_count": sum(1 for row in rows if row.get("candidate_truncated")),
        "reviewed_without_teacher_distribution": sum(1 for row in rows if not row.get("has_teacher_distribution")),
        "lambda_metrics": lambda_metrics,
    }
    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-jsonl", action="append", required=True)
    parser.add_argument("--q-checkpoint", required=True)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--lambdas", default="0.00,0.05,0.10,0.20,0.35,0.50")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    metrics = evaluate(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
