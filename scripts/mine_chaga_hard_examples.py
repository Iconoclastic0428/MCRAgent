#!/usr/bin/env python3
"""Mine model-vs-CHAGA hard examples for targeted finetuning."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from evaluate_transformer_chaga_review import (
    collect_original_prediction_rows,
    load_checkpoint,
    load_evaluation_examples,
    resolve_eval_max_candidates,
    select_reviewed_examples,
)
from train_transformer_candidate import normalize_teacher_action


def action_family(action: str | None) -> str:
    normalized = normalize_teacher_action(action)
    return normalized.split()[0] if normalized else ""


def split_actions(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(action for action in (normalize_teacher_action(part) for part in str(value).split("|")) if action)


def classify_prediction_row(row: dict) -> str:
    if bool(row.get("relaxed_match")):
        return "accepted"
    predicted = normalize_teacher_action(row.get("predicted_normalized") or row.get("predicted_action"))
    top1 = normalize_teacher_action(row.get("chaga_top1_action"))
    top3 = split_actions(row.get("chaga_top3_actions"))
    if not predicted or not top1:
        return "unmapped"
    if action_family(predicted) != action_family(top1):
        return "wrong_family"
    if action_family(top1) in {"CHI", "PENG", "GANG", "BUGANG", "HU", "PASS"}:
        return "claim_or_hu_error"
    if predicted in top3 and not bool(row.get("teacher_accept_top3")):
        return "top3_but_not_relaxed"
    return "same_family_not_top3"


def hard_example_weight(row: dict) -> float:
    error_type = str(row.get("error_type") or classify_prediction_row(row))
    weights = {
        "accepted": 1.0,
        "unmapped": 2.0,
        "top3_but_not_relaxed": 3.0,
        "wrong_family": 4.0,
        "claim_or_hu_error": 4.0,
        "same_family_not_top3": 6.0,
    }
    return weights.get(error_type, 2.0)


def enrich_prediction_row(row: dict) -> dict:
    enriched = dict(row)
    error_type = classify_prediction_row(enriched)
    enriched["error_type"] = error_type
    enriched["teacher_family"] = action_family(enriched.get("chaga_top1_action"))
    enriched["predicted_family"] = action_family(enriched.get("predicted_normalized"))
    enriched["sample_weight"] = hard_example_weight(enriched)
    return enriched


def summarize_rows(rows: list[dict]) -> dict:
    error_counts = Counter(str(row.get("error_type") or "") for row in rows)
    hard_rows = [row for row in rows if row.get("error_type") != "accepted"]
    return {
        "rows": len(rows),
        "hard_rows": len(hard_rows),
        "error_counts": dict(sorted(error_counts.items())),
        "hard_sample_weight_sum": sum(float(row.get("sample_weight", 1.0)) for row in hard_rows),
    }


def mine_hard_examples(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_checkpoint(Path(args.checkpoint), device)
    examples, load_summary = load_evaluation_examples(args, config)
    reviewed = select_reviewed_examples(examples)
    max_candidates = resolve_eval_max_candidates(config, args.max_candidates)
    rows = collect_original_prediction_rows(
        model,
        reviewed,
        max_candidates=max_candidates,
        batch_size=args.batch_size,
        device=device,
    )
    enriched = [enrich_prediction_row(row) for row in rows]
    if args.mismatches_only:
        enriched_to_write = [row for row in enriched if row.get("error_type") != "accepted"]
    else:
        enriched_to_write = enriched
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as dst:
        for row in enriched_to_write:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "format": "mcr_chaga_hard_examples_v1",
        "checkpoint": args.checkpoint,
        "raw": args.raw,
        "review_audit_jsonl": args.review_audit_jsonl,
        "examples": len(examples),
        "reviewed_examples": len(reviewed),
        "written_rows": len(enriched_to_write),
        "evaluated_candidate_width": max_candidates,
        "load_summary": load_summary,
        **summarize_rows(enriched),
    }
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw", action="append", required=True)
    parser.add_argument("--review-audit-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-len", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=235)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--no-rule-features", action="store_true")
    parser.add_argument("--mismatches-only", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    summary = mine_hard_examples(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
