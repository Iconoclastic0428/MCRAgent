#!/usr/bin/env python3
"""Summarize CHAGA reviewed train/validation coverage before L40 finetuning."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from train_transformer_candidate import (
    TransformerExample,
    hu_gated_candidate_mask,
    load_examples,
    load_review_target_lookup,
    normalize_teacher_action,
)


def action_family(action: str | None) -> str:
    normalized = normalize_teacher_action(action)
    return normalized.split()[0] if normalized else "UNKNOWN"


def bucket_turn(turn: int) -> str:
    if turn <= 10:
        return "turn=0-10"
    if turn <= 30:
        return "turn=11-30"
    if turn <= 60:
        return "turn=31-60"
    return "turn=61+"


def bucket_candidate_count(count: int) -> str:
    if count <= 5:
        return "candidates=1-5"
    if count <= 20:
        return "candidates=6-20"
    if count <= 60:
        return "candidates=21-60"
    return "candidates=61+"


def row_from_example(example: TransformerExample) -> dict:
    top1 = example.teacher_candidate_norms[0] if example.teacher_candidate_norms else ""
    candidate_count = int(hu_gated_candidate_mask(example.action_mask, allow_hu=example.allow_hu).sum())
    return {
        "record_id": getattr(example, "record_id", ""),
        "session_id": getattr(example, "session_id", ""),
        "player": int(example.player),
        "turn": int(example.turn),
        "teacher_family": action_family(top1),
        "candidate_count": candidate_count,
        "turn_bucket": bucket_turn(int(example.turn)),
        "candidate_count_bucket": bucket_candidate_count(candidate_count),
        "relaxed_region": "first_four_play" if bool(example.teacher_accept_top3) else "other",
    }


def slice_key(row: dict) -> str:
    return "|".join(
        [
            f"family={row.get('teacher_family', 'UNKNOWN')}",
            str(row.get("turn_bucket") or bucket_turn(int(row.get("turn", 0)))),
            str(row.get("candidate_count_bucket") or bucket_candidate_count(int(row.get("candidate_count", 0)))),
            f"region={row.get('relaxed_region', 'other')}",
        ]
    )


def summarize_coverage_rows(
    train_rows: list[dict],
    val_rows: list[dict],
    *,
    min_train_to_val_ratio: float = 5.0,
) -> dict:
    train_slices = Counter(slice_key(row) for row in train_rows)
    val_slices = Counter(slice_key(row) for row in val_rows)
    overall_ratio = len(train_rows) / max(1, len(val_rows))
    worst_slice = None
    worst_ratio = None
    for key, val_count in val_slices.items():
        ratio = train_slices.get(key, 0) / max(1, val_count)
        if worst_ratio is None or ratio < worst_ratio:
            worst_ratio = ratio
            worst_slice = key
    return {
        "format": "mcr_chaga_coverage_v1",
        "train_reviewed_states": len(train_rows),
        "val_reviewed_states": len(val_rows),
        "train_by_family": dict(sorted(Counter(row.get("teacher_family", "UNKNOWN") for row in train_rows).items())),
        "val_by_family": dict(sorted(Counter(row.get("teacher_family", "UNKNOWN") for row in val_rows).items())),
        "train_by_slice": dict(sorted(train_slices.items())),
        "val_by_slice": dict(sorted(val_slices.items())),
        "gates": {
            "overall_train_to_val_ratio": {
                "ratio": overall_ratio,
                "required": min_train_to_val_ratio,
                "passed": overall_ratio >= min_train_to_val_ratio,
            },
            "slice_train_to_val_ratio": {
                "worst_slice": worst_slice,
                "worst_ratio": worst_ratio,
                "required": min_train_to_val_ratio,
                "passed": worst_ratio is not None and worst_ratio >= min_train_to_val_ratio,
            },
        },
    }


def load_reviewed_rows(raw_paths: list[str], audit_path: str, *, max_examples: int | None, no_rule_features: bool) -> tuple[list[dict], dict]:
    lookup = load_review_target_lookup(Path(audit_path))
    examples, summary = load_examples(
        [Path(path) for path in raw_paths],
        history_len=80,
        max_examples=max_examples,
        teacher_lookup=lookup,
        reviewed_only=False,
        compute_rule_features=not no_rule_features,
    )
    reviewed = [example for example in examples if example.teacher_candidate_norms]
    return [row_from_example(example) for example in reviewed], summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-raw", action="append", required=True)
    parser.add_argument("--val-raw", action="append", required=True)
    parser.add_argument("--review-audit-jsonl", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--min-train-to-val-ratio", type=float, default=5.0)
    parser.add_argument("--no-rule-features", action="store_true")
    args = parser.parse_args()

    train_rows, train_load_summary = load_reviewed_rows(
        args.train_raw,
        args.review_audit_jsonl,
        max_examples=args.max_train_examples,
        no_rule_features=args.no_rule_features,
    )
    val_rows, val_load_summary = load_reviewed_rows(
        args.val_raw,
        args.review_audit_jsonl,
        max_examples=args.max_val_examples,
        no_rule_features=args.no_rule_features,
    )
    summary = summarize_coverage_rows(
        train_rows,
        val_rows,
        min_train_to_val_ratio=args.min_train_to_val_ratio,
    )
    summary["train_load_summary"] = train_load_summary
    summary["val_load_summary"] = val_load_summary
    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
