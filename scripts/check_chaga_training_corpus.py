#!/usr/bin/env python3
"""Dry-run gates for CHAGA-reviewed Transformer training corpora."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from train_transformer_candidate import (
    DEFAULT_HISTORY_LEN,
    FeatureAgent,
    TransformerExample,
    action_response,
    collate_transformer_examples,
    filter_reviewed_examples,
    has_mapped_teacher_distribution,
    hu_gated_candidate_mask,
    is_reviewed_example,
    load_examples,
    load_review_target_lookup,
    normalize_teacher_action,
)


SPLITS = ("train", "val", "test")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def raw_sessions(path: Path) -> set[str]:
    return {
        str(row.get("belongs") or "")
        for row in iter_jsonl(path)
        if row.get("belongs")
    }


def audit_sessions(path: Path) -> set[str]:
    return {
        str(row.get("session_id") or "")
        for row in iter_jsonl(path)
        if row.get("session_id")
    }


def assert_session_disjoint(sessions_by_split: dict[str, set[str]]) -> None:
    seen: dict[str, str] = {}
    overlaps: list[tuple[str, str, str]] = []
    for split, sessions in sessions_by_split.items():
        for session in sessions:
            previous = seen.get(session)
            if previous is not None:
                overlaps.append((session, previous, split))
            else:
                seen[session] = split
    if overlaps:
        preview = ", ".join(f"{session}:{left}/{right}" for session, left, right in overlaps[:10])
        raise ValueError(f"session overlap across splits: {preview}")


def assert_train_val_audit_excludes_test(train_val_sessions: set[str], test_sessions: set[str]) -> None:
    overlap = sorted(train_val_sessions & test_sessions)
    if overlap:
        raise ValueError(f"train-val audit contains test sessions: {overlap[:10]}")


def load_reviewed_split(
    *,
    split: str,
    raw_path: Path,
    audit_path: Path,
    history_len: int,
    teacher_temperature: float,
) -> tuple[list[TransformerExample], dict]:
    lookup = load_review_target_lookup(audit_path)
    examples, load_summary = load_examples(
        [raw_path],
        history_len=history_len,
        teacher_lookup=lookup,
        teacher_temperature=teacher_temperature,
    )
    reviewed = [example for example in examples if is_reviewed_example(example)]
    reviewed_with_distribution, filter_summary = filter_reviewed_examples(
        examples,
        require_distribution=True,
    )
    summary = {
        "split": split,
        "raw_path": str(raw_path),
        "audit_path": str(audit_path),
        "raw_sessions": len(raw_sessions(raw_path)),
        "audit_sessions": len(audit_sessions(audit_path)),
        "examples": len(examples),
        "reviewed_examples": len(reviewed),
        "reviewed_examples_with_distribution": len(reviewed_with_distribution),
        "load_summary": load_summary,
        "review_filter_summary": filter_summary,
    }
    return reviewed, summary


def validate_reviewed_examples(
    split: str,
    reviewed_examples: list[TransformerExample],
    *,
    max_candidates: int,
    chunk_size: int = 512,
) -> dict[str, int | str]:
    if max_candidates < FeatureAgent.ACT_SIZE:
        raise ValueError(f"{split}: max_candidates must be {FeatureAgent.ACT_SIZE} for reviewed CHAGA gates")
    if not reviewed_examples:
        raise ValueError(f"{split}: reviewed_examples must be non-empty")

    counts: Counter[str] = Counter()
    for example in reviewed_examples:
        counts["reviewed_examples"] += 1
        if not example.teacher_candidate_norms:
            counts["missing_original_candidates"] += 1
        if not has_mapped_teacher_distribution(example):
            counts["without_mapped_teacher_distribution"] += 1
        if example.teacher_accept_top3:
            counts["accept_top3_rows"] += 1
            top1 = normalize_teacher_action(example.teacher_candidate_norms[0] if example.teacher_candidate_norms else "")
            if not top1.startswith("PLAY "):
                counts["top3_relaxation_non_play"] += 1
        gated_candidates = [int(action) for action in np.flatnonzero(hu_gated_candidate_mask(example.action_mask, allow_hu=example.allow_hu) > 0)]
        if len(gated_candidates) > max_candidates:
            counts["candidate_truncation_count"] += 1
        accepted_norms = accepted_teacher_norms(example)
        if "HU" in accepted_norms:
            counts["accepted_hu_rows"] += 1
            legal_hu = any(normalize_teacher_action(action_response(action)) == "HU" for action in gated_candidates)
            if not legal_hu:
                counts["accepted_hu_outside_legal_mask"] += 1
            if not example.allow_hu:
                counts["accepted_hu_when_allow_hu_false"] += 1

    for chunk in chunks(reviewed_examples, chunk_size):
        batch = collate_transformer_examples(chunk, max_candidates=max_candidates)
        has_original = batch["has_teacher_original"].bool()
        has_accept = batch["has_teacher_accept_set"].bool()
        top1_mask = batch["teacher_original_top1_mask"].bool()
        accept_mask = batch["teacher_accept_mask"].bool()
        counts["missing_has_teacher_original"] += int((~has_original).sum().item())
        counts["missing_has_teacher_accept_set"] += int((~has_accept).sum().item())
        counts["empty_top1_mask"] += int((~top1_mask.any(dim=1)).sum().item())
        counts["empty_accept_mask"] += int((~accept_mask.any(dim=1)).sum().item())

    failures = {
        "missing_original_candidates": "missing original CHAGA candidates",
        "without_mapped_teacher_distribution": "without mapped teacher distribution",
        "candidate_truncation_count": "candidate truncation",
        "top3_relaxation_non_play": "top-3 relaxation on non-PLAY top1",
        "accepted_hu_outside_legal_mask": "accepted HU outside legal mask",
        "accepted_hu_when_allow_hu_false": "accepted HU when allow_hu=false",
        "empty_accept_mask": "empty accepted candidate mask",
        "empty_top1_mask": "empty original top1 mask",
        "missing_has_teacher_original": "missing original teacher mask",
        "missing_has_teacher_accept_set": "missing accepted teacher mask",
    }
    for key in failures:
        counts.setdefault(key, 0)
    for key, message in failures.items():
        if counts.get(key, 0):
            raise ValueError(f"{split}: {message}: {counts[key]}")
    counts["max_candidates"] = max_candidates
    counts["split"] = split
    return dict(counts)


def accepted_teacher_norms(example: TransformerExample) -> set[str]:
    candidates = (
        example.teacher_candidate_norms[:3]
        if example.teacher_accept_top3
        else example.teacher_candidate_norms[:1]
    )
    return {
        normalize_teacher_action(action)
        for action in candidates
        if normalize_teacher_action(action)
    }


def chunks(items: list[TransformerExample], chunk_size: int) -> Iterable[list[TransformerExample]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def enforce_minimums(summary: dict, *, min_reviewed: int, min_sessions: int) -> None:
    if int(summary["reviewed_examples"]) < min_reviewed:
        raise ValueError(
            f"{summary['split']}: reviewed_examples {summary['reviewed_examples']} < required {min_reviewed}"
        )
    if int(summary["audit_sessions"]) < min_sessions:
        raise ValueError(
            f"{summary['split']}: audit_sessions {summary['audit_sessions']} < required {min_sessions}"
        )


def check_corpus(args: argparse.Namespace) -> dict:
    paths = {
        "train": (Path(args.train_raw), Path(args.train_audit)),
        "val": (Path(args.val_raw), Path(args.val_audit)),
    }
    if args.test_raw and args.test_audit:
        paths["test"] = (Path(args.test_raw), Path(args.test_audit))

    raw_sessions_by_split = {split: raw_sessions(raw_path) for split, (raw_path, _) in paths.items()}
    audit_sessions_by_split = {split: audit_sessions(audit_path) for split, (_, audit_path) in paths.items()}
    assert_session_disjoint(raw_sessions_by_split)
    assert_session_disjoint(audit_sessions_by_split)
    if args.train_val_audit and "test" in audit_sessions_by_split:
        assert_train_val_audit_excludes_test(audit_sessions(Path(args.train_val_audit)), audit_sessions_by_split["test"])

    split_summaries = {}
    min_reviewed = {"train": args.min_train_reviewed, "val": args.min_val_reviewed, "test": args.min_test_reviewed}
    min_sessions = {"train": args.min_train_sessions, "val": args.min_val_sessions, "test": args.min_test_sessions}
    for split, (raw_path, audit_path) in paths.items():
        reviewed, load_summary = load_reviewed_split(
            split=split,
            raw_path=raw_path,
            audit_path=audit_path,
            history_len=args.history_len,
            teacher_temperature=args.teacher_temperature,
        )
        enforce_minimums(
            load_summary,
            min_reviewed=min_reviewed.get(split, 0),
            min_sessions=min_sessions.get(split, 0),
        )
        gate_summary = validate_reviewed_examples(
            split,
            reviewed,
            max_candidates=args.max_candidates,
            chunk_size=args.chunk_size,
        )
        split_summaries[split] = {**load_summary, "gate_summary": gate_summary}

    return {
        "format": "mcr_chaga_training_corpus_gate_v1",
        "max_candidates": args.max_candidates,
        "history_len": args.history_len,
        "teacher_temperature": args.teacher_temperature,
        "splits": split_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-raw", required=True)
    parser.add_argument("--val-raw", required=True)
    parser.add_argument("--test-raw")
    parser.add_argument("--train-audit", required=True)
    parser.add_argument("--val-audit", required=True)
    parser.add_argument("--test-audit")
    parser.add_argument("--train-val-audit")
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--history-len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=FeatureAgent.ACT_SIZE)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--min-train-reviewed", type=int, default=25000)
    parser.add_argument("--min-val-reviewed", type=int, default=5000)
    parser.add_argument("--min-test-reviewed", type=int, default=5000)
    parser.add_argument("--min-train-sessions", type=int, default=50)
    parser.add_argument("--min-val-sessions", type=int, default=10)
    parser.add_argument("--min-test-sessions", type=int, default=10)
    args = parser.parse_args()

    summary = check_corpus(args)
    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
