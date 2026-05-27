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
    TransformerCandidateModel,
    TransformerExample,
    TransformerRawDataset,
    collate_transformer_examples,
    evaluate_model,
    load_examples,
    load_review_target_lookup,
)


def select_reviewed_examples(examples: Iterable[TransformerExample]) -> list[TransformerExample]:
    return [example for example in examples if example.teacher_action_distribution is not None]


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


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_checkpoint(Path(args.checkpoint), device)
    lookup = load_review_target_lookup(Path(args.review_audit_jsonl))
    examples, load_summary = load_examples(
        [Path(path) for path in args.raw],
        history_len=int(config.get("history_len", args.history_len)),
        max_records_per_source=args.max_records_per_source,
        max_examples=args.max_examples,
        teacher_lookup=lookup,
        teacher_temperature=args.teacher_temperature,
    )
    reviewed = select_reviewed_examples(examples)
    if not reviewed:
        raise ValueError("no reviewed examples matched the checkpoint evaluation inputs")
    max_candidates = int(config.get("max_candidates", args.max_candidates))
    loader = DataLoader(
        TransformerRawDataset(reviewed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_transformer_examples(items, max_candidates=max_candidates),
    )
    metrics = evaluate_model(model, loader, device)
    return {
        "format": "mcr_transformer_chaga_review_eval_v1",
        "checkpoint": args.checkpoint,
        "raw": args.raw,
        "review_audit_jsonl": args.review_audit_jsonl,
        "examples": len(examples),
        "reviewed_examples": len(reviewed),
        "load_summary": load_summary,
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw", action="append", required=True)
    parser.add_argument("--review-audit-jsonl", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-len", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
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
