#!/usr/bin/env python3
"""Visualize where Transformer predictions diverge from CHAGA review candidates."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate_transformer_chaga_review import load_checkpoint, select_reviewed_examples
from train_transformer_candidate import (
    TransformerRawDataset,
    action_response,
    collate_transformer_examples,
    load_examples,
    load_review_target_lookup,
)


def aggregate_by_turn(rows: list[dict]) -> dict[int, dict]:
    stats: dict[int, dict] = {}
    for row in rows:
        turn = int(row["turn"])
        bucket = stats.setdefault(
            turn,
            {
                "turn": turn,
                "total": 0,
                "top1_match": 0,
                "top3_match": 0,
                "relaxed_match": 0,
                "top1_mismatch": 0,
                "top3_mismatch": 0,
                "relaxed_mismatch": 0,
            },
        )
        bucket["total"] += 1
        for key in ("top1_match", "top3_match", "relaxed_match"):
            if row.get(key):
                bucket[key] += 1
    for bucket in stats.values():
        total = bucket["total"]
        bucket["top1_mismatch"] = total - bucket["top1_match"]
        bucket["top3_mismatch"] = total - bucket["top3_match"]
        bucket["relaxed_mismatch"] = total - bucket["relaxed_match"]
        bucket["top1_mismatch_rate"] = bucket["top1_mismatch"] / total if total else None
        bucket["top3_mismatch_rate"] = bucket["top3_mismatch"] / total if total else None
        bucket["relaxed_mismatch_rate"] = bucket["relaxed_mismatch"] / total if total else None
    return dict(sorted(stats.items()))


def render_svg(stats: dict[int, dict], *, title: str) -> str:
    width = max(900, 80 + 22 * max(1, len(stats)))
    height = 520
    margin_left = 70
    margin_bottom = 90
    plot_w = width - margin_left - 30
    plot_h = height - 120
    max_total = max((bucket["total"] for bucket in stats.values()), default=1)
    bar_gap = 4
    bar_w = max(6, (plot_w - bar_gap * max(0, len(stats) - 1)) / max(1, len(stats)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfa"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{html.escape(title)}</text>',
        '<text x="20" y="260" transform="rotate(-90 20 260)" font-family="Arial, sans-serif" font-size="13" fill="#444">reviewed decisions</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 30}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="70" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        '<g font-family="Arial, sans-serif" font-size="11" fill="#444">',
    ]
    for tick in range(0, max_total + 1, max(1, max_total // 5)):
        y = height - margin_bottom - (tick / max_total) * plot_h
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - 30}" y2="{y:.1f}" stroke="#e4e4e0" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
    parts.append("</g>")

    for index, (turn, bucket) in enumerate(stats.items()):
        x = margin_left + index * (bar_w + bar_gap)
        total_h = (bucket["total"] / max_total) * plot_h
        mismatch_h = (bucket["relaxed_mismatch"] / max_total) * plot_h
        y_total = height - margin_bottom - total_h
        y_mismatch = height - margin_bottom - mismatch_h
        tooltip = (
            f"turn {turn}: total={bucket['total']}, relaxed mismatches={bucket['relaxed_mismatch']}, "
            f"relaxed mismatch rate={bucket['relaxed_mismatch_rate']:.1%}"
        )
        parts.extend(
            [
                f"<g><title>{html.escape(tooltip)}</title>",
                f'<rect x="{x:.1f}" y="{y_total:.1f}" width="{bar_w:.1f}" height="{total_h:.1f}" fill="#d9e2ec"/>',
                f'<rect x="{x:.1f}" y="{y_mismatch:.1f}" width="{bar_w:.1f}" height="{mismatch_h:.1f}" fill="#c94f4f"/>',
                f'<text x="{x + bar_w / 2:.1f}" y="{height - margin_bottom + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" transform="rotate(65 {x + bar_w / 2:.1f} {height - margin_bottom + 18})">turn {turn}</text>',
                "</g>",
            ]
        )
    legend_y = height - 38
    parts.extend(
        [
            f'<rect x="{margin_left}" y="{legend_y - 12}" width="14" height="14" fill="#d9e2ec"/><text x="{margin_left + 20}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12">reviewed model decisions</text>',
            f'<rect x="{margin_left + 210}" y="{legend_y - 12}" width="14" height="14" fill="#c94f4f"/><text x="{margin_left + 230}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12">relaxed mismatches vs CHAGA</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def collect_prediction_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
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
        raise ValueError("no reviewed examples matched")
    max_candidates = int(config.get("max_candidates", args.max_candidates))
    loader = DataLoader(
        TransformerRawDataset(reviewed),
        batch_size=args.batch_size,
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
                pred_action = int(batch["candidate_actions"][index, pred_slot].item())
                teacher_dist = batch["teacher_target_dist"][index]
                positive = torch.nonzero(teacher_dist > 0, as_tuple=False).flatten()
                top1_slot = int(torch.argmax(teacher_dist).item())
                top3_slots = set(torch.topk(teacher_dist, k=min(3, int(positive.numel()))).indices.tolist())
                top1_match = pred_slot == top1_slot
                top3_match = pred_slot in top3_slots
                relaxed_match = top1_match or (bool(batch["teacher_accept_top3"][index].item()) and top3_match)
                rows.append(
                    {
                        "turn": int(example.turn),
                        "player": int(example.player),
                        "response": example.response,
                        "predicted_action": action_response(pred_action),
                        "chaga_top1_action": action_response(int(batch["candidate_actions"][index, top1_slot].item())),
                        "teacher_accept_top3": bool(batch["teacher_accept_top3"][index].item()),
                        "top1_match": top1_match,
                        "top3_match": top3_match,
                        "relaxed_match": relaxed_match,
                    }
                )
    return rows, {"examples": len(examples), "reviewed_examples": len(reviewed), "load_summary": load_summary}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw", action="append", required=True)
    parser.add_argument("--review-audit-jsonl", required=True)
    parser.add_argument("--svg-out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-len", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows, info = collect_prediction_rows(args)
    stats = aggregate_by_turn(rows)
    svg = render_svg(stats, title="Transformer relaxed mismatches vs CHAGA review candidates by turn")
    svg_path = Path(args.svg_out)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")
    write_csv(Path(args.csv_out), rows)
    summary = {"checkpoint": args.checkpoint, "raw": args.raw, **info, "by_turn": stats}
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
