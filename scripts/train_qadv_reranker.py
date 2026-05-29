#!/usr/bin/env python3
"""Train a small Q-adversarial reranker from qadv_hard_v1 rows."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from mine_chaga_hard_examples import assert_qadv_candidate_digest, validate_qadv_hard_example
from qadv_reranker import QAdvReranker, qadv_family_id, qadv_total_loss, select_qadv_action
from train_transformer_candidate import CANDIDATE_RULE_FEATURES, FeatureAgent


def open_text_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def load_qadv_rows(paths: list[Path], *, strict: bool = True, max_rows: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open_text_maybe_gzip(path) as src:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                assert_qadv_candidate_digest(row)
                if strict:
                    validate_qadv_hard_example(row)
                rows.append(row)
                if max_rows is not None and len(rows) >= int(max_rows):
                    return rows
    return rows


class QAdvDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        if not rows:
            raise ValueError("empty QADV dataset")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def collate_qadv_rows(rows: list[dict]) -> dict[str, torch.Tensor | list[dict]]:
    if not rows:
        raise ValueError("empty QADV batch")
    width = max(len(row.get("candidate_actions") or []) for row in rows)
    batch_size = len(rows)
    candidate_actions = torch.zeros((batch_size, width), dtype=torch.long)
    candidate_family_ids = torch.zeros((batch_size, width), dtype=torch.long)
    candidate_rule_features = torch.zeros((batch_size, width, CANDIDATE_RULE_FEATURES), dtype=torch.float32)
    base_logits = torch.zeros((batch_size, width), dtype=torch.float32)
    base_ranks = torch.full((batch_size, width), float(width), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    accepted_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    hard_negative_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    teacher_dist = torch.zeros((batch_size, width), dtype=torch.float32)
    candidate_is_hu = torch.zeros((batch_size, width), dtype=torch.bool)
    allow_hu = torch.zeros((batch_size,), dtype=torch.bool)
    scalar_features = torch.zeros((batch_size, 3), dtype=torch.float32)
    sample_weight = torch.ones((batch_size,), dtype=torch.float32)

    for row_index, row in enumerate(rows):
        actions = [int(action) for action in row.get("candidate_actions") or []]
        norms = [str(norm) for norm in row.get("candidate_norms") or []]
        accepted = {int(action) for action in row.get("accepted_action_ids") or []}
        hard_negative = {int(action) for action in row.get("hard_negative_action_ids") or []}
        rule_features = list(row.get("candidate_rule_features") or [])
        logits = [float(value) for value in row.get("base_logits") or []]
        ranks = [float(value) for value in row.get("base_ranks") or []]
        dist = [float(value) for value in row.get("teacher_target_dist") or []]
        for slot, action in enumerate(actions):
            candidate_actions[row_index, slot] = action
            candidate_family_ids[row_index, slot] = qadv_family_id(norms[slot] if slot < len(norms) else "")
            candidate_mask[row_index, slot] = True
            accepted_mask[row_index, slot] = action in accepted
            hard_negative_mask[row_index, slot] = action in hard_negative
            candidate_is_hu[row_index, slot] = (norms[slot].upper() if slot < len(norms) else "") == "HU"
            if slot < len(rule_features):
                feature_values = list(rule_features[slot])[:CANDIDATE_RULE_FEATURES]
                candidate_rule_features[row_index, slot, : len(feature_values)] = torch.tensor(feature_values)
            if slot < len(logits):
                base_logits[row_index, slot] = logits[slot]
            if slot < len(ranks):
                base_ranks[row_index, slot] = ranks[slot]
            if slot < len(dist):
                teacher_dist[row_index, slot] = max(0.0, dist[slot])
        dist_total = float(teacher_dist[row_index].sum().item())
        if dist_total > 0.0:
            teacher_dist[row_index] /= dist_total
        scalar = list(row.get("scalar_features") or [])[:3]
        scalar_features[row_index, : len(scalar)] = torch.tensor([float(value) for value in scalar])
        allow_hu[row_index] = bool(row.get("allow_hu", False))
        sample_weight[row_index] = float(row.get("sample_weight", 1.0))

    return {
        "candidate_actions": candidate_actions,
        "candidate_family_ids": candidate_family_ids,
        "candidate_rule_features": candidate_rule_features,
        "base_logits": base_logits,
        "base_ranks": base_ranks,
        "candidate_mask": candidate_mask,
        "accepted_mask": accepted_mask,
        "hard_negative_mask": hard_negative_mask,
        "teacher_dist": teacher_dist,
        "candidate_is_hu": candidate_is_hu,
        "allow_hu": allow_hu,
        "scalar_features": scalar_features,
        "sample_weight": sample_weight,
        "rows": rows,
    }


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def split_rows(rows: list[dict], *, val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_count = int(len(shuffled) * float(val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def _metric_value(value, *, default: float = float("-inf")) -> float:
    if value is None:
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if torch.isfinite(torch.tensor(value)).item() else float(default)


def qadv_tiebreak(metrics: dict) -> float:
    return _metric_value(metrics.get("val_changed_to_accepted_rate"), default=0.0) - _metric_value(
        metrics.get("val_changed_from_accepted_to_wrong_rate"),
        default=0.0,
    )


def is_better_qadv_epoch(
    candidate: dict,
    incumbent: dict | None,
    *,
    monitor_metric: str,
    min_delta: float,
) -> bool:
    if incumbent is None:
        return monitor_metric in candidate and candidate.get(monitor_metric) is not None
    candidate_primary = _metric_value(candidate.get(monitor_metric))
    incumbent_primary = _metric_value(incumbent.get(monitor_metric))
    if candidate_primary > incumbent_primary + float(min_delta):
        return True
    if abs(candidate_primary - incumbent_primary) <= float(min_delta):
        return qadv_tiebreak(candidate) > qadv_tiebreak(incumbent)
    return False


def model_state_cpu(model: QAdvReranker) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def qadv_checkpoint_payload(
    *,
    state_dict: dict[str, torch.Tensor],
    model_config: dict,
    metrics: dict,
) -> dict:
    return {
        "kind": "qadv_reranker",
        "schema": "mcr_qadv_reranker_v1",
        "model_state": state_dict,
        "config": model_config,
        "metrics": metrics,
    }


def accepted_accuracy(model: QAdvReranker, loader: DataLoader, device: torch.device, *, lambda_q: float) -> dict:
    model.eval()
    total = correct = changed = changed_to_accepted = changed_from_accepted = low_fan_hu = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            q_scores = model(batch)
            pred = select_qadv_action(
                batch["base_logits"],
                q_scores,
                batch["candidate_mask"],
                lambda_q=lambda_q,
                candidate_is_hu=batch["candidate_is_hu"],
                allow_hu=batch["allow_hu"],
            )
            base_pred = select_qadv_action(
                batch["base_logits"],
                q_scores,
                batch["candidate_mask"],
                lambda_q=0.0,
                candidate_is_hu=batch["candidate_is_hu"],
                allow_hu=batch["allow_hu"],
            )
            for row_index, slot in enumerate(pred.detach().cpu().tolist()):
                base_slot = int(base_pred[row_index].detach().cpu().item())
                total += 1
                accepted = bool(raw_batch["accepted_mask"][row_index, slot].item())
                base_accepted = bool(raw_batch["accepted_mask"][row_index, base_slot].item())
                if accepted:
                    correct += 1
                if slot != base_slot:
                    changed += 1
                    if accepted and not base_accepted:
                        changed_to_accepted += 1
                    if base_accepted and not accepted:
                        changed_from_accepted += 1
                if bool(raw_batch["candidate_is_hu"][row_index, slot].item()) and not bool(raw_batch["allow_hu"][row_index].item()):
                    low_fan_hu += 1
    return {
        "samples": total,
        "accepted_accuracy": correct / total if total else None,
        "changed_from_base_rate": changed / total if total else None,
        "changed_to_accepted_rate": changed_to_accepted / total if total else None,
        "changed_from_accepted_to_wrong_rate": changed_from_accepted / total if total else None,
        "low_fan_hu_count": low_fan_hu,
    }


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = load_qadv_rows([Path(path) for path in args.hard_jsonl], max_rows=args.max_rows)
    eval_paths = list(getattr(args, "eval_hard_jsonl", []) or [])
    if eval_paths:
        train_rows = rows
        val_rows = load_qadv_rows([Path(path) for path in eval_paths])
    else:
        train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)
    if not train_rows:
        raise ValueError("no QADV training rows")
    if eval_paths and not val_rows:
        raise ValueError("no QADV external validation rows")
    model_config = {
        "action_vocab_size": FeatureAgent.ACT_SIZE,
        "rule_feature_size": CANDIDATE_RULE_FEATURES,
        "scalar_feature_size": 3,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    model = QAdvReranker(
        action_vocab_size=int(model_config["action_vocab_size"]),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    sampler = None
    shuffle = True
    if args.weighted_sampler:
        weights = torch.tensor([float(row.get("sample_weight", 1.0)) for row in train_rows], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        QAdvDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_qadv_rows,
    )
    val_loader = (
        DataLoader(
            QAdvDataset(val_rows),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_qadv_rows,
        )
        if val_rows
        else None
    )

    history: list[dict] = []
    best_epoch_metrics: dict | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_total = 0.0
        batches = 0
        component_totals: dict[str, float] = {}
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            q_scores = model(batch)
            loss, components = qadv_total_loss(
                base_logits=batch["base_logits"],
                q_scores=q_scores,
                candidate_mask=batch["candidate_mask"],
                accepted_mask=batch["accepted_mask"],
                hard_negative_mask=batch["hard_negative_mask"],
                teacher_dist=batch["teacher_dist"],
                train_lambda=args.train_lambda,
                margin=args.margin,
                accept_weight=args.accept_weight,
                pair_weight=args.pair_weight,
                cql_weight=args.cql_weight,
                soft_weight=args.soft_weight,
                q_l2_weight=args.q_l2_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_total += float(loss.detach().cpu().item())
            batches += 1
            for key, value in components.items():
                component_totals[key] = component_totals.get(key, 0.0) + float(value)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, batches),
            **{f"train_{key}": value / max(1, batches) for key, value in component_totals.items()},
        }
        if val_loader is not None:
            epoch_metrics.update({f"val_{key}": value for key, value in accepted_accuracy(model, val_loader, device, lambda_q=args.eval_lambda).items()})
        if is_better_qadv_epoch(
            epoch_metrics,
            best_epoch_metrics,
            monitor_metric=args.monitor_metric,
            min_delta=args.min_delta,
        ):
            best_epoch_metrics = dict(epoch_metrics)
            best_state = model_state_cpu(model)
        history.append(epoch_metrics)

    train_eval_loader = DataLoader(
        QAdvDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_qadv_rows,
    )
    metrics = {
        "format": "mcr_qadv_reranker_train_v1",
        "hard_jsonl": args.hard_jsonl,
        "eval_hard_jsonl": eval_paths,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "epochs": args.epochs,
        "history": history,
        "train_eval": accepted_accuracy(model, train_eval_loader, device, lambda_q=args.eval_lambda),
        "model_config": model_config,
        "monitor_metric": args.monitor_metric,
        "min_delta": args.min_delta,
        "best_epoch": int(best_epoch_metrics["epoch"]) if best_epoch_metrics else None,
        "best_metric": (
            _metric_value(best_epoch_metrics.get(args.monitor_metric))
            if best_epoch_metrics
            else None
        ),
        "best_tiebreak": qadv_tiebreak(best_epoch_metrics) if best_epoch_metrics else None,
        "best_model_out": args.best_model_out,
    }
    if val_loader is not None:
        metrics["val_eval"] = accepted_accuracy(model, val_loader, device, lambda_q=args.eval_lambda)

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(qadv_checkpoint_payload(state_dict=model.state_dict(), model_config=model_config, metrics=metrics), model_path)
    if args.best_model_out:
        best_path = Path(args.best_model_out)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        if best_state is None:
            best_state = model_state_cpu(model)
        torch.save(qadv_checkpoint_payload(state_dict=best_state, model_config=model_config, metrics=metrics), best_path)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-jsonl", action="append", required=True)
    parser.add_argument("--eval-hard-jsonl", action="append", default=[])
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--best-model-out", default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--monitor-metric", default="val_accepted_accuracy")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--train-lambda", type=float, default=1.0)
    parser.add_argument("--eval-lambda", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--accept-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--cql-weight", type=float, default=0.03)
    parser.add_argument("--soft-weight", type=float, default=0.05)
    parser.add_argument("--q-l2-weight", type=float, default=0.0001)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
