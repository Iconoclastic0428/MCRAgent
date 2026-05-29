#!/usr/bin/env python3
"""Mine model-vs-CHAGA hard examples for targeted finetuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections import deque
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate_transformer_chaga_review import (
    collect_original_prediction_rows,
    load_checkpoint,
    load_evaluation_examples,
    resolve_eval_max_candidates,
    select_reviewed_examples,
)
from train_transformer_candidate import (
    CANDIDATE_RULE_FEATURES,
    FeatureAgent,
    TransformerRawDataset,
    action_response,
    collate_transformer_examples,
    normalize_teacher_action,
    normalize_teacher_candidates,
    _is_first_four_discard_review_entry,
    _review_checks_pass,
)


QADV_SCHEMA = "qadv_hard_v1"
QADV_HARD_WEIGHTS = {
    "accepted": 1.0,
    "top3_but_not_accepted_under_strict_rule": 4.0,
    "wrong_family": 5.0,
    "same_family_top5_rank_2_5": 4.0,
    "same_family_not_chaga_top5": 6.0,
    "high_margin_mismatch": 6.0,
    "claim_or_hu_error": 3.0,
    "unmapped": 2.0,
}


def action_family(action: str | None) -> str:
    normalized = normalize_teacher_action(action)
    return normalized.split()[0] if normalized else ""


def split_actions(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(action for action in (normalize_teacher_action(part) for part in str(value).split("|")) if action)


def qadv_accepted_norms(
    teacher_candidate_norms: tuple[str, ...] | list[str],
    *,
    play_ordinal: int | None = None,
    relaxed_play_top3_count: int = 4,
    accept_top3: bool | None = None,
) -> tuple[str, ...]:
    """Return the strict QADV accepted set for one reviewed CHAGA decision."""

    normalized = tuple(normalize_teacher_action(action) for action in teacher_candidate_norms if normalize_teacher_action(action))
    if not normalized:
        return ()
    top1 = normalized[0]
    if not top1.startswith("PLAY "):
        return normalized[:1]
    if accept_top3 is None:
        accept_top3 = play_ordinal is not None and 1 <= int(play_ordinal) <= int(relaxed_play_top3_count)
    return normalized[:3] if accept_top3 else normalized[:1]


def qadv_candidate_digest(candidate_actions: list[int] | tuple[int, ...], candidate_norms: list[str] | tuple[str, ...]) -> str:
    payload = [
        {"action": int(action), "norm": normalize_teacher_action(norm)}
        for action, norm in zip(candidate_actions, candidate_norms)
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_qadv_candidate_digest(row: dict) -> None:
    expected = qadv_candidate_digest(row.get("candidate_actions") or [], row.get("candidate_norms") or [])
    if str(row.get("candidate_digest") or "") != expected:
        raise ValueError(f"candidate digest mismatch for {row.get('example_key') or '<unknown>'}")


def _ranked_indices(scores: list[float], *, limit: int | None = None) -> list[int]:
    ranked = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
    return ranked if limit is None else ranked[:limit]


def _qadv_family(norm: str | None) -> str:
    normalized = normalize_teacher_action(norm)
    return normalized.split()[0] if normalized else ""


def _qadv_hard_weight(error_type: str, hard_negative_reasons: dict[str, list[str]] | None = None) -> float:
    weight = float(QADV_HARD_WEIGHTS.get(error_type, 2.0))
    if hard_negative_reasons:
        for reasons in hard_negative_reasons.values():
            if "high_margin_teacher_disagreement" in reasons:
                weight = max(weight, QADV_HARD_WEIGHTS["high_margin_mismatch"])
    return weight


def _filter_low_fan_hu_candidates(row: dict) -> dict:
    if bool(row.get("allow_hu", False)):
        return dict(row)
    candidate_actions = list(row.get("candidate_actions") or [])
    candidate_norms = [normalize_teacher_action(norm) for norm in (row.get("candidate_norms") or [])]
    keep = [index for index, norm in enumerate(candidate_norms) if norm != "HU"]
    if len(keep) == len(candidate_actions):
        return dict(row)

    filtered = dict(row)
    for key in ["candidate_actions", "candidate_norms", "candidate_rule_features", "base_logits", "teacher_target_dist"]:
        values = list(row.get(key) or [])
        filtered[key] = [values[index] for index in keep if index < len(values)]
    return filtered


def _teacher_distribution_for_candidates(row: dict, candidate_norms: list[str]) -> list[float]:
    raw = list(row.get("teacher_target_dist") or [])
    if len(raw) == len(candidate_norms):
        total = float(sum(max(0.0, float(value)) for value in raw))
        if total > 0.0:
            return [max(0.0, float(value)) / total for value in raw]
    return [0.0 for _ in candidate_norms]


def _action_ids_for_norms(
    candidate_actions: list[int],
    candidate_norms: list[str],
    accepted_norms: tuple[str, ...],
) -> list[int]:
    accepted = set(accepted_norms)
    return [
        int(action)
        for action, norm in zip(candidate_actions, candidate_norms)
        if normalize_teacher_action(norm) in accepted
    ]


def build_qadv_hard_example(
    source_row: dict,
    *,
    relaxed_play_top3_count: int = 4,
    high_margin_prob_gap: float = 0.35,
) -> dict:
    """Build one cached-output QADV row from a reviewed model prediction row."""

    source_row = _filter_low_fan_hu_candidates(source_row)
    candidate_actions = [int(action) for action in (source_row.get("candidate_actions") or [])]
    candidate_norms = [normalize_teacher_action(norm) for norm in (source_row.get("candidate_norms") or [])]
    base_logits = [float(value) for value in (source_row.get("base_logits") or [])]
    if not (len(candidate_actions) == len(candidate_norms) == len(base_logits)):
        raise ValueError("candidate actions, norms, and base logits must have equal length")

    rule_features = list(source_row.get("candidate_rule_features") or [])
    if len(rule_features) != len(candidate_actions):
        rule_features = [[0.0] * CANDIDATE_RULE_FEATURES for _ in candidate_actions]
    teacher_candidate_norms = tuple(
        normalize_teacher_action(action)
        for action in (
            source_row.get("teacher_candidate_norms")
            or split_actions(source_row.get("chaga_top5_actions"))
            or split_actions(source_row.get("chaga_top3_actions"))
            or [source_row.get("chaga_top1_action")]
        )
        if normalize_teacher_action(action)
    )
    accepted_norms = qadv_accepted_norms(
        teacher_candidate_norms,
        play_ordinal=source_row.get("play_ordinal"),
        relaxed_play_top3_count=relaxed_play_top3_count,
        accept_top3=bool(source_row.get("teacher_accept_top3", False)),
    )
    accepted_action_ids = _action_ids_for_norms(candidate_actions, candidate_norms, accepted_norms)
    accepted_set = set(accepted_action_ids)

    ranked_slots = _ranked_indices(base_logits)
    top3_slots = ranked_slots[:3]
    base_top1_slot = top3_slots[0] if top3_slots else None
    base_top1_norm = candidate_norms[base_top1_slot] if base_top1_slot is not None else ""
    base_top3_norms = [candidate_norms[index] for index in top3_slots]

    hard_negative_reasons: dict[str, list[str]] = {}

    def add_negative(action_id: int, reason: str) -> None:
        if int(action_id) in accepted_set:
            return
        key = str(int(action_id))
        hard_negative_reasons.setdefault(key, [])
        if reason not in hard_negative_reasons[key]:
            hard_negative_reasons[key].append(reason)

    for rank, slot in enumerate(top3_slots):
        add_negative(candidate_actions[slot], "base_top1_wrong" if rank == 0 else "base_top3_wrong")

    strict_top1 = accepted_norms == teacher_candidate_norms[:1]
    if strict_top1:
        for norm in teacher_candidate_norms[1:3]:
            for action, candidate_norm in zip(candidate_actions, candidate_norms):
                if normalize_teacher_action(candidate_norm) == norm:
                    add_negative(int(action), "chaga_rank_2_3_not_strict_accepted")

    teacher_dist = _teacher_distribution_for_candidates(source_row, candidate_norms)
    sorted_dist = sorted(teacher_dist, reverse=True)
    high_margin = len(sorted_dist) >= 2 and (float(sorted_dist[0]) - float(sorted_dist[1])) >= float(high_margin_prob_gap)
    if high_margin and base_top1_slot is not None and candidate_actions[base_top1_slot] not in accepted_set:
        add_negative(candidate_actions[base_top1_slot], "high_margin_teacher_disagreement")

    top1_norm = teacher_candidate_norms[0] if teacher_candidate_norms else ""
    predicted_norm = normalize_teacher_action(base_top1_norm)
    if predicted_norm and predicted_norm in set(accepted_norms):
        error_type = "accepted"
    elif not predicted_norm or not top1_norm:
        error_type = "unmapped"
    elif _qadv_family(predicted_norm) != _qadv_family(top1_norm):
        error_type = "wrong_family"
    elif _qadv_family(top1_norm) in {"CHI", "PENG", "GANG", "BUGANG", "HU", "PASS", "ABANDON"}:
        error_type = "claim_or_hu_error"
    elif predicted_norm in set(teacher_candidate_norms[:3]) and predicted_norm not in set(accepted_norms):
        error_type = "top3_but_not_accepted_under_strict_rule"
    elif predicted_norm in set(teacher_candidate_norms[:5]):
        error_type = "same_family_top5_rank_2_5"
    else:
        error_type = "same_family_not_chaga_top5"

    hard_negative_action_ids = sorted(int(action) for action in hard_negative_reasons)
    base_rank_by_slot = {slot: rank for rank, slot in enumerate(ranked_slots)}
    row = {
        "schema": QADV_SCHEMA,
        "example_key": str(source_row.get("example_key") or ""),
        "record_id": str(source_row.get("record_id") or ""),
        "session_id": str(source_row.get("session_id") or ""),
        "turn": int(source_row.get("turn") or 0),
        "player": int(source_row.get("player") or 0),
        "request": str(source_row.get("request") or ""),
        "response": str(source_row.get("response") or ""),
        "allow_hu": bool(source_row.get("allow_hu", False)),
        "candidate_actions": candidate_actions,
        "candidate_norms": candidate_norms,
        "candidate_digest": qadv_candidate_digest(candidate_actions, candidate_norms),
        "candidate_rule_features": rule_features,
        "scalar_features": [float(value) for value in (source_row.get("scalar_features") or [0.0, 0.0, 0.0])],
        "base_logits": base_logits,
        "base_ranks": [float(base_rank_by_slot.get(slot, len(candidate_actions))) for slot in range(len(candidate_actions))],
        "base_top1_action": predicted_norm,
        "base_top3_actions": base_top3_norms,
        "teacher_candidate_norms": list(teacher_candidate_norms),
        "teacher_target_dist": teacher_dist,
        "teacher_accept_top3": bool(source_row.get("teacher_accept_top3", False)),
        "accepted_norms": list(accepted_norms),
        "accepted_action_ids": accepted_action_ids,
        "hard_negative_action_ids": hard_negative_action_ids,
        "hard_negative_reasons": {key: sorted(value) for key, value in sorted(hard_negative_reasons.items())},
        "error_type": error_type,
        "sample_weight": _qadv_hard_weight(error_type, hard_negative_reasons),
        "is_hard": error_type != "accepted",
        "has_teacher_distribution": bool(source_row.get("has_teacher_distribution", False)),
        "candidate_truncated": bool(source_row.get("candidate_truncated", False)),
    }
    return row


def validate_qadv_hard_example(row: dict) -> None:
    if row.get("schema") != QADV_SCHEMA:
        raise ValueError(f"unsupported qadv schema: {row.get('schema')!r}")
    lengths = {
        len(row.get("candidate_actions") or []),
        len(row.get("candidate_norms") or []),
        len(row.get("base_logits") or []),
        len(row.get("base_ranks") or []),
        len(row.get("candidate_rule_features") or []),
    }
    if len(lengths) != 1:
        raise ValueError("candidate arrays have inconsistent lengths")
    assert_qadv_candidate_digest(row)
    if not row.get("accepted_action_ids"):
        raise ValueError(f"empty accepted set for {row.get('example_key') or '<unknown>'}")
    accepted = {int(action) for action in row.get("accepted_action_ids") or []}
    hard = {int(action) for action in row.get("hard_negative_action_ids") or []}
    if accepted & hard:
        raise ValueError("accepted action cannot also be a hard negative")
    if row.get("error_type") != "accepted" and not hard:
        raise ValueError(f"empty hard negative set for mismatch {row.get('example_key') or '<unknown>'}")
    if not bool(row.get("allow_hu", False)) and "HU" in {normalize_teacher_action(norm) for norm in row.get("candidate_norms") or []}:
        raise ValueError(f"low-fan Hu candidate survived QADV gating for {row.get('example_key') or '<unknown>'}")


def summarize_qadv_rows(rows: list[dict]) -> dict:
    gates = {
        "empty_accepted_sets": 0,
        "action_outside_mask": 0,
        "low_fan_hu": 0,
        "empty_hard_negative_sets_for_mismatches": 0,
        "candidate_truncation_count": 0,
    }
    for row in rows:
        if not row.get("accepted_action_ids"):
            gates["empty_accepted_sets"] += 1
        if int(row.get("turn") or 0) < 0:
            gates["action_outside_mask"] += 1
        if not bool(row.get("allow_hu", False)) and "HU" in {normalize_teacher_action(norm) for norm in row.get("candidate_norms") or []}:
            gates["low_fan_hu"] += 1
        if row.get("error_type") != "accepted" and not row.get("hard_negative_action_ids"):
            gates["empty_hard_negative_sets_for_mismatches"] += 1
        if row.get("candidate_truncated"):
            gates["candidate_truncation_count"] += 1
    return {
        "schema": QADV_SCHEMA,
        "rows": len(rows),
        "hard_rows": sum(1 for row in rows if row.get("error_type") != "accepted"),
        "error_counts": dict(sorted(Counter(str(row.get("error_type") or "") for row in rows).items())),
        "hard_sample_weight_sum": sum(
            float(row.get("sample_weight", 1.0)) for row in rows if row.get("error_type") != "accepted"
        ),
        **gates,
    }


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


class QAdvFullReviewLookup:
    def __init__(self, entries_by_key: dict[tuple, list[dict]]) -> None:
        self.entries_by_key = {tuple(key): deque(value) for key, value in entries_by_key.items()}

    def _keys_for_kwargs(self, **kwargs) -> tuple[tuple, ...]:
        turn = str(kwargs.get("turn") if kwargs.get("turn") is not None else "")
        base_key = (
            str(kwargs.get("record_id") or ""),
            str(kwargs.get("player")),
            str(kwargs.get("request") or ""),
            normalize_teacher_action(str(kwargs.get("response") or "")),
        )
        return (
            (*base_key, turn),
            (*base_key, ""),
            base_key,
        )

    def __call__(self, **kwargs) -> dict | None:
        for key in self._keys_for_kwargs(**kwargs):
            queue = self.entries_by_key.get(key)
            if queue:
                return dict(queue.popleft())
        return None


def load_qadv_full_review_lookup(path: Path) -> QAdvFullReviewLookup:
    entries: dict[tuple, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if not line.strip():
                continue
            entry = json.loads(line)
            checks = entry.get("checks") or {}
            if not _review_checks_pass(checks):
                continue
            raw_candidates = entry.get("chaga_top5_candidates") or []
            if not raw_candidates:
                continue
            state_context = entry.get("state_context") or {}
            turn_value = state_context.get("turn", entry.get("state_turn", entry.get("turn", "")))
            key = (
                str(entry.get("record_id") or ""),
                str(entry.get("seat")),
                str(state_context.get("request") or ""),
                normalize_teacher_action(
                    str(
                        state_context.get("state_actual_response")
                        or entry.get("state_actual_response")
                        or entry.get("human_action")
                        or ""
                    )
                ),
                str(turn_value if turn_value is not None else ""),
            )
            entries.setdefault(key, []).append(
                {
                    "candidates": raw_candidates,
                    "candidate_norms": normalize_teacher_candidates(raw_candidates),
                    "accept_top3": _is_first_four_discard_review_entry(entry),
                    "play_ordinal": entry.get("play_ordinal"),
                }
            )
    return QAdvFullReviewLookup(entries)


def _teacher_dist_from_full_candidates(
    full_candidates: list,
    candidate_norms: list[str],
    *,
    temperature: float,
) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    score_by_norm: dict[str, float] = {}
    for item in full_candidates:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        norm = normalize_teacher_action(str(item[1]))
        if not norm:
            continue
        try:
            score = float(item[0])
        except (TypeError, ValueError):
            continue
        score_by_norm[norm] = max(score_by_norm.get(norm, float("-inf")), score)
    matched_scores = [score_by_norm.get(norm, float("-inf")) for norm in candidate_norms]
    finite_scores = [score for score in matched_scores if score != float("-inf")]
    if not finite_scores:
        return [0.0 for _ in candidate_norms]
    max_score = max(finite_scores)
    weights = []
    for score in matched_scores:
        if score == float("-inf"):
            weights.append(0.0)
        else:
            weights.append(float(torch.exp(torch.tensor((score - max_score) / float(temperature))).item()))
    total = float(sum(weights))
    if total <= 0.0:
        return [0.0 for _ in candidate_norms]
    return [float(weight) / total for weight in weights]


def collect_qadv_prediction_rows(
    model,
    reviewed,
    *,
    full_review_lookup: QAdvFullReviewLookup | None,
    max_candidates: int,
    batch_size: int,
    device: torch.device,
    teacher_temperature: float,
    relaxed_play_top3_count: int,
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
            logits = logits.detach().cpu()
            for index, example in enumerate(current_examples):
                mask = batch["candidate_mask"][index].bool()
                slots = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                candidate_actions = [int(batch["candidate_actions"][index, slot].item()) for slot in slots]
                candidate_norms = [normalize_teacher_action(action_response(action)) for action in candidate_actions]
                full_review = (
                    full_review_lookup(
                        record_id=getattr(example, "record_id", ""),
                        player=int(example.player),
                        request=getattr(example, "request", ""),
                        response=example.response,
                        turn=int(example.turn),
                    )
                    if full_review_lookup is not None
                    else None
                )
                if full_review:
                    teacher_candidate_norms = tuple(full_review.get("candidate_norms") or ())
                    teacher_accept_top3 = bool(full_review.get("accept_top3", False))
                    play_ordinal = full_review.get("play_ordinal")
                    teacher_target_dist = _teacher_dist_from_full_candidates(
                        full_review.get("candidates") or [],
                        candidate_norms,
                        temperature=teacher_temperature,
                    )
                    has_teacher_distribution = any(value > 0.0 for value in teacher_target_dist)
                else:
                    teacher_candidate_norms = tuple(getattr(example, "teacher_candidate_norms", ()))
                    teacher_accept_top3 = bool(getattr(example, "teacher_accept_top3", False))
                    play_ordinal = None
                    teacher_values = batch["teacher_target_dist"][index, slots].float().tolist()
                    total = float(sum(max(0.0, value) for value in teacher_values))
                    teacher_target_dist = [max(0.0, value) / total for value in teacher_values] if total > 0.0 else [0.0 for _ in slots]
                    has_teacher_distribution = bool(getattr(example, "teacher_action_distribution", None) is not None)

                source_row = {
                    "example_key": "|".join(
                        [
                            str(getattr(example, "record_id", "") or ""),
                            str(int(example.player)),
                            str(int(example.turn)),
                            str(getattr(example, "request", "") or ""),
                        ]
                    ),
                    "record_id": str(getattr(example, "record_id", "") or ""),
                    "session_id": str(getattr(example, "session_id", "") or ""),
                    "turn": int(example.turn),
                    "player": int(example.player),
                    "request": str(getattr(example, "request", "") or ""),
                    "response": str(example.response),
                    "candidate_actions": candidate_actions,
                    "candidate_norms": candidate_norms,
                    "candidate_rule_features": batch["candidate_rule_features"][index, slots, :].float().tolist(),
                    "base_logits": logits[index, slots].float().tolist(),
                    "scalar_features": batch["scalar_features"][index].float().tolist(),
                    "teacher_candidate_norms": teacher_candidate_norms,
                    "teacher_target_dist": teacher_target_dist,
                    "teacher_accept_top3": teacher_accept_top3,
                    "allow_hu": bool(getattr(example, "allow_hu", False)),
                    "has_teacher_distribution": has_teacher_distribution,
                    "play_ordinal": play_ordinal,
                    "candidate_truncated": int(mask.sum().item()) >= int(max_candidates) and int(example.action_mask.sum()) > int(max_candidates),
                }
                rows.append(
                    build_qadv_hard_example(
                        source_row,
                        relaxed_play_top3_count=relaxed_play_top3_count,
                    )
                )
    return rows


def mine_hard_examples(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_checkpoint(Path(args.checkpoint), device)
    examples, load_summary = load_evaluation_examples(args, config)
    reviewed = select_reviewed_examples(examples)
    max_candidates = resolve_eval_max_candidates(config, args.max_candidates)
    if args.output_format == "qadv":
        full_review_lookup = load_qadv_full_review_lookup(Path(args.review_audit_jsonl))
        enriched = collect_qadv_prediction_rows(
            model,
            reviewed,
            full_review_lookup=full_review_lookup,
            max_candidates=max_candidates,
            batch_size=args.batch_size,
            device=device,
            teacher_temperature=args.teacher_temperature,
            relaxed_play_top3_count=args.relaxed_play_top3_count,
        )
        if args.mismatches_only:
            enriched_to_write = [row for row in enriched if row.get("error_type") != "accepted"]
        else:
            enriched_to_write = enriched
        qadv_summary = summarize_qadv_rows(enriched)
        if args.fail_on_gate:
            gate_failures = {
                key: value
                for key, value in qadv_summary.items()
                if key
                in {
                    "empty_accepted_sets",
                    "action_outside_mask",
                    "low_fan_hu",
                    "empty_hard_negative_sets_for_mismatches",
                    "candidate_truncation_count",
                }
                and int(value) > 0
            }
            if gate_failures:
                raise ValueError(f"QADV hard-example gate failed: {gate_failures}")
    else:
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
        qadv_summary = None
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as dst:
        for row in enriched_to_write:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "format": QADV_SCHEMA if args.output_format == "qadv" else "mcr_chaga_hard_examples_v1",
        "checkpoint": args.checkpoint,
        "raw": args.raw,
        "review_audit_jsonl": args.review_audit_jsonl,
        "examples": len(examples),
        "reviewed_examples": len(reviewed),
        "written_rows": len(enriched_to_write),
        "evaluated_candidate_width": max_candidates,
        "load_summary": load_summary,
        **(qadv_summary if qadv_summary is not None else summarize_rows(enriched)),
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
    parser.add_argument("--output-format", choices=["qadv", "legacy"], default="qadv")
    parser.add_argument("--relaxed-play-top3-count", type=int, default=4)
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    summary = mine_hard_examples(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
