#!/usr/bin/env python3
"""Build QADV terminal-return rows from historical tziakcha trajectories."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from evaluate_transformer_chaga_review import load_checkpoint, resolve_eval_max_candidates
from mine_chaga_hard_examples import (
    QAdvFullReviewLookup,
    _teacher_dist_from_full_candidates,
    load_qadv_full_review_lookup,
    qadv_accepted_norms,
    qadv_candidate_digest,
)
from train_transformer_candidate import (
    CANDIDATE_RULE_FEATURES,
    FeatureAgent,
    TransformerExample,
    TransformerRawDataset,
    action_response,
    build_transformer_examples_from_record,
    collate_transformer_examples,
    normalize_teacher_action,
)


QADV_TERMINAL_SCHEMA = "qadv_terminal_v1"


def open_text_maybe_gzip(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8-sig")
    return path.open(mode, encoding="utf-8-sig")


def _response_from_entry(entry) -> str:
    if isinstance(entry, dict):
        return str(entry.get("response") or entry.get("raw") or "")
    return str(entry or "")


def _score_for_player(record: dict, player: int) -> float:
    scores = record.get("scores") or {}
    value = scores.get(str(player), scores.get(player, 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _placement_delta(record: dict, player: int) -> float:
    scores = []
    raw_scores = record.get("scores") or {}
    for seat in range(4):
        scores.append((seat, _coerce_float(raw_scores.get(str(seat), raw_scores.get(seat, 0.0)))))
    scores.sort(key=lambda item: item[1], reverse=True)
    for rank, (seat, _) in enumerate(scores):
        if seat == int(player):
            return float(1.5 - rank)
    return 0.0


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _terminal_fan(record: dict, winner: int | None) -> tuple[int | None, int | None]:
    if winner is None:
        return None, None
    step = record.get("step") or {}
    y = step.get("y") or record.get("y") or []
    if isinstance(y, list) and 0 <= int(winner) < len(y):
        payload = y[int(winner)]
        if isinstance(payload, dict) and "f" in payload:
            fan = int(payload["f"])
            return fan, fan
    return None, None


def _last_can_hu(logs: list, player: int) -> bool:
    for index in range(len(logs) - 1, -1, -1):
        output = (logs[index].get("output") if isinstance(logs[index], dict) else None) or {}
        display = output.get("display") or {}
        can_hu = display.get("canHu")
        if isinstance(can_hu, list) and 0 <= int(player) < len(can_hu):
            try:
                return float(can_hu[int(player)]) >= 8.0
            except (TypeError, ValueError):
                return False
    return False


def _parse_request_actor(request: str) -> int | None:
    parts = str(request or "").split()
    if len(parts) >= 2 and parts[0] == "3":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _is_self_draw_request(request: str, winner: int, display: dict) -> bool:
    parts = str(request or "").split()
    if parts and parts[0] == "2":
        return True
    return str(display.get("action") or "").upper() == "DRAW" and int(display.get("player", -1)) == int(winner)


def extract_terminal_outcomes(record: dict) -> dict[int, dict]:
    """Return per-player terminal outcome labels for one converted tziakcha record."""

    logs = record.get("logs") or []
    winner: int | None = None
    winner_request = ""
    winner_display: dict = {}
    final_turn = max(0, len(logs) // 2 - 1)
    for turn in range(0, len(logs) - 1, 2):
        output = (logs[turn].get("output") if isinstance(logs[turn], dict) else None) or {}
        requests = output.get("content") or {}
        response_log = logs[turn + 1]
        if not isinstance(response_log, dict):
            continue
        for player_text, entry in response_log.items():
            response = normalize_teacher_action(_response_from_entry(entry))
            if response == "HU":
                try:
                    winner = int(player_text)
                except (TypeError, ValueError):
                    continue
                winner_request = str((requests or {}).get(str(winner), ""))
                winner_display = dict(output.get("display") or {})
                final_turn = turn // 2

    result_type = "HU" if winner is not None else "HUANG"
    discarder = None
    is_self_draw = False
    hu_turn = final_turn if winner is not None else None
    fan, base_fan = _terminal_fan(record, winner)
    low_fan_hu = bool(winner is not None and base_fan is not None and int(base_fan) < 8)
    if winner is not None:
        is_self_draw = _is_self_draw_request(winner_request, winner, winner_display)
        if not is_self_draw:
            discarder = _parse_request_actor(winner_request)

    outcomes: dict[int, dict] = {}
    for player in range(4):
        player_won = winner is not None and int(player) == int(winner)
        player_dealt_in = discarder is not None and int(player) == int(discarder)
        end_wait = bool(result_type == "HUANG" and _last_can_hu(logs, player))
        wait_when_deal_in = bool(player_dealt_in and _last_can_hu(logs, player))
        outcomes[player] = {
            "result_type": result_type,
            "winner": winner,
            "discarder": discarder,
            "is_self_draw": is_self_draw,
            "final_turn": int(final_turn),
            "hu_turn": hu_turn,
            "player_won": player_won,
            "player_dealt_in": player_dealt_in,
            "score_delta": _score_for_player(record, player),
            "placement_delta": _placement_delta(record, player),
            "end_wait": end_wait,
            "wait_when_deal_in": wait_when_deal_in,
            "fan": fan,
            "base_fan": base_fan,
            "low_fan_hu": low_fan_hu,
        }
    return outcomes


def terminal_return(terminal: dict, *, decision_turn: int, gamma: float = 0.995) -> dict:
    final_turn = int(terminal.get("final_turn") or decision_turn)
    hu_turn = terminal.get("hu_turn")
    player_won = bool(terminal.get("player_won", False))
    player_dealt_in = bool(terminal.get("player_dealt_in", False))
    end_wait = bool(terminal.get("end_wait", False))
    wait_when_deal_in = bool(terminal.get("wait_when_deal_in", False))
    point_delta_norm = max(-1.0, min(1.0, float(terminal.get("score_delta") or 0.0) / 64.0))
    hu_turn_norm = float(hu_turn) / float(final_turn) if player_won and hu_turn is not None and final_turn > 0 else 0.0
    components = {
        "point_delta": point_delta_norm,
        "win": 0.20 if player_won else 0.0,
        "end_wait": 0.10 if end_wait else 0.0,
        "wait_when_deal_in": 0.05 if wait_when_deal_in else 0.0,
        "deal_in": -0.15 if player_dealt_in else 0.0,
        "hu_turn": -0.10 * hu_turn_norm if player_won else 0.0,
    }
    total = float(sum(components.values()))
    discounted = total * (float(gamma) ** max(0, final_turn - int(decision_turn)))
    return {
        "point_delta_norm": point_delta_norm,
        "terminal_return": total,
        "discounted_return": discounted,
        "components": components,
    }


def _rank_values(scores: list[float]) -> list[float]:
    ranked = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
    ranks = [float(len(scores)) for _ in scores]
    for rank, slot in enumerate(ranked):
        ranks[slot] = float(rank)
    return ranks


def _ids_for_norms(candidate_actions: list[int], candidate_norms: list[str], norms: Iterable[str]) -> list[int]:
    wanted = {normalize_teacher_action(norm) for norm in norms if normalize_teacher_action(norm)}
    return [
        int(action)
        for action, norm in zip(candidate_actions, candidate_norms)
        if normalize_teacher_action(norm) in wanted
    ]


def build_qadv_terminal_row(source_row: dict, terminal: dict, *, gamma: float = 0.995) -> dict:
    candidate_actions = [int(action) for action in source_row.get("candidate_actions") or []]
    candidate_norms = [normalize_teacher_action(norm) for norm in source_row.get("candidate_norms") or []]
    base_logits = [float(value) for value in source_row.get("base_logits") or []]
    if not (len(candidate_actions) == len(candidate_norms) == len(base_logits)):
        raise ValueError("candidate actions, norms, and base logits must have equal length")
    rule_features = list(source_row.get("candidate_rule_features") or [])
    if len(rule_features) != len(candidate_actions):
        rule_features = [[0.0] * CANDIDATE_RULE_FEATURES for _ in candidate_actions]
    base_ranks = [float(value) for value in (source_row.get("base_ranks") or [])]
    if len(base_ranks) != len(candidate_actions):
        base_ranks = _rank_values(base_logits)
    scalar_features = [float(value) for value in (source_row.get("scalar_features") or [])[:3]]
    while len(scalar_features) < 3:
        scalar_features.append(0.0)

    chosen_norm = normalize_teacher_action(
        source_row.get("actual_response_norm")
        or source_row.get("response")
        or source_row.get("chosen_norm")
        or ""
    )
    chosen_action_ids = _ids_for_norms(candidate_actions, candidate_norms, [chosen_norm])
    chosen_action_id = chosen_action_ids[0] if chosen_action_ids else None
    chosen_is_legal = chosen_action_id is not None
    accepted_norms = [
        normalize_teacher_action(norm)
        for norm in (source_row.get("accepted_norms") or [])
        if normalize_teacher_action(norm)
    ]
    accepted_action_ids = [int(action) for action in (source_row.get("accepted_action_ids") or [])]
    if not accepted_action_ids and accepted_norms:
        accepted_action_ids = _ids_for_norms(candidate_actions, candidate_norms, accepted_norms)
    teacher_candidate_norms = [
        normalize_teacher_action(norm)
        for norm in (source_row.get("teacher_candidate_norms") or [])
        if normalize_teacher_action(norm)
    ]
    base_top_slot = max(range(len(base_logits)), key=lambda slot: base_logits[slot]) if base_logits else None
    candidate_digest = qadv_candidate_digest(candidate_actions, candidate_norms)
    allow_hu = bool(source_row.get("allow_hu", False))
    accepted_hu_outside_gate = (not allow_hu) and any(
        candidate_norms[index] == "HU"
        for index, action in enumerate(candidate_actions)
        if int(action) in set(accepted_action_ids)
    )
    return_values = terminal_return(terminal, decision_turn=int(source_row.get("turn") or 0), gamma=gamma)

    row = {
        "schema": QADV_TERMINAL_SCHEMA,
        "schema_version": QADV_TERMINAL_SCHEMA,
        "example_key": str(source_row.get("example_key") or ""),
        "candidate_digest": candidate_digest,
        "state_key": {
            "record_id": str(source_row.get("record_id") or ""),
            "session_id": str(source_row.get("session_id") or ""),
            "player": int(source_row.get("player") or 0),
            "turn": int(source_row.get("turn") or 0),
            "request": str(source_row.get("request") or ""),
            "actual_response_norm": chosen_norm,
            "play_ordinal": source_row.get("play_ordinal"),
        },
        "legal": {
            "candidate_action_ids": candidate_actions,
            "candidate_norms": candidate_norms,
            "candidate_families": [norm.split()[0] if norm else "" for norm in candidate_norms],
            "allow_hu": allow_hu,
            "candidate_count": len(candidate_actions),
            "candidate_truncation_count": 1 if bool(source_row.get("candidate_truncated", False)) else 0,
        },
        "base": {
            "pred_action_id": candidate_actions[base_top_slot] if base_top_slot is not None else None,
            "pred_norm": candidate_norms[base_top_slot] if base_top_slot is not None else "",
            "base_logits": base_logits,
            "base_ranks": base_ranks,
        },
        "chosen": {
            "action_id": chosen_action_id,
            "norm": chosen_norm,
            "is_legal": chosen_is_legal,
        },
        "chaga": {
            "has_review": bool(source_row.get("has_teacher_distribution") or teacher_candidate_norms),
            "candidate_norms": teacher_candidate_norms,
            "accepted_norms": accepted_norms,
            "accepted_action_ids": accepted_action_ids,
            "accept_top3": bool(source_row.get("teacher_accept_top3", False)),
            "top1_norm": teacher_candidate_norms[0] if teacher_candidate_norms else "",
            "top1_top2_margin": _top1_top2_margin(source_row.get("teacher_target_dist") or []),
        },
        "terminal": dict(terminal),
        "return": return_values,
        "safety": {
            "action_outside_mask": not chosen_is_legal,
            "accepted_hu_outside_gate": accepted_hu_outside_gate,
            "low_fan_hu": bool(terminal.get("low_fan_hu", False)),
        },
        # Flattened fields make this file easy to mix into the existing QADV dataloader later.
        "candidate_actions": candidate_actions,
        "candidate_norms": candidate_norms,
        "candidate_rule_features": rule_features,
        "base_logits": base_logits,
        "base_ranks": base_ranks,
        "scalar_features": scalar_features,
        "accepted_action_ids": accepted_action_ids,
        "hard_negative_action_ids": [],
        "teacher_target_dist": [float(value) for value in (source_row.get("teacher_target_dist") or [])],
        "allow_hu": allow_hu,
        "chosen_action_id": chosen_action_id,
        "chosen_norm": chosen_norm,
        "discounted_return": return_values["discounted_return"],
        "terminal_return": return_values["terminal_return"],
        "sample_weight": float(source_row.get("sample_weight", 1.0)),
    }
    return row


def _top1_top2_margin(values: Iterable) -> float | None:
    numeric = sorted([float(value) for value in values if _is_number(value)], reverse=True)
    if len(numeric) < 2:
        return None
    return float(numeric[0] - numeric[1])


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def validate_qadv_terminal_row(row: dict) -> None:
    if row.get("schema_version") != QADV_TERMINAL_SCHEMA:
        raise ValueError(f"unsupported terminal schema: {row.get('schema_version')!r}")
    legal = row.get("legal") or {}
    candidate_actions = [int(action) for action in legal.get("candidate_action_ids") or []]
    candidate_norms = [normalize_teacher_action(norm) for norm in legal.get("candidate_norms") or []]
    if not candidate_actions:
        raise ValueError("empty legal candidate row")
    if len(candidate_actions) != len(candidate_norms):
        raise ValueError("candidate action/norm length mismatch")
    if row.get("candidate_digest") != qadv_candidate_digest(candidate_actions, candidate_norms):
        raise ValueError(f"candidate digest mismatch for {row.get('example_key') or '<unknown>'}")
    safety = row.get("safety") or {}
    if safety.get("action_outside_mask"):
        raise ValueError(f"chosen action outside legal mask for {row.get('example_key') or '<unknown>'}")
    if safety.get("accepted_hu_outside_gate"):
        raise ValueError(f"accepted Hu outside fan gate for {row.get('example_key') or '<unknown>'}")
    if safety.get("low_fan_hu"):
        raise ValueError(f"low-fan Hu terminal row for {row.get('example_key') or '<unknown>'}")
    if not bool(legal.get("allow_hu", False)) and "HU" in candidate_norms:
        raise ValueError(f"low-fan Hu candidate survived terminal gating for {row.get('example_key') or '<unknown>'}")


def summarize_terminal_rows(
    rows: list[dict],
    *,
    min_rows: int = 25_000,
    min_games: int = 500,
    min_return_std: float = 0.03,
) -> dict:
    returns = [float((row.get("return") or {}).get("discounted_return", row.get("discounted_return", 0.0))) for row in rows]
    record_ids = {str((row.get("state_key") or {}).get("record_id") or row.get("record_id") or "") for row in rows}
    record_ids.discard("")
    terminal_by_game: dict[str, dict] = {}
    for row in rows:
        record_id = str((row.get("state_key") or {}).get("record_id") or "")
        if record_id and record_id not in terminal_by_game:
            terminal_by_game[record_id] = dict(row.get("terminal") or {})
    return_std = float(statistics.pstdev(returns)) if len(returns) > 1 else 0.0
    nonzero_score_games = sum(1 for terminal in terminal_by_game.values() if abs(float(terminal.get("score_delta") or 0.0)) > 0.0)
    action_outside_mask = sum(1 for row in rows if (row.get("safety") or {}).get("action_outside_mask"))
    low_fan_hu = sum(1 for row in rows if (row.get("safety") or {}).get("low_fan_hu"))
    accepted_hu_outside_gate = sum(1 for row in rows if (row.get("safety") or {}).get("accepted_hu_outside_gate"))
    candidate_truncation_count = sum(int((row.get("legal") or {}).get("candidate_truncation_count") or 0) for row in rows)
    empty_legal_candidate_rows = sum(1 for row in rows if not (row.get("legal") or {}).get("candidate_action_ids"))
    missing_terminal_result = sum(1 for row in rows if not (row.get("terminal") or {}).get("result_type"))
    gate_failures = {}
    if len(rows) < int(min_rows):
        gate_failures["rows"] = len(rows)
    if len(record_ids) < int(min_games):
        gate_failures["games"] = len(record_ids)
    if nonzero_score_games <= 0:
        gate_failures["nonzero_score_games"] = nonzero_score_games
    if return_std < float(min_return_std):
        gate_failures["return_std"] = return_std
    for key, value in {
        "candidate_truncation_count": candidate_truncation_count,
        "action_outside_mask": action_outside_mask,
        "low_fan_hu": low_fan_hu,
        "accepted_hu_outside_gate": accepted_hu_outside_gate,
        "empty_legal_candidate_rows": empty_legal_candidate_rows,
        "missing_terminal_result": missing_terminal_result,
    }.items():
        if int(value) > 0:
            gate_failures[key] = int(value)
    return {
        "schema": QADV_TERMINAL_SCHEMA,
        "rows": len(rows),
        "games": len(record_ids),
        "nonzero_score_games": nonzero_score_games,
        "return_mean": float(sum(returns) / len(returns)) if returns else None,
        "return_std": return_std,
        "score_delta_mean": (
            float(sum(float((row.get("terminal") or {}).get("score_delta") or 0.0) for row in rows) / len(rows))
            if rows
            else None
        ),
        "candidate_truncation_count": candidate_truncation_count,
        "action_outside_mask": action_outside_mask,
        "low_fan_hu": low_fan_hu,
        "accepted_hu_outside_gate": accepted_hu_outside_gate,
        "empty_legal_candidate_rows": empty_legal_candidate_rows,
        "missing_terminal_result": missing_terminal_result,
        "hu_games": sum(1 for terminal in terminal_by_game.values() if terminal.get("result_type") == "HU"),
        "huang_games": sum(1 for terminal in terminal_by_game.values() if terminal.get("result_type") == "HUANG"),
        "deal_in_rows": sum(1 for row in rows if (row.get("terminal") or {}).get("player_dealt_in")),
        "end_wait_rows": sum(1 for row in rows if (row.get("terminal") or {}).get("end_wait")),
        "wait_when_deal_in_rows": sum(1 for row in rows if (row.get("terminal") or {}).get("wait_when_deal_in")),
        "rows_by_action_family": dict(
            sorted(Counter(str(row.get("chosen_norm") or "").split()[0] for row in rows).items())
        ),
        "rows_with_chaga_review": sum(1 for row in rows if (row.get("chaga") or {}).get("has_review")),
        "rows_without_chaga_review": sum(1 for row in rows if not (row.get("chaga") or {}).get("has_review")),
        "gate_failures": gate_failures,
        "gate_passed": not gate_failures,
    }


def iter_jsonl_records(paths: Iterable[Path], *, max_records_per_source: int | None = None) -> Iterable[dict]:
    for path in paths:
        seen = 0
        with open_text_maybe_gzip(path) as src:
            for line in src:
                if not line.strip():
                    continue
                if max_records_per_source is not None and seen >= int(max_records_per_source):
                    break
                seen += 1
                yield json.loads(line)


def _source_row_from_example(
    example: TransformerExample,
    batch: dict,
    logits: torch.Tensor,
    index: int,
    *,
    max_candidates: int,
    full_review_lookup: QAdvFullReviewLookup | None,
    teacher_temperature: float,
) -> dict:
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
        accepted_norms = qadv_accepted_norms(
            teacher_candidate_norms,
            play_ordinal=play_ordinal,
            accept_top3=teacher_accept_top3,
        )
    else:
        teacher_candidate_norms = ()
        teacher_accept_top3 = False
        play_ordinal = None
        teacher_target_dist = [0.0 for _ in candidate_norms]
        accepted_norms = ()
    accepted_action_ids = _ids_for_norms(candidate_actions, candidate_norms, accepted_norms)
    return {
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
        "base_ranks": _rank_values(logits[index, slots].float().tolist()),
        "scalar_features": batch["scalar_features"][index].float().tolist(),
        "teacher_candidate_norms": teacher_candidate_norms,
        "teacher_target_dist": teacher_target_dist,
        "teacher_accept_top3": teacher_accept_top3,
        "accepted_norms": list(accepted_norms),
        "accepted_action_ids": accepted_action_ids,
        "allow_hu": bool(getattr(example, "allow_hu", False)),
        "has_teacher_distribution": bool(full_review),
        "play_ordinal": play_ordinal,
        "candidate_truncated": int(mask.sum().item()) >= int(max_candidates) and int(example.action_mask.sum()) > int(max_candidates),
    }


def build_terminal_rows(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_checkpoint(Path(args.checkpoint), device)
    model.eval()
    max_candidates = resolve_eval_max_candidates(config, args.max_candidates)
    full_review_lookup = (
        load_qadv_full_review_lookup(Path(args.review_audit_jsonl))
        if args.review_audit_jsonl
        else None
    )
    rows: list[dict] = []
    load_stats: Counter[str] = Counter()
    records_seen = 0
    for record in iter_jsonl_records([Path(path) for path in args.raw], max_records_per_source=args.max_records_per_source):
        records_seen += 1
        outcomes = extract_terminal_outcomes(record)
        examples, stats = build_transformer_examples_from_record(
            record,
            history_len=int(config.get("history_len", args.history_len)),
            teacher_lookup=None,
            teacher_temperature=args.teacher_temperature,
            reviewed_only=False,
            compute_rule_features=not bool(args.no_rule_features),
        )
        load_stats.update(stats)
        if args.max_examples is not None and len(rows) >= int(args.max_examples):
            break
        if args.progress_every_records and records_seen % int(args.progress_every_records) == 0:
            print(
                json.dumps(
                    {
                        "event": "qadv_terminal_progress",
                        "records_seen": records_seen,
                        "rows": len(rows),
                        "last_record_id": str(record.get("source_record_id") or record.get("match_id") or record.get("id") or ""),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if not examples:
            continue
        loader = DataLoader(
            TransformerRawDataset(examples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda items: collate_transformer_examples(items, max_candidates=max_candidates),
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                current = examples[offset : offset + int(batch["target_index"].numel())]
                offset += len(current)
                device_batch = {key: value.to(device) for key, value in batch.items()}
                logits, _ = model(device_batch)
                logits = logits.detach().cpu()
                for index, example in enumerate(current):
                    terminal = outcomes.get(int(example.player))
                    if terminal is None:
                        continue
                    source_row = _source_row_from_example(
                        example,
                        batch,
                        logits,
                        index,
                        max_candidates=max_candidates,
                        full_review_lookup=full_review_lookup,
                        teacher_temperature=args.teacher_temperature,
                    )
                    row = build_qadv_terminal_row(source_row, terminal, gamma=args.gamma)
                    rows.append(row)
                    if args.max_examples is not None and len(rows) >= int(args.max_examples):
                        break
                if args.max_examples is not None and len(rows) >= int(args.max_examples):
                    break
        if args.max_examples is not None and len(rows) >= int(args.max_examples):
            break

    for row in rows:
        validate_qadv_terminal_row(row)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_text_maybe_gzip(out_path, "wt") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "format": QADV_TERMINAL_SCHEMA,
        "checkpoint": args.checkpoint,
        "raw": args.raw,
        "review_audit_jsonl": args.review_audit_jsonl,
        "records_seen": records_seen,
        "load_stats": dict(sorted(load_stats.items())),
        "out_jsonl": str(out_path),
        **summarize_terminal_rows(
            rows,
            min_rows=args.min_rows,
            min_games=args.min_games,
            min_return_std=args.min_return_std,
        ),
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
    parser.add_argument("--review-audit-jsonl", default=None)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-len", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=FeatureAgent.ACT_SIZE)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--no-rule-features", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--min-rows", type=int, default=25_000)
    parser.add_argument("--min-games", type=int, default=500)
    parser.add_argument("--min-return-std", type=float, default=0.03)
    parser.add_argument("--progress-every-records", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    summary = build_terminal_rows(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_gate and summary.get("gate_failures"):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
