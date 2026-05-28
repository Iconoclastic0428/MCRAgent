#!/usr/bin/env python3
"""Evaluate a deterministic mahjong-algorithm baseline against CHAGA reviews."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from build_lawlorentz_dataset import (
    BotzoneFeatureRuntime,
    actual_response,
    response_to_valid_action,
)
from evaluate_transformer_chaga_review import score_original_chaga_match
from lawlorentz_policy import LawlorentzEffectiveScorer
from train_transformer_candidate import (
    FeatureAgent,
    ReviewTarget,
    _advance_runtime_after_response,
    hu_gated_candidate_mask,
    load_review_target_lookup,
    normalize_teacher_action,
    normalize_teacher_candidates,
)
from visualize_transformer_chaga_review_turns import aggregate_by_turn, render_svg


ScoreActionResponse = Callable[[int], str]


def response_family(response: str | None) -> str:
    normalized = normalize_teacher_action(response)
    if not normalized:
        return "UNKNOWN"
    head = normalized.split()[0]
    if head == "PONG":
        return "PENG"
    if head in {"KANG", "KONG"}:
        return "GANG"
    return head


def choose_algorithmic_action(
    action_mask: np.ndarray,
    candidate_rule_features: np.ndarray,
    *,
    allow_hu: bool,
    action_to_response: ScoreActionResponse,
) -> tuple[int, str]:
    gated = hu_gated_candidate_mask(action_mask, action_to_response, allow_hu=allow_hu)
    best: tuple[float, str, int, str] | None = None
    for action in np.flatnonzero(gated > 0):
        action = int(action)
        response = action_to_response(action)
        features = candidate_rule_features[action] if action < len(candidate_rule_features) else np.zeros((7,))
        score = algorithmic_action_score(response, features, allow_hu=allow_hu)
        candidate = (score, normalize_teacher_action(response), -action, response)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("no legal algorithmic action after Hu gate")
    _, _, neg_action, response = best
    return -neg_action, response


def algorithmic_action_score(
    response: str,
    candidate_rule_features: Iterable[float],
    *,
    allow_hu: bool,
) -> float:
    family = response_family(response)
    features = list(candidate_rule_features)
    while len(features) < 7:
        features.append(0.0)
    if family == "HU":
        return 1_000_000.0 if allow_hu else -1_000_000.0
    if family == "PLAY":
        return (
            1_000.0
            + 50_000.0 * float(features[3])
            + 10_000.0 * float(features[4])
            + 2_000.0 * float(features[5])
            + 500.0 * float(features[6])
        )
    if family in {"GANG", "BUGANG"}:
        return 700.0
    if family in {"PENG", "CHI"}:
        return 100.0
    if family == "PASS":
        return 0.0
    return -100.0


def predict_algorithmic_response_from_state(
    agent: FeatureAgent,
    obs: dict,
    *,
    levels: int,
) -> str:
    actions = [int(action) for action in np.flatnonzero(obs["action_mask"] > 0)]
    responses = [agent.action2response(action) for action in actions]
    if any(response_family(response) == "HU" for response in responses):
        return "Hu"

    play = _choose_play_with_agent(agent, obs, levels=levels)
    if play is not None:
        return f"Play {play[0]}"

    pass_profile = _profile_key(agent, levels=levels)
    best_claim: tuple[tuple, str] | None = None
    for response in responses:
        family = response_family(response)
        if family not in {"PENG", "CHI"}:
            continue
        trial = copy.deepcopy(agent)
        claim_request = f"Player {agent.seatWind} {response}"
        try:
            claim_obs = trial.request2obs(claim_request)
        except Exception:
            continue
        claim_play = _choose_play_with_agent(trial, claim_obs, levels=levels)
        if claim_play is None:
            continue
        _discard, key = claim_play
        if key <= pass_profile:
            continue
        normalized = normalize_teacher_action(response)
        candidate = (key, normalized or response)
        if best_claim is None or candidate > best_claim:
            best_claim = candidate
    if best_claim is not None:
        return best_claim[1].title()

    for response in responses:
        if response_family(response) in {"GANG", "BUGANG"}:
            return response
    return "Pass"


def _choose_play_with_agent(
    agent: FeatureAgent,
    obs: dict,
    *,
    levels: int,
) -> tuple[str, tuple] | None:
    actions = [int(action) for action in np.flatnonzero(obs["action_mask"] > 0)]
    play_tiles = []
    for action in actions:
        response = agent.action2response(action)
        if response_family(response) == "PLAY":
            parts = response.split()
            if len(parts) >= 2:
                play_tiles.append(parts[1])
    if not play_tiles:
        return None
    scorer = LawlorentzEffectiveScorer(
        packs=tuple(agent.packs[0]),
        shown_tiles=Counter(agent.shownTiles),
        seat_wind=int(agent.seatWind),
        prevalent_wind=0,
        levels=levels,
    )
    return max(
        ((tile, scorer.discard_key(agent.hand, tile)) for tile in play_tiles),
        key=lambda item: item[1],
    )


def _profile_key(agent: FeatureAgent, *, levels: int) -> tuple:
    scorer = LawlorentzEffectiveScorer(
        packs=tuple(agent.packs[0]),
        shown_tiles=Counter(agent.shownTiles),
        seat_wind=int(agent.seatWind),
        prevalent_wind=0,
        levels=levels,
    )
    return scorer.profile(tuple(agent.hand)).key()


def collect_algorithmic_prediction_rows(
    raw_paths: list[Path],
    review_audit_jsonl: Path,
    *,
    levels: int,
    max_records_per_source: int | None = None,
    max_examples: int | None = None,
) -> tuple[list[dict], dict]:
    lookup = load_review_target_lookup(review_audit_jsonl)
    rows: list[dict] = []
    stats: Counter[str] = Counter()
    for raw_path in raw_paths:
        with raw_path.open("r", encoding="utf-8-sig") as src:
            source_records = 0
            for line in src:
                if max_records_per_source is not None and source_records >= max_records_per_source:
                    break
                if not line.strip():
                    continue
                source_records += 1
                record = json.loads(line)
                stats["records"] += 1
                _collect_record_rows(record, lookup, rows, stats, levels=levels, max_examples=max_examples)
                if max_examples is not None and len(rows) >= max_examples:
                    return rows, dict(stats)
    return rows, dict(stats)


def _collect_record_rows(
    record: dict,
    lookup,
    rows: list[dict],
    stats: Counter[str],
    *,
    levels: int,
    max_examples: int | None,
) -> None:
    runtimes = [BotzoneFeatureRuntime() for _ in range(4)]
    skip_requests: list[Counter[str]] = [Counter() for _ in range(4)]
    train_players_raw = record.get("train_players")
    train_players = {str(player) for player in train_players_raw} if train_players_raw is not None else None
    record_id = str(record.get("source_record_id") or record.get("match_id") or record.get("id") or "")
    logs = record.get("logs") or []

    for turn in range(0, len(logs) - 1, 2):
        if max_examples is not None and len(rows) >= max_examples:
            return
        output = logs[turn].get("output") or {}
        requests = output.get("content") or {}
        if not isinstance(requests, dict):
            continue
        response_log = logs[turn + 1]
        if not isinstance(response_log, dict):
            continue
        for player_text, request in requests.items():
            if max_examples is not None and len(rows) >= max_examples:
                return
            try:
                player = int(player_text)
            except (TypeError, ValueError):
                stats["bad_player"] += 1
                continue
            if player < 0 or player >= 4:
                stats["bad_player"] += 1
                continue
            request = str(request)
            if skip_requests[player][request] > 0:
                skip_requests[player][request] -= 1
                stats["skipped_duplicate_self_event"] += 1
                continue
            actual = actual_response(response_log, str(player))
            try:
                obs = runtimes[player].observe(request)
            except Exception as exc:
                stats[f"observe_error:{type(exc).__name__}"] += 1
                continue
            action = None
            if obs is not None and actual is not None and runtimes[player].agent is not None:
                action = response_to_valid_action(runtimes[player].agent, obs, request, actual)
                if train_players is None or str(player) in train_players:
                    _maybe_append_review_row(
                        rows,
                        lookup,
                        record_id=record_id,
                        player=player,
                        turn=turn // 2,
                        request=request,
                        actual=actual,
                        obs=obs,
                        agent=runtimes[player].agent,
                        levels=levels,
                    )
            if actual is not None:
                _advance_runtime_after_response(
                    runtimes[player],
                    request,
                    actual,
                    action,
                    skip_requests[player],
                    stats,
                )


def _maybe_append_review_row(
    rows: list[dict],
    lookup,
    *,
    record_id: str,
    player: int,
    turn: int,
    request: str,
    actual: str,
    obs: dict,
    agent: FeatureAgent,
    levels: int,
) -> None:
    target = lookup(
        record_id=record_id,
        player=player,
        turn=turn,
        request=request,
        response=actual,
        obs=obs,
    )
    if not isinstance(target, ReviewTarget):
        return
    candidate_norms = normalize_teacher_candidates(target.candidates)
    if not candidate_norms:
        return
    predicted = predict_algorithmic_response_from_state(agent, obs, levels=levels)
    scores = score_original_chaga_match(
        predicted,
        teacher_top1_norm=candidate_norms[0],
        teacher_top3_norms=candidate_norms[:3],
        accept_top3=bool(target.accept_top3),
    )
    rows.append(
        {
            "turn": int(turn),
            "player": int(player),
            "response": normalize_teacher_action(actual),
            "predicted_action": predicted,
            "predicted_normalized": normalize_teacher_action(predicted),
            "predicted_family": response_family(predicted),
            "chaga_top1_action": candidate_norms[0],
            "chaga_family": response_family(candidate_norms[0]),
            "chaga_top3_actions": "|".join(candidate_norms[:3]),
            "teacher_accept_top3": bool(target.accept_top3),
            **scores,
        }
    )


def aggregate_algorithmic_candidate_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    top1 = sum(1 for row in rows if row.get("top1_match"))
    top3 = sum(1 for row in rows if row.get("top3_match"))
    relaxed = sum(1 for row in rows if row.get("relaxed_match"))
    by_family = {}
    for family in sorted({response_family(row.get("chaga_top1_action")) for row in rows}):
        family_rows = [row for row in rows if response_family(row.get("chaga_top1_action")) == family]
        family_total = len(family_rows)
        family_top1 = sum(1 for row in family_rows if row.get("top1_match"))
        family_top3 = sum(1 for row in family_rows if row.get("top3_match"))
        family_relaxed = sum(1 for row in family_rows if row.get("relaxed_match"))
        by_family[family] = {
            "samples": family_total,
            "top1_accuracy": family_top1 / family_total if family_total else None,
            "top3_inclusion": family_top3 / family_total if family_total else None,
            "relaxed_accuracy": family_relaxed / family_total if family_total else None,
        }
    return {
        "original_samples": total,
        "original_top1_accuracy": top1 / total if total else None,
        "original_top3_inclusion": top3 / total if total else None,
        "original_relaxed_accuracy": relaxed / total if total else None,
        "by_family": by_family,
    }


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_algorithmic_baseline(args: argparse.Namespace) -> dict:
    raw_paths = [Path(path) for path in args.raw]
    rows, load_stats = collect_algorithmic_prediction_rows(
        raw_paths,
        Path(args.review_audit_jsonl),
        levels=args.lawlorentz_levels,
        max_records_per_source=args.max_records_per_source,
        max_examples=args.max_examples,
    )
    if not rows:
        raise ValueError("no CHAGA reviewed rows matched the algorithmic evaluation inputs")
    stats = aggregate_by_turn(rows)
    summary = {
        "format": "mcr_algorithmic_chaga_review_eval_v1",
        "baseline": "PyMahjongGB mahjong-algorithm shanten/fan effective-tile scorer",
        "raw": [str(path) for path in raw_paths],
        "review_audit_jsonl": args.review_audit_jsonl,
        "lawlorentz_levels": args.lawlorentz_levels,
        "load_stats": load_stats,
        **aggregate_algorithmic_candidate_metrics(rows),
        "by_turn": stats,
    }
    write_rows_csv(Path(args.csv_out), rows)
    svg = render_svg(stats, title=args.title)
    svg_path = Path(args.svg_out)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="append", required=True)
    parser.add_argument("--review-audit-jsonl", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--svg-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--lawlorentz-levels", type=int, default=3)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--title",
        default="Mahjong-algorithm relaxed mismatches vs CHAGA review candidates by turn",
    )
    args = parser.parse_args()
    summary = evaluate_algorithmic_baseline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
