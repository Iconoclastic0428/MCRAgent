#!/usr/bin/env python3
"""Summarize official-judge match output into fixed-league feasibility metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from official_judge_match import DEFAULT_ALEO, DEFAULT_JUDGE, DEFAULT_SAMPLE, run_match_set


POLICY_CHOICES = [
    "lawlorentz_effective",
    "lawlorentz_model",
    "fallback",
    "shanten",
    "model",
    "transformer",
    "json",
    "aleo",
    "sample",
]


def _display(result: dict[str, Any]) -> dict[str, Any]:
    return (result.get("final_output") or {}).get("display") or {}


def _terminal_action(result: dict[str, Any]) -> str:
    display = _display(result)
    return str(display.get("action") or result.get("terminal_reason") or "UNKNOWN")


def _display_player(result: dict[str, Any]) -> int | None:
    try:
        return int(_display(result).get("player"))
    except (TypeError, ValueError):
        return None


def infer_deal_in_player(result: dict[str, Any]) -> int | None:
    """Infer the single discarder from official score deltas when possible.

    The judge display used by this harness does not expose the loser that dealt
    into Hu. If the three non-winning players have equal score deltas, treat the
    hand as self-draw-like. If exactly one non-winner has the strict minimum
    delta, report that player as an inferred deal-in.
    """

    if _terminal_action(result) != "HU":
        return None
    winner = _display_player(result)
    scores = result.get("scores") or []
    if winner is None or len(scores) < 4:
        return None
    loser_scores: list[tuple[int, float]] = [
        (player, float(score))
        for player, score in enumerate(scores[:4])
        if player != winner
    ]
    if not loser_scores:
        return None
    values = [score for _, score in loser_scores]
    if len(set(values)) == 1:
        return None
    minimum = min(values)
    minimum_players = [player for player, score in loser_scores if score == minimum]
    return minimum_players[0] if len(minimum_players) == 1 else None


def end_wait_players(result: dict[str, Any]) -> list[int]:
    """Return players reported as waiting in judge display `canHu` fields."""

    can_hu = _display(result).get("canHu")
    if not isinstance(can_hu, list):
        return []
    players: list[int] = []
    for player, value in enumerate(can_hu[:4]):
        try:
            if int(value) >= 0:
                players.append(player)
        except (TypeError, ValueError):
            continue
    return players


def _target_diagnostics(results: list[dict[str, Any]], target_player: int) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for result in results:
        diagnostics = result.get("policy_diagnostics") or []
        if not (0 <= target_player < len(diagnostics)):
            continue
        for key, value in (diagnostics[target_player] or {}).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    return totals


def _mean(values: list[int | float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_feasibility(
    official_summary: dict[str, Any],
    *,
    target_player: int = 0,
) -> dict[str, Any]:
    results = list(official_summary.get("results") or [])
    games = len(results)
    hu_turns: list[int | float] = []
    target_hu_turns: list[int | float] = []
    target_hu_fans: list[int] = []
    hu_count = 0
    target_hu_count = 0
    low_fan_hu_count = 0
    all_low_fan_hu_count = 0
    deal_in_count = 0
    target_deal_in_count = 0
    wait_when_deal_in_count = 0
    end_wait_count = 0
    target_end_wait_count = 0

    for result in results:
        action = _terminal_action(result)
        winner = _display_player(result)
        turn = result.get("turns")
        if action == "HU":
            hu_count += 1
            if isinstance(turn, (int, float)):
                hu_turns.append(turn)
            fan = _display(result).get("fanCnt")
            if fan is not None:
                try:
                    if int(fan) < 8:
                        all_low_fan_hu_count += 1
                except (TypeError, ValueError):
                    pass
            if winner == target_player:
                target_hu_count += 1
                if isinstance(turn, (int, float)):
                    target_hu_turns.append(turn)
                if fan is not None:
                    try:
                        fan_int = int(fan)
                        target_hu_fans.append(fan_int)
                        if fan_int < 8:
                            low_fan_hu_count += 1
                    except (TypeError, ValueError):
                        pass

        deal_in_player = infer_deal_in_player(result)
        waiting_players = end_wait_players(result)
        end_wait_count += len(waiting_players)
        if target_player in waiting_players:
            target_end_wait_count += 1
        if deal_in_player is not None:
            deal_in_count += 1
            if deal_in_player == target_player:
                target_deal_in_count += 1
                if target_player in waiting_players:
                    wait_when_deal_in_count += 1

    average_scores = official_summary.get("average_scores") or []
    target_diagnostics = _target_diagnostics(results, target_player)
    if not target_diagnostics:
        diagnostics_totals = official_summary.get("policy_diagnostics_totals") or []
        if 0 <= target_player < len(diagnostics_totals):
            target_diagnostics = dict(diagnostics_totals[target_player] or {})

    illegal_predictions = int(target_diagnostics.get("illegal_predictions", 0))
    fan_check_rejects = int(target_diagnostics.get("fan_check_rejects", 0))
    fan_check_accepts = int(target_diagnostics.get("fan_check_accepts", 0))
    fan_check_errors = int(target_diagnostics.get("fan_check_errors", 0))
    fan_check_missing = int(target_diagnostics.get("fan_check_missing", 0))

    return {
        "format": "mcr_fixed_league_feasibility_v1",
        "source_policy": official_summary.get("policy"),
        "source_model": official_summary.get("model"),
        "opponent": official_summary.get("opponent"),
        "opponent_model": official_summary.get("opponent_model"),
        "target_player": target_player,
        "games": games,
        "average_point_delta": (
            float(average_scores[target_player])
            if 0 <= target_player < len(average_scores)
            else None
        ),
        "hu_count": hu_count,
        "hu_rate": hu_count / games if games else None,
        "target_hu_count": target_hu_count,
        "target_hu_rate": target_hu_count / games if games else None,
        "average_hu_turn": _mean(hu_turns),
        "target_average_hu_turn": _mean(target_hu_turns),
        "deal_in_count": deal_in_count,
        "deal_in_rate": deal_in_count / games if games else None,
        "target_deal_in_count": target_deal_in_count,
        "target_deal_in_rate": target_deal_in_count / games if games else None,
        "end_wait_count": end_wait_count,
        "end_wait_rate": end_wait_count / (games * 4) if games else None,
        "target_end_wait_count": target_end_wait_count,
        "target_end_wait_rate": target_end_wait_count / games if games else None,
        "wait_when_deal_in_count": wait_when_deal_in_count,
        "wait_when_deal_in_rate": (
            wait_when_deal_in_count / target_deal_in_count if target_deal_in_count else None
        ),
        "illegal_prediction_count": illegal_predictions,
        "action_outside_legal_mask_count": illegal_predictions,
        "fan_check_reject_count": fan_check_rejects,
        "fan_check_accept_count": fan_check_accepts,
        "fan_check_error_count": fan_check_errors,
        "fan_check_missing_count": fan_check_missing,
        "low_fan_hu_count": low_fan_hu_count,
        "all_players_low_fan_hu_count": all_low_fan_hu_count,
        "target_hu_fans": target_hu_fans,
        "min_target_hu_fan": min(target_hu_fans) if target_hu_fans else None,
        "max_target_hu_fan": max(target_hu_fans) if target_hu_fans else None,
        "target_policy_diagnostics": target_diagnostics,
        "notes": [
            "Deal-in is inferred from a unique strict minimum loser score on HU hands.",
            "End-wait is inferred from judge display canHu entries with values >= 0.",
            "Low-fan Hu counts use display fanCnt when present; the live policy still gates Hu before action selection.",
        ],
    }


def _run_or_load_official_summary(args: argparse.Namespace) -> dict[str, Any]:
    if args.summary_json:
        return json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    return run_match_set(args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--target-player", type=int, default=0)
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="fallback")
    parser.add_argument("--model", default=None)
    parser.add_argument("--opponent", choices=POLICY_CHOICES, default="sample")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--raw", default="data/raw/botzone_mcr_1000.jsonl")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--judge", default=str(DEFAULT_JUDGE))
    parser.add_argument("--aleo-exe", default=str(DEFAULT_ALEO))
    parser.add_argument("--sample-exe", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--lawlorentz-levels", type=int, default=1)
    parser.add_argument("--out", default="runs/fixed_league_feasibility.json")
    parser.add_argument("--official-out", default=None)
    args = parser.parse_args()

    official_summary = _run_or_load_official_summary(args)
    feasibility = summarize_feasibility(official_summary, target_player=args.target_player)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feasibility, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.official_out:
        official_out_path = Path(args.official_out)
        official_out_path.parent.mkdir(parents=True, exist_ok=True)
        official_out_path.write_text(
            json.dumps(official_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(feasibility, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
