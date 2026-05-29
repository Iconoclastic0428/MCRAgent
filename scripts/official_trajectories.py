#!/usr/bin/env python3
"""Generate reward-weighted training trajectories from official judge matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_judge_match import load_initdata, make_policy, run_match

POLICY_CHOICES = ["fallback", "shanten", "model", "transformer", "json", "aleo", "sample"]


def rewards_from_scores(scores: list[int | float]) -> list[float]:
    positive_total = sum(float(score) for score in scores if float(score) > 0)
    rewards: list[float] = []
    for score in scores:
        value = float(score)
        if value <= -30:
            rewards.append(-1.0)
        elif value > 0 and positive_total > 0:
            rewards.append(value / positive_total)
        else:
            rewards.append(0.0)
    return rewards


def finish_metadata_from_output(final_output: dict | None) -> dict:
    display = (final_output or {}).get("display") or {}
    return {
        "action": display.get("action"),
        "winner": display.get("player"),
        "fan_count": int(display.get("fanCnt") or 0),
        "score": display.get("score"),
    }


def game_from_official_result(result: dict, match_id: str) -> dict:
    trajectory: list[dict] = []
    log = result.get("log") or []
    for index in range(0, len(log) - 1, 2):
        output = (log[index].get("output") or {}).get("content") or {}
        responses = log[index + 1]
        for player_text in sorted(output, key=lambda value: int(value)):
            response_item = responses.get(player_text) or {}
            trajectory.append(
                {
                    "player": int(player_text),
                    "request": str(output[player_text]),
                    "response": str(response_item.get("response", "PASS")),
                }
            )
    scores = result.get("scores") or [0, 0, 0, 0]
    return {
        "match_id": match_id,
        "scores": scores,
        "rewards": rewards_from_scores(scores),
        "finish": finish_metadata_from_output(result.get("final_output")),
        "terminal_reason": result.get("terminal_reason"),
        "turns": result.get("turns"),
        "trajectory": trajectory,
    }


def run_trajectory_set(args: argparse.Namespace) -> dict:
    games = []
    score_totals = [0.0, 0.0, 0.0, 0.0]
    initdata_items = load_initdata(Path(args.raw), limit=args.games, offset=args.offset)
    qadv_model = getattr(args, "qadv_model", None)
    qadv_lambda = float(getattr(args, "qadv_lambda", 0.0) or 0.0)
    opponent_qadv_model = getattr(args, "opponent_qadv_model", None)
    opponent_qadv_lambda = float(getattr(args, "opponent_qadv_lambda", 0.0) or 0.0)
    for index, initdata in enumerate(initdata_items):
        policies = [
            make_policy(
                args.policy,
                args.model,
                qadv_model=qadv_model,
                qadv_lambda=qadv_lambda,
                aleo_exe=args.aleo_exe,
                sample_exe=args.sample_exe,
            ),
            make_policy(
                args.opponent,
                args.opponent_model,
                qadv_model=opponent_qadv_model,
                qadv_lambda=opponent_qadv_lambda,
                aleo_exe=args.aleo_exe,
                sample_exe=args.sample_exe,
            ),
            make_policy(
                args.opponent,
                args.opponent_model,
                qadv_model=opponent_qadv_model,
                qadv_lambda=opponent_qadv_lambda,
                aleo_exe=args.aleo_exe,
                sample_exe=args.sample_exe,
            ),
            make_policy(
                args.opponent,
                args.opponent_model,
                qadv_model=opponent_qadv_model,
                qadv_lambda=opponent_qadv_lambda,
                aleo_exe=args.aleo_exe,
                sample_exe=args.sample_exe,
            ),
        ]
        result = run_match(
            policies,
            initdata,
            exe_path=args.judge,
            max_turns=args.max_turns,
        )
        for player, score in enumerate(result["scores"]):
            score_totals[player] += float(score)
        games.append(game_from_official_result(result, match_id=f"official-{index}"))
    return {
        "policy": args.policy,
        "model": args.model,
        "qadv_model": qadv_model,
        "qadv_lambda": qadv_lambda,
        "opponent": args.opponent,
        "opponent_model": args.opponent_model,
        "opponent_qadv_model": opponent_qadv_model,
        "opponent_qadv_lambda": opponent_qadv_lambda,
        "judge": args.judge,
        "raw": args.raw,
        "offset": args.offset,
        "games": len(games),
        "score_totals": score_totals,
        "average_scores": [score / len(games) if games else 0.0 for score in score_totals],
        "results": games,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="model")
    parser.add_argument("--model", default="models/composite_feature_draw_reaction_nonpass50_1000.pkl")
    parser.add_argument("--qadv-model", default=None)
    parser.add_argument("--qadv-lambda", type=float, default=0.0)
    parser.add_argument("--opponent", choices=POLICY_CHOICES, default="shanten")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--opponent-qadv-model", default=None)
    parser.add_argument("--opponent-qadv-lambda", type=float, default=0.0)
    parser.add_argument("--raw", default="data/raw/botzone_mcr_1000.jsonl")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--judge", default="build/official_judge/mcr_judge.exe")
    parser.add_argument("--aleo-exe", default="build/aleo_bot.exe")
    parser.add_argument("--sample-exe", default="build/official_sample_bot.exe")
    parser.add_argument("--out", default="runs/official_trajectories.json")
    args = parser.parse_args()

    summary = run_trajectory_set(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
