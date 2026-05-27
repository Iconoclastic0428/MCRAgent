#!/usr/bin/env python3
"""Lightweight Botzone-protocol self-play for MCR policy iteration.

This is not the official judge. It is a deterministic Python harness for
generating policy trajectories and comparing current policies while the C++
judge/fan calculator is not buildable on this host.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

from hand_features import min_shanten
from policy_bot import BotzonePolicy, ShantenHeuristicPredictor, SklearnPredictor


PolicyFactory = Callable[[], BotzonePolicy]


ALL_TILES = (
    [f"{suit}{rank}" for suit in ("W", "B", "T") for rank in range(1, 10) for _ in range(4)]
    + [f"F{rank}" for rank in range(1, 5) for _ in range(4)]
    + [f"J{rank}" for rank in range(1, 4) for _ in range(4)]
)


def reaction_priority(response: str) -> int:
    action = response.split()[0] if response else "PASS"
    return {"PASS": 0, "CHI": 1, "PENG": 2, "GANG": 3, "HU": 4}.get(action, 0)


def choose_reaction_winner(actor: int, responses: dict[int, str]) -> tuple[int, str] | None:
    candidates = [
        (reaction_priority(response), (player - actor) % 4, player, response)
        for player, response in responses.items()
        if player != actor and reaction_priority(response) > 0
    ]
    if not candidates:
        return None
    priority, _, player, response = max(candidates, key=lambda item: (item[0], -item[1]))
    return player, response


def make_policy_factory(kind: str, model: str | None = None) -> PolicyFactory:
    def factory() -> BotzonePolicy:
        if kind == "fallback":
            return BotzonePolicy()
        if kind == "shanten":
            return BotzonePolicy(ShantenHeuristicPredictor())
        if kind == "model":
            if model is None:
                raise ValueError("model path is required for model policy")
            return BotzonePolicy(SklearnPredictor(Path(model)))
        raise ValueError(f"unknown policy kind: {kind}")

    return factory


def build_wall(seed: int) -> list[str]:
    wall = list(ALL_TILES)
    rng = random.Random(seed)
    rng.shuffle(wall)
    return wall


def run_game(policy_factories: list[PolicyFactory], seed: int, max_turns: int = 200) -> dict:
    policies = [factory() for factory in policy_factories]
    wall = build_wall(seed)
    trajectory: list[dict] = []

    quan = seed % 4
    for player, policy in enumerate(policies):
        request = f"0 {player} {quan}"
        response = policy.respond(request)
        trajectory.append({"player": player, "request": request, "response": response})

    for player, policy in enumerate(policies):
        hand = [wall.pop() for _ in range(13)]
        request = "1 0 0 0 0 " + " ".join(hand)
        response = policy.respond(request)
        trajectory.append({"player": player, "request": request, "response": response})

    current = 0
    turns = 0
    winner: int | None = None
    terminal_reason = "turn_limit"

    while wall and turns < max_turns and winner is None:
        turns += 1
        drawn = wall.pop()
        request = f"2 {drawn}"
        response = policies[current].respond(request)
        trajectory.append({"player": current, "request": request, "response": response})

        action = response.split()[0] if response else "PASS"
        if action == "HU":
            winner = current
            terminal_reason = "hu"
            break
        if action in {"GANG", "BUGANG"}:
            continue
        if action != "PLAY":
            current = (current + 1) % 4
            continue

        discard = response.split()[1]
        reaction_result = resolve_reactions(
            actor=current,
            event=f"3 {current} PLAY {discard}",
            policies=policies,
            trajectory=trajectory,
        )
        if reaction_result is None:
            current = (current + 1) % 4
            continue

        claimant, claim_response = reaction_result
        claim_action = claim_response.split()[0]
        if claim_action == "HU":
            winner = claimant
            terminal_reason = "hu"
            break
        if claim_action in {"PENG", "CHI"} and len(claim_response.split()) >= 2:
            follow_discard = claim_response.split()[-1]
            nested = resolve_reactions(
                actor=claimant,
                event=f"3 {claimant} {claim_action} {follow_discard}",
                policies=policies,
                trajectory=trajectory,
            )
            if nested is not None and nested[1].split()[0] == "HU":
                winner = nested[0]
                terminal_reason = "hu"
                break
            current = (claimant + 1) % 4
        else:
            current = claimant

    if winner is None:
        terminal_reason = "wall" if not wall else terminal_reason
    rewards = terminal_rewards(policies, winner)
    return {
        "seed": seed,
        "turns": turns,
        "terminal_reason": terminal_reason,
        "winner": winner,
        "rewards": rewards,
        "remaining_wall": len(wall),
        "trajectory": trajectory,
    }


def resolve_reactions(
    actor: int, event: str, policies: list[BotzonePolicy], trajectory: list[dict]
) -> tuple[int, str] | None:
    responses: dict[int, str] = {}
    for offset in range(1, 4):
        player = (actor + offset) % 4
        response = policies[player].respond(event)
        responses[player] = response
        trajectory.append({"player": player, "request": event, "response": response})
    return choose_reaction_winner(actor, responses)


def terminal_rewards(policies: list[BotzonePolicy], winner: int | None) -> list[float]:
    if winner is not None:
        return [3.0 if player == winner else -1.0 for player in range(4)]
    shanten = [min_shanten(policy.hand.elements()) for policy in policies]
    best = min(shanten)
    winners = [index for index, value in enumerate(shanten) if value == best]
    return [1.0 / len(winners) if player in winners else -1.0 / (4 - len(winners) or 1) for player in range(4)]


def run_match_set(args: argparse.Namespace) -> dict:
    factory = make_policy_factory(args.policy, args.model)
    opponents = [make_policy_factory(args.opponent, args.opponent_model) for _ in range(3)]
    results = []
    score_totals = [0.0, 0.0, 0.0, 0.0]
    wins = [0, 0, 0, 0]
    for game_index in range(args.games):
        factories = [factory, *opponents]
        result = run_game(factories, seed=args.seed + game_index, max_turns=args.max_turns)
        results.append(result if args.keep_trajectories else {k: v for k, v in result.items() if k != "trajectory"})
        for player, reward in enumerate(result["rewards"]):
            score_totals[player] += reward
        if result["winner"] is not None:
            wins[result["winner"]] += 1
    summary = {
        "policy": args.policy,
        "model": args.model,
        "opponent": args.opponent,
        "opponent_model": args.opponent_model,
        "games": args.games,
        "seed": args.seed,
        "score_totals": score_totals,
        "average_rewards": [value / args.games for value in score_totals],
        "wins": wins,
        "results": results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["fallback", "shanten", "model"], default="fallback")
    parser.add_argument("--model", default=None)
    parser.add_argument("--opponent", choices=["fallback", "shanten", "model"], default="fallback")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--keep-trajectories", action="store_true")
    parser.add_argument("--out", default="runs/selfplay_eval.json")
    args = parser.parse_args()

    summary = run_match_set(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
