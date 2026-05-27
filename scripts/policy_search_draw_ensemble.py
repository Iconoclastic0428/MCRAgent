#!/usr/bin/env python3
"""Cross-entropy style policy search over draw-ensemble weights."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from official_judge_match import run_match_set


def load_pickle(path: Path | str) -> dict:
    with Path(path).open("rb") as src:
        return pickle.load(src)


def normalize_weights(weights: Sequence[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must have a positive total")
    return [float(weight) / total for weight in weights]


def parse_weight_vector(text: str) -> list[float]:
    weights = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not weights:
        raise ValueError("weight vector must not be empty")
    return weights


def candidate_reward(summary: dict) -> tuple[float, float]:
    games = int(summary.get("games") or 0)
    wins = summary.get("wins") or [0]
    average_scores = summary.get("average_scores") or [0.0]
    win_rate = float(wins[0]) / games if games else 0.0
    return win_rate, float(average_scores[0])


def rank_candidate_summaries(summaries: Sequence[dict]) -> list[dict]:
    return sorted(summaries, key=candidate_reward, reverse=True)


def write_candidate_policy(
    draw_model_paths: Sequence[Path | str],
    draw_weights: Sequence[float],
    reaction_model_path: Path | str,
    out_path: Path | str,
    prefer_hu: bool = True,
    reaction_thresholds: dict | None = None,
) -> dict:
    if len(draw_model_paths) != len(draw_weights):
        raise ValueError("draw model and draw weight counts must match")
    normalized = normalize_weights([float(weight) for weight in draw_weights])
    payload = {
        "kind": "draw_ensemble_composite_policy",
        "draw_payloads": [load_pickle(path) for path in draw_model_paths],
        "draw_weights": normalized,
        "reaction_payload": load_pickle(reaction_model_path),
        "components": {
            "draw_models": [str(path) for path in draw_model_paths],
            "reaction_model": str(reaction_model_path),
        },
        "policy_search": {
            "algorithm": "cross_entropy_weight_search",
            "reward": "official_sample_win_rate_then_average_score",
        },
    }
    if prefer_hu:
        payload["prefer_hu"] = True
    if reaction_thresholds:
        payload["reaction_thresholds"] = dict(reaction_thresholds)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as dst:
        pickle.dump(payload, dst)
    return payload


def _weight_tag(weights: Sequence[float]) -> str:
    return "_".join(f"{int(round(weight * 1000)):03d}" for weight in weights)


def dedupe_weight_vectors(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    seen: set[tuple[float, ...]] = set()
    deduped: list[list[float]] = []
    for vector in vectors:
        normalized = normalize_weights(vector)
        key = tuple(round(weight, 6) for weight in normalized)
        if key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def random_weight_vector(count: int, rng: random.Random) -> list[float]:
    return normalize_weights([rng.expovariate(1.0) for _ in range(count)])


def perturb_weights(weights: Sequence[float], noise: float, rng: random.Random) -> list[float]:
    return normalize_weights([max(0.0001, weight + rng.gauss(0.0, noise)) for weight in weights])


def make_population(
    seed_vectors: Sequence[Sequence[float]],
    draw_count: int,
    population_size: int,
    noise: float,
    rng: random.Random,
) -> list[list[float]]:
    population: list[list[float]] = []
    population.extend(dedupe_weight_vectors(seed_vectors))
    if not population:
        population.append(normalize_weights([1.0] * draw_count))

    seed_index = 0
    while len(population) < population_size:
        if noise > 0 and seed_index < len(population):
            population.append(perturb_weights(population[seed_index], noise, rng))
            seed_index += 1
        else:
            population.append(random_weight_vector(draw_count, rng))
    return dedupe_weight_vectors(population)[:population_size]


def evaluate_candidate(weights: Sequence[float], args: argparse.Namespace, iteration: int, index: int) -> dict:
    normalized = normalize_weights(weights)
    tag = _weight_tag(normalized)
    stem = f"{args.prefix}_iter{iteration:02d}_cand{index:02d}_{tag}"
    model_path = Path(args.model_dir) / f"{stem}.pkl"
    summary_path = Path(args.out_dir) / f"{stem}.json"

    if args.reuse_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    write_candidate_policy(
        draw_model_paths=args.draw_model,
        draw_weights=normalized,
        reaction_model_path=args.reaction_model,
        out_path=model_path,
        prefer_hu=args.prefer_hu,
    )
    match_args = SimpleNamespace(
        policy="model",
        model=str(model_path),
        opponent=args.opponent,
        opponent_model=args.opponent_model,
        raw=args.raw,
        games=args.games,
        offset=args.offset,
        max_turns=args.max_turns,
        judge=args.judge,
        aleo_exe=args.aleo_exe,
        sample_exe=args.sample_exe,
    )
    summary = run_match_set(match_args)
    summary["candidate"] = {
        "model": str(model_path),
        "draw_weights": normalized,
        "draw_models": list(args.draw_model),
        "reaction_model": args.reaction_model,
        "iteration": iteration,
        "index": index,
        "reward": candidate_reward(summary),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def search(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    draw_count = len(args.draw_model)
    seed_vectors = [parse_weight_vector(text) for text in args.candidate_weights]
    if args.base_weight:
        seed_vectors.insert(0, parse_weight_vector(args.base_weight))
    if not seed_vectors:
        seed_vectors = [normalize_weights([1.0] * draw_count)]

    all_summaries: list[dict] = []
    current_seeds = seed_vectors
    noise = args.noise
    for iteration in range(args.iterations):
        population = make_population(current_seeds, draw_count, args.population, noise, rng)
        iteration_summaries = [
            evaluate_candidate(weights, args, iteration, index)
            for index, weights in enumerate(population)
        ]
        all_summaries.extend(iteration_summaries)
        elites = rank_candidate_summaries(iteration_summaries)[: max(1, args.elite)]
        current_seeds = [summary["candidate"]["draw_weights"] for summary in elites]
        noise *= args.noise_decay

    ranked = rank_candidate_summaries(all_summaries)
    report = {
        "algorithm": "cross_entropy_weight_search",
        "reward": "official_sample_win_rate_then_average_score",
        "draw_models": list(args.draw_model),
        "reaction_model": args.reaction_model,
        "games_per_candidate": args.games,
        "iterations": args.iterations,
        "population": args.population,
        "best": ranked[0] if ranked else None,
        "ranked_candidates": [
            {
                "model": summary["candidate"]["model"],
                "draw_weights": summary["candidate"]["draw_weights"],
                "wins": summary.get("wins"),
                "average_scores": summary.get("average_scores"),
                "games": summary.get("games"),
                "reward": candidate_reward(summary),
            }
            for summary in ranked
        ],
    }
    report_path = Path(args.out_dir) / f"{args.prefix}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw-model", action="append", required=True)
    parser.add_argument("--reaction-model", required=True)
    parser.add_argument("--candidate-weights", action="append", default=[])
    parser.add_argument("--base-weight", default=None)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--noise-decay", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--raw", default="data/raw/botzone_mcr_1000.jsonl")
    parser.add_argument("--opponent", choices=["fallback", "shanten", "model", "json", "aleo", "sample"], default="sample")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--judge", default="build/official_judge/mcr_judge.exe")
    parser.add_argument("--aleo-exe", default="build/aleo_bot.exe")
    parser.add_argument("--sample-exe", default="build/official_sample_bot.exe")
    parser.add_argument("--out-dir", default="runs/policy_search_draw_ensemble")
    parser.add_argument("--model-dir", default="models/policy_search_draw_ensemble")
    parser.add_argument("--prefix", default="draw_policy_search")
    parser.add_argument("--prefer-hu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    report = search(args)
    printable = dict(report)
    best = printable.get("best") or {}
    if best:
        printable["best"] = {
            "candidate": best.get("candidate"),
            "games": best.get("games"),
            "wins": best.get("wins"),
            "average_scores": best.get("average_scores"),
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
