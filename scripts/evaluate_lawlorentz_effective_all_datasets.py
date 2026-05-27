#!/usr/bin/env python3
"""Evaluate the Lawlorentz effective policy across preserved MCR datasets."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from lawlorentz_policy import LawlorentzEffectivePolicy, LawlorentzModelPolicy
from official_judge_match import run_match_set
from train_bc_policy import derive_action_type


DEFAULT_PATTERNS = [
    "data/raw/*.jsonl",
    "data/eval/*.jsonl",
]


def discover_datasets(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(".").glob(pattern))
    return sorted({path.resolve() for path in paths if path.is_file() and _dataset_kind(path) != "unsupported"})


def dependency_status() -> dict:
    lawlorentz_dir = Path("external/Chinese-Standard-Mahjong-DRL")
    return {
        "PyMahjongGB": importlib.util.find_spec("MahjongGB") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "lawlorentz_dir": lawlorentz_dir.exists(),
        "lawlorentz_checkpoint_files": sorted(
            str(path) for path in lawlorentz_dir.glob("checkpoint/*") if path.is_file()
        ),
    }


def evaluate_dataset(path: Path, args: argparse.Namespace) -> dict:
    kind = _dataset_kind(path)
    if kind == "replay":
        result = evaluate_replay_dataset(
            path,
            args.games_per_dataset,
            args.lawlorentz_levels,
            policy=getattr(args, "policy", "lawlorentz_effective"),
            model=getattr(args, "model", None),
        )
        result["evaluation_kind"] = "replay_agreement"
        return result
    if kind != "initdata":
        return {"evaluation_kind": "unsupported", "error": "dataset does not contain initdata or Botzone logs"}
    games = args.games_per_dataset if args.games_per_dataset else None
    match_args = SimpleNamespace(
        policy=getattr(args, "policy", "lawlorentz_effective"),
        model=getattr(args, "model", None),
        opponent=args.opponent,
        opponent_model=None,
        raw=str(path),
        games=games,
        offset=args.offset,
        max_turns=args.max_turns,
        judge=args.judge,
        aleo_exe=args.aleo_exe,
        sample_exe=args.sample_exe,
        lawlorentz_levels=args.lawlorentz_levels,
    )
    summary = run_match_set(match_args)
    result = {key: value for key, value in summary.items() if key != "results"}
    result["evaluation_kind"] = "official_sample_gate"
    return result


def evaluate_replay_dataset(
    path: Path,
    limit: int | None,
    levels: int,
    *,
    policy: str = "lawlorentz_effective",
    model: str | None = None,
) -> dict:
    metrics = {
        "matches": 0,
        "examples": 0,
        "exact_matches": 0,
        "action_type_matches": 0,
        "active_draw_examples": 0,
        "active_draw_exact_matches": 0,
        "active_draw_action_type_matches": 0,
        "reaction_examples": 0,
        "reaction_exact_matches": 0,
        "reaction_action_type_matches": 0,
        "actual_hu_responses": 0,
        "predicted_hu_responses": 0,
        "actual_hu_matches": 0,
        "predicted_hu_matches": 0,
        "actual_hu_turn_sum": 0,
        "predicted_hu_turn_sum": 0,
    }
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if limit is not None and metrics["matches"] >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            logs = record.get("logs") or []
            if not logs:
                continue
            metrics["matches"] += 1
            actual_first_hu_turn: int | None = None
            predicted_first_hu_turn: int | None = None
            policies = {
                str(player): _make_replay_policy(policy, levels=levels, model=model)
                for player in range(4)
            }
            for turn in range(0, len(logs) - 1, 2):
                turn_number = turn // 2 + 1
                requests = (logs[turn].get("output") or {}).get("content") or {}
                responses = logs[turn + 1]
                for player, request in requests.items():
                    actual = _actual_response(responses, str(player))
                    if actual is None:
                        continue
                    predicted = policies[str(player)].respond(str(request))
                    _record_replay_prediction(metrics, str(request), predicted, actual)
                    if derive_action_type(actual) == "HU":
                        metrics["actual_hu_responses"] += 1
                        if actual_first_hu_turn is None:
                            actual_first_hu_turn = turn_number
                    if derive_action_type(predicted) == "HU":
                        metrics["predicted_hu_responses"] += 1
                        if predicted_first_hu_turn is None:
                            predicted_first_hu_turn = turn_number
            if actual_first_hu_turn is not None:
                metrics["actual_hu_matches"] += 1
                metrics["actual_hu_turn_sum"] += actual_first_hu_turn
            if predicted_first_hu_turn is not None:
                metrics["predicted_hu_matches"] += 1
                metrics["predicted_hu_turn_sum"] += predicted_first_hu_turn
    _add_rates(metrics)
    return metrics


def run_all(args: argparse.Namespace) -> dict:
    patterns = args.dataset_pattern or list(DEFAULT_PATTERNS)
    datasets = discover_datasets(patterns)
    results = []
    for dataset in datasets:
        try:
            result = evaluate_dataset(dataset, args)
            result["dataset"] = str(dataset)
            results.append(result)
        except Exception as exc:
            results.append({"dataset": str(dataset), "error": str(exc)})
    return {
        "policy": getattr(args, "policy", "lawlorentz_effective"),
        "model": getattr(args, "model", None),
        "lawlorentz_levels": args.lawlorentz_levels,
        "games_per_dataset": args.games_per_dataset,
        "dependencies": dependency_status(),
        "dataset_patterns": patterns,
        "datasets": [str(path) for path in datasets],
        "results": results,
    }


def _make_replay_policy(policy: str, *, levels: int, model: str | None):
    if policy == "lawlorentz_effective":
        return LawlorentzEffectivePolicy(levels=levels)
    if policy == "lawlorentz_model":
        if model is None:
            raise ValueError("--model is required for lawlorentz_model")
        return LawlorentzModelPolicy(model, device="cpu")
    raise ValueError(f"unsupported replay policy: {policy}")


def count_initdata_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "initdata" in record:
                count += 1
    return count


def _dataset_kind(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig") as src:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "initdata" in record:
                    return "initdata"
                if "logs" in record:
                    return "replay"
                return "unsupported"
    except Exception:
        return "unsupported"
    return "unsupported"


def _actual_response(response_log: dict, player: str) -> str | None:
    item = response_log.get(str(player)) or {}
    response = item.get("raw") or item.get("response")
    return None if response is None else str(response).strip()


def _record_replay_prediction(metrics: dict, request: str, predicted: str, actual: str) -> None:
    metrics["examples"] += 1
    if predicted == actual:
        metrics["exact_matches"] += 1
    if derive_action_type(predicted) == derive_action_type(actual):
        metrics["action_type_matches"] += 1
    if request.startswith("2 "):
        metrics["active_draw_examples"] += 1
        if predicted == actual:
            metrics["active_draw_exact_matches"] += 1
        if derive_action_type(predicted) == derive_action_type(actual):
            metrics["active_draw_action_type_matches"] += 1
    if request.startswith("3 "):
        metrics["reaction_examples"] += 1
        if predicted == actual:
            metrics["reaction_exact_matches"] += 1
        if derive_action_type(predicted) == derive_action_type(actual):
            metrics["reaction_action_type_matches"] += 1


def _add_rates(metrics: dict) -> None:
    for numerator, denominator, key in [
        ("exact_matches", "examples", "exact_accuracy"),
        ("action_type_matches", "examples", "action_type_accuracy"),
        ("active_draw_exact_matches", "active_draw_examples", "active_draw_exact_accuracy"),
        ("active_draw_action_type_matches", "active_draw_examples", "active_draw_action_type_accuracy"),
        ("reaction_exact_matches", "reaction_examples", "reaction_exact_accuracy"),
        ("reaction_action_type_matches", "reaction_examples", "reaction_action_type_accuracy"),
        ("actual_hu_responses", "examples", "actual_hu_response_rate"),
        ("predicted_hu_responses", "examples", "predicted_hu_response_rate"),
        ("actual_hu_matches", "matches", "actual_hu_rate"),
        ("predicted_hu_matches", "matches", "predicted_hu_rate"),
    ]:
        metrics[key] = metrics[numerator] / metrics[denominator] if metrics[denominator] else None
    metrics["average_actual_hu_turn"] = (
        metrics["actual_hu_turn_sum"] / metrics["actual_hu_matches"]
        if metrics["actual_hu_matches"]
        else None
    )
    metrics["average_predicted_hu_turn"] = (
        metrics["predicted_hu_turn_sum"] / metrics["predicted_hu_matches"]
        if metrics["predicted_hu_matches"]
        else None
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-pattern", action="append", default=None)
    parser.add_argument("--policy", choices=["lawlorentz_effective", "lawlorentz_model"], default="lawlorentz_effective")
    parser.add_argument("--model", default=None)
    parser.add_argument("--games-per-dataset", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--lawlorentz-levels", type=int, default=1)
    parser.add_argument("--opponent", choices=["sample", "fallback", "shanten", "lawlorentz_effective"], default="sample")
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--judge", default="build/official_judge/mcr_judge.exe")
    parser.add_argument("--aleo-exe", default="build/aleo_bot.exe")
    parser.add_argument("--sample-exe", default="build/official_sample_bot.exe")
    parser.add_argument("--out", default="runs/lawlorentz_effective_all_datasets_report.json")
    args = parser.parse_args()

    report = run_all(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
