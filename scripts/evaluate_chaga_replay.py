#!/usr/bin/env python3
"""Evaluate replay agreement on Tziakcha records for CHAGA02-CHAGA08 seats."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

from tziakcha_records import convert_record, record_step
from train_bc_policy import derive_action_type


try:
    from policy_bot import BotzonePolicy, SklearnPredictor
except ImportError:  # pragma: no cover - optional for tests that do not use sklearn policies.
    BotzonePolicy = None
    SklearnPredictor = None


DEFAULT_PLAYER_RE = re.compile(r"^CHAGA0[2-8](?:$|\D)", re.IGNORECASE)


def empty_metrics() -> dict:
    return {
        "records_seen": 0,
        "records_evaluated": 0,
        "selected_player_record_appearances": 0,
        "policy_calls": 0,
        "selected_decisions": 0,
        "selected_exact_matches": 0,
        "selected_action_type_matches": 0,
        "selected_actual_hu_decisions": 0,
        "selected_predicted_hu_decisions": 0,
        "selected_actual_hu_exact_matches": 0,
        "selected_actual_play_decisions": 0,
        "selected_predicted_play_decisions": 0,
        "selected_actual_play_exact_matches": 0,
        "selected_actual_play_action_type_matches": 0,
        "records_with_selected_actual_hu": 0,
        "records_with_selected_predicted_hu": 0,
        "selected_actual_hu_turn_sum": 0,
        "selected_predicted_hu_turn_sum": 0,
        "selected_player_records": {},
        "selected_player_decisions": {},
        "selected_actual_action_counts": {},
        "selected_predicted_action_counts": {},
        "per_player": {},
        "player_extract_errors": [],
        "convert_errors": [],
    }


def empty_player_metrics() -> dict:
    return {
        "selected_decisions": 0,
        "selected_exact_matches": 0,
        "selected_action_type_matches": 0,
        "selected_actual_hu_decisions": 0,
        "selected_predicted_hu_decisions": 0,
        "selected_actual_hu_exact_matches": 0,
        "selected_actual_play_decisions": 0,
        "selected_predicted_play_decisions": 0,
        "selected_actual_play_exact_matches": 0,
        "selected_actual_play_action_type_matches": 0,
        "selected_actual_action_counts": {},
        "selected_predicted_action_counts": {},
    }


def selected_players_from_raw_record(record: dict, pattern: re.Pattern[str] = DEFAULT_PLAYER_RE) -> dict[str, str]:
    step = record_step(record)
    selected: dict[str, str] = {}
    for index, player in enumerate(step.get("p") or []):
        if isinstance(player, dict):
            name = str(player.get("n", "")).strip()
        else:
            name = str(player).strip()
        if pattern.search(name):
            selected[str(index)] = name
    return selected


def evaluate_converted_record_for_players(
    record: dict,
    selected_players: dict[str, str],
    *,
    policy_factory: Callable[[], object],
    metrics: dict | None = None,
) -> dict:
    metrics = metrics or empty_metrics()
    metrics["records_evaluated"] += 1
    metrics["selected_player_record_appearances"] += len(selected_players)
    for name in selected_players.values():
        _inc(metrics["selected_player_records"], name)

    policies = {str(player): policy_factory() for player in range(4)}
    actual_first_hu_turn: int | None = None
    predicted_first_hu_turn: int | None = None

    logs = record.get("logs") or []
    for turn in range(0, len(logs) - 1, 2):
        turn_number = turn // 2 + 1
        requests = ((logs[turn].get("output") or {}).get("content") or {})
        responses = logs[turn + 1]
        for player, request in requests.items():
            player = str(player)
            actual = _actual_response(responses, player)
            if actual is None:
                continue
            policy = policies.setdefault(player, policy_factory())
            predicted = str(policy.respond(str(request))).strip()
            metrics["policy_calls"] += 1
            if player not in selected_players:
                continue
            _record_selected_prediction(
                metrics,
                player_name=selected_players[player],
                predicted=predicted,
                actual=actual,
            )
            if derive_action_type(actual) == "HU" and actual_first_hu_turn is None:
                actual_first_hu_turn = turn_number
            if derive_action_type(predicted) == "HU" and predicted_first_hu_turn is None:
                predicted_first_hu_turn = turn_number

    if actual_first_hu_turn is not None:
        metrics["records_with_selected_actual_hu"] += 1
        metrics["selected_actual_hu_turn_sum"] += actual_first_hu_turn
    if predicted_first_hu_turn is not None:
        metrics["records_with_selected_predicted_hu"] += 1
        metrics["selected_predicted_hu_turn_sum"] += predicted_first_hu_turn
    return metrics


def evaluate_raw_file(
    path: Path,
    *,
    player_pattern: re.Pattern[str] = DEFAULT_PLAYER_RE,
    limit: int | None = None,
    policy_factory: Callable[[], object],
    converter: Callable[[dict], dict] = convert_record,
) -> dict:
    metrics = empty_metrics()
    with path.open("r", encoding="utf-8-sig") as src:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            if limit is not None and metrics["records_evaluated"] >= limit:
                break
            metrics["records_seen"] += 1
            record = json.loads(line)
            try:
                selected = selected_players_from_raw_record(record, player_pattern)
            except Exception as exc:
                metrics["player_extract_errors"].append({"line": line_number, "error": str(exc)})
                continue
            if not selected:
                continue
            try:
                converted = converter(record)
            except Exception as exc:
                metrics["convert_errors"].append({"line": line_number, "error": str(exc)})
                continue
            evaluate_converted_record_for_players(
                converted,
                selected,
                policy_factory=policy_factory,
                metrics=metrics,
            )
    add_rates(metrics)
    return metrics


def evaluate_prepared_file(
    path: Path,
    *,
    limit: int | None = None,
    policy_factory: Callable[[], object],
) -> dict:
    metrics = empty_metrics()
    with path.open("r", encoding="utf-8-sig") as src:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            if limit is not None and metrics["records_evaluated"] >= limit:
                break
            metrics["records_seen"] += 1
            record = json.loads(line)
            selected = {
                str(player): str(name)
                for player, name in (record.get("train_player_names") or {}).items()
            }
            if not selected:
                metrics["player_extract_errors"].append(
                    {"line": line_number, "error": "missing train_player_names"}
                )
                continue
            evaluate_converted_record_for_players(
                record,
                selected,
                policy_factory=policy_factory,
                metrics=metrics,
            )
    add_rates(metrics)
    return metrics


def add_rates(metrics: dict) -> None:
    _add_prediction_rates(metrics)
    for player_metrics in metrics["per_player"].values():
        _add_prediction_rates(player_metrics)
    for numerator, denominator, key in [
        ("records_with_selected_actual_hu", "records_evaluated", "record_selected_actual_hu_rate"),
        ("records_with_selected_predicted_hu", "records_evaluated", "record_selected_predicted_hu_rate"),
    ]:
        metrics[key] = _rate(metrics[numerator], metrics[denominator])
    metrics["average_selected_actual_hu_turn"] = _rate(
        metrics["selected_actual_hu_turn_sum"],
        metrics["records_with_selected_actual_hu"],
    )
    metrics["average_selected_predicted_hu_turn"] = _rate(
        metrics["selected_predicted_hu_turn_sum"],
        metrics["records_with_selected_predicted_hu"],
    )


def make_policy_factory(policy: str, *, levels: int, model: str | None):
    if policy == "lawlorentz_effective":
        from lawlorentz_policy import LawlorentzEffectivePolicy

        return lambda: LawlorentzEffectivePolicy(levels=levels)
    if policy == "lawlorentz_model":
        if model is None:
            raise ValueError("--model is required when --policy lawlorentz_model")
        from lawlorentz_policy import LawlorentzModelPolicy

        return lambda: LawlorentzModelPolicy(model, device="cpu")
    if policy == "botzone_model":
        if model is None:
            raise ValueError("--model is required when --policy botzone_model")
        if BotzonePolicy is None or SklearnPredictor is None:
            raise ImportError("policy_bot is unavailable")
        model_path = Path(model)
        return lambda: BotzonePolicy(SklearnPredictor(model_path))
    raise ValueError(f"unsupported policy: {policy}")


def _record_selected_prediction(metrics: dict, *, player_name: str, predicted: str, actual: str) -> None:
    player_metrics = metrics["per_player"].setdefault(player_name, empty_player_metrics())
    _record_prediction_counts(metrics, predicted=predicted, actual=actual)
    _record_prediction_counts(player_metrics, predicted=predicted, actual=actual)
    _inc(metrics["selected_player_decisions"], player_name)


def _record_prediction_counts(metrics: dict, *, predicted: str, actual: str) -> None:
    actual_type = derive_action_type(actual)
    predicted_type = derive_action_type(predicted)
    metrics["selected_decisions"] += 1
    _inc(metrics["selected_actual_action_counts"], actual_type)
    _inc(metrics["selected_predicted_action_counts"], predicted_type)
    if predicted == actual:
        metrics["selected_exact_matches"] += 1
    if predicted_type == actual_type:
        metrics["selected_action_type_matches"] += 1
    if actual_type == "HU":
        metrics["selected_actual_hu_decisions"] += 1
        if predicted == actual:
            metrics["selected_actual_hu_exact_matches"] += 1
    if predicted_type == "HU":
        metrics["selected_predicted_hu_decisions"] += 1
    if actual_type == "PLAY":
        metrics["selected_actual_play_decisions"] += 1
        if predicted == actual:
            metrics["selected_actual_play_exact_matches"] += 1
        if predicted_type == actual_type:
            metrics["selected_actual_play_action_type_matches"] += 1
    if predicted_type == "PLAY":
        metrics["selected_predicted_play_decisions"] += 1


def _add_prediction_rates(metrics: dict) -> None:
    for numerator, denominator, key in [
        ("selected_exact_matches", "selected_decisions", "selected_exact_accuracy"),
        ("selected_action_type_matches", "selected_decisions", "selected_action_type_accuracy"),
        ("selected_actual_hu_decisions", "selected_decisions", "selected_actual_hu_decision_rate"),
        ("selected_predicted_hu_decisions", "selected_decisions", "selected_predicted_hu_decision_rate"),
        ("selected_actual_hu_exact_matches", "selected_actual_hu_decisions", "selected_hu_exact_match_rate"),
        ("selected_actual_play_exact_matches", "selected_actual_play_decisions", "selected_play_exact_match_rate"),
        (
            "selected_actual_play_action_type_matches",
            "selected_actual_play_decisions",
            "selected_play_action_type_match_rate",
        ),
    ]:
        metrics[key] = _rate(metrics[numerator], metrics[denominator])


def _actual_response(response_log: dict, player: str) -> str | None:
    item = response_log.get(str(player)) or {}
    response = item.get("raw") or item.get("response")
    return None if response is None else str(response).strip()


def _inc(values: dict[str, int], key: str, amount: int = 1) -> None:
    values[key] = values.get(key, 0) + amount


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=None)
    parser.add_argument("--converted", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--policy",
        choices=["lawlorentz_effective", "lawlorentz_model", "botzone_model"],
        default="lawlorentz_effective",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--lawlorentz-levels", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--player-regex", default=DEFAULT_PLAYER_RE.pattern)
    args = parser.parse_args()

    pattern = re.compile(args.player_regex, re.IGNORECASE)
    policy_factory = make_policy_factory(args.policy, levels=args.lawlorentz_levels, model=args.model)
    if args.converted:
        metrics = evaluate_prepared_file(
            Path(args.converted),
            limit=args.limit,
            policy_factory=policy_factory,
        )
        source_path = args.converted
        source_kind = "converted"
    elif args.raw:
        metrics = evaluate_raw_file(
            Path(args.raw),
            player_pattern=pattern,
            limit=args.limit,
            policy_factory=policy_factory,
        )
        source_path = args.raw
        source_kind = "raw"
    else:
        raise ValueError("either --raw or --converted is required")
    report = {
        "raw": args.raw,
        "converted": args.converted,
        "source": source_path,
        "source_kind": source_kind,
        "policy": args.policy,
        "model": args.model,
        "lawlorentz_levels": args.lawlorentz_levels,
        "player_regex": args.player_regex,
        "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
