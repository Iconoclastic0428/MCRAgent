#!/usr/bin/env python3
"""Generate official-judge rollout rows for future QADV return training.

This is an observability/logger path only. The rows are deliberately gated so
all-Huang or unsafe official-league runs cannot become Q-return training data.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from official_judge_match import DEFAULT_ALEO, DEFAULT_JUDGE, DEFAULT_SAMPLE
from official_judge_match import load_initdata, make_policy, run_match
from official_judge_match import placement_rewards_from_scores

QADV_ROLLOUT_SCHEMA = "qadv_league_rollout_v1"
SEAT_LABELS = ("A", "B", "C", "D")
POLICY_CHOICES = {
    "fallback",
    "shanten",
    "model",
    "transformer",
    "json",
    "aleo",
    "sample",
    "lawlorentz_effective",
    "lawlorentz_model",
}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    policy: str
    model: str | None = None
    qadv_model: str | None = None
    qadv_lambda: float = 0.0

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["qadv_lambda"] == 0.0 and data["qadv_model"] is None:
            data.pop("qadv_model")
            data.pop("qadv_lambda")
        return data


def loads_jsonl_line(line: str) -> dict[str, Any]:
    return json.loads(line)


def parse_policy_spec(text: str) -> PolicySpec:
    fields: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"policy spec part must be key=value: {part!r}")
        key, value = part.split("=", 1)
        fields[key.strip().replace("-", "_")] = value.strip()

    name = fields.get("name")
    policy = fields.get("policy")
    if not name:
        raise ValueError("policy spec requires name=...")
    if not policy:
        raise ValueError("policy spec requires policy=...")
    if policy not in POLICY_CHOICES:
        raise ValueError(f"unknown policy in spec {name!r}: {policy!r}")
    qadv_lambda = float(fields.get("qadv_lambda", "0") or 0.0)
    return PolicySpec(
        name=name,
        policy=policy,
        model=fields.get("model") or None,
        qadv_model=fields.get("qadv_model") or None,
        qadv_lambda=qadv_lambda,
    )


def _parse_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def terminal_result_from_official(result: dict[str, Any]) -> dict[str, Any]:
    final_output = result.get("final_output") or {}
    display = final_output.get("display") or {}
    action = str(display.get("action") or result.get("terminal_reason") or "UNKNOWN").upper()
    winner = _parse_int(display.get("player"))
    fan_count = _parse_int(display.get("fanCnt"))
    base_fan_count = _parse_int(display.get("baseFanCnt") or display.get("baseFan"))
    flower_count = _parse_int(display.get("flowerCnt") or display.get("flowers"))
    discarder = _parse_int(
        display.get("discarder")
        or display.get("from")
        or display.get("offer")
        or display.get("fromPlayer")
    )
    scores = [float(score) for score in (result.get("scores") or [0, 0, 0, 0])]
    return {
        "terminal_reason": result.get("terminal_reason"),
        "action": action,
        "winner": winner,
        "discarder": discarder,
        "self_draw": action == "HU" and discarder is None and winner is not None,
        "turns": result.get("turns"),
        "scores": scores,
        "fan_count": fan_count,
        "base_fan_count": base_fan_count,
        "flower_count": flower_count,
    }


def _return_fields(
    *,
    terminal: dict[str, Any],
    player: int,
    decision_turn: int,
    gamma: float,
) -> dict[str, Any]:
    scores = terminal.get("scores") or [0, 0, 0, 0]
    score_delta = float(scores[player]) if player < len(scores) else 0.0
    final_turn = int(terminal.get("turns") or 0)
    player_won = terminal.get("action") == "HU" and terminal.get("winner") == player
    player_dealt_in = terminal.get("action") == "HU" and terminal.get("discarder") == player
    hu_turn_norm = (float(final_turn) / max(1.0, float(final_turn))) if player_won and final_turn > 0 else 0.0
    point_delta_norm = _clip(score_delta / 64.0, -1.0, 1.0)
    components = {
        "point_delta_norm": point_delta_norm,
        "player_won": 1.0 if player_won else 0.0,
        "end_wait": 0.0,
        "wait_when_deal_in": 0.0,
        "player_dealt_in": 1.0 if player_dealt_in else 0.0,
        "hu_turn_norm_if_win": hu_turn_norm if player_won else 0.0,
    }
    terminal_return = (
        1.00 * components["point_delta_norm"]
        + 0.20 * components["player_won"]
        + 0.10 * components["end_wait"]
        + 0.05 * components["wait_when_deal_in"]
        - 0.15 * components["player_dealt_in"]
        - 0.10 * components["hu_turn_norm_if_win"]
    )
    discount_steps = max(0, final_turn - int(decision_turn))
    return {
        "score_delta": score_delta,
        "point_delta_norm": point_delta_norm,
        "terminal_return": terminal_return,
        "discounted_return": terminal_return * (float(gamma) ** discount_steps),
        "discount_steps": discount_steps,
        "components": components,
    }


def _low_fan_hu(terminal: dict[str, Any]) -> bool:
    action = str(terminal.get("action") or "").upper()
    if action not in {"HU", "WH"}:
        return False
    gate_fan = terminal.get("base_fan_count")
    if gate_fan is None:
        gate_fan = terminal.get("fan_count")
    return gate_fan is not None and int(gate_fan) < 8


def _illegal_hu(terminal: dict[str, Any]) -> bool:
    action = str(terminal.get("action") or "").upper()
    return action == "WH" or _low_fan_hu(terminal)


def _response_action(response: str) -> str:
    return str(response or "").strip().split(" ", 1)[0].upper()


def _winner_discard_turn(game: dict[str, Any]) -> int | None:
    """Return the winner's discard-cycle turn, on the roughly 1..22 MCR scale.

    The official judge's raw ``turns`` counter advances on every judge output
    and includes reaction windows, so it is much larger than a player's
    draw/discard turn count. For Hu timing we count the winner's completed
    ``PLAY`` responses, and count the terminal Hu opportunity as the next turn.
    """

    terminal = game.get("terminal_result") or {}
    if str(terminal.get("action") or "").upper() != "HU":
        return None
    winner = _parse_int(terminal.get("winner"))
    if winner is None or winner < 0 or winner > 3:
        return None
    discard_counts = [0, 0, 0, 0]
    for row in game.get("rows", []) or []:
        player = _parse_int(row.get("player"))
        if player is None or player < 0 or player > 3:
            continue
        action = _response_action(row.get("response", ""))
        if action == "HU" and player == winner:
            return discard_counts[player] + 1
        if action == "PLAY":
            discard_counts[player] += 1
    return discard_counts[winner] + 1


def _rows_from_result(
    *,
    result: dict[str, Any],
    match_id: str,
    game_index: int,
    seat_specs: list[PolicySpec],
    policy_pool: list[PolicySpec],
    gamma: float,
) -> list[dict[str, Any]]:
    terminal = terminal_result_from_official(result)
    pool_public = [spec.to_public_dict() for spec in policy_pool]
    seat_names = [spec.name for spec in seat_specs]
    rows: list[dict[str, Any]] = []
    decision_index = 0
    log = result.get("log") or []
    for turn_index in range(0, len(log) - 1, 2):
        output = (log[turn_index].get("output") or {}).get("content") or {}
        responses = log[turn_index + 1]
        for player_text in sorted(output, key=lambda value: int(value)):
            player = int(player_text)
            response_item = responses.get(player_text) or {}
            response = str(response_item.get("response", "PASS")).strip() or "PASS"
            returns = _return_fields(
                terminal=terminal,
                player=player,
                decision_turn=turn_index // 2,
                gamma=gamma,
            )
            rows.append(
                {
                    "schema_version": QADV_ROLLOUT_SCHEMA,
                    "match_id": match_id,
                    "game_index": game_index,
                    "decision_index": decision_index,
                    "turn_index": turn_index // 2,
                    "player": player,
                    "policy_name": seat_specs[player].name,
                    "seat_policy_names": seat_names,
                    "policy_pool": pool_public,
                    "request": str(output[player_text]),
                    "response": response,
                    "terminal_result": terminal,
                    "return_fields": returns,
                    "safety": {
                        "action_outside_mask": bool(response_item.get("verdict") not in {None, "OK"}),
                        "low_fan_hu": _low_fan_hu(terminal),
                        "illegal_hu": _illegal_hu(terminal),
                    },
                }
            )
            decision_index += 1
    return rows


def _policy_specs_for_game(policy_specs: list[PolicySpec], game_index: int) -> list[PolicySpec]:
    if not policy_specs:
        raise ValueError("at least one policy spec is required")
    return [policy_specs[(game_index + seat) % len(policy_specs)] for seat in range(4)]


def _make_policies(
    seat_specs: list[PolicySpec],
    *,
    aleo_exe: str,
    sample_exe: str,
    lawlorentz_levels: int,
):
    return [
        make_policy(
            spec.policy,
            spec.model,
            qadv_model=spec.qadv_model,
            qadv_lambda=spec.qadv_lambda,
            aleo_exe=aleo_exe,
            sample_exe=sample_exe,
            lawlorentz_levels=lawlorentz_levels,
        )
        for spec in seat_specs
    ]


def _merge_numeric_diagnostics(target: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    for key, value in diagnostics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value


def aggregate_policy_diagnostics_by_name(
    games: list[dict[str, Any]],
    policy_specs: list[PolicySpec],
) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {spec.name: {} for spec in policy_specs}
    for game in games:
        seat_names = list(game.get("seat_policy_names") or [])
        diagnostics_by_seat = list(game.get("policy_diagnostics") or [])
        for seat, diagnostics in enumerate(diagnostics_by_seat[:4]):
            name = seat_names[seat] if seat < len(seat_names) else f"seat{seat}"
            _merge_numeric_diagnostics(totals.setdefault(name, {}), diagnostics or {})
    return totals


def aggregate_policy_diagnostics_by_seat(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: list[dict[str, Any]] = []
    for game in games:
        for seat, diagnostics in enumerate(list(game.get("policy_diagnostics") or [])[:4]):
            while len(totals) <= seat:
                totals.append({})
            _merge_numeric_diagnostics(totals[seat], diagnostics or {})
    return totals


def summarize_games(
    games: list[dict[str, Any]],
    *,
    policy_specs: list[PolicySpec],
    min_games: int,
    min_rows: int,
    min_nonzero_score_rate: float,
    min_return_std: float,
    require_policy_pool_size: int,
    max_huang_rate: float | None = None,
    checked_policy_name: str | None = None,
    baseline_policy_names: list[str] | None = None,
    min_checked_hu_lift: float | None = None,
) -> dict[str, Any]:
    rows = [row for game in games for row in game.get("rows", [])]
    terminal_counts: dict[str, int] = {}
    low_fan_hu_count = 0
    illegal_hu_count = 0
    wrong_hu_count = 0
    nonzero_score_games = 0
    placement_reward_totals = [0.0, 0.0, 0.0, 0.0]
    seat_raw_score_totals = [0.0, 0.0, 0.0, 0.0]
    seat_hu_counts = [0, 0, 0, 0]
    seat_hu_turns: list[list[float]] = [[], [], [], []]
    policy_seat_games = {spec.name: 0 for spec in policy_specs}
    policy_hu_counts = {spec.name: 0 for spec in policy_specs}
    policy_hu_turns: dict[str, list[float]] = {spec.name: [] for spec in policy_specs}
    policy_raw_hu_turns: dict[str, list[float]] = {spec.name: [] for spec in policy_specs}
    policy_placement_totals = {spec.name: 0.0 for spec in policy_specs}
    policy_diagnostics_totals = aggregate_policy_diagnostics_by_name(games, policy_specs)
    seat_policy_diagnostics_totals = aggregate_policy_diagnostics_by_seat(games)
    seat_coverage = {str(seat): 0 for seat in range(4)}
    returns: list[float] = []
    action_outside_mask_count = 0
    for game in games:
        terminal = game.get("terminal_result") or {}
        action = str(terminal.get("action") or "UNKNOWN").upper()
        terminal_counts[action] = terminal_counts.get(action, 0) + 1
        scores = terminal.get("scores") or game.get("scores") or []
        if any(float(score) != 0.0 for score in scores):
            nonzero_score_games += 1
        if _low_fan_hu(terminal):
            low_fan_hu_count += 1
        if _illegal_hu(terminal):
            illegal_hu_count += 1
        if action == "WH":
            wrong_hu_count += 1
        placement_rewards = placement_rewards_from_scores(scores)
        for seat, reward in enumerate(placement_rewards):
            placement_reward_totals[seat] += float(reward)
        for seat in range(4):
            if seat < len(scores):
                seat_raw_score_totals[seat] += float(scores[seat])
        seat_names = list(game.get("seat_policy_names") or [])
        winner = _parse_int(terminal.get("winner"))
        raw_turns = terminal.get("turns")
        hu_discard_turn = _winner_discard_turn(game)
        if action == "HU" and isinstance(winner, int) and 0 <= winner < 4:
            seat_hu_counts[winner] += 1
            if isinstance(hu_discard_turn, (int, float)):
                seat_hu_turns[winner].append(float(hu_discard_turn))
        for seat, name in enumerate(seat_names[:4]):
            policy_seat_games.setdefault(name, 0)
            policy_hu_counts.setdefault(name, 0)
            policy_hu_turns.setdefault(name, [])
            policy_raw_hu_turns.setdefault(name, [])
            policy_placement_totals.setdefault(name, 0.0)
            policy_seat_games[name] += 1
            if seat < len(placement_rewards):
                policy_placement_totals[name] += float(placement_rewards[seat])
            if action == "HU" and winner == seat:
                policy_hu_counts[name] += 1
                if isinstance(hu_discard_turn, (int, float)):
                    policy_hu_turns[name].append(float(hu_discard_turn))
                if isinstance(raw_turns, (int, float)):
                    policy_raw_hu_turns[name].append(float(raw_turns))
    for row in rows:
        seat_coverage[str(row["player"])] += 1
        returns.append(float((row.get("return_fields") or {}).get("discounted_return", 0.0)))
        if (row.get("safety") or {}).get("action_outside_mask"):
            action_outside_mask_count += 1
    return_std = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    nonzero_score_rate = nonzero_score_games / len(games) if games else 0.0
    huang_games = terminal_counts.get("HUANG", 0)
    huang_rate = huang_games / len(games) if games else None
    policy_hu_rates = {
        name: policy_hu_counts.get(name, 0) / len(games) if games else None
        for name in policy_hu_counts
    }
    baseline_names = list(baseline_policy_names or [])
    baseline_rates = [
        float(policy_hu_rates[name])
        for name in baseline_names
        if name in policy_hu_rates and policy_hu_rates[name] is not None
    ]
    baseline_average_hu_rate = sum(baseline_rates) / len(baseline_rates) if baseline_rates else None
    checked_hu_rate = (
        policy_hu_rates.get(checked_policy_name)
        if checked_policy_name
        else None
    )
    if (
        checked_hu_rate is not None
        and baseline_average_hu_rate is not None
        and baseline_average_hu_rate > 0
    ):
        checked_hu_lift_vs_baseline = (checked_hu_rate / baseline_average_hu_rate) - 1.0
    elif checked_hu_rate is not None and baseline_average_hu_rate == 0:
        checked_hu_lift_vs_baseline = None
    else:
        checked_hu_lift_vs_baseline = None
    failures: dict[str, Any] = {}
    if len(games) < int(min_games):
        failures["games"] = len(games)
    if len(rows) < int(min_rows):
        failures["rows"] = len(rows)
    if nonzero_score_rate < float(min_nonzero_score_rate):
        failures["nonzero_score_rate"] = nonzero_score_rate
    if return_std < float(min_return_std):
        failures["return_std"] = return_std
    if max_huang_rate is not None and huang_rate is not None and huang_rate > float(max_huang_rate):
        failures["huang_rate"] = huang_rate
    if checked_policy_name:
        if checked_hu_rate is None:
            failures["checked_policy_missing"] = checked_policy_name
        elif baseline_names and baseline_average_hu_rate is None:
            failures["baseline_policy_missing"] = baseline_names
        elif (
            min_checked_hu_lift is not None
            and baseline_average_hu_rate is not None
            and baseline_average_hu_rate > 0
            and checked_hu_rate < baseline_average_hu_rate * (1.0 + float(min_checked_hu_lift))
        ):
            failures["checked_policy_hu_lift"] = {
                "checked_policy": checked_policy_name,
                "checked_hu_rate": checked_hu_rate,
                "baseline_policy_names": baseline_names,
                "baseline_average_hu_rate": baseline_average_hu_rate,
                "required_lift": float(min_checked_hu_lift),
                "actual_lift": checked_hu_lift_vs_baseline,
            }
    if low_fan_hu_count:
        failures["low_fan_hu_count"] = low_fan_hu_count
    if illegal_hu_count:
        failures["illegal_hu_count"] = illegal_hu_count
    if action_outside_mask_count:
        failures["action_outside_mask_count"] = action_outside_mask_count
    if len(policy_specs) < int(require_policy_pool_size):
        failures["policy_pool_size"] = len(policy_specs)
    return {
        "format": "mcr_qadv_league_rollouts_v1",
        "schema_version": QADV_ROLLOUT_SCHEMA,
        "games": len(games),
        "rows": len(rows),
        "policy_pool_names": [spec.name for spec in policy_specs],
        "policy_pool": [spec.to_public_dict() for spec in policy_specs],
        "terminal_action_counts": terminal_counts,
        "hu_games": terminal_counts.get("HU", 0),
        "hu_rate": terminal_counts.get("HU", 0) / len(games) if games else None,
        "huang_games": huang_games,
        "huang_rate": huang_rate,
        "nonzero_score_games": nonzero_score_games,
        "nonzero_score_rate": nonzero_score_rate,
        "return_std": return_std,
        "low_fan_hu_count": low_fan_hu_count,
        "illegal_hu_count": illegal_hu_count,
        "wrong_hu_count": wrong_hu_count,
        "action_outside_mask_count": action_outside_mask_count,
        "placement_reward_schema": "score rank reward 4/2/1/0; tied nonzero ranks split rewards; all-zero games score 0",
        "placement_reward_totals_4_2_1_0": placement_reward_totals,
        "average_placement_rewards_4_2_1_0": [
            reward / len(games) if games else 0.0 for reward in placement_reward_totals
        ],
        "seat_labels": {str(seat): SEAT_LABELS[seat] for seat in range(4)},
        "seat_raw_score_totals": {
            SEAT_LABELS[seat]: seat_raw_score_totals[seat] for seat in range(4)
        },
        "seat_average_raw_scores": {
            SEAT_LABELS[seat]: seat_raw_score_totals[seat] / len(games) if games else None
            for seat in range(4)
        },
        "seat_hu_counts": {SEAT_LABELS[seat]: seat_hu_counts[seat] for seat in range(4)},
        "seat_hu_rates": {
            SEAT_LABELS[seat]: seat_hu_counts[seat] / len(games) if games else None
            for seat in range(4)
        },
        "seat_average_hu_turns": {
            SEAT_LABELS[seat]: (sum(seat_hu_turns[seat]) / len(seat_hu_turns[seat]) if seat_hu_turns[seat] else None)
            for seat in range(4)
        },
        "policy_seat_games": policy_seat_games,
        "policy_hu_counts": policy_hu_counts,
        "policy_hu_rate_denominator": "total_games",
        "policy_hu_rates": policy_hu_rates,
        "policy_hu_rates_per_seat_game": {
            name: policy_hu_counts.get(name, 0) / games_seen if games_seen else None
            for name, games_seen in policy_seat_games.items()
        },
        "policy_hu_turn_definition": (
            "winner discard-cycle turn: completed PLAY responses by the winning seat, "
            "plus one for the terminal Hu opportunity"
        ),
        "policy_average_hu_turns": {
            name: (sum(turns) / len(turns) if turns else None)
            for name, turns in policy_hu_turns.items()
        },
        "policy_average_raw_judge_hu_turns": {
            name: (sum(turns) / len(turns) if turns else None)
            for name, turns in policy_raw_hu_turns.items()
        },
        "policy_placement_reward_totals_4_2_1_0": policy_placement_totals,
        "policy_average_final_score_rewards_4_2_1_0": {
            name: policy_placement_totals.get(name, 0.0) / games_seen if games_seen else None
            for name, games_seen in policy_seat_games.items()
        },
        "self_play_guideline": {
            "max_huang_rate": max_huang_rate,
            "checked_policy_name": checked_policy_name,
            "baseline_policy_names": baseline_names,
            "baseline_average_hu_rate": baseline_average_hu_rate,
            "checked_policy_hu_rate": checked_hu_rate,
            "min_checked_hu_lift": min_checked_hu_lift,
            "checked_policy_hu_lift_vs_baseline": checked_hu_lift_vs_baseline,
            "huang_rate_passed": (
                None if max_huang_rate is None or huang_rate is None else huang_rate <= float(max_huang_rate)
            ),
            "checked_hu_lift_passed": (
                None
                if min_checked_hu_lift is None
                or checked_hu_rate is None
                or baseline_average_hu_rate is None
                or baseline_average_hu_rate <= 0
                else checked_hu_rate >= baseline_average_hu_rate * (1.0 + float(min_checked_hu_lift))
            ),
        },
        "policy_diagnostics_totals": policy_diagnostics_totals,
        "seat_policy_diagnostics_totals": seat_policy_diagnostics_totals,
        "seat_coverage": seat_coverage,
        "gate_passed": not failures,
        "gate_failures": failures,
    }


def run_rollout_set(args: argparse.Namespace) -> dict[str, Any]:
    policy_specs = [parse_policy_spec(item) for item in args.policy_spec]
    initdata_items = load_initdata(Path(args.raw), limit=args.games, offset=args.offset)
    games: list[dict[str, Any]] = []
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for game_index, initdata in enumerate(initdata_items):
            seat_specs = _policy_specs_for_game(policy_specs, game_index)
            policies = _make_policies(
                seat_specs,
                aleo_exe=args.aleo_exe,
                sample_exe=args.sample_exe,
                lawlorentz_levels=int(args.lawlorentz_levels),
            )
            result = run_match(
                policies,
                initdata,
                exe_path=args.judge,
                max_turns=int(args.max_turns),
            )
            rows = _rows_from_result(
                result=result,
                match_id=f"official-league-{int(args.offset) + game_index:06d}",
                game_index=game_index,
                seat_specs=seat_specs,
                policy_pool=policy_specs,
                gamma=float(getattr(args, "gamma", 0.995)),
            )
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            games.append(
                {
                    "scores": result.get("scores") or [0, 0, 0, 0],
                    "terminal_result": terminal_result_from_official(result),
                    "seat_policy_names": [spec.name for spec in seat_specs],
                    "policy_diagnostics": result.get("policy_diagnostics") or [],
                    "rows": rows,
                }
            )
    summary = summarize_games(
        games,
        policy_specs=policy_specs,
        min_games=int(args.min_games),
        min_rows=int(args.min_rows),
        min_nonzero_score_rate=float(args.min_nonzero_score_rate),
        min_return_std=float(args.min_return_std),
        require_policy_pool_size=int(args.require_policy_pool_size),
        max_huang_rate=getattr(args, "max_huang_rate", None),
        checked_policy_name=getattr(args, "checked_policy_name", None),
        baseline_policy_names=list(getattr(args, "baseline_policy_name", None) or []),
        min_checked_hu_lift=getattr(args, "min_checked_hu_lift", None),
    )
    summary.update(
        {
            "raw": args.raw,
            "offset": int(args.offset),
            "out_jsonl": str(out_path),
        }
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if getattr(args, "fail_on_gate", False) and not summary["gate_passed"]:
        raise SystemExit(f"rollout gate failed: {summary['gate_failures']}")
    return summary


def _default_policy_specs(args: argparse.Namespace) -> list[str]:
    specs = [f"name=base,policy=transformer,model={args.model}"]
    if args.qadv_model:
        specs.append(
            f"name=qadv005,policy=transformer,model={args.model},"
            f"qadv_model={args.qadv_model},qadv_lambda=0.05"
        )
        specs.append(
            f"name=qadv010,policy=transformer,model={args.model},"
            f"qadv_model={args.qadv_model},qadv_lambda=0.10"
        )
    specs.append("name=shanten,policy=shanten")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-spec", action="append", default=None)
    parser.add_argument("--model", default="models/transformer_candidate_finetune_medhard_l40_20260528b.pt")
    parser.add_argument("--qadv-model", default=None)
    parser.add_argument("--raw", default="data/raw/botzone_mcr_1000.jsonl")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--judge", default=str(DEFAULT_JUDGE))
    parser.add_argument("--aleo-exe", default=str(DEFAULT_ALEO))
    parser.add_argument("--sample-exe", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--lawlorentz-levels", type=int, default=1)
    parser.add_argument("--out-jsonl", default="data/processed/qadv/qadv_league_rollouts_v1.jsonl")
    parser.add_argument("--summary-out", default="data/processed/qadv/qadv_league_rollouts_v1_summary.json")
    parser.add_argument("--min-games", type=int, default=200)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-nonzero-score-rate", type=float, default=0.05)
    parser.add_argument("--min-return-std", type=float, default=0.03)
    parser.add_argument("--require-policy-pool-size", type=int, default=2)
    parser.add_argument("--max-huang-rate", type=float, default=None)
    parser.add_argument("--checked-policy-name", default=None)
    parser.add_argument("--baseline-policy-name", action="append", default=[])
    parser.add_argument("--min-checked-hu-lift", type=float, default=None)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if not args.policy_spec:
        args.policy_spec = _default_policy_specs(args)
    summary = run_rollout_set(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
