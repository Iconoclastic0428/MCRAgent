#!/usr/bin/env python3
"""Audit CHAGA review-row alignment against reconstructed model states."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from build_lawlorentz_dataset import (
    BotzoneFeatureRuntime,
    _is_trainable_response,
    _safe_apply_own_response,
    _valid_observation,
    actual_response,
    response_to_valid_action,
)
from evaluate_chaga_replay import DEFAULT_PLAYER_RE, selected_players_from_raw_record
from tziakcha_records import (
    chi_tiles,
    convert_record,
    event_request,
    is_promoted_gang,
    meld_tile,
    parsed_actions,
    record_step,
    tile_id_to_botzone_symbol,
)


ACTION_NAMES = {
    1: "Abandon",
    2: "Play",
    3: "Chi",
    4: "Peng",
    5: "Gang",
    6: "Hu",
}


@dataclass
class DecisionState:
    player: int
    turn: int
    request: str
    actual: str
    action: int | None
    legal_actions: list[str]
    obs_summary: dict
    render: str
    window: str
    current_actor: int | None
    offered_tile: str | None
    drawn_tile: str | None
    hand_size: int | None
    hand_tiles: list[str]


def normalized_action(action: str | None) -> str:
    if not action:
        return ""
    parts = str(action).strip().split()
    if not parts:
        return ""
    head = parts[0].upper()
    if head == "PLAY" and len(parts) >= 2:
        return f"PLAY {parts[1].upper()}"
    if head == "CHI" and len(parts) >= 2:
        return f"CHI {parts[1].upper()}"
    if head == "PENG":
        return "PENG"
    if head in {"GANG", "BUGANG"}:
        return head
    if head == "HU":
        return "HU"
    if head == "PASS":
        return "PASS"
    if head == "ABANDON":
        return "ABANDON"
    return " ".join(part.upper() for part in parts)


def actual_action_from_review_row(row: dict, offered_tile: str | None = None) -> str | None:
    kind = int(row.get("r", 0) or 0)
    value = int(row.get("v", 0) or 0)
    if kind == 1:
        return "Abandon"
    if value == 0 and kind in {3, 4, 5, 6}:
        return "Pass"
    if kind == 2:
        if value < 0:
            return None
        return f"Play {tile_id_to_botzone_symbol(value)}"
    if kind == 3:
        if offered_tile is not None:
            try:
                middle = tile_id_to_botzone_symbol(chi_tiles(value, _tile_symbol_to_nominal_id(offered_tile))[1])
                return f"Chi {middle}"
            except Exception:
                pass
        return "Chi"
    if kind == 4:
        return "Peng"
    if kind == 5:
        return "BuGang" if is_promoted_gang(value) else "Gang"
    if kind == 6:
        return "Hu"
    return None


def relaxed_candidate_match(
    actual: str,
    candidates: list,
    *,
    play_ordinal: int | None,
) -> bool:
    normalized = normalized_action(actual)
    top_n = 3 if normalized.startswith("PLAY ") and play_ordinal is not None and play_ordinal <= 6 else 1
    return normalized in normalized_candidates(candidates, limit=top_n)


def normalized_candidates(candidates: list, *, limit: int) -> set[str]:
    out: set[str] = set()
    for item in candidates[:limit]:
        if isinstance(item, list | tuple) and len(item) >= 2:
            out.add(normalized_action(str(item[1])))
    return out


def fetch_review(session_id: str, seat: int, cache_dir: Path, *, timeout: float = 30.0) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{session_id}_seat{seat}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        url = f"https://tc-api.pesiu.org/review/?id={session_id}&seat={seat}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        data = json.loads(urllib.request.urlopen(request, timeout=timeout).read().decode("utf-8"))
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.05)
    if isinstance(data, list):
        return data
    if data.get("code") not in {None, 0}:
        raise ValueError(data.get("message") or f"review API error for {session_id} seat {seat}")
    return list(data.get("data") or [])


def iter_decision_states(record: dict) -> tuple[list[DecisionState], Counter[str]]:
    runtimes = [BotzoneFeatureRuntime() for _ in range(4)]
    skip_requests: list[Counter[str]] = [Counter() for _ in range(4)]
    states: list[DecisionState] = []
    stats: Counter[str] = Counter()
    logs = record.get("logs") or []
    for turn in range(0, len(logs) - 1, 2):
        output = logs[turn].get("output") or {}
        requests = output.get("content") or {}
        if not isinstance(requests, dict):
            continue
        response_log = logs[turn + 1]
        if not isinstance(response_log, dict):
            continue
        for player_text, request in requests.items():
            try:
                player = int(player_text)
            except (TypeError, ValueError):
                stats["bad_player"] += 1
                continue
            request = str(request)
            if skip_requests[player][request] > 0:
                skip_requests[player][request] -= 1
                stats["skipped_duplicate_self_event"] += 1
                continue
            response = actual_response(response_log, str(player))
            try:
                obs = runtimes[player].observe(request)
            except Exception as exc:
                stats[f"observe_error:{type(exc).__name__}"] += 1
                continue
            if obs is None or response is None:
                continue
            if not _is_trainable_response(request, response):
                stats["untrainable_response"] += 1
                continue
            action = response_to_valid_action(runtimes[player].agent, obs, request, response)
            legal_actions = _legal_action_strings(runtimes[player].agent, obs)
            states.append(
                DecisionState(
                    player=player,
                    turn=turn // 2,
                    request=request,
                    actual=response,
                    action=action,
                    legal_actions=legal_actions,
                    obs_summary=_obs_summary(runtimes[player].agent, obs),
                    render=_render_state(player, request, response, runtimes[player].agent, obs),
                    window=_request_window(request),
                    current_actor=_request_actor(request, player),
                    offered_tile=_request_offered_tile(request),
                    drawn_tile=_request_drawn_tile(request),
                    hand_size=len(getattr(runtimes[player].agent, "hand", []) or []),
                    hand_tiles=list(getattr(runtimes[player].agent, "hand", []) or []),
                )
            )
            stats["states"] += 1
            if action is None:
                stats["actual_not_in_mask"] += 1
            runtimes[player].remember_response(request, action)
            for skipped in _safe_apply_own_response(runtimes[player], request, response, stats):
                skip_requests[player][skipped] += 1
    return states, stats


def audit_records(
    raw_path: Path,
    *,
    cache_dir: Path,
    sample_size: int,
    seed: int,
    player_pattern: re.Pattern[str] = DEFAULT_PLAYER_RE,
) -> tuple[list[dict], dict]:
    all_entries: list[dict] = []
    summary: Counter[str] = Counter()
    alignment_failures: list[dict] = []
    session_review_cache: dict[tuple[str, int], list[dict]] = {}
    raw_records = list(_iter_jsonl(raw_path))
    session_api_seats = build_session_api_seat_maps(raw_records)

    for raw_record in raw_records:
        summary["records_seen"] += 1
        try:
            step = record_step(raw_record)
            selected = selected_players_from_raw_record(raw_record, player_pattern)
        except Exception as exc:
            summary["record_extract_errors"] += 1
            alignment_failures.append({"record": raw_record.get("id"), "error": str(exc)})
            continue
        if not selected:
            continue
        try:
            converted = convert_record(raw_record)
            decision_states, state_stats = iter_decision_states(converted)
        except Exception as exc:
            summary["state_build_errors"] += 1
            alignment_failures.append({"record": raw_record.get("id"), "error": str(exc)})
            continue
        summary.update({f"state_{key}": value for key, value in state_stats.items()})

        record_id = str(raw_record.get("id") or "")
        session_id = str(raw_record.get("belongs") or "")
        rr = int(step.get("i", 0) or 0)
        raw_actions = parsed_actions(step)
        states_by_player = defaultdict(list)
        for state in decision_states:
            states_by_player[state.player].append(state)

        for player_text, player_name in selected.items():
            seat = int(player_text)
            api_seat = session_api_seats.get(session_id, {}).get(player_name, seat)
            key = (session_id, api_seat)
            try:
                if key not in session_review_cache:
                    session_review_cache[key] = fetch_review(session_id, api_seat, cache_dir)
            except Exception as exc:
                summary["review_fetch_errors"] += 1
                alignment_failures.append(
                    {
                        "record": record_id,
                        "session": session_id,
                        "seat": seat,
                        "api_seat": api_seat,
                        "error": str(exc),
                    }
                )
                continue
            rows = [
                row
                for row in session_review_cache[key]
                if int(row.get("rr", -1) or -1) == rr and (row.get("extra") or {}).get("candidates")
            ]
            row_play_ordinal = 0
            cursors: dict[str, int] = defaultdict(int)
            for row in rows:
                summary["review_rows_with_candidates"] += 1
                ri = int(row.get("ri", -1) or -1)
                expected = expected_context_from_row(row, raw_actions)
                actual = actual_action_from_review_row(row, expected.get("offered_tile"))
                if not actual:
                    summary["rows_without_actual"] += 1
                    continue
                if normalized_action(actual).startswith("PLAY "):
                    row_play_ordinal += 1
                    play_ordinal: int | None = row_play_ordinal
                else:
                    play_ordinal = None
                state = find_matching_state(
                    states_by_player[seat],
                    row=row,
                    actual=actual,
                    expected=expected,
                    cursors=cursors,
                )
                if state is None:
                    summary["alignment_missing_state"] += 1
                    alignment_failures.append(
                        {
                            "record": record_id,
                            "session": session_id,
                            "rr": rr,
                            "ri": ri,
                            "seat": seat,
                            "api_seat": api_seat,
                            "actual": actual,
                            "expected": expected,
                        }
                    )
                    continue
                entry = build_audit_entry(
                    raw_record=raw_record,
                    player_name=player_name,
                    api_seat=api_seat,
                    row=row,
                    actual=actual,
                    state=state,
                    expected=expected,
                    play_ordinal=play_ordinal,
                )
                all_entries.append(entry)
                update_summary(summary, entry)

    entries = list(all_entries)
    if len(entries) > sample_size:
        entries = random.Random(seed).sample(entries, sample_size)
    result = build_summary(summary, all_entries, entries, alignment_failures, sample_size, seed)
    return entries, result


def build_session_api_seat_maps(raw_records: list[dict]) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for record in raw_records:
        session_id = str(record.get("belongs") or "")
        if not session_id or session_id in maps:
            continue
        try:
            step = record_step(record)
        except Exception:
            continue
        if int(step.get("i", 0) or 0) != 0:
            continue
        mapping: dict[str, int] = {}
        for index, player in enumerate(step.get("p") or []):
            name = str(player.get("n", "") if isinstance(player, dict) else player).strip()
            if name:
                mapping[name] = index
        maps[session_id] = mapping
    return maps


def expected_context_from_row(row: dict, raw_actions: list[dict]) -> dict:
    ri = int(row.get("ri", -1) or -1)
    kind = int(row.get("r", 0) or 0)
    out = {"ri_in_range": 0 <= ri < len(raw_actions), "window": "unknown"}
    if not out["ri_in_range"]:
        return out
    action = raw_actions[ri]
    out.update(
        {
            "raw_player": action["player"],
            "raw_type": action["type"],
            "raw_data": action["data"],
        }
    )
    if kind == 2:
        out["window"] = "draw"
        if action["type"] == 7:
            out["drawn_tile"] = tile_id_to_botzone_symbol(action["data"] & 0xFF)
            out["current_actor"] = action["player"]
        elif action["type"] == 2:
            out["current_actor"] = action["player"]
            out["actual_tile"] = tile_id_to_botzone_symbol(action["data"] & 0xFF)
        return out
    if action["type"] == 2:
        out["window"] = "claim"
        out["current_actor"] = action["player"]
        out["offered_tile"] = tile_id_to_botzone_symbol(action["data"] & 0xFF)
        out["request_action"] = "PLAY"
    elif action["type"] == 5:
        tile = meld_tile(action["data"])
        out["window"] = "claim"
        out["current_actor"] = action["player"]
        out["offered_tile"] = tile_id_to_botzone_symbol(tile)
        out["request_action"] = "BUGANG" if is_promoted_gang(action["data"]) else "GANG"
    return out


def find_matching_state(
    states: list[DecisionState],
    *,
    row: dict,
    actual: str,
    expected: dict,
    cursors: dict[str, int],
) -> DecisionState | None:
    kind = int(row.get("r", 0) or 0)
    expected_window = "draw" if kind == 2 else "claim"
    key = f"{expected_window}:{normalized_action(actual)}"
    start = cursors[key]
    for index in range(start, len(states)):
        state = states[index]
        if state.window != expected_window:
            continue
        if not _state_context_matches_expected(state, expected):
            continue
        if kind == 2 and normalized_action(state.actual) != normalized_action(actual):
            continue
        if kind != 2 and _claim_family(normalized_action(state.actual)) != _claim_family(normalized_action(actual)):
            continue
        cursors[key] = index + 1
        return state
    return None


def build_audit_entry(
    *,
    raw_record: dict,
    player_name: str,
    api_seat: int,
    row: dict,
    actual: str,
    state: DecisionState,
    expected: dict,
    play_ordinal: int | None,
) -> dict:
    candidates = (row.get("extra") or {}).get("candidates") or []
    legal_norm = {normalized_action(action) for action in state.legal_actions}
    candidate_norm_top5 = [normalized_action(item[1]) for item in candidates[:5] if len(item) >= 2]
    actual_norm = normalized_action(actual)
    checks = {
        "offered_tile_matches": _tile_matches(state.offered_tile, expected.get("offered_tile")),
        "drawn_tile_matches": _tile_matches(state.drawn_tile, expected.get("drawn_tile")),
        "current_actor_matches": expected.get("current_actor") is None
        or state.current_actor == expected.get("current_actor"),
        "window_matches": state.window == expected.get("window"),
        "hand_size_mod_ok": _hand_size_mod_ok(state),
        "actual_in_legal_mask": actual_norm in legal_norm or _claim_family(actual_norm) in legal_norm,
        "top1_in_legal_mask": bool(candidate_norm_top5) and (
            candidate_norm_top5[0] in legal_norm or _claim_family(candidate_norm_top5[0]) in legal_norm
        ),
    }
    return {
        "record_id": str(raw_record.get("id") or ""),
        "session_id": str(raw_record.get("belongs") or ""),
        "rr": row.get("rr"),
        "ri": row.get("ri"),
        "seat": state.player,
        "state_turn": state.turn,
        "state_action": state.actual,
        "api_seat": api_seat,
        "player_name": player_name,
        "render": state.render,
        "model_observation": state.obs_summary,
        "legal_action_mask": state.legal_actions,
        "human_action": actual,
        "human_action_normalized": actual_norm,
        "state_actual_response": state.actual,
        "chaga_top5_candidates": candidates[:5],
        "chaga_top5_normalized": candidate_norm_top5,
        "play_ordinal": play_ordinal,
        "match_top1": actual_norm in normalized_candidates(candidates, limit=1),
        "match_top3": actual_norm in normalized_candidates(candidates, limit=3),
        "match_top5": actual_norm in normalized_candidates(candidates, limit=5),
        "match_relaxed_first6_top3": relaxed_candidate_match(actual, candidates, play_ordinal=play_ordinal),
        "checks": checks,
        "expected_context": expected,
        "state_context": {
            "turn": state.turn,
            "state_actual_response": state.actual,
            "request": state.request,
            "window": state.window,
            "current_actor": state.current_actor,
            "offered_tile": state.offered_tile,
            "drawn_tile": state.drawn_tile,
            "hand_size": state.hand_size,
            "hand_tiles": state.hand_tiles,
        },
    }


def update_summary(summary: Counter[str], entry: dict) -> None:
    summary["aligned_review_states"] += 1
    action = entry["human_action_normalized"].split()[0]
    summary[f"action:{action}"] += 1
    if entry["match_top1"]:
        summary["match_top1"] += 1
    if entry["match_top3"]:
        summary["match_top3"] += 1
    if entry["match_top5"]:
        summary["match_top5"] += 1
    if entry["match_relaxed_first6_top3"]:
        summary["match_relaxed_first6_top3"] += 1
    for key, value in entry["checks"].items():
        if value:
            summary[f"check_pass:{key}"] += 1
        else:
            summary[f"check_fail:{key}"] += 1


def build_summary(
    counts: Counter[str],
    all_entries: list[dict],
    sampled_entries: list[dict],
    alignment_failures: list[dict],
    sample_size: int,
    seed: int,
) -> dict:
    total = counts["aligned_review_states"]
    sampled_counts = Counter()
    for entry in sampled_entries:
        update_summary(sampled_counts, entry)
    return {
        "sample_size_requested": sample_size,
        "sample_size_written": len(sampled_entries),
        "sample_seed": seed,
        "all_aligned_review_states": total,
        "all_rates": _rates_from_counts(counts),
        "sample_rates": _rates_from_counts(sampled_counts),
        "counts": dict(counts),
        "sample_counts": dict(sampled_counts),
        "alignment_failure_count": len(alignment_failures),
        "alignment_failures_preview": alignment_failures[:50],
    }


def _rates_from_counts(counts: Counter[str]) -> dict:
    total = counts["aligned_review_states"]
    return {
        "top1": _rate(counts["match_top1"], total),
        "top3": _rate(counts["match_top3"], total),
        "top5": _rate(counts["match_top5"], total),
        "relaxed_first6_top3_else_top1": _rate(counts["match_relaxed_first6_top3"], total),
        "offered_tile_match": _rate(counts["check_pass:offered_tile_matches"], total),
        "drawn_tile_match": _rate(counts["check_pass:drawn_tile_matches"], total),
        "current_actor_match": _rate(counts["check_pass:current_actor_matches"], total),
        "window_match": _rate(counts["check_pass:window_matches"], total),
        "hand_size_mod_ok": _rate(counts["check_pass:hand_size_mod_ok"], total),
        "actual_in_legal_mask": _rate(counts["check_pass:actual_in_legal_mask"], total),
        "top1_in_legal_mask": _rate(counts["check_pass:top1_in_legal_mask"], total),
    }


def _legal_action_strings(agent, obs: dict) -> list[str]:
    if agent is None:
        return []
    out = []
    for action in np.flatnonzero(obs.get("action_mask", []) > 0):
        try:
            out.append(agent.action2response(int(action)))
        except Exception:
            continue
    return out


def _obs_summary(agent, obs: dict) -> dict:
    observation = obs.get("observation")
    mask = obs.get("action_mask")
    shanten = None
    if observation is not None:
        try:
            from feature import FeatureAgent

            shanten_vec = observation[FeatureAgent.OFFSET_OBS["SHANTEN"], 0, :7]
            shanten = int(np.argmax(shanten_vec))
        except Exception:
            shanten = None
    return {
        "observation_shape": list(observation.shape) if observation is not None else None,
        "nonzero_observation_values": int(np.count_nonzero(observation)) if observation is not None else None,
        "legal_action_count": int(np.sum(mask)) if mask is not None else None,
        "shanten_index": shanten,
        "hand": list(getattr(agent, "hand", []) or []),
        "packs": [list(pack) for pack in (getattr(agent, "packs", [[]]) or [[]])[0]],
        "shown_tile_count": len(getattr(agent, "shownTiles", []) or []),
    }


def _render_state(player: int, request: str, response: str, agent, obs: dict) -> str:
    hand = " ".join(getattr(agent, "hand", []) or [])
    legal_count = int(np.sum(obs.get("action_mask", []))) if isinstance(obs, dict) else 0
    return f"seat={player} request='{request}' hand=[{hand}] legal_count={legal_count} human='{response}'"


def _request_window(request: str) -> str:
    tokens = request.split()
    if tokens and tokens[0] == "2":
        return "draw"
    if tokens and tokens[0] == "3":
        return "claim"
    return "other"


def _request_actor(request: str, player: int) -> int | None:
    tokens = request.split()
    if not tokens:
        return None
    if tokens[0] == "2":
        return player
    if tokens[0] == "3" and len(tokens) >= 2:
        return int(tokens[1])
    return None


def _request_offered_tile(request: str) -> str | None:
    tokens = request.split()
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2].upper() not in {"DRAW"}:
        return tokens[-1]
    return None


def _request_drawn_tile(request: str) -> str | None:
    tokens = request.split()
    if len(tokens) >= 2 and tokens[0] == "2":
        return tokens[1]
    return None


def _state_context_matches_expected(state: DecisionState, expected: dict) -> bool:
    if expected.get("window") in {"draw", "claim"} and state.window != expected["window"]:
        return False
    if expected.get("current_actor") is not None and state.current_actor != expected["current_actor"]:
        return False
    if expected.get("offered_tile") is not None and state.offered_tile != expected["offered_tile"]:
        return False
    if expected.get("drawn_tile") is not None and state.drawn_tile != expected["drawn_tile"]:
        return False
    return True


def _hand_size_mod_ok(state: DecisionState) -> bool:
    if state.hand_size is None:
        return False
    if state.window == "draw":
        return state.hand_size % 3 == 2
    if state.window == "claim":
        return state.hand_size % 3 == 1
    return True


def _tile_matches(actual: str | None, expected: str | None) -> bool:
    return expected is None or actual == expected


def _claim_family(action: str) -> str:
    head = action.split()[0] if action else ""
    return head if head in {"CHI", "PENG", "GANG", "BUGANG", "HU", "PASS", "ABANDON"} else action


def _tile_symbol_to_nominal_id(tile: str) -> int:
    suit = tile[0].upper()
    rank = int(tile[1:])
    if suit == "W":
        return (rank - 1) << 2
    if suit == "T":
        return (rank + 8) << 2
    if suit == "B":
        return (rank + 17) << 2
    if suit == "F":
        return (rank + 26) << 2
    if suit == "J":
        return (rank + 30) << 2
    if suit == "H":
        return 135 + rank
    return 0


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--cache-dir", default="data/raw/chaga_reviews")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    entries, summary = audit_records(
        Path(args.raw),
        cache_dir=Path(args.cache_dir),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n" for entry in entries),
        encoding="utf-8",
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
