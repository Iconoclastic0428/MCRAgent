#!/usr/bin/env python3
"""Seat-rotated head-to-head benchmark for persistent text-mode bot wrappers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

PLACEMENT_REWARDS_4_2_1_0 = (4.0, 2.0, 1.0, 0.0)


def judge_env() -> dict[str, str]:
    env = os.environ.copy()
    prefix = Path(".conda/mcr-cpp").resolve()
    additions = [
        prefix / "Library/mingw-w64/bin",
        prefix / "Library/usr/bin",
        prefix / "Library/bin",
        prefix / "Scripts",
    ]
    env["PATH"] = ";".join(str(path) for path in additions) + ";" + env.get("PATH", "")
    return env


def run_judge(log: list[dict], initdata: dict, exe_path: Path | str) -> dict:
    payload = {"log": log, "initdata": initdata}
    proc = subprocess.run(
        [str(exe_path)],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        env=judge_env(),
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"judge failed rc={proc.returncode}: {proc.stderr[:500]}")
    return json.loads(proc.stdout)


def _hu_fan_count_for_player(output: dict, player: int) -> int | float | None:
    display = output.get("display") or {}
    can_hu = display.get("canHu")
    if not isinstance(can_hu, list) or not 0 <= player < len(can_hu):
        return None
    fan_count = can_hu[player]
    return fan_count if isinstance(fan_count, (int, float)) else None


def _blocked_hu_fallback_response(output: dict, player: int) -> str:
    display = output.get("display") or {}
    action = str(display.get("action") or "").upper()
    actor = display.get("player")
    tile = str(display.get("tile") or "")
    try:
        actor_id = int(actor) if actor is not None else None
    except (TypeError, ValueError):
        actor_id = None
    if action == "DRAW" and actor_id == player and tile and not tile.startswith("H"):
        return f"PLAY {tile}"
    return "PASS"


def sanitize_response_for_judge(
    output: dict,
    player: int,
    response: str,
    *,
    flower_count: int | None = None,
) -> tuple[str, str]:
    head = response.strip().split(" ", 1)[0].upper() if response.strip() else "PASS"
    if head != "HU":
        return response, "OK"
    fan_count = _hu_fan_count_for_player(output, player)
    if fan_count is not None and fan_count < 8:
        return _blocked_hu_fallback_response(output, player), f"HU_BLOCKED_FAN_LT_8:{fan_count}"
    if fan_count is not None and flower_count is not None:
        base_fan = fan_count - max(0, int(flower_count))
        if base_fan < 8:
            return _blocked_hu_fallback_response(output, player), f"HU_BLOCKED_BASE_FAN_LT_8:{base_fan}/{fan_count}"
    return response, "OK"


def build_response_log(output: dict, policies: list[object]) -> dict[str, dict[str, str]]:
    response_log: dict[str, dict[str, str]] = {}
    for player, request in (output.get("content") or {}).items():
        index = int(player)
        raw_response = policies[index].respond(str(request))
        flower_count_getter = getattr(policies[index], "hu_guard_flower_count", None)
        flower_count = flower_count_getter() if callable(flower_count_getter) else None
        response, guard_reason = sanitize_response_for_judge(
            output,
            index,
            raw_response,
            flower_count=flower_count,
        )
        item = {"response": response, "raw": raw_response, "verdict": "OK"}
        if guard_reason != "OK":
            item["guard"] = guard_reason
        response_log[str(player)] = item
    return response_log


def scores_from_finish(output: dict) -> list[int | float]:
    content = output.get("content") or {}
    return [content.get(str(player), 0) for player in range(4)]


def run_match(
    policies: list[object],
    initdata: dict,
    *,
    exe_path: Path | str,
    max_turns: int = 500,
) -> dict:
    log: list[dict] = []
    final_output: dict | None = None
    turns = 0
    for turns in range(1, max_turns + 1):
        output = run_judge(log, initdata, exe_path)
        command = output.get("command")
        if command == "finish":
            final_output = output
            break
        if command != "request":
            raise RuntimeError(f"unexpected judge command: {command}")
        response_log = build_response_log(output, policies)
        log.append(
            {
                "output": output,
                "response": response_log,
            }
        )
    else:
        raise RuntimeError(f"match exceeded max_turns={max_turns}")

    if final_output is None:
        raise RuntimeError("judge never returned finish")

    return {
        "final_output": final_output,
        "scores": scores_from_finish(final_output),
        "turns": turns,
        "log": log,
        "terminal_reason": final_output.get("display", {}).get("action"),
    }


def placement_rewards_from_scores(scores: list[int | float]) -> list[float]:
    if len(scores) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    values = [float(score) for score in scores[:4]]
    if all(value == values[0] for value in values):
        return [0.0, 0.0, 0.0, 0.0]
    rewards = [0.0, 0.0, 0.0, 0.0]
    ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    index = 0
    while index < len(ranked):
        score = ranked[index][1]
        end = index + 1
        while end < len(ranked) and ranked[end][1] == score:
            end += 1
        reward = sum(PLACEMENT_REWARDS_4_2_1_0[index:end]) / (end - index)
        for seat, _ in ranked[index:end]:
            rewards[seat] = reward
        index = end
    return rewards


def load_initdata(path: Path, limit: int | None = None, offset: int = 0) -> list[dict]:
    initdata = []
    seen = 0
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            if seen < offset:
                seen += 1
                continue
            record = json.loads(line)
            initdata.append(record["initdata"])
            seen += 1
            if limit is not None and len(initdata) >= limit:
                break
    return initdata


def infer_hu_kind(scores: list[int | float], winner: int) -> tuple[str, int | None]:
    winner_score = scores[winner]
    loser_scores = [scores[index] for index in range(4) if index != winner]
    if len(set(loser_scores)) == 1 and winner_score == -3 * loser_scores[0]:
        return "self_draw", None
    deal_in = min((index for index in range(4) if index != winner), key=lambda index: scores[index])
    return "discard", deal_in


def summarize_tracked(results: list[dict], tracked_seat: int) -> dict:
    total = len(results)
    total_score = 0.0
    total_reward = 0.0
    hu = 0
    self_draw = 0
    discard_hu = 0
    huang = 0
    deal_in = 0
    first_place = 0
    terminal_actions: Counter[str] = Counter()
    for result in results:
        scores = [float(value) for value in result.get("scores") or [0, 0, 0, 0]]
        total_score += scores[tracked_seat]
        rewards = placement_rewards_from_scores(scores)
        total_reward += rewards[tracked_seat]
        if scores[tracked_seat] == max(scores) and scores[tracked_seat] > 0:
            first_place += 1
        display = ((result.get("final_output") or {}).get("display") or {})
        action = str(display.get("action") or result.get("terminal_reason") or "UNKNOWN")
        terminal_actions[action] += 1
        if action == "HUANG":
            huang += 1
        if action != "HU":
            continue
        winner = int(display.get("player"))
        kind, loser = infer_hu_kind(scores, winner)
        if winner == tracked_seat:
            hu += 1
            if kind == "self_draw":
                self_draw += 1
            else:
                discard_hu += 1
        elif kind == "discard" and loser == tracked_seat:
            deal_in += 1
    return {
        "games": total,
        "tracked_seat": tracked_seat,
        "average_score": total_score / total if total else 0.0,
        "average_placement_reward_4_2_1_0": total_reward / total if total else 0.0,
        "hu_games": hu,
        "self_draw_games": self_draw,
        "discard_hu_games": discard_hu,
        "deal_in_games": deal_in,
        "huang_games": huang,
        "first_place_games": first_place,
        "hu_rate": hu / total if total else 0.0,
        "self_draw_rate": self_draw / total if total else 0.0,
        "discard_hu_rate": discard_hu / total if total else 0.0,
        "deal_in_rate": deal_in / total if total else 0.0,
        "huang_rate": huang / total if total else 0.0,
        "first_place_rate": first_place / total if total else 0.0,
        "terminal_actions": dict(terminal_actions),
    }


class PersistentTextBotPolicy:
    def __init__(self, script_path: str | Path, *, max_restarts: int = 3) -> None:
        self.script_path = str(script_path)
        self.max_restarts = int(max_restarts)
        self.requests: list[str] = []
        self.player_id: int | None = None
        self.flower_counts = [0, 0, 0, 0]
        self.restart_count = 0
        self.proc = self._spawn()

    def _spawn(self) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, self.script_path, "--protocol", "text"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=judge_env(),
        )
        assert proc.stdin is not None
        proc.stdin.write("1\n")
        proc.stdin.flush()
        return proc

    def _restart_and_replay(self) -> None:
        self.close()
        self.proc = self._spawn()
        self.restart_count += 1
        replayed = list(self.requests)
        self.requests.clear()
        self.player_id = None
        self.flower_counts = [0, 0, 0, 0]
        for request in replayed:
            self.respond(request)

    def _observe_request_for_hu_guard(self, request: str) -> None:
        tokens = request.strip().split()
        if not tokens:
            return
        if tokens[0] == "0" and len(tokens) >= 2:
            try:
                self.player_id = int(tokens[1])
            except ValueError:
                self.player_id = None
            return
        if tokens[0] == "1" and len(tokens) >= 5:
            try:
                self.flower_counts = [int(value) for value in tokens[1:5]]
            except ValueError:
                self.flower_counts = [0, 0, 0, 0]
            return
        if tokens[0] == "3" and len(tokens) >= 4 and tokens[2].upper() == "BUHUA":
            try:
                actor = int(tokens[1])
            except ValueError:
                return
            if 0 <= actor < 4:
                self.flower_counts[actor] += 1

    def hu_guard_flower_count(self) -> int | None:
        if self.player_id is None or not 0 <= self.player_id < 4:
            return None
        return self.flower_counts[self.player_id]

    def respond(self, request: str) -> str:
        attempts = 0
        while True:
            if self.proc.stdin is None or self.proc.stdout is None:
                raise RuntimeError("bot subprocess stdio is unavailable")
            self.proc.stdin.write(request + "\n")
            self.proc.stdin.flush()
            response = "PASS"
            while True:
                line = self.proc.stdout.readline()
                if line == "":
                    stderr = ""
                    if self.proc.stderr is not None:
                        stderr = self.proc.stderr.read()
                    if attempts >= self.max_restarts:
                        raise RuntimeError(
                            "bot subprocess exited while waiting for response "
                            f"after {attempts} restart(s); last request={request!r}; "
                            f"stderr: {stderr[:2000]}"
                        )
                    attempts += 1
                    self._restart_and_replay()
                    break
                text = line.strip()
                if text == ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<":
                    self._observe_request_for_hu_guard(request)
                    self.requests.append(request)
                    return response
                if text:
                    response = text

    def close(self) -> None:
        if self.proc.poll() is None:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-script", required=True)
    parser.add_argument("--opponent-script", required=True)
    parser.add_argument("--raw", default="data/eval/botzone_mcr_first64_suit_permuted_384.jsonl")
    parser.add_argument("--games-per-seat", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--judge", default="build/official_judge/mcr_judge.exe")
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    initdata_items = load_initdata(
        Path(args.raw),
        limit=int(args.games_per_seat),
        offset=int(args.offset),
    )
    all_results: list[dict] = []
    seat_summaries: list[dict] = []

    def write_progress(partial_seat: int | None = None, partial_results: list[dict] | None = None) -> dict:
        current_seat_summaries = list(seat_summaries)
        if partial_seat is not None and partial_results is not None:
            current_seat_summaries.append(
                {
                    "seat": partial_seat,
                    "tracked": summarize_tracked(partial_results, partial_seat),
                }
            )
        total_games = len(all_results)
        if partial_results is not None:
            total_games += len(partial_results)
        total_score = 0.0
        total_reward = 0.0
        hu = 0
        self_draw = 0
        discard_hu = 0
        huang = 0
        deal_in = 0
        first_place = 0
        terminal_actions: Counter[str] = Counter()
        for seat_summary in current_seat_summaries:
            tracked = seat_summary["tracked"]
            games = int(tracked["games"])
            total_score += float(tracked["average_score"]) * games
            total_reward += float(tracked["average_placement_reward_4_2_1_0"]) * games
            hu += int(tracked["hu_games"])
            self_draw += int(tracked["self_draw_games"])
            discard_hu += int(tracked["discard_hu_games"])
            huang += int(tracked["huang_games"])
            deal_in += int(tracked["deal_in_games"])
            first_place += int(tracked["first_place_games"])
            terminal_actions.update(tracked["terminal_actions"])
        aggregate = {
            "games": total_games,
            "average_score": total_score / total_games if total_games else 0.0,
            "average_placement_reward_4_2_1_0": total_reward / total_games if total_games else 0.0,
            "hu_rate": hu / total_games if total_games else 0.0,
            "self_draw_rate": self_draw / total_games if total_games else 0.0,
            "discard_hu_rate": discard_hu / total_games if total_games else 0.0,
            "deal_in_rate": deal_in / total_games if total_games else 0.0,
            "huang_rate": huang / total_games if total_games else 0.0,
            "first_place_rate": first_place / total_games if total_games else 0.0,
            "terminal_actions": dict(terminal_actions),
        }
        payload = {
            "policy_script": str(args.policy_script),
            "opponent_script": str(args.opponent_script),
            "raw": str(args.raw),
            "games_per_seat": int(args.games_per_seat),
            "total_games": total_games,
            "seat_summaries": current_seat_summaries,
            "aggregate_tracked_policy": aggregate,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    for seat in range(4):
        seat_results: list[dict] = []
        policies = []
        try:
            for player in range(4):
                script = args.policy_script if player == seat else args.opponent_script
                policies.append(PersistentTextBotPolicy(script))
            for game_index, initdata in enumerate(initdata_items):
                result = run_match(
                    policies,
                    initdata,
                    exe_path=args.judge,
                    max_turns=int(args.max_turns),
                )
                compact = {key: value for key, value in result.items() if key != "log"}
                compact["game_index"] = game_index
                seat_results.append(compact)
                if args.progress_every > 0 and (game_index + 1) % int(args.progress_every) == 0:
                    partial_payload = write_progress(seat, seat_results)
                    print(
                        json.dumps(
                            {
                                "event": "benchmark_progress",
                                "seat": seat,
                                "game_index": game_index,
                                "games_completed": partial_payload["total_games"],
                                "aggregate": partial_payload["aggregate_tracked_policy"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            for policy in policies:
                policy.close()
        all_results.extend(seat_results)
        seat_summaries.append(
            {
                "seat": seat,
                "tracked": summarize_tracked(seat_results, seat),
            }
        )
        payload = write_progress()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
