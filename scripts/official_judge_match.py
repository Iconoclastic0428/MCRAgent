#!/usr/bin/env python3
"""Run policies through the official Chinese-Standard-Mahjong judge binary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from lawlorentz_policy import LawlorentzEffectivePolicy, LawlorentzModelPolicy
from policy_bot import BotzonePolicy, ShantenHeuristicPredictor, SklearnPredictor


DEFAULT_JUDGE = Path("build/official_judge/mcr_judge.exe")
DEFAULT_ALEO = Path("build/aleo_bot.exe")
DEFAULT_SAMPLE = Path("build/official_sample_bot.exe")


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


def run_judge(
    log: list[dict],
    initdata: dict,
    exe_path: Path | str | None = None,
) -> dict:
    exe = Path(exe_path or DEFAULT_JUDGE)
    payload = {"log": log, "initdata": initdata}
    proc = subprocess.run(
        [str(exe)],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        env=judge_env(),
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"judge failed rc={proc.returncode}: {proc.stderr[:500]}")
    return json.loads(proc.stdout)


def build_response_log(output: dict, policies: list[BotzonePolicy]) -> dict:
    response_log: dict[str, dict[str, str]] = {}
    for player, request in (output.get("content") or {}).items():
        index = int(player)
        response = policies[index].respond(str(request))
        response_log[str(player)] = {"response": response, "raw": response, "verdict": "OK"}
    return response_log


def scores_from_finish(output: dict) -> list[int | float]:
    content = output.get("content") or {}
    return [content.get(str(player), 0) for player in range(4)]


JudgeFunc = Callable[[list[dict], dict, Path | str | None], dict]


def run_aleo_process(exe_path: Path | str, payload: str) -> str:
    proc = subprocess.run(
        [str(exe_path)],
        input=payload,
        text=True,
        capture_output=True,
        env=judge_env(),
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Aleo bot failed rc={proc.returncode}: {proc.stderr[:500]}")
    return proc.stdout


class AleoProcessPolicy:
    def __init__(self, exe_path: Path | str = DEFAULT_ALEO, runner=run_aleo_process):
        self.exe_path = str(exe_path)
        self.runner = runner
        self.requests: list[str] = []
        self.responses: list[str] = []
        self.error_count = 0
        self.last_error = ""

    def respond(self, request: str) -> str:
        lines = [str(len(self.requests) + 1)]
        for old_request, old_response in zip(self.requests, self.responses):
            lines.extend([old_request, old_response])
        lines.append(request)
        payload = "\n".join(lines) + "\n"
        try:
            response = extract_botzone_action(self.runner(self.exe_path, payload))
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            response = "PASS"
        self.requests.append(request)
        self.responses.append(response)
        return response

    def diagnostics(self) -> dict[str, str | int]:
        return {
            "kind": "aleo",
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


def extract_botzone_action(output: str) -> str:
    valid_actions = {"PASS", "HU", "PLAY", "PENG", "CHI", "GANG", "BUGANG"}
    for line in reversed(output.splitlines()):
        response = line.strip()
        if response and response.split()[0].upper() in valid_actions:
            return response
    return "PASS"


def run_json_bot_process(exe_path: Path | str, payload: str) -> str:
    exe = Path(exe_path)
    command = [sys.executable, str(exe)] if exe.suffix.lower() in {".py", ".pyz", ".zip"} else [str(exe)]
    proc = subprocess.run(
        command,
        input=payload,
        text=True,
        capture_output=True,
        env=judge_env(),
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"JSON bot failed rc={proc.returncode}: {proc.stderr[:500]}")
    return proc.stdout


class BotzoneJsonProcessPolicy:
    def __init__(self, exe_path: Path | str = DEFAULT_SAMPLE, runner=run_json_bot_process):
        self.exe_path = str(exe_path)
        self.runner = runner
        self.requests: list[str] = []
        self.responses: list[str] = []

    def respond(self, request: str) -> str:
        current_requests = [*self.requests, request]
        payload = json.dumps(
            {"requests": current_requests, "responses": self.responses},
            ensure_ascii=False,
        )
        output = json.loads(self.runner(self.exe_path, payload))
        response = str(output.get("response", "PASS")).strip()
        self.requests.append(request)
        self.responses.append(response)
        return response


def run_match(
    policies: list[BotzonePolicy],
    initdata: dict,
    judge_func: JudgeFunc = run_judge,
    exe_path: Path | str | None = None,
    max_turns: int = 500,
) -> dict:
    log: list[dict] = []
    for turn in range(max_turns):
        output = judge_func(log, initdata, exe_path)
        if output.get("command") == "finish":
            return {
                "terminal_reason": "finish",
                "turns": turn,
                "scores": scores_from_finish(output),
                "final_output": output,
                "policy_diagnostics": collect_policy_diagnostics(policies),
                "log": log,
            }
        wrapped_output = {"output": output}
        response_log = build_response_log(output, policies)
        log.extend([wrapped_output, response_log])
    return {
        "terminal_reason": "turn_limit",
        "turns": max_turns,
        "scores": [0, 0, 0, 0],
        "final_output": None,
        "policy_diagnostics": collect_policy_diagnostics(policies),
        "log": log,
    }


def collect_policy_diagnostics(policies: list[object]) -> list[dict]:
    diagnostics = []
    for policy in policies:
        if hasattr(policy, "diagnostics"):
            diagnostics.append(policy.diagnostics())
        else:
            diagnostics.append({})
    return diagnostics


def aggregate_policy_diagnostics(results: list[dict]) -> list[dict[str, int | float]]:
    totals: list[dict[str, int | float]] = []
    for result in results:
        for player, diagnostics in enumerate(result.get("policy_diagnostics") or []):
            while len(totals) <= player:
                totals.append({})
            for key, value in diagnostics.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    totals[player][key] = totals[player].get(key, 0) + value
    return totals


def summarize_terminals(results: list[dict]) -> dict:
    terminal_actions: dict[str, int] = {}
    player0_hu_fans: list[int] = []
    hu_count = 0
    hu_turns: list[int | float] = []
    player0_hu_count = 0
    player0_hu_turns: list[int | float] = []
    for result in results:
        display = ((result.get("final_output") or {}).get("display") or {})
        action = str(display.get("action") or result.get("terminal_reason") or "UNKNOWN")
        terminal_actions[action] = terminal_actions.get(action, 0) + 1
        turn = result.get("turns")
        try:
            player = int(display.get("player"))
        except (TypeError, ValueError):
            player = None
        if action == "HU":
            hu_count += 1
            if isinstance(turn, (int, float)):
                hu_turns.append(turn)
        if action == "HU" and player == 0:
            player0_hu_count += 1
            if isinstance(turn, (int, float)):
                player0_hu_turns.append(turn)
            fan = display.get("fanCnt")
            if fan is not None:
                player0_hu_fans.append(int(fan))
    total = len(results)
    return {
        "terminal_actions": terminal_actions,
        "hu_count": hu_count,
        "hu_rate": hu_count / total if total else None,
        "average_hu_turn": sum(hu_turns) / len(hu_turns) if hu_turns else None,
        "player0_hu_count": player0_hu_count,
        "player0_hu_rate": player0_hu_count / total if total else None,
        "player0_average_hu_turn": (
            sum(player0_hu_turns) / len(player0_hu_turns) if player0_hu_turns else None
        ),
        "player0_hu_fans": player0_hu_fans,
        "min_player0_hu_fan": min(player0_hu_fans) if player0_hu_fans else None,
        "max_player0_hu_fan": max(player0_hu_fans) if player0_hu_fans else None,
    }


def make_policy(
    kind: str,
    model: str | None = None,
    aleo_exe: str | Path = DEFAULT_ALEO,
    sample_exe: str | Path = DEFAULT_SAMPLE,
    lawlorentz_levels: int = 1,
) -> BotzonePolicy | AleoProcessPolicy | BotzoneJsonProcessPolicy | LawlorentzEffectivePolicy | LawlorentzModelPolicy:
    if kind == "lawlorentz_effective":
        return LawlorentzEffectivePolicy(levels=lawlorentz_levels)
    if kind == "lawlorentz_model":
        if model is None:
            raise ValueError("--model is required for lawlorentz_model policy")
        return LawlorentzModelPolicy(model)
    if kind == "fallback":
        return BotzonePolicy()
    if kind == "shanten":
        return BotzonePolicy(ShantenHeuristicPredictor())
    if kind == "model":
        if model is None:
            raise ValueError("--model is required for model policy")
        return BotzonePolicy(SklearnPredictor(Path(model)))
    if kind == "aleo":
        return AleoProcessPolicy(aleo_exe)
    if kind == "json":
        if model is None:
            raise ValueError("--model is required for json policy")
        return BotzoneJsonProcessPolicy(model)
    if kind == "sample":
        return BotzoneJsonProcessPolicy(sample_exe)
    raise ValueError(f"unknown policy kind: {kind}")


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


def run_match_set(args: argparse.Namespace) -> dict:
    games = []
    score_totals = [0.0, 0.0, 0.0, 0.0]
    wins = [0, 0, 0, 0]
    initdata_items = load_initdata(Path(args.raw), limit=args.games, offset=args.offset)
    lawlorentz_levels = int(getattr(args, "lawlorentz_levels", 1))
    for index, initdata in enumerate(initdata_items):
        policies = [
            make_policy(args.policy, args.model, args.aleo_exe, args.sample_exe, lawlorentz_levels),
            make_policy(
                args.opponent,
                args.opponent_model,
                args.aleo_exe,
                args.sample_exe,
                lawlorentz_levels,
            ),
            make_policy(
                args.opponent,
                args.opponent_model,
                args.aleo_exe,
                args.sample_exe,
                lawlorentz_levels,
            ),
            make_policy(
                args.opponent,
                args.opponent_model,
                args.aleo_exe,
                args.sample_exe,
                lawlorentz_levels,
            ),
        ]
        result = run_match(
            policies,
            initdata,
            exe_path=args.judge,
            max_turns=args.max_turns,
        )
        compact = {key: value for key, value in result.items() if key != "log"}
        compact["game_index"] = index
        games.append(compact)
        for player, score in enumerate(result["scores"]):
            score_totals[player] += float(score)
        best = max(result["scores"])
        for player, score in enumerate(result["scores"]):
            if score == best and score > 0:
                wins[player] += 1

    return {
        "policy": args.policy,
        "model": args.model,
        "opponent": args.opponent,
        "opponent_model": args.opponent_model,
        "judge": args.judge,
        "raw": args.raw,
        "games": len(games),
        "score_totals": score_totals,
        "average_scores": [score / len(games) if games else 0.0 for score in score_totals],
        "wins": wins,
        "policy_diagnostics_totals": aggregate_policy_diagnostics(games),
        **summarize_terminals(games),
        "results": games,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["lawlorentz_effective", "lawlorentz_model", "fallback", "shanten", "model", "json", "aleo", "sample"],
        default="fallback",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--opponent",
        choices=["lawlorentz_effective", "lawlorentz_model", "fallback", "shanten", "model", "json", "aleo", "sample"],
        default="fallback",
    )
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--raw", default="data/raw/botzone_mcr_1000.jsonl")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--judge", default=str(DEFAULT_JUDGE))
    parser.add_argument("--aleo-exe", default=str(DEFAULT_ALEO))
    parser.add_argument("--sample-exe", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--lawlorentz-levels", type=int, default=1)
    parser.add_argument("--out", default="runs/official_judge_match.json")
    args = parser.parse_args()

    summary = run_match_set(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
