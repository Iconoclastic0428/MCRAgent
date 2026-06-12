"""Collect Tjong self-play logs through the official judge."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from .policy_bot import TjongCheckpointPredictor, respond_json_with_predictor


def add_scripts_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))


def collect(args: argparse.Namespace) -> dict:
    add_scripts_to_path()
    from official_judge_match import load_initdata, run_match  # noqa: PLC0415
    from official_fan import OfficialFanChecker  # noqa: PLC0415

    initdata_items = load_initdata(Path(args.raw), limit=args.games, offset=args.offset)
    raw_out = Path(args.out_raw)
    fan_out = Path(args.out_fan_items)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    fan_out.parent.mkdir(parents=True, exist_ok=True)
    fan_checker = OfficialFanChecker.default()
    if getattr(args, "fan_checker", None):
        fan_checker = OfficialFanChecker(args.fan_checker)
    predictor = TjongCheckpointPredictor(
        args.checkpoint,
        device=args.device,
        require_encoding_version=args.require_encoding_version,
        require_paper_config=args.require_paper_config,
    )
    accumulator = SelfplaySummaryAccumulator()
    games = 0
    with raw_out.open("w", encoding="utf-8") as raw_file, fan_out.open("w", encoding="utf-8") as fan_file:
        for index, initdata in enumerate(initdata_items):
            policies = [
                TjongBotzoneJsonReplayPolicy(predictor, fan_checker=fan_checker) for _ in range(4)
            ]
            result = run_match(policies, initdata, exe_path=args.judge, max_turns=args.max_turns)
            match_id = f"tjong-selfplay-{args.offset + index}"
            record = {
                "match_id": match_id,
                "game": "Chinese-Standard-Mahjong",
                "initdata": initdata,
                "scores": {str(player): result["scores"][player] for player in range(4)},
                "logs": result.get("log") or [],
                "final_output": result.get("final_output"),
                "terminal_reason": result.get("terminal_reason"),
                "turn_count": result.get("turns"),
            }
            raw_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            fan_file.write(json.dumps(extract_fan_items(record), ensure_ascii=False, separators=(",", ":")) + "\n")
            raw_file.flush()
            fan_file.flush()
            accumulator.add(record)
            games += 1
            if games == 1 or games % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "selfplay_progress",
                            "games": games,
                            "requested_games": args.games,
                            "offset": args.offset,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    summary = accumulator.to_summary()
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "required_encoding_version": args.require_encoding_version,
            "require_paper_config": args.require_paper_config,
            "raw": args.raw,
            "requested_games": args.games,
            "offset": args.offset,
            "max_turns": args.max_turns,
            "games": games,
            "out_raw": str(raw_out),
            "out_fan_items": str(fan_out),
            "policy_interface": "botzone_json_replay_inprocess",
        }
    )
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


class TjongBotzoneJsonReplayPolicy:
    """In-process Botzone JSON policy used by official-judge self-play.

    This keeps the rollout contract identical to the deployable JSON bot:
    every decision is made from the full Botzone ``requests``/``responses``
    arrays, then replayed through the same runtime encoder used at training
    time. Keeping it in-process avoids spawning Python for every action.
    """

    def __init__(self, predictor, *, fan_checker=None):
        self.predictor = predictor
        self.fan_checker = fan_checker
        self.requests: list[str] = []
        self.responses: list[str] = []
        self.error_count = 0
        self.last_error = ""

    def respond(self, request: str) -> str:
        payload = {
            "requests": [*self.requests, str(request)],
            "responses": list(self.responses),
        }
        response = respond_json_with_predictor(
            payload,
            self.predictor,
            fan_checker=self.fan_checker,
        ).strip()
        self.requests.append(str(request))
        self.responses.append(response)
        return response

    def diagnostics(self) -> dict[str, int | str]:
        return {
            "kind": "tjong_botzone_json_replay",
            "decisions": len(self.responses),
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


def extract_fan_items(record: dict) -> dict:
    display = (((record.get("final_output") or {}).get("display")) or {})
    fans = []
    for item in display.get("fan") or []:
        fans.append(
            {
                "score": float(item.get("value", 0.0)) * float(item.get("cnt", 1.0)),
                "name": str(item.get("name", "")),
                # Exact per-fan tile attribution is not provided by Botzone display.
                # This collector writes the fan values and names; a separate attribution
                # source is still required for faithful fan-backward population.
                "tiles": [],
            }
        )
    return {"match_id": record.get("match_id"), "fans": fans}


class SelfplaySummaryAccumulator:
    def __init__(self) -> None:
        self.total = 0
        self.terminal_actions: dict[str, int] = {}
        self.hu_turns: list[float] = []
        self.hu_fan_totals: list[float] = []
        self.fan_breakdown: Counter[str] = Counter()
        self.fan_occurrences: Counter[str] = Counter()
        self.fan_score_totals: Counter[str] = Counter()
        self.deal_in_counts = [0, 0, 0, 0]
        self.per_player = {
            str(player): {
                "score_label": chr(ord("A") + player),
                "hu_count": 0,
                "hu_turns": [],
                "hu_fan_totals": [],
                "raw_score_total": 0.0,
                "fan_breakdown": Counter(),
                "fan_occurrences": Counter(),
                "fan_score_totals": Counter(),
                "position_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
            }
            for player in range(4)
        }

    def add(self, record: dict) -> None:
        self.total += 1
        display = (((record.get("final_output") or {}).get("display")) or {})
        action = str(display.get("action") or "UNKNOWN").upper()
        self.terminal_actions[action] = self.terminal_actions.get(action, 0) + 1
        scores = record_scores(record)
        positions = placement_ranks(scores)
        for player, score in enumerate(scores):
            player_summary = self.per_player[str(player)]
            player_summary["raw_score_total"] += float(score)
            player_summary["position_counts"][str(positions[player])] += 1
        if action == "HU":
            winner = display_player(display)
            turn = numeric_turn(record.get("turn_count"))
            if turn is not None:
                self.hu_turns.append(turn)
            if winner is not None:
                self.per_player[str(winner)]["hu_count"] += 1
                if turn is not None:
                    self.per_player[str(winner)]["hu_turns"].append(turn)
                fan_total = numeric_turn(display.get("fanCnt"))
                if fan_total is not None:
                    self.hu_fan_totals.append(fan_total)
                    self.per_player[str(winner)]["hu_fan_totals"].append(fan_total)
                self._add_fans(display, winner)
            loser = infer_deal_in_loser(scores, winner)
            if loser is not None:
                self.deal_in_counts[loser] += 1

    def _add_fans(self, display: dict, winner: int) -> None:
        player_item = self.per_player[str(winner)]
        seen_names: set[str] = set()
        for fan in display.get("fan") or []:
            name = str(fan.get("name") or "").strip() or "UNKNOWN"
            count = numeric_turn(fan.get("cnt"))
            value = numeric_turn(fan.get("value"))
            count = float(count) if count is not None else 1.0
            value = float(value) if value is not None else 0.0
            seen_names.add(name)
            self.fan_occurrences[name] += count
            self.fan_score_totals[name] += value * count
            player_item["fan_occurrences"][name] += count
            player_item["fan_score_totals"][name] += value * count
        for name in seen_names:
            self.fan_breakdown[name] += 1
            player_item["fan_breakdown"][name] += 1

    def to_summary(self) -> dict:
        hu_count = int(self.terminal_actions.get("HU", 0))
        huang_count = int(self.terminal_actions.get("HUANG", 0))
        score_table = []
        per_player_summary = {}
        for player in range(4):
            item = self.per_player[str(player)]
            player_hu_turns = [float(value) for value in item["hu_turns"]]
            player_fan_totals = [float(value) for value in item["hu_fan_totals"]]
            raw_score_total = float(item["raw_score_total"])
            player_summary = {
                "score_label": item["score_label"],
                "hu_count": int(item["hu_count"]),
                "deal_in_count": int(self.deal_in_counts[player]),
                "position_counts": dict(item["position_counts"]),
                "player": player,
                "hu_rate": float(item["hu_count"] / self.total) if self.total else None,
                "deal_in_rate": float(self.deal_in_counts[player] / self.total) if self.total else None,
                "average_hu_turn": (
                    float(sum(player_hu_turns) / len(player_hu_turns)) if player_hu_turns else None
                ),
                "average_hu_fan": (
                    float(sum(player_fan_totals) / len(player_fan_totals)) if player_fan_totals else None
                ),
                "average_raw_score": float(raw_score_total / self.total) if self.total else None,
                "fan_breakdown": fan_breakdown_summary(
                    item["fan_breakdown"],
                    item["fan_occurrences"],
                    item["fan_score_totals"],
                ),
            }
            per_player_summary[str(player)] = player_summary
            score_table.append(dict(player_summary))
        return {
            "games": self.total,
            "terminal_actions": dict(self.terminal_actions),
            "hu_count": hu_count,
            "hu_rate": float(hu_count / self.total) if self.total else None,
            "hu_turn_average": float(sum(self.hu_turns) / len(self.hu_turns)) if self.hu_turns else None,
            "hu_fan_average": (
                float(sum(self.hu_fan_totals) / len(self.hu_fan_totals)) if self.hu_fan_totals else None
            ),
            "hu_fan_min": float(min(self.hu_fan_totals)) if self.hu_fan_totals else None,
            "hu_fan_max": float(max(self.hu_fan_totals)) if self.hu_fan_totals else None,
            "huang_count": huang_count,
            "huang_rate": float(huang_count / self.total) if self.total else None,
            "hu_rate_definition": "seat HU count divided by total games; all four player rates sum to total HU rate",
            "hu_turn_definition": "average judge turn_count among games where the seat HU",
            "deal_in_definition": "seat count as unique lowest non-winner score on HU terminal divided by total games; self-draw/tied loser terminals count as no deal-in",
            "raw_score_definition": "average official judge final raw score over all games for that seat",
            "fan_breakdown": fan_breakdown_summary(
                self.fan_breakdown,
                self.fan_occurrences,
                self.fan_score_totals,
            ),
            "per_player": per_player_summary,
            "score_table": score_table,
        }


def summarize(records: Iterable[dict]) -> dict:
    accumulator = SelfplaySummaryAccumulator()
    for record in records:
        accumulator.add(record)
    return accumulator.to_summary()


def record_scores(record: dict) -> list[float]:
    scores = record.get("scores")
    if isinstance(scores, dict):
        return [float(scores.get(str(player), 0.0) or 0.0) for player in range(4)]
    if isinstance(scores, list):
        padded = [*scores, 0.0, 0.0, 0.0, 0.0]
        return [float(padded[player] or 0.0) for player in range(4)]
    display_scores = (((record.get("final_output") or {}).get("display")) or {}).get("score")
    if isinstance(display_scores, list):
        padded = [*display_scores, 0.0, 0.0, 0.0, 0.0]
        return [float(padded[player] or 0.0) for player in range(4)]
    return [0.0, 0.0, 0.0, 0.0]


def placement_ranks(scores: list[float]) -> list[int]:
    return [1 + sum(1 for other in scores if float(other) > float(score)) for score in scores[:4]]


def display_player(display: dict) -> int | None:
    try:
        player = int(display.get("player"))
    except (TypeError, ValueError):
        return None
    return player if 0 <= player < 4 else None


def numeric_turn(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_deal_in_loser(scores: list[float], winner: int | None) -> int | None:
    if winner is None or not 0 <= int(winner) < 4 or len(scores) < 4:
        return None
    others = [(player, float(score)) for player, score in enumerate(scores[:4]) if player != int(winner)]
    if not others:
        return None
    minimum = min(score for _, score in others)
    losers = [player for player, score in others if score == minimum]
    return losers[0] if len(losers) == 1 else None


def fan_breakdown_summary(
    hand_counts: Counter[str],
    occurrence_counts: Counter[str],
    score_totals: Counter[str],
) -> dict[str, dict[str, float | int]]:
    names = sorted(set(hand_counts) | set(occurrence_counts) | set(score_totals))
    return {
        name: {
            "hu_hands": int(hand_counts.get(name, 0)),
            "occurrences": float(occurrence_counts.get(name, 0.0)),
            "score_total": float(score_totals.get(name, 0.0)),
        }
        for name in names
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw", default="data/raw/botzone_mcr_5000.jsonl")
    parser.add_argument("--games", type=int, default=763358)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--judge", default="build/official_judge/mcr_judge.exe")
    parser.add_argument("--fan-checker", default=None)
    parser.add_argument("--out-raw", default="data/processed/tjong/tjong_selfplay_raw.jsonl")
    parser.add_argument("--out-fan-items", default="data/processed/tjong/tjong_selfplay_fan_items.jsonl")
    parser.add_argument("--summary-out", default="runs/tjong_selfplay_summary.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--require-paper-config", action="store_true")
    args = parser.parse_args()
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
