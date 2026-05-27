#!/usr/bin/env python3
"""Build Lawlorentz-format supervised data from Botzone-style raw logs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


LAWLORENTZ_DIR = Path(__file__).resolve().parents[1] / "external" / "Chinese-Standard-Mahjong-DRL"
if str(LAWLORENTZ_DIR) not in sys.path:
    sys.path.insert(0, str(LAWLORENTZ_DIR))

from feature import FeatureAgent  # noqa: E402


DRAW_ACTIONS = {"PLAY", "HU", "GANG", "BUGANG"}
REACTION_ACTIONS = {"PASS", "HU", "CHI", "PENG", "GANG"}


@dataclass
class LawlorentzExample:
    obs: np.ndarray
    mask: np.ndarray
    act: int
    player: int
    response: str
    kind: str


class BotzoneFeatureRuntime:
    """Replay one player's Botzone requests using Lawlorentz FeatureAgent."""

    def __init__(self) -> None:
        self.agent: FeatureAgent | None = None
        self.seat_wind = 0
        self.prevalent_wind = 0
        self.zimo = False
        self.angang: str | None = None

    def observe(self, request: str) -> np.ndarray | dict | None:
        tokens = request.strip().split()
        if not tokens:
            return None
        if tokens[0] == "0":
            self.seat_wind = int(tokens[1])
            self.prevalent_wind = int(tokens[2]) if len(tokens) >= 3 else 0
            self.agent = FeatureAgent(self.seat_wind)
            self.agent.request2obs(f"Wind {self.prevalent_wind}")
            self.zimo = False
            self.angang = None
            return None

        self._ensure_agent()
        assert self.agent is not None

        if tokens[0] == "1":
            self.agent.request2obs(" ".join(["Deal", *tokens[5:]]))
            return None
        if tokens[0] == "2":
            return self.agent.request2obs(f"Draw {tokens[1]}")
        if tokens[0] != "3" or len(tokens) < 3:
            return None

        actor = int(tokens[1])
        action = tokens[2].upper()
        if action == "DRAW":
            self.agent.request2obs(f"Player {actor} Draw")
            self.zimo = True
            return None
        if action == "GANG":
            if actor == self.seat_wind and self.angang:
                self.agent.request2obs(f"Player {actor} AnGang {self.angang}")
            elif self.zimo:
                self.agent.request2obs(f"Player {actor} AnGang")
            else:
                self.agent.request2obs(f"Player {actor} Gang")
            self.angang = None
            return None
        if action == "BUGANG" and len(tokens) >= 4:
            obs = self.agent.request2obs(f"Player {actor} BuGang {tokens[3]}")
            return None if actor == self.seat_wind else obs

        self.zimo = False
        if action == "CHI":
            self.agent.request2obs(f"Player {actor} Chi {tokens[3]}")
        elif action == "PENG":
            self.agent.request2obs(f"Player {actor} Peng")
        obs = self.agent.request2obs(f"Player {actor} Play {tokens[-1]}")
        return None if actor == self.seat_wind else obs

    def remember_response(self, request: str, action: int | None) -> None:
        if action is None or self.agent is None:
            return
        tokens = request.strip().split()
        if not tokens:
            return
        response = self.agent.action2response(int(action))
        parts = response.split()
        if tokens[0] == "2" and len(parts) == 2 and parts[0] == "Gang":
            self.angang = parts[1]
        elif parts and parts[0] != "Gang":
            self.angang = None

    def apply_own_response(self, request: str, response: str) -> list[str]:
        if self.agent is None:
            return []
        request_tokens = request.strip().split()
        response_tokens = response.strip().split()
        if not request_tokens or not response_tokens:
            return []
        head = response_tokens[0].upper()
        skips: list[str] = []

        if request_tokens[0] == "2" and head == "PLAY" and len(response_tokens) >= 2:
            discard = response_tokens[1]
            self.agent.request2obs(f"Player {self.seat_wind} Play {discard}")
            skips.append(f"3 {self.seat_wind} PLAY {discard}")
            return skips

        if request_tokens[0] == "3" and head == "PENG" and len(response_tokens) >= 2:
            discard = response_tokens[-1]
            self.agent.request2obs(f"Player {self.seat_wind} Peng")
            self.agent.request2obs(f"Player {self.seat_wind} Play {discard}")
            skips.append(f"3 {self.seat_wind} PENG {discard}")
            skips.append(f"3 {self.seat_wind} PLAY {discard}")
            return skips

        if request_tokens[0] == "3" and head == "CHI" and len(response_tokens) >= 3:
            middle = response_tokens[1]
            discard = response_tokens[-1]
            self.agent.request2obs(f"Player {self.seat_wind} Chi {middle}")
            self.agent.request2obs(f"Player {self.seat_wind} Play {discard}")
            skips.append(f"3 {self.seat_wind} CHI {middle} {discard}")
            skips.append(f"3 {self.seat_wind} PLAY {discard}")
            return skips

        if request_tokens[0] == "3" and head == "GANG":
            self.agent.request2obs(f"Player {self.seat_wind} Gang")
            skips.append(f"3 {self.seat_wind} GANG")
            return skips

        return skips

    def _ensure_agent(self) -> None:
        if self.agent is None:
            self.agent = FeatureAgent(self.seat_wind)
            self.agent.request2obs(f"Wind {self.prevalent_wind}")


def actual_response(response_log: dict, player: str) -> str | None:
    item = response_log.get(str(player))
    if not isinstance(item, dict):
        return None
    raw = item.get("raw") or item.get("response") or item.get("content")
    if raw is None:
        return None
    response = str(raw).strip()
    return response or None


def iter_lawlorentz_examples(record: dict) -> tuple[list[LawlorentzExample], Counter[str]]:
    runtimes = [BotzoneFeatureRuntime() for _ in range(4)]
    skip_requests: list[Counter[str]] = [Counter() for _ in range(4)]
    examples: list[LawlorentzExample] = []
    stats: Counter[str] = Counter()
    logs = record.get("logs") or []
    train_players_raw = record.get("train_players")
    train_players = (
        {str(player) for player in train_players_raw}
        if train_players_raw is not None
        else None
    )

    for turn in range(0, len(logs) - 1, 2):
        output = (logs[turn].get("output") or {})
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
            if player < 0 or player >= 4:
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
            if action is None:
                stats[f"unmapped:{response.split()[0].upper()}"] += 1
                continue
            if train_players is not None and str(player) not in train_players:
                stats["filtered_train_player_examples"] += 1
                runtimes[player].remember_response(request, action)
                for skipped in _safe_apply_own_response(runtimes[player], request, response, stats):
                    skip_requests[player][skipped] += 1
                continue
            if not _valid_observation(obs):
                stats["invalid_observation"] += 1
                runtimes[player].remember_response(request, action)
                for skipped in _safe_apply_own_response(runtimes[player], request, response, stats):
                    skip_requests[player][skipped] += 1
                continue
            if int(np.sum(obs["action_mask"])) == 1:
                stats["single_action_mask"] += 1
                runtimes[player].remember_response(request, action)
                for skipped in _safe_apply_own_response(runtimes[player], request, response, stats):
                    skip_requests[player][skipped] += 1
                continue
            examples.append(
                LawlorentzExample(
                    obs=obs["observation"].astype(np.int8),
                    mask=obs["action_mask"].astype(np.int8),
                    act=int(action),
                    player=player,
                    response=_label_response(runtimes[player].agent, action),
                    kind="primary",
                )
            )
            stats["examples"] += 1
            stats[f"action:{examples[-1].response.split()[0]}"] += 1
            runtimes[player].remember_response(request, action)
            examples.extend(_claim_discard_examples(runtimes[player].agent, response, stats, player))
            for skipped in _safe_apply_own_response(runtimes[player], request, response, stats):
                skip_requests[player][skipped] += 1

    return examples, stats


def _safe_apply_own_response(
    runtime: BotzoneFeatureRuntime,
    request: str,
    response: str,
    stats: Counter[str],
) -> list[str]:
    try:
        return runtime.apply_own_response(request, response)
    except Exception as exc:
        stats[f"apply_response_error:{type(exc).__name__}"] += 1
        return []


def _is_trainable_response(request: str, response: str) -> bool:
    head = response.split()[0].upper()
    tokens = request.strip().split()
    if not tokens:
        return False
    if tokens[0] == "2":
        return head in DRAW_ACTIONS
    if tokens[0] == "3":
        return head in REACTION_ACTIONS
    return False


def _valid_observation(obs: dict) -> bool:
    observation = obs.get("observation")
    mask = obs.get("action_mask")
    if observation is None or mask is None:
        return False
    if tuple(observation.shape) != (FeatureAgent.OBS_SIZE, 4, 9):
        return False
    if tuple(mask.shape) != (FeatureAgent.ACT_SIZE,):
        return False
    shanten = observation[FeatureAgent.OFFSET_OBS["SHANTEN"], 0, :7]
    return int(np.sum(shanten)) == 1


def response_to_valid_action(
    agent: FeatureAgent | None,
    obs: dict,
    request: str,
    response: str,
) -> int | None:
    if agent is None:
        return None
    normalized = _response_key(response)
    valid_actions = [int(action) for action in np.flatnonzero(obs["action_mask"] > 0)]
    exact_matches: list[int] = []
    fallback_matches: list[int] = []
    request_head = request.strip().split()[0] if request.strip() else ""
    for action in valid_actions:
        lawlorentz_response = agent.action2response(action)
        key = _lawlorentz_key(lawlorentz_response)
        if key == normalized:
            exact_matches.append(action)
        elif _compatible_response(key, normalized, request_head):
            fallback_matches.append(action)
    if exact_matches:
        return exact_matches[0]
    if fallback_matches:
        return fallback_matches[0]
    return None


def _claim_discard_examples(
    agent: FeatureAgent | None,
    response: str,
    stats: Counter[str],
    player: int,
) -> list[LawlorentzExample]:
    if agent is None:
        return []
    tokens = response.strip().split()
    if not tokens:
        return []
    head = tokens[0].upper()
    if head == "CHI" and len(tokens) >= 3:
        claim_request = f"Player {agent.seatWind} Chi {tokens[1]}"
        discard = tokens[-1]
    elif head == "PENG" and len(tokens) >= 2:
        claim_request = f"Player {agent.seatWind} Peng"
        discard = tokens[-1]
    else:
        return []
    try:
        trial = copy.deepcopy(agent)
        obs = trial.request2obs(claim_request)
    except Exception as exc:
        stats[f"claim_discard_error:{type(exc).__name__}"] += 1
        return []
    action = response_to_valid_action(trial, obs, f"2 {discard}", f"PLAY {discard}")
    if action is None:
        stats["claim_discard_unmapped"] += 1
        return []
    if not _valid_observation(obs):
        stats["claim_discard_invalid_observation"] += 1
        return []
    if int(np.sum(obs["action_mask"])) == 1:
        stats["claim_discard_single_action_mask"] += 1
        return []
    stats["examples"] += 1
    stats["action:Play"] += 1
    stats["claim_discard_examples"] += 1
    return [
        LawlorentzExample(
            obs=obs["observation"].astype(np.int8),
            mask=obs["action_mask"].astype(np.int8),
            act=int(action),
            player=player,
            response=_label_response(trial, action),
            kind="claim_discard",
        )
    ]


def _response_key(response: str) -> tuple[str, str | None]:
    tokens = response.strip().split()
    if not tokens:
        return ("PASS", None)
    head = tokens[0].upper()
    if head in {"PASS", "HU"}:
        return (head, None)
    if head == "PLAY" and len(tokens) >= 2:
        return ("PLAY", tokens[1])
    if head == "CHI" and len(tokens) >= 2:
        return ("CHI", tokens[1])
    if head == "PENG":
        return ("PENG", None)
    if head == "GANG":
        return ("GANG", tokens[1] if len(tokens) >= 2 else None)
    if head == "BUGANG" and len(tokens) >= 2:
        return ("BUGANG", tokens[1])
    return (head, tokens[1] if len(tokens) >= 2 else None)


def _lawlorentz_key(response: str) -> tuple[str, str | None]:
    tokens = response.strip().split()
    if not tokens:
        return ("PASS", None)
    head = tokens[0]
    if head == "Pass":
        return ("PASS", None)
    if head == "Hu":
        return ("HU", None)
    if head == "Play" and len(tokens) >= 2:
        return ("PLAY", tokens[1])
    if head == "Chi" and len(tokens) >= 2:
        return ("CHI", tokens[1])
    if head == "Peng":
        return ("PENG", None)
    if head == "Gang":
        return ("GANG", tokens[1] if len(tokens) >= 2 else None)
    if head == "BuGang" and len(tokens) >= 2:
        return ("BUGANG", tokens[1])
    return (head.upper(), tokens[1] if len(tokens) >= 2 else None)


def _compatible_response(
    lawlorentz_key: tuple[str, str | None],
    botzone_key: tuple[str, str | None],
    request_head: str,
) -> bool:
    law_head, law_tile = lawlorentz_key
    bot_head, bot_tile = botzone_key
    if law_head == "PASS" and bot_head == "PASS":
        return True
    if law_head == "HU" and bot_head == "HU":
        return True
    if law_head == "PENG" and bot_head == "PENG":
        return True
    if law_head == "CHI" and bot_head == "CHI" and law_tile == bot_tile:
        return True
    if law_head == "GANG" and bot_head == "GANG":
        if law_tile is None or bot_tile is None:
            return True
        return law_tile == bot_tile
    if law_head == "BUGANG" and bot_head in {"BUGANG", "GANG"} and request_head == "2":
        return law_tile == bot_tile
    return False


def _label_response(agent: FeatureAgent | None, action: int) -> str:
    if agent is None:
        return str(action)
    return agent.action2response(int(action))


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.shard_index = 0
        self.counts: list[int] = []
        self.obs: list[np.ndarray] = []
        self.mask: list[np.ndarray] = []
        self.act: list[int] = []

    def add(self, example: LawlorentzExample) -> None:
        self.obs.append(example.obs)
        self.mask.append(example.mask)
        self.act.append(example.act)
        if len(self.act) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.act:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.shard_index}.npz"
        np.savez(
            path,
            obs=np.stack(self.obs).astype(np.int8),
            mask=np.stack(self.mask).astype(np.int8),
            act=np.array(self.act, dtype=np.int64),
        )
        self.counts.append(len(self.act))
        self.shard_index += 1
        self.obs.clear()
        self.mask.clear()
        self.act.clear()


def iter_records(path: Path, limit: int | None = None) -> Iterable[dict]:
    seen = 0
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            yield json.loads(line)
            seen += 1
            if limit is not None and seen >= limit:
                break


def build_dataset(
    raw_paths: list[Path],
    out_dir: Path,
    *,
    shard_size: int = 20_000,
    max_records_per_source: int | None = None,
    max_examples: int | None = None,
) -> dict:
    cooked_dir = out_dir / "cooked_data_without0"
    writer = ShardWriter(cooked_dir, shard_size)
    totals: Counter[str] = Counter()
    source_summaries: list[dict] = []

    for raw_path in raw_paths:
        source_stats: Counter[str] = Counter()
        records = 0
        for record in iter_records(raw_path, limit=max_records_per_source):
            records += 1
            try:
                examples, stats = iter_lawlorentz_examples(record)
            except Exception as exc:
                source_stats[f"record_error:{type(exc).__name__}"] += 1
                continue
            source_stats.update(stats)
            for example in examples:
                if max_examples is not None and totals["examples_written"] >= max_examples:
                    break
                writer.add(example)
                totals["examples_written"] += 1
            if max_examples is not None and totals["examples_written"] >= max_examples:
                break
        source_stats["records"] = records
        source_stats["path"] = str(raw_path)
        source_summaries.append(dict(source_stats))
        if max_examples is not None and totals["examples_written"] >= max_examples:
            break

    writer.flush()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "count.json").write_text(json.dumps(writer.counts), encoding="utf-8")
    manifest = {
        "format": "lawlorentz_cooked_npz_v1",
        "feature_agent_commit": "52e680174e48f900d299341e5c04e5ae3f5cc623",
        "raw_paths": [str(path) for path in raw_paths],
        "out_dir": str(out_dir),
        "cooked_dir": str(cooked_dir),
        "shard_size": shard_size,
        "shards": len(writer.counts),
        "counts": writer.counts,
        "examples": int(sum(writer.counts)),
        "max_records_per_source": max_records_per_source,
        "max_examples": max_examples,
        "source_summaries": source_summaries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="append", required=True, help="Botzone-style raw JSONL path; repeatable")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=20_000)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    manifest = build_dataset(
        [Path(path) for path in args.raw],
        Path(args.out_dir),
        shard_size=args.shard_size,
        max_records_per_source=args.max_records_per_source,
        max_examples=args.max_examples,
    )
    printable = {key: value for key, value in manifest.items() if key != "source_summaries"}
    printable["source_summaries"] = manifest["source_summaries"]
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
