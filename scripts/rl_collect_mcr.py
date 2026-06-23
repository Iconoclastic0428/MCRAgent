#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_AGENT_DIR = (
    WORKSPACE_ROOT
    / "third_party"
    / "mahjong-agent-2025-aug12-excludespecial-3pkl"
    / "feature-agent"
)
if str(WORKSPACE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))
if str(FEATURE_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_AGENT_DIR))

from benchmark_json_policies import (  # noqa: E402
    infer_hu_kind,
    judge_env,
    load_initdata,
    sanitize_response_for_judge,
    scores_from_finish,
)
from feature import FeatureAgent  # noqa: E402
from feature_repo_json_runtime import ReplayFeatureJsonBot  # noqa: E402
from rl_model_utils import load_selfvec_model  # noqa: E402


DEFAULT_CHECKPOINT = (
    WORKSPACE_ROOT
    / "models"
    / "feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a"
    / "16.pkl"
)


def run_judge(log: list[dict[str, Any]], initdata: dict[str, Any], exe_path: Path) -> dict[str, Any]:
    import subprocess

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


def botzone_to_feature_response(response: str) -> str:
    parts = response.strip().split()
    if not parts:
        return "Pass"
    head = parts[0].upper()
    if head == "PASS":
        return "Pass"
    if head == "HU":
        return "Hu"
    if head == "PLAY" and len(parts) >= 2:
        return f"Play {parts[1]}"
    if head == "GANG":
        return "Gang" if len(parts) < 2 else f"Gang {parts[1]}"
    if head == "BUGANG" and len(parts) >= 2:
        return f"BuGang {parts[1]}"
    if head == "PENG" and len(parts) >= 2:
        return f"Peng {parts[1]}"
    if head == "CHI" and len(parts) >= 3:
        return " ".join(["Chi", *parts[1:]])
    return "Pass"


class LoggingFeaturePolicy(ReplayFeatureJsonBot):
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        epsilon: float = 0.0,
        temperature: float = 0.05,
        top_p: float = 1.0,
        record: bool = True,
        policy_name: str = "trainee",
        legal_dueling_mean: bool = False,
        seed: int = 0,
    ) -> None:
        self.FeatureAgent = FeatureAgent
        self.policy_name = policy_name
        self.epsilon = float(epsilon)
        self.temperature = max(1e-6, float(temperature))
        self.top_p = float(top_p)
        self.record = bool(record)
        self.legal_dueling_mean = bool(legal_dueling_mean)
        self.rng = random.Random(seed)
        self.pending_transitions: list[dict[str, Any]] = []
        self.blocked_hu_count = 0
        self.illegal_action_count = 0
        model = load_selfvec_model(checkpoint, device="cpu", mixed_kernel_input=True, dueling_head=True)
        model.eval()
        super().__init__(model, obs_mode="vec")

    def _new_agent(self, seat_wind: int):
        return self.FeatureAgent(seat_wind)

    def start_game(self) -> None:
        self.pending_transitions.clear()

    def _select_action(self, masked_q: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask.astype(bool))
        if legal.size == 0:
            return 0
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            logits = masked_q[legal].astype(np.float64) / self.temperature
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= probs.sum()
            if 0.0 < self.top_p < 1.0:
                order = np.argsort(-probs)
                cumulative = np.cumsum(probs[order])
                keep = order[cumulative <= self.top_p]
                if keep.size == 0:
                    keep = order[:1]
                probs2 = probs[keep]
                probs2 /= probs2.sum()
                return int(legal[int(self.rng.choices(list(keep), weights=probs2, k=1)[0])])
            return int(self.rng.choices(list(legal), weights=probs, k=1)[0])
        return int(legal[np.argmax(masked_q[legal])])

    def _predict_response(self, obs: dict[str, Any]) -> str:
        obs_array = np.asarray(obs["observation"], dtype=np.float32)
        vec_array = np.asarray(obs["vec"], dtype=np.float32)
        mask_array = np.asarray(obs["action_mask"], dtype=np.float32)
        with torch.no_grad():
            out = self.model(
                {
                    "is_training": False,
                    "return_raw_logits": True,
                    "legal_dueling_mean": self.legal_dueling_mean,
                    "obs": {
                        "observation": torch.from_numpy(np.expand_dims(obs_array, 0)),
                        "vec": torch.from_numpy(np.expand_dims(vec_array, 0)),
                        "action_mask": torch.from_numpy(np.expand_dims(mask_array, 0)),
                    },
                }
            )
        masked_q = out["masked_q"].detach().cpu().numpy().reshape(-1)
        action = self._select_action(masked_q, mask_array)
        if mask_array[action] <= 0:
            self.illegal_action_count += 1
            raise RuntimeError(f"selected illegal action {action}")
        if self.record:
            self.pending_transitions.append(
                {
                    "obs": obs_array.copy(),
                    "vec": vec_array.copy(),
                    "mask": mask_array.astype(np.uint8).copy(),
                    "action": int(action),
                    "seat": int(self.seat_wind if self.seat_wind is not None else -1),
                    "policy_name": self.policy_name,
                }
            )
        assert self.agent is not None
        return str(self.agent.action2response(action))

    def override_last_action(self, response: str) -> None:
        if not self.record or not self.pending_transitions or self.agent is None:
            return
        feature_response = botzone_to_feature_response(response)
        action = int(self.agent.response2action(feature_response))
        last = self.pending_transitions[-1]
        if last["mask"][action] <= 0:
            self.illegal_action_count += 1
            raise RuntimeError(f"guard fallback produced illegal action {action}: {response}")
        last["action"] = action
        last["guarded"] = True
        self.blocked_hu_count += 1

    def finalize_game(
        self,
        *,
        scores: list[int | float],
        display: dict[str, Any],
        game_id: int,
        reward_scale: float,
        reward_clip: float,
    ) -> list[dict[str, Any]]:
        seat = int(self.seat_wind if self.seat_wind is not None else -1)
        terminal_score = float(scores[seat]) if 0 <= seat < len(scores) else 0.0
        reward = max(-reward_clip, min(reward_clip, terminal_score / reward_scale))
        action = str(display.get("action") or "").upper()
        tag = 0
        if action == "HU" and display.get("player") is not None:
            winner = int(display["player"])
            kind, loser = infer_hu_kind(scores, winner)
            if winner == seat:
                tag = 1 if kind == "self_draw" else 2
            elif kind == "discard" and loser == seat:
                tag = 3
            else:
                tag = 4
        elif action == "HUANG":
            tag = 5
        done = len(self.pending_transitions)
        finalized: list[dict[str, Any]] = []
        for index, item in enumerate(self.pending_transitions):
            out = dict(item)
            out["reward"] = float(reward)
            out["steps_to_done"] = int(done - index - 1)
            out["game_id"] = int(game_id)
            out["terminal_score"] = float(terminal_score)
            out["terminal_tag"] = int(tag)
            finalized.append(out)
        self.pending_transitions.clear()
        return finalized


def build_response_log(output: dict[str, Any], policies: list[LoggingFeaturePolicy]) -> dict[str, dict[str, str]]:
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
        if guard_reason != "OK" and hasattr(policies[index], "override_last_action"):
            policies[index].override_last_action(response)
        item = {"response": response, "raw": raw_response, "verdict": "OK"}
        if guard_reason != "OK":
            item["guard"] = guard_reason
        response_log[str(player)] = item
    return response_log


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.buffer: list[dict[str, Any]] = []
        self.shard_index = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add_many(self, rows: list[dict[str, Any]]) -> None:
        self.buffer.extend(rows)
        while len(self.buffer) >= self.shard_size:
            self.flush(self.shard_size)

    def flush(self, count: int | None = None) -> None:
        if not self.buffer:
            return
        if count is None:
            count = len(self.buffer)
        rows = self.buffer[:count]
        del self.buffer[:count]
        path = self.out_dir / f"shard_{self.shard_index:06d}.npz"
        tmp = self.out_dir / f"shard_{self.shard_index:06d}.tmp.npz"
        obs = np.stack([row["obs"] for row in rows]).astype(np.uint8)
        vec = np.stack([row["vec"] for row in rows]).astype(np.float16)
        mask = np.stack([row["mask"] for row in rows]).astype(np.uint8)
        action = np.asarray([row["action"] for row in rows], dtype=np.int64)
        if not np.all(mask[np.arange(len(rows)), action] > 0):
            raise RuntimeError("refusing to write shard with illegal action")
        np.savez_compressed(
            tmp,
            obs=obs,
            vec=vec,
            mask=mask,
            action=action,
            reward=np.asarray([row["reward"] for row in rows], dtype=np.float32),
            steps_to_done=np.asarray([row["steps_to_done"] for row in rows], dtype=np.int32),
            seat=np.asarray([row["seat"] for row in rows], dtype=np.int8),
            game_id=np.asarray([row["game_id"] for row in rows], dtype=np.int64),
            terminal_score=np.asarray([row["terminal_score"] for row in rows], dtype=np.float32),
            terminal_tag=np.asarray([row["terminal_tag"] for row in rows], dtype=np.int8),
        )
        os.replace(tmp, path)
        self.shard_index += 1


def run_game(
    *,
    initdata: dict[str, Any],
    policies: list[LoggingFeaturePolicy],
    trainee: LoggingFeaturePolicy,
    game_id: int,
    judge: Path,
    reward_scale: float,
    reward_clip: float,
    max_turns: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    log: list[dict[str, Any]] = []
    trainee.start_game()
    for turn in range(max_turns):
        output = run_judge(log, initdata, judge)
        if output.get("command") == "finish":
            scores = scores_from_finish(output)
            rows = trainee.finalize_game(
                scores=scores,
                display=output.get("display") or {},
                game_id=game_id,
                reward_scale=reward_scale,
                reward_clip=reward_clip,
            )
            return rows, {"turns": turn, "scores": scores, "display": output.get("display") or {}}
        if output.get("command") != "request":
            raise RuntimeError(f"unexpected judge command: {output.get('command')}")
        response_log = build_response_log(output, policies)
        log.extend([{"output": output}, response_log])
    raise RuntimeError(f"game exceeded max_turns={max_turns}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--baseline-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--raw", default=str(WORKSPACE_ROOT / "data" / "eval" / "botzone_mcr_first64_suit_permuted_384.jsonl"))
    parser.add_argument("--judge", default=str(WORKSPACE_ROOT / "build" / "official_judge" / "mcr_judge.exe"))
    parser.add_argument("--games-per-seat", type=int, default=256)
    parser.add_argument("--initdata-offset", type=int, default=0)
    parser.add_argument("--seats", default="0,1,2,3", help="Comma-separated trainee seats, or 'all'.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reward-scale", type=float, default=64.0)
    parser.add_argument("--reward-clip", type=float, default=4.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--legal-dueling-mean", action="store_true")
    args = parser.parse_args()

    if str(args.seats).lower() == "all":
        trainee_seats = [0, 1, 2, 3]
    else:
        trainee_seats = [int(part) for part in str(args.seats).split(",") if part.strip()]
    if not trainee_seats or any(seat < 0 or seat > 3 for seat in trainee_seats):
        raise ValueError(f"--seats must contain seats in 0..3, got {args.seats!r}")

    initdata_items = load_initdata(
        Path(args.raw),
        limit=int(args.games_per_seat),
        offset=int(args.initdata_offset),
    )
    if not initdata_items:
        raise RuntimeError(f"no initdata loaded from {args.raw}")
    out_dir = Path(args.out)
    writer = ShardWriter(out_dir, int(args.shard_size))
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "raw": str(Path(args.raw).resolve()),
        "judge": str(Path(args.judge).resolve()),
        "games_per_seat_requested": int(args.games_per_seat),
        "games_per_seat_loaded": len(initdata_items),
        "initdata_offset": int(args.initdata_offset),
        "trainee_seats": trainee_seats,
        "reward_scale": float(args.reward_scale),
        "reward_clip": float(args.reward_clip),
        "epsilon": float(args.epsilon),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "legal_dueling_mean": bool(args.legal_dueling_mean),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    total_games = 0
    total_transitions = 0
    start = time.time()
    for trainee_seat in trainee_seats:
        policies: list[LoggingFeaturePolicy] = []
        trainee: LoggingFeaturePolicy | None = None
        for seat in range(4):
            if seat == trainee_seat:
                policy = LoggingFeaturePolicy(
                    args.checkpoint,
                    epsilon=args.epsilon,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    record=True,
                    policy_name="trainee",
                    legal_dueling_mean=args.legal_dueling_mean,
                    seed=1000 + trainee_seat,
                )
                trainee = policy
            else:
                policy = LoggingFeaturePolicy(
                    args.baseline_checkpoint,
                    epsilon=0.0,
                    temperature=args.temperature,
                    top_p=1.0,
                    record=False,
                    policy_name="baseline",
                    legal_dueling_mean=False,
                    seed=2000 + seat,
                )
            policies.append(policy)
        assert trainee is not None
        try:
            for game_index, initdata in enumerate(initdata_items):
                game_id = trainee_seat * 1_000_000_000 + int(args.initdata_offset) + game_index
                rows, result = run_game(
                    initdata=initdata,
                    policies=policies,
                    trainee=trainee,
                    game_id=game_id,
                    judge=Path(args.judge),
                    reward_scale=float(args.reward_scale),
                    reward_clip=float(args.reward_clip),
                    max_turns=int(args.max_turns),
                )
                writer.add_many(rows)
                total_games += 1
                total_transitions += len(rows)
                print(
                    json.dumps(
                        {
                            "event": "collect_progress",
                            "seat": trainee_seat,
                            "game_index": game_index,
                            "total_games": total_games,
                            "total_transitions": total_transitions,
                            "terminal_action": result["display"].get("action"),
                            "elapsed_s": time.time() - start,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        finally:
            pass
    writer.flush()
    summary = {
        **metadata,
        "total_games": total_games,
        "total_transitions": total_transitions,
        "shards": writer.shard_index,
        "elapsed_s": time.time() - start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
