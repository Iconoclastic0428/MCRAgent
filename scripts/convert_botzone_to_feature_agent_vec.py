#!/usr/bin/env python3
"""Convert Botzone JSONL logs into the feature-agent supervised format.

The GoddessLuBoYan/mahjong-agent-2025 baseline trains a 235-way action model
from FeatureAgent observations. This converter keeps that state encoder and
label space, but reads our Botzone-format JSONL records directly and writes
compact shards instead of one small NPZ per match.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACTION_NAMES = ("PASS", "HU", "PLAY", "GANG", "ANGANG", "BUGANG", "PENG", "CHI")


@dataclass
class AgentRuntime:
    agent: Any
    zimo: bool = False
    pending_angang_tile: str | None = None


@dataclass
class ConversionStats:
    records_seen: int = 0
    records_written: int = 0
    records_skipped: int = 0
    examples_seen: int = 0
    examples_kept: int = 0
    single_action_skipped: int = 0
    label_outside_mask: Counter[str] = field(default_factory=Counter)
    action_seen: Counter[str] = field(default_factory=Counter)
    action_kept: Counter[str] = field(default_factory=Counter)
    response_seen: Counter[str] = field(default_factory=Counter)
    unsupported_requests: Counter[str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)
    shard_examples: list[int] = field(default_factory=list)

    def merge_examples(self, other: "ConversionStats") -> None:
        self.examples_seen += other.examples_seen
        self.examples_kept += other.examples_kept
        self.single_action_skipped += other.single_action_skipped
        self.label_outside_mask.update(other.label_outside_mask)
        self.action_seen.update(other.action_seen)
        self.action_kept.update(other.action_kept)
        self.response_seen.update(other.response_seen)
        self.unsupported_requests.update(other.unsupported_requests)
        self.errors.update(other.errors)

    def to_json(self) -> dict[str, Any]:
        return {
            "format": "feature_agent_vec_shards_v1",
            "records_seen": self.records_seen,
            "records_written": self.records_written,
            "records_skipped": self.records_skipped,
            "examples_seen": self.examples_seen,
            "examples_kept": self.examples_kept,
            "single_action_skipped": self.single_action_skipped,
            "label_outside_mask": dict(sorted(self.label_outside_mask.items())),
            "action_seen": dict(sorted(self.action_seen.items())),
            "action_kept": dict(sorted(self.action_kept.items())),
            "response_seen": dict(sorted(self.response_seen.items())),
            "unsupported_requests": dict(sorted(self.unsupported_requests.items())),
            "errors": dict(sorted(self.errors.items())),
            "shard_examples": self.shard_examples,
        }


class ShardWriter:
    def __init__(self, out_dir: Path, samples_per_shard: int) -> None:
        self.out_dir = out_dir
        self.samples_per_shard = int(samples_per_shard)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index = 0
        self.obs: list[np.ndarray] = []
        self.mask: list[np.ndarray] = []
        self.vec: list[np.ndarray] = []
        self.act: list[int] = []
        self.shards: list[dict[str, Any]] = []

    def add(self, obs: dict[str, np.ndarray], action: int) -> None:
        self.obs.append(np.asarray(obs["observation"], dtype=np.int8))
        self.mask.append(np.asarray(obs["action_mask"], dtype=np.int8))
        self.vec.append(np.asarray(obs["vec"], dtype=np.float16))
        self.act.append(int(action))
        if len(self.act) >= self.samples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.act:
            return
        path = self.out_dir / f"feature_agent_vec_shard_{self.shard_index:04d}.npz"
        np.savez(
            path,
            obs=np.stack(self.obs).astype(np.int8),
            mask=np.stack(self.mask).astype(np.int8),
            vec=np.stack(self.vec).astype(np.float16),
            act=np.asarray(self.act, dtype=np.int16),
        )
        self.shards.append({"path": path.name, "examples": len(self.act)})
        self.shard_index += 1
        self.obs.clear()
        self.mask.clear()
        self.vec.clear()
        self.act.clear()

    def write_index(self, stats: ConversionStats, *, source: str) -> Path:
        self.flush()
        index = {
            **stats.to_json(),
            "source": source,
            "sharded": True,
            "compact_storage": True,
            "samples_per_shard": self.samples_per_shard,
            "examples": sum(int(item["examples"]) for item in self.shards),
            "shards": self.shards,
        }
        index_path = self.out_dir / "index.json"
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        return index_path


class SampleBuffer:
    def __init__(self) -> None:
        self.samples: list[tuple[dict[str, np.ndarray], int]] = []

    def add(self, obs: dict[str, np.ndarray], action: int) -> None:
        self.samples.append((obs, int(action)))

    def commit_to(self, writer: ShardWriter) -> None:
        for obs, action in self.samples:
            writer.add(obs, action)


def add_feature_agent_path(feature_agent_dir: Path):
    resolved = str(feature_agent_dir.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from feature import FeatureAgent  # noqa: PLC0415

    return FeatureAgent


def iter_jsonl(path: Path, *, offset: int = 0, limit: int | None = None) -> Iterable[dict[str, Any]]:
    yielded = 0
    with path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src):
            if line_number < offset or not line.strip():
                continue
            yield json.loads(line)
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def response_text(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("raw") or entry.get("response") or "PASS").strip() or "PASS"
    return str(entry or "PASS").strip() or "PASS"


def response_head(response: str) -> str:
    parts = response.strip().split()
    return parts[0].upper() if parts else "PASS"


def request_event_tile(tokens: list[str]) -> str | None:
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2].upper() in {"PLAY", "PENG"}:
        return tokens[-1]
    if len(tokens) >= 5 and tokens[0] == "3" and tokens[2].upper() == "CHI":
        return tokens[-1]
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2].upper() == "BUGANG":
        return tokens[3]
    return None


def feature_label_response(request: str, response: str) -> str | None:
    request_tokens = request.strip().split()
    response_tokens = response.strip().split()
    if not response_tokens:
        return "Pass"
    action = response_tokens[0].upper()
    if action == "PASS":
        return "Pass"
    if action == "HU":
        return "Hu"
    if request_tokens and request_tokens[0] == "2":
        if action == "PLAY" and len(response_tokens) >= 2:
            return f"Play {response_tokens[1]}"
        if action == "GANG" and len(response_tokens) >= 2:
            return f"AnGang {response_tokens[1]}"
        if action == "BUGANG" and len(response_tokens) >= 2:
            return f"BuGang {response_tokens[1]}"
    if len(request_tokens) >= 3 and request_tokens[0] == "3":
        event_tile = request_event_tile(request_tokens)
        if action == "GANG" and event_tile:
            return f"Gang {event_tile}"
        if action == "PENG" and event_tile:
            return f"Peng {event_tile}"
        if action == "CHI" and event_tile and len(response_tokens) >= 2:
            return f"Chi {event_tile} {response_tokens[1]}"
    return None


def action_family(agent: Any, action_id: int) -> str:
    response = agent.action2response(int(action_id))
    head = response.split()[0] if response else "Pass"
    if head == "Pass":
        return "PASS"
    if head == "Hu":
        return "HU"
    if head == "Play":
        return "PLAY"
    if head == "Gang":
        return "GANG"
    if head == "AnGang":
        return "ANGANG"
    if head == "BuGang":
        return "BUGANG"
    if head == "Peng":
        return "PENG"
    if head == "Chi":
        return "CHI"
    return head.upper()


def maybe_add_sample(
    writer: ShardWriter | SampleBuffer,
    stats: ConversionStats,
    agent: Any,
    obs: dict[str, np.ndarray] | None,
    label: str | None,
) -> None:
    if obs is None or label is None:
        return
    action_id = int(agent.response2action(label))
    family = action_family(agent, action_id)
    stats.examples_seen += 1
    stats.action_seen[family] += 1
    mask = np.asarray(obs.get("action_mask"), dtype=np.int8)
    if int(mask.sum()) <= 1:
        stats.single_action_skipped += 1
        return
    if action_id < 0 or action_id >= len(mask) or int(mask[action_id]) <= 0:
        stats.label_outside_mask[family] += 1
        return
    writer.add(obs, action_id)
    stats.examples_kept += 1
    stats.action_kept[family] += 1


def observe_request(runtime: AgentRuntime, seat: int, request: str, stats: ConversionStats) -> dict[str, np.ndarray] | None:
    tokens = request.strip().split()
    if not tokens:
        return None
    agent = runtime.agent
    if tokens[0] == "0" and len(tokens) >= 3:
        agent.request2obs(f"Wind {tokens[2]}")
        return None
    if tokens[0] == "1":
        hand_tiles = [tile for tile in tokens[5:] if not tile.startswith("H")]
        if len(hand_tiles) != len(tokens[5:]):
            stats.unsupported_requests["DEAL_FLOWER_TILES"] += len(tokens[5:]) - len(hand_tiles)
        agent.request2obs(" ".join(["Deal", *hand_tiles[:13]]))
        return None
    if tokens[0] == "2" and len(tokens) >= 2:
        if tokens[1].startswith("H"):
            stats.unsupported_requests["FLOWER_DRAW"] += 1
            return None
        runtime.zimo = False
        return agent.request2obs(f"Draw {tokens[1]}")
    if tokens[0] != "3" or len(tokens) < 3:
        return None

    try:
        actor = int(tokens[1])
    except ValueError:
        return None
    event = tokens[2].upper()
    if event == "DRAW":
        runtime.zimo = True
        agent.request2obs(f"Player {actor} Draw")
        return None
    if event == "BUHUA":
        stats.unsupported_requests["BUHUA"] += 1
        return None
    if event == "GANG":
        if actor == seat and runtime.pending_angang_tile:
            agent.request2obs(f"Player {actor} AnGang {runtime.pending_angang_tile}")
            runtime.pending_angang_tile = None
        elif runtime.zimo:
            agent.request2obs(f"Player {actor} AnGang")
        else:
            agent.request2obs(f"Player {actor} Gang")
        runtime.zimo = False
        return None
    if event == "BUGANG" and len(tokens) >= 4:
        runtime.zimo = False
        return agent.request2obs(f"Player {actor} BuGang {tokens[3]}")
    if event == "PLAY" and len(tokens) >= 4:
        runtime.zimo = False
        return agent.request2obs(f"Player {actor} Play {tokens[3]}")
    if event == "PENG" and len(tokens) >= 4:
        runtime.zimo = False
        agent.request2obs(f"Player {actor} Peng")
        return agent.request2obs(f"Player {actor} Play {tokens[3]}")
    if event == "CHI" and len(tokens) >= 5:
        runtime.zimo = False
        agent.request2obs(f"Player {actor} Chi {tokens[3]}")
        return agent.request2obs(f"Player {actor} Play {tokens[4]}")
    stats.unsupported_requests[event] += 1
    return None


def add_post_claim_discard_sample(
    writer: ShardWriter,
    stats: ConversionStats,
    runtime: AgentRuntime,
    seat: int,
    response: str,
) -> None:
    parts = response.strip().split()
    if not parts:
        return
    action = parts[0].upper()
    agent = runtime.agent
    if action == "PENG" and len(parts) >= 2:
        post_obs = agent.request2obs(f"Player {seat} Peng")
        maybe_add_sample(writer, stats, agent, post_obs, f"Play {parts[1]}")
        agent.request2obs(f"Player {seat} UnPeng")
    elif action == "CHI" and len(parts) >= 3:
        middle = parts[1]
        discard = parts[2]
        post_obs = agent.request2obs(f"Player {seat} Chi {middle}")
        maybe_add_sample(writer, stats, agent, post_obs, f"Play {discard}")
        agent.request2obs(f"Player {seat} UnChi {middle}")


def convert_record(
    record: dict[str, Any],
    *,
    FeatureAgent,
    writer: ShardWriter,
    stats: ConversionStats,
) -> int:
    runtimes = [AgentRuntime(FeatureAgent(seat)) for seat in range(4)]
    before = stats.examples_kept
    logs = record.get("logs") or []
    for index in range(0, len(logs) - 1, 2):
        output = (logs[index] or {}).get("output") or {}
        content = output.get("content") or {}
        responses = logs[index + 1] if isinstance(logs[index + 1], dict) else {}
        for seat in range(4):
            request = str(content.get(str(seat), ""))
            response = response_text(responses.get(str(seat), {}))
            stats.response_seen[response_head(response)] += 1
            obs = observe_request(runtimes[seat], seat, request, stats)
            label = feature_label_response(request, response)
            maybe_add_sample(writer, stats, runtimes[seat].agent, obs, label)
            if request.startswith("2 ") and response_head(response) == "GANG":
                parts = response.split()
                if len(parts) >= 2:
                    runtimes[seat].pending_angang_tile = parts[1]
            add_post_claim_discard_sample(writer, stats, runtimes[seat], seat, response)
    return stats.examples_kept - before


def convert(args: argparse.Namespace) -> dict[str, Any]:
    FeatureAgent = add_feature_agent_path(Path(args.feature_agent_dir))
    out_dir = Path(args.out_dir)
    writer = ShardWriter(out_dir, samples_per_shard=args.samples_per_shard)
    stats = ConversionStats()
    for record in iter_jsonl(Path(args.raw), offset=args.offset, limit=args.limit):
        stats.records_seen += 1
        record_stats = ConversionStats()
        record_buffer = SampleBuffer()
        try:
            kept = convert_record(record, FeatureAgent=FeatureAgent, writer=record_buffer, stats=record_stats)
        except Exception as exc:  # Keep a bad record from poisoning a whole shard.
            stats.records_skipped += 1
            stats.errors[type(exc).__name__] += 1
            if args.strict:
                raise
            continue
        record_buffer.commit_to(writer)
        stats.merge_examples(record_stats)
        if kept > 0:
            stats.records_written += 1
        if stats.records_seen == 1 or stats.records_seen % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "feature_agent_vec_progress",
                        "records_seen": stats.records_seen,
                        "examples_kept": stats.examples_kept,
                        "action_kept": dict(sorted(stats.action_kept.items())),
                        "label_outside_mask": dict(sorted(stats.label_outside_mask.items())),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    index_path = writer.write_index(stats, source=str(args.raw))
    summary = json.loads(index_path.read_text(encoding="utf-8"))
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--feature-agent-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-shard", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
