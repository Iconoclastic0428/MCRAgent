"""Tensorize Botzone MCR logs into the Tjong paper feature shapes.

The paper states the feature sizes but does not publish the original
preprocessing code. This converter keeps the replay logic explicit and records
the few assumptions in the emitted summary.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from .actions import ACTION_NAMES, ACTION_TO_INDEX, flatten_claim
from .encoding import build_game_features, stack_hidden_tile_features, stack_visible_tile_features
from .tiles import SUIT_TO_ID, TILE_NAMES, is_suited, suited_rank, tile_id


PASS_ACTION = ACTION_TO_INDEX["PASS"]
DISCARD_ACTION = ACTION_TO_INDEX["DISCARD"]
HU_ACTION = ACTION_TO_INDEX["HU"]
CLAIM_ACTIONS = {
    "CHI": "CHOW",
    "PENG": "PONG",
    "GANG": "MINGKONG",
    "BUGANG": "BUKONG",
}
TRAINABLE_REQUEST_HEADS = {"2", "3"}
TENSOR_ENCODING_VERSION = "tjong_cit2_12298_v3_hidden_concealed_kong"
SHARD_INDEX_FORMAT = "tjong_tensor_shards_v1"
HIDDEN_TILE_ROW_NAMES = (
    "opponent_hand_1",
    "opponent_hand_2",
    "opponent_hand_3",
    "concealed_kongs",
    "remaining_wall",
)


@dataclass
class MemoryFrame:
    visible_tiles: torch.Tensor
    game_features: torch.Tensor
    previous_action: int
    reward: float = 0.0


@dataclass
class EncodedExample:
    visible_tiles: torch.Tensor
    game_features: torch.Tensor
    rewards: torch.Tensor
    previous_actions: torch.Tensor
    sub_visible_tiles: torch.Tensor
    sub_game_features: torch.Tensor
    sub_rewards: torch.Tensor
    sub_previous_actions: torch.Tensor
    hidden_tiles: torch.Tensor
    action_label: int
    claim_label: int
    discard_label: int
    value_target: float
    match_id: str
    player: int
    turn_index: int
    request: str
    response: str
    kind: str


@dataclass
class ReplayState:
    """Public replay state plus global tiles available in offline logs."""

    prevailing_wind: int = 0
    hands: list[Counter[str]] = field(default_factory=lambda: [Counter() for _ in range(4)])
    discards: list[Counter[str]] = field(default_factory=lambda: [Counter() for _ in range(4)])
    pengs: list[Counter[str]] = field(default_factory=lambda: [Counter() for _ in range(4)])
    chows: list[Counter[str]] = field(default_factory=lambda: [Counter() for _ in range(4)])
    kongs: list[Counter[str]] = field(default_factory=lambda: [Counter() for _ in range(4)])
    concealed_kongs: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    concealed_kong_tiles: list[Counter[str]] = field(default_factory=lambda: [Counter() for _ in range(4)])
    tile_counts: list[int] = field(default_factory=lambda: [21, 21, 21, 21])
    wall_remaining: Counter[str] = field(default_factory=Counter)
    claimable_discard: tuple[int, str] | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ReplayState":
        state = cls()
        initdata = record.get("initdata") or {}
        if "quan" in initdata:
            state.prevailing_wind = int(initdata["quan"])
        walltiles = str(initdata.get("walltiles") or "").split()
        if walltiles:
            state.wall_remaining = Counter(tile for tile in walltiles if tile in TILE_NAMES)
        else:
            state.wall_remaining = Counter({tile: 4 for tile in TILE_NAMES})
        return state

    def apply_display(self, display: dict[str, Any]) -> None:
        action = str(display.get("action") or "").upper()
        if "quan" in display:
            self.prevailing_wind = int(display["quan"])
        if "tileCnt" in display and isinstance(display["tileCnt"], list):
            self.tile_counts = [int(value) for value in display["tileCnt"][:4]]

        if action == "INIT":
            return
        if action == "DEAL":
            hands = display.get("hand") or []
            for player, tiles in enumerate(hands[:4]):
                self.hands[player] = Counter(tile for tile in tiles if tile in TILE_NAMES)
                self._remove_from_wall(self.hands[player].elements())
            self.claimable_discard = None
            return
        if action == "DRAW":
            self.claimable_discard = None
            player = _display_player(display)
            tile = display.get("tile")
            if player is not None and tile in TILE_NAMES:
                self.hands[player][tile] += 1
                self._remove_from_wall([tile])
            return
        if action == "PLAY":
            player = _display_player(display)
            tile = display.get("tile")
            if player is not None and tile in TILE_NAMES:
                self._remove_from_hand(player, tile, 1)
                self.discards[player][tile] += 1
                self.claimable_discard = (player, tile)
            return
        if action == "CHI":
            self._apply_chow_display(display)
            return
        if action == "PENG":
            self._apply_pong_display(display)
            return
        if action == "GANG":
            self._apply_kong_display(display)
            return
        if action == "BUGANG":
            player = _display_player(display)
            tile = display.get("tile")
            self.claimable_discard = None
            if player is not None and tile in TILE_NAMES:
                self._remove_from_hand(player, tile, 1)
                self.pengs[player][tile] = max(0, self.pengs[player][tile] - 1)
                self.kongs[player][tile] += 1
            return
        if action in {"HU", "HUANG"}:
            self.claimable_discard = None

    def apply_deal_requests(self, requests: dict[str, Any]) -> None:
        for player_text, request in requests.items():
            try:
                player = int(player_text)
            except (TypeError, ValueError):
                continue
            if player < 0 or player >= 4:
                continue
            tokens = str(request).strip().split()
            if len(tokens) < 6 or tokens[0] != "1":
                continue
            tiles = [tile for tile in tokens[5:] if tile in TILE_NAMES]
            self.hands[player] = Counter(tiles)
            self._remove_from_wall(self.hands[player].elements())
        self.claimable_discard = None

    def after_claim(self, player: int, request: str, response: str) -> "ReplayState | None":
        """Return a cloned state after the claim but before its forced discard."""

        parts = response.strip().split()
        request_tokens = request.strip().split()
        if not parts or not request_tokens or request_tokens[0] != "3":
            return None
        head = parts[0].upper()
        event_tile = request_event_tile(request_tokens)
        if event_tile not in TILE_NAMES:
            return None

        clone = copy.deepcopy(self)
        if head == "PENG" and len(parts) >= 2:
            clone._consume_claimable_discard(event_tile)
            clone._remove_from_hand(player, event_tile, 2)
            clone.pengs[player][event_tile] += 1
            clone.claimable_discard = None
            return clone
        if head == "CHI" and len(parts) >= 3:
            middle = parts[1]
            if not _valid_chow_pair(middle, event_tile):
                return None
            clone._consume_claimable_discard(event_tile)
            for tile, count in chow_needed_from_hand(middle, event_tile).items():
                clone._remove_from_hand(player, tile, count)
            for tile in chow_sequence(middle):
                clone.chows[player][tile] += 1
            clone.claimable_discard = None
            return clone
        return None

    def encode(self, player: int, action_mask: Iterable[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows: dict[str, list[int] | torch.Tensor] = {}
        rows["hand_self"] = _counter_to_ids(self.hands[player])
        live_counts = self.live_counts_for(player)
        rows["available_tile_mask"] = (live_counts > 0).float()

        for rel in range(4):
            absolute = (player + rel) % 4
            rows[f"discard_p{rel}"] = _counter_to_ids(self.discards[absolute])
            rows[f"peng_p{rel}"] = _counter_to_ids(_meld_tiles(self.pengs[absolute], 3))
            rows[f"chow_p{rel}"] = _counter_to_ids(self.chows[absolute])
            rows[f"kong_p{rel}"] = _counter_to_ids(_meld_tiles(self.kongs[absolute], 4))
            rows[f"remaining_tiles_p{rel}"] = live_counts

        visible = stack_visible_tile_features(rows)
        opponents = [(player + rel) % 4 for rel in range(1, 4)]
        game = build_game_features(
            prevailing_wind=self.prevailing_wind,
            seat_wind=player,
            opponent_concealed_kongs=[float(self.concealed_kongs[other]) for other in opponents],
            remaining_tile_counts=[float(self.tile_counts[(player + rel) % 4]) for rel in range(4)],
            hand_tile_counts=[float(sum(self.hands[(player + rel) % 4].values())) for rel in range(4)],
            action_mask=list(action_mask),
        )
        hidden = stack_hidden_tile_features(
            [
                _counter_to_tensor(self.hands[opponents[0]]),
                _counter_to_tensor(self.hands[opponents[1]]),
                _counter_to_tensor(self.hands[opponents[2]]),
                _counter_to_tensor(sum((self.concealed_kong_tiles[index] for index in range(4)), Counter())),
                _counter_to_tensor(self.wall_remaining),
            ]
        )
        return visible, game, hidden

    def live_counts_for(self, player: int) -> torch.Tensor:
        visible = Counter()
        visible.update(self.hands[player])
        for counter in self.discards:
            visible.update(counter)
        for counter in self.pengs:
            visible.update(_meld_tiles(counter, 3))
        for counter in self.chows:
            visible.update(counter)
        for counter in self.kongs:
            visible.update(_meld_tiles(counter, 4))
        return torch.clamp(torch.full((len(TILE_NAMES),), 4.0) - _counter_to_tensor(visible), min=0.0)

    def _apply_chow_display(self, display: dict[str, Any]) -> None:
        player = _display_player(display)
        middle = display.get("tileCHI")
        discard = display.get("tile")
        event_tile = self.claimable_discard[1] if self.claimable_discard else None
        self.claimable_discard = None
        if player is None or middle not in TILE_NAMES or event_tile not in TILE_NAMES:
            return
        self._consume_claimable_discard(event_tile)
        for tile, count in chow_needed_from_hand(middle, event_tile).items():
            self._remove_from_hand(player, tile, count)
        for tile in chow_sequence(middle):
            self.chows[player][tile] += 1
        if discard in TILE_NAMES:
            self._remove_from_hand(player, discard, 1)
            self.discards[player][discard] += 1
            self.claimable_discard = (player, discard)

    def _apply_pong_display(self, display: dict[str, Any]) -> None:
        player = _display_player(display)
        discard = display.get("tile")
        event_tile = self.claimable_discard[1] if self.claimable_discard else None
        self.claimable_discard = None
        if player is None or event_tile not in TILE_NAMES:
            return
        self._consume_claimable_discard(event_tile)
        self._remove_from_hand(player, event_tile, 2)
        self.pengs[player][event_tile] += 1
        if discard in TILE_NAMES:
            self._remove_from_hand(player, discard, 1)
            self.discards[player][discard] += 1
            self.claimable_discard = (player, discard)

    def _apply_kong_display(self, display: dict[str, Any]) -> None:
        player = _display_player(display)
        tile = display.get("tile")
        if player is None or tile not in TILE_NAMES:
            self.claimable_discard = None
            return
        if self.claimable_discard and self.claimable_discard[1] == tile:
            self._consume_claimable_discard(tile)
            self._remove_from_hand(player, tile, 3)
        else:
            self._remove_from_hand(player, tile, 4)
            self.concealed_kongs[player] += 1
            self.concealed_kong_tiles[player][tile] += 4
        self.kongs[player][tile] += 1
        self.claimable_discard = None

    def _consume_claimable_discard(self, tile: str) -> None:
        if self.claimable_discard and self.claimable_discard[1] == tile:
            actor, _ = self.claimable_discard
            if self.discards[actor][tile] > 0:
                self.discards[actor][tile] -= 1
                if self.discards[actor][tile] <= 0:
                    del self.discards[actor][tile]

    def _remove_from_wall(self, tiles: Iterable[str]) -> None:
        for tile in tiles:
            if self.wall_remaining[tile] > 0:
                self.wall_remaining[tile] -= 1
                if self.wall_remaining[tile] <= 0:
                    del self.wall_remaining[tile]

    def _remove_from_hand(self, player: int, tile: str, count: int) -> None:
        if tile not in TILE_NAMES:
            return
        self.hands[player][tile] -= int(count)
        if self.hands[player][tile] <= 0:
            del self.hands[player][tile]


def tensorize_file(
    in_path: Path,
    out_path: Path | None,
    *,
    summary_out: Path | None = None,
    corpus_validation: Path | None = None,
    max_matches: int | None = None,
    memory_len: int = 4,
    include_single_action: bool = False,
    single_action_discard_stride: int = 1,
    streaming: bool = False,
    compact_metadata: bool = False,
    compact_storage: bool = False,
    shard_dir: Path | None = None,
    shard_index_out: Path | None = None,
    shard_max_examples: int = 50000,
    progress_every: int = 0,
) -> dict[str, Any]:
    if shard_dir is not None:
        return tensorize_file_sharded(
            in_path,
            shard_dir,
            index_out=shard_index_out,
            summary_out=summary_out,
            corpus_validation=corpus_validation,
            max_matches=max_matches,
            memory_len=memory_len,
            include_single_action=include_single_action,
            single_action_discard_stride=single_action_discard_stride,
            compact_metadata=compact_metadata,
            compact_storage=compact_storage,
            shard_max_examples=shard_max_examples,
            progress_every=progress_every,
        )
    if out_path is None:
        raise ValueError("out_path is required unless shard_dir is provided")
    if streaming:
        return tensorize_file_streaming(
            in_path,
            out_path,
            summary_out=summary_out,
            corpus_validation=corpus_validation,
            max_matches=max_matches,
            memory_len=memory_len,
            include_single_action=include_single_action,
            single_action_discard_stride=single_action_discard_stride,
            compact_metadata=compact_metadata,
            progress_every=progress_every,
        )

    examples: list[EncodedExample] = []
    stats: Counter[str] = Counter()

    with _open_text(in_path) as src:
        for line_no, line in enumerate(src, start=1):
            if max_matches is not None and stats["matches"] >= max_matches:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_examples, record_stats = tensorize_record(
                    record,
                    memory_len=memory_len,
                    include_single_action=include_single_action,
                    single_action_discard_stride=single_action_discard_stride,
                )
            except Exception as exc:
                stats[f"record_error:{type(exc).__name__}"] += 1
                stats["record_error_lines"] += 1
                continue
            stats["matches"] += 1
            stats.update(record_stats)
            examples.extend(record_examples)
            if progress_every > 0 and stats["matches"] % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "stage": "tensorize_collect",
                            "matches": int(stats["matches"]),
                            "examples": int(len(examples)),
                            "line": int(line_no),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _stack_examples(examples, stats)
    encoding_schema = tensor_encoding_schema()
    payload["encoding_schema"] = encoding_schema
    corpus_validation_payload = None
    if corpus_validation is not None:
        corpus_validation_payload = json.loads(corpus_validation.read_text(encoding="utf-8"))
        payload["corpus_validation"] = corpus_validation_payload
    _finalize_metadata(payload)
    torch.save(payload, out_path)

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "corpus_validation": corpus_validation_payload,
        "examples": len(examples),
        "stats": dict(sorted(stats.items())),
        "encoding_schema": encoding_schema,
        "memory_len": memory_len,
        "include_single_action": include_single_action,
        "single_action_discard_stride": int(single_action_discard_stride),
        "assumptions": [
            "Chow claim indices use suit * 21 + (middle_rank - 2) * 3 + (offer_position - 1), matching the public Lawlorentz/Botzone action layout.",
            "The policy-visible remaining-tile rows are live tile counts from the acting player's visible perspective.",
            "The value hidden matrix stores three opponent hands, concealed Kong tile counts, and remaining wall tiles to match the paper's 5 x 34 hidden/global feature shape.",
            "Tile-decision heads receive sub-state tensors whose past frames match the action-decision memory and whose current frame is conditioned on the chosen action.",
            "Claim responses with forced discards produce one claim-family example plus one post-claim discard example; the post-claim discard sub-state uses the replayed state after the meld.",
        ],
    }
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def tensorize_file_streaming(
    in_path: Path,
    out_path: Path,
    *,
    summary_out: Path | None = None,
    corpus_validation: Path | None = None,
    max_matches: int | None = None,
    memory_len: int = 4,
    include_single_action: bool = False,
    single_action_discard_stride: int = 1,
    compact_metadata: bool = False,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Tensorize without retaining every EncodedExample object before saving.

    The paper-scale corpus is large enough that the legacy list-then-stack path
    can OOM before torch.save. This two-pass writer keeps only the final tensors
    plus one record's examples in memory.
    """

    stats = _scan_tensorize_stats(
        in_path,
        max_matches=max_matches,
        memory_len=memory_len,
        include_single_action=include_single_action,
        single_action_discard_stride=single_action_discard_stride,
        progress_every=progress_every,
    )
    total_examples = int(stats.get("examples", 0))
    payload = _allocate_payload(total_examples, stats, compact_metadata=compact_metadata)
    second_stats = Counter()
    written = 0

    with _open_text(in_path) as src:
        for line_no, line in enumerate(src, start=1):
            if max_matches is not None and second_stats["matches"] >= max_matches:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_examples, record_stats = tensorize_record(
                    record,
                    memory_len=memory_len,
                    include_single_action=include_single_action,
                    single_action_discard_stride=single_action_discard_stride,
                )
            except Exception as exc:
                second_stats[f"record_error:{type(exc).__name__}"] += 1
                second_stats["record_error_lines"] += 1
                continue
            second_stats["matches"] += 1
            second_stats.update(record_stats)
            for example in record_examples:
                _write_example(payload, written, example)
                written += 1
            if progress_every > 0 and second_stats["matches"] % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "stage": "tensorize_write",
                            "matches": int(second_stats["matches"]),
                            "examples_written": int(written),
                            "line": int(line_no),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if written != total_examples:
        raise RuntimeError(f"streaming tensorized {written} examples, expected {total_examples}")
    if int(second_stats.get("examples", 0)) != total_examples:
        raise RuntimeError(
            f"streaming stats mismatch: second pass examples={second_stats.get('examples', 0)}, "
            f"first pass examples={total_examples}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding_schema = tensor_encoding_schema()
    payload["encoding_schema"] = encoding_schema
    corpus_validation_payload = None
    if corpus_validation is not None:
        corpus_validation_payload = json.loads(corpus_validation.read_text(encoding="utf-8"))
        payload["corpus_validation"] = corpus_validation_payload
    _finalize_metadata(payload)
    torch.save(payload, out_path)

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "corpus_validation": corpus_validation_payload,
        "examples": total_examples,
        "stats": dict(sorted(stats.items())),
        "encoding_schema": encoding_schema,
        "memory_len": memory_len,
        "include_single_action": include_single_action,
        "single_action_discard_stride": int(single_action_discard_stride),
        "streaming": True,
        "compact_metadata": bool(compact_metadata),
        "assumptions": [
            "Chow claim indices use suit * 21 + (middle_rank - 2) * 3 + (offer_position - 1), matching the public Lawlorentz/Botzone action layout.",
            "The policy-visible remaining-tile rows are live tile counts from the acting player's visible perspective.",
            "The value hidden matrix stores three opponent hands, concealed Kong tile counts, and remaining wall tiles to match the paper's 5 x 34 hidden/global feature shape.",
            "Tile-decision heads receive sub-state tensors whose past frames match the action-decision memory and whose current frame is conditioned on the chosen action.",
            "Claim responses with forced discards produce one claim-family example plus one post-claim discard example; the post-claim discard sub-state uses the replayed state after the meld.",
        ],
    }
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def tensorize_file_sharded(
    in_path: Path,
    shard_dir: Path,
    *,
    index_out: Path | None = None,
    summary_out: Path | None = None,
    corpus_validation: Path | None = None,
    max_matches: int | None = None,
    memory_len: int = 4,
    include_single_action: bool = False,
    single_action_discard_stride: int = 1,
    compact_metadata: bool = False,
    compact_storage: bool = False,
    shard_max_examples: int = 50000,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Tensorize into immutable supervised shards plus a JSON index.

    Keeping every single-action discard is required for the paper-style
    supervised imitation objective, but a single monolithic tensor can exhaust
    both RAM and PVC space. This writer flushes bounded shards and records a
    small index manifest that the supervised trainer can stream shard by shard.
    """

    if shard_max_examples <= 0:
        raise ValueError("shard_max_examples must be positive")
    index_out = index_out or (shard_dir / "index.json")
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)
    if index_out.exists():
        index_out.unlink()

    stats: Counter[str] = Counter()
    shard_stats: Counter[str] = Counter()
    buffer: list[EncodedExample] = []
    shards: list[dict[str, Any]] = []
    encoding_schema = tensor_encoding_schema()
    corpus_validation_payload = None
    if corpus_validation is not None:
        corpus_validation_payload = json.loads(corpus_validation.read_text(encoding="utf-8"))

    def flush_shard() -> None:
        nonlocal buffer, shard_stats
        if not buffer:
            return
        shard_index = len(shards)
        shard_path = shard_dir / f"shard_{shard_index:05d}.pt"
        payload = _payload_from_examples(buffer, shard_stats, compact_metadata=compact_metadata)
        payload["encoding_schema"] = encoding_schema
        payload["format"] = "tjong_tensor_shard_v1"
        payload["shard_index"] = shard_index
        payload["examples"] = len(buffer)
        if compact_storage:
            _compact_supervised_payload(payload)
        _finalize_metadata(payload)
        torch.save(payload, shard_path)
        shards.append(
            {
                "path": shard_path.name,
                "examples": len(buffer),
                "sha256": _sha256_file(shard_path),
                "stats": dict(sorted(shard_stats.items())),
            }
        )
        print(
            json.dumps(
                {
                    "stage": "tensorize_shard_write",
                    "shard": shard_index,
                    "examples": len(buffer),
                    "path": str(shard_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        buffer = []
        shard_stats = Counter()

    with _open_text(in_path) as src:
        for line_no, line in enumerate(src, start=1):
            if max_matches is not None and stats["matches"] >= max_matches:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_examples, record_stats = tensorize_record(
                    record,
                    memory_len=memory_len,
                    include_single_action=include_single_action,
                    single_action_discard_stride=single_action_discard_stride,
                )
            except Exception as exc:
                stats[f"record_error:{type(exc).__name__}"] += 1
                stats["record_error_lines"] += 1
                continue
            if buffer and len(buffer) + len(record_examples) > shard_max_examples:
                flush_shard()
            stats["matches"] += 1
            stats.update(record_stats)
            shard_stats["matches"] += 1
            shard_stats.update(record_stats)
            buffer.extend(record_examples)
            if progress_every > 0 and stats["matches"] % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "stage": "tensorize_sharded_collect",
                            "matches": int(stats["matches"]),
                            "examples": int(stats.get("examples", 0)),
                            "line": int(line_no),
                            "shards": len(shards),
                            "buffer_examples": len(buffer),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        flush_shard()

    summary = {
        "format": SHARD_INDEX_FORMAT,
        "input": str(in_path),
        "output": str(index_out),
        "shard_dir": str(shard_dir),
        "corpus_validation": corpus_validation_payload,
        "examples": int(stats.get("examples", 0)),
        "stats": dict(sorted(stats.items())),
        "encoding_schema": encoding_schema,
        "memory_len": memory_len,
        "include_single_action": include_single_action,
        "single_action_discard_stride": int(single_action_discard_stride),
        "sharded": True,
        "compact_metadata": bool(compact_metadata),
        "compact_storage": bool(compact_storage),
        "shard_max_examples": int(shard_max_examples),
        "shards": shards,
        "assumptions": [
            "Chow claim indices use suit * 21 + (middle_rank - 2) * 3 + (offer_position - 1), matching the public Lawlorentz/Botzone action layout.",
            "The policy-visible remaining-tile rows are live tile counts from the acting player's visible perspective.",
            "The value hidden matrix stores three opponent hands, concealed Kong tile counts, and remaining wall tiles to match the paper's 5 x 34 hidden/global feature shape.",
            "Tile-decision heads receive sub-state tensors whose past frames match the action-decision memory and whose current frame is conditioned on the chosen action.",
            "Claim responses with forced discards produce one claim-family example plus one post-claim discard example; the post-claim discard sub-state uses the replayed state after the meld.",
            "Compact supervised shards store count/mask features and labels as uint8; rewards and value targets are reconstructed as zero tensors by supervised CE training.",
        ],
    }
    index_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _scan_tensorize_stats(
    in_path: Path,
    *,
    max_matches: int | None,
    memory_len: int,
    include_single_action: bool,
    single_action_discard_stride: int,
    progress_every: int,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    with _open_text(in_path) as src:
        for line_no, line in enumerate(src, start=1):
            if max_matches is not None and stats["matches"] >= max_matches:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_examples, record_stats = tensorize_record(
                    record,
                    memory_len=memory_len,
                    include_single_action=include_single_action,
                    single_action_discard_stride=single_action_discard_stride,
                )
            except Exception as exc:
                stats[f"record_error:{type(exc).__name__}"] += 1
                stats["record_error_lines"] += 1
                continue
            stats["matches"] += 1
            stats.update(record_stats)
            if len(record_examples) != int(record_stats.get("examples", len(record_examples))):
                raise RuntimeError(
                    f"record example count mismatch at line {line_no}: "
                    f"len={len(record_examples)} stats={record_stats.get('examples')}"
                )
            if progress_every > 0 and stats["matches"] % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "stage": "tensorize_scan",
                            "matches": int(stats["matches"]),
                            "examples": int(stats.get("examples", 0)),
                            "line": int(line_no),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return stats


def tensorize_record(
    record: dict[str, Any],
    *,
    memory_len: int = 4,
    include_single_action: bool = False,
    single_action_discard_stride: int = 1,
) -> tuple[list[EncodedExample], Counter[str]]:
    state = ReplayState.from_record(record)
    histories: list[deque[MemoryFrame]] = [deque(maxlen=memory_len - 1) for _ in range(4)]
    last_action = [PASS_ACTION] * 4
    stats: Counter[str] = Counter()
    examples: list[EncodedExample] = []
    logs = record.get("logs") or []
    scores = record.get("scores") or {}
    match_id = str(record.get("match_id") or "")

    for turn in range(0, len(logs) - 1, 2):
        request_log = logs[turn]
        response_log = logs[turn + 1]
        if not isinstance(request_log, dict) or not isinstance(response_log, dict):
            continue
        output = request_log.get("output") or {}
        requests = output.get("content") or {}
        display = output.get("display") or {}
        if not isinstance(requests, dict) or not isinstance(display, dict):
            continue
        state.apply_display(display)
        if str(display.get("action") or "").upper() == "DEAL" and not display.get("hand"):
            state.apply_deal_requests(requests)

        for player_text, request in requests.items():
            try:
                player = int(player_text)
            except (TypeError, ValueError):
                stats["bad_player"] += 1
                continue
            if player < 0 or player >= 4:
                stats["bad_player"] += 1
                continue
            response = actual_response(response_log, player_text)
            if response is None:
                stats["missing_response"] += 1
                continue
            request = str(request)
            request_tokens = request.strip().split()
            if not request_tokens or request_tokens[0] not in TRAINABLE_REQUEST_HEADS:
                stats["nontrainable_request"] += 1
                continue
            label = response_to_labels(request, response)
            if label is None:
                stats[f"unmapped:{response.split()[0].upper() if response.split() else 'EMPTY'}"] += 1
                continue
            action_mask = action_type_mask(player, request_tokens, state, display)
            if not include_single_action and sum(1 for value in action_mask if value > 0.0) <= 1:
                if label[0] == DISCARD_ACTION and single_action_discard_stride > 0:
                    stats["single_action_discard_seen"] += 1
                    if (stats["single_action_discard_seen"] - 1) % int(single_action_discard_stride) == 0:
                        stats["single_action_discard_kept"] += 1
                    else:
                        stats["single_action_discard_sampled_out"] += 1
                        continue
                else:
                    stats["single_action_mask"] += 1
                    continue
            if action_mask[label[0]] <= 0.0:
                stats[f"label_outside_mask:{ACTION_NAMES[label[0]]}"] += 1
                continue

            example = make_example(
                state=state,
                histories=histories,
                player=player,
                request=request,
                response=response,
                label=label,
                action_mask=action_mask,
                match_id=match_id,
                turn_index=turn // 2,
                value_target=float(scores.get(str(player), 0.0)),
                previous_action=last_action[player],
                kind="primary",
                memory_len=memory_len,
            )
            examples.append(example)
            _append_history(histories[player], example, previous_action=label[0])
            last_action[player] = label[0]
            stats["examples"] += 1
            stats[f"action:{ACTION_NAMES[label[0]]}"] += 1

            post_claim = maybe_make_claim_discard_example(
                state=state,
                histories=histories,
                player=player,
                request=request,
                response=response,
                match_id=match_id,
                turn_index=turn // 2,
                value_target=float(scores.get(str(player), 0.0)),
                claim_action=label[0],
                memory_len=memory_len,
            )
            if post_claim is not None:
                examples.append(post_claim)
                _append_history(histories[player], post_claim, previous_action=DISCARD_ACTION)
                last_action[player] = DISCARD_ACTION
                stats["examples"] += 1
                stats["claim_discard_examples"] += 1
                stats["action:DISCARD"] += 1

    return examples, stats


def make_example(
    *,
    state: ReplayState,
    histories: list[deque[MemoryFrame]],
    player: int,
    request: str,
    response: str,
    label: tuple[int, int, int],
    action_mask: list[float],
    match_id: str,
    turn_index: int,
    value_target: float,
    previous_action: int,
    kind: str,
    memory_len: int,
    sub_state: ReplayState | None = None,
    sub_action_mask: list[float] | None = None,
    sub_previous_action: int | None = None,
) -> EncodedExample:
    visible, game, hidden = state.encode(player, action_mask)
    tile_state = state if sub_state is None else sub_state
    tile_action_mask = selected_action_mask(label[0]) if sub_action_mask is None else sub_action_mask
    sub_visible, sub_game, _ = tile_state.encode(player, tile_action_mask)
    frames = list(histories[player])[-(memory_len - 1) :] + [
        MemoryFrame(visible, game, previous_action=previous_action, reward=0.0)
    ]
    sub_frames = list(histories[player])[-(memory_len - 1) :] + [
        MemoryFrame(
            sub_visible,
            sub_game,
            previous_action=label[0] if sub_previous_action is None else sub_previous_action,
            reward=0.0,
        )
    ]
    while len(frames) < memory_len:
        frames.insert(0, zero_memory_frame(previous_action=PASS_ACTION))
    while len(sub_frames) < memory_len:
        sub_frames.insert(0, zero_memory_frame(previous_action=PASS_ACTION))
    return EncodedExample(
        visible_tiles=torch.stack([frame.visible_tiles for frame in frames], dim=0),
        game_features=torch.stack([frame.game_features for frame in frames], dim=0),
        rewards=torch.tensor([frame.reward for frame in frames], dtype=torch.float32),
        previous_actions=torch.tensor([frame.previous_action for frame in frames], dtype=torch.long),
        sub_visible_tiles=torch.stack([frame.visible_tiles for frame in sub_frames], dim=0),
        sub_game_features=torch.stack([frame.game_features for frame in sub_frames], dim=0),
        sub_rewards=torch.tensor([frame.reward for frame in sub_frames], dtype=torch.float32),
        sub_previous_actions=torch.tensor([frame.previous_action for frame in sub_frames], dtype=torch.long),
        hidden_tiles=hidden,
        action_label=int(label[0]),
        claim_label=int(label[1]),
        discard_label=int(label[2]),
        value_target=float(value_target),
        match_id=match_id,
        player=int(player),
        turn_index=int(turn_index),
        request=request,
        response=response,
        kind=kind,
    )


def maybe_make_claim_discard_example(
    *,
    state: ReplayState,
    histories: list[deque[MemoryFrame]],
    player: int,
    request: str,
    response: str,
    match_id: str,
    turn_index: int,
    value_target: float,
    claim_action: int,
    memory_len: int,
) -> EncodedExample | None:
    discard = claim_response_discard(response)
    if discard not in TILE_NAMES:
        return None
    post_claim_state = state.after_claim(player, request, response)
    if post_claim_state is None:
        return None
    mask = [0.0] * len(ACTION_NAMES)
    mask[DISCARD_ACTION] = 1.0
    return make_example(
        state=post_claim_state,
        histories=histories,
        player=player,
        request=f"{request} -> {response}",
        response=f"PLAY {discard}",
        label=(DISCARD_ACTION, 0, tile_id(discard)),
        action_mask=mask,
        match_id=match_id,
        turn_index=turn_index,
        value_target=value_target,
        previous_action=claim_action,
        kind="post_claim_discard",
        memory_len=memory_len,
        sub_state=post_claim_state,
        sub_action_mask=mask,
        sub_previous_action=DISCARD_ACTION,
    )


def response_to_labels(request: str, response: str) -> tuple[int, int, int] | None:
    parts = response.strip().split()
    if not parts:
        return None
    head = parts[0].upper()
    request_tokens = request.strip().split()
    event_tile = request_event_tile(request_tokens)
    if head == "PASS":
        return (PASS_ACTION, 0, 0)
    if head == "HU":
        return (HU_ACTION, 0, 0)
    if head == "PLAY" and len(parts) >= 2 and parts[1] in TILE_NAMES:
        return (DISCARD_ACTION, 0, tile_id(parts[1]))
    if head == "CHI" and len(parts) >= 3 and event_tile in TILE_NAMES:
        middle = parts[1]
        if not _valid_chow_pair(middle, event_tile):
            return None
        return (ACTION_TO_INDEX["CHOW"], flatten_claim("CHOW", chow_claim_index(middle, event_tile)), 0)
    if head == "PENG" and event_tile in TILE_NAMES:
        return (ACTION_TO_INDEX["PONG"], flatten_claim("PONG", tile_id(event_tile)), 0)
    if head == "GANG":
        if request_tokens and request_tokens[0] == "2":
            tile = parts[1] if len(parts) >= 2 else event_tile
            if tile not in TILE_NAMES:
                return None
            return (ACTION_TO_INDEX["ANKONG"], flatten_claim("ANKONG", tile_id(tile)), 0)
        if event_tile not in TILE_NAMES:
            return None
        return (ACTION_TO_INDEX["MINGKONG"], flatten_claim("MINGKONG", tile_id(event_tile)), 0)
    if head == "BUGANG" and len(parts) >= 2 and parts[1] in TILE_NAMES:
        return (ACTION_TO_INDEX["BUKONG"], flatten_claim("BUKONG", tile_id(parts[1])), 0)
    return None


def action_type_mask(
    player: int,
    request_tokens: list[str],
    state: ReplayState,
    display: dict[str, Any],
) -> list[float]:
    mask = [0.0] * len(ACTION_NAMES)
    if not request_tokens:
        mask[PASS_ACTION] = 1.0
        return mask
    can_hu = display.get("canHu")
    hu_legal = isinstance(can_hu, list) and player < len(can_hu) and _can_hu_value(can_hu[player])

    if request_tokens[0] == "2":
        if hu_legal:
            mask[HU_ACTION] = 1.0
        if state.hands[player]:
            mask[DISCARD_ACTION] = 1.0
        if any(count >= 4 for count in state.hands[player].values()):
            mask[ACTION_TO_INDEX["ANKONG"]] = 1.0
        if any(state.pengs[player][tile] > 0 and state.hands[player][tile] > 0 for tile in state.pengs[player]):
            mask[ACTION_TO_INDEX["BUKONG"]] = 1.0
        return mask

    mask[PASS_ACTION] = 1.0
    if hu_legal:
        mask[HU_ACTION] = 1.0
    if len(request_tokens) < 3:
        return mask
    action = request_tokens[2].upper()
    event_tile = request_event_tile(request_tokens)
    if action not in {"PLAY", "PENG", "CHI"} or event_tile not in TILE_NAMES:
        return mask
    try:
        actor = int(request_tokens[1])
    except (TypeError, ValueError):
        return mask
    if actor == player:
        return mask
    if state.hands[player][event_tile] >= 2:
        mask[ACTION_TO_INDEX["PONG"]] = 1.0
    if state.hands[player][event_tile] >= 3:
        mask[ACTION_TO_INDEX["MINGKONG"]] = 1.0
    if player == (actor + 1) % 4 and any_legal_chow(state.hands[player], event_tile):
        mask[ACTION_TO_INDEX["CHOW"]] = 1.0
    return mask


def actual_response(response_log: dict[str, Any], player: str) -> str | None:
    item = response_log.get(str(player))
    if not isinstance(item, dict):
        return None
    raw = item.get("raw") or item.get("response") or item.get("content")
    if raw is None:
        return None
    response = str(raw).strip()
    return response or None


def request_event_tile(tokens: list[str]) -> str | None:
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2].upper() in {"PLAY", "PENG", "CHI"}:
        return tokens[-1]
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2].upper() == "BUGANG":
        return tokens[3]
    return None


def claim_response_discard(response: str) -> str | None:
    parts = response.strip().split()
    if len(parts) >= 2 and parts[0].upper() == "PENG":
        return parts[-1]
    if len(parts) >= 3 and parts[0].upper() == "CHI":
        return parts[-1]
    return None


def chow_claim_index(middle: str, event_tile: str) -> int:
    if not _valid_chow_pair(middle, event_tile):
        raise ValueError(f"invalid chow pair: middle={middle}, event={event_tile}")
    suit_offset = SUIT_TO_ID[middle[0]] * 21
    middle_rank = suited_rank(middle)
    offer_position = suited_rank(event_tile) - middle_rank + 2
    return suit_offset + (middle_rank - 2) * 3 + (offer_position - 1)


def chow_sequence(middle: str) -> list[str]:
    rank = suited_rank(middle)
    if rank < 2 or rank > 8:
        raise ValueError(f"chow middle rank must be in [2, 8], got {middle}")
    return [f"{middle[0]}{rank - 1}", middle, f"{middle[0]}{rank + 1}"]


def chow_needed_from_hand(middle: str, event_tile: str) -> Counter[str]:
    needed = Counter(chow_sequence(middle))
    needed[event_tile] -= 1
    for tile in list(needed):
        if needed[tile] <= 0:
            del needed[tile]
    return needed


def any_legal_chow(hand: Counter[str], event_tile: str) -> bool:
    if not is_suited(event_tile):
        return False
    suit = event_tile[0]
    rank = suited_rank(event_tile)
    for middle_rank in range(rank - 1, rank + 2):
        middle = f"{suit}{middle_rank}"
        if not _valid_chow_pair(middle, event_tile):
            continue
        needed = chow_needed_from_hand(middle, event_tile)
        if all(hand[tile] >= count for tile, count in needed.items()):
            return True
    return False


def _valid_chow_pair(middle: str, event_tile: str) -> bool:
    if not is_suited(middle) or not is_suited(event_tile):
        return False
    if middle[0] != event_tile[0]:
        return False
    middle_rank = suited_rank(middle)
    event_rank = suited_rank(event_tile)
    return 2 <= middle_rank <= 8 and abs(event_rank - middle_rank) <= 1


def zero_memory_frame(previous_action: int = PASS_ACTION) -> MemoryFrame:
    return MemoryFrame(
        visible_tiles=torch.zeros(22, 34, dtype=torch.float32),
        game_features=torch.zeros(24, dtype=torch.float32),
        previous_action=int(previous_action),
        reward=0.0,
    )


def selected_action_mask(action: int) -> list[float]:
    mask = [0.0] * len(ACTION_NAMES)
    mask[int(action)] = 1.0
    return mask


def _append_history(history: deque[MemoryFrame], example: EncodedExample, *, previous_action: int) -> None:
    history.append(
        MemoryFrame(
            visible_tiles=example.visible_tiles[-1].clone(),
            game_features=example.game_features[-1].clone(),
            previous_action=int(previous_action),
            reward=0.0,
        )
    )


def _stack_examples(examples: list[EncodedExample], stats: Counter[str]) -> dict[str, Any]:
    if not examples:
        empty = {
            "visible_tiles": torch.zeros(0, 4, 22, 34),
            "game_features": torch.zeros(0, 4, 24),
            "rewards": torch.zeros(0, 4),
            "previous_actions": torch.zeros(0, 4, dtype=torch.long),
            "sub_visible_tiles": torch.zeros(0, 4, 22, 34),
            "sub_game_features": torch.zeros(0, 4, 24),
            "sub_rewards": torch.zeros(0, 4),
            "sub_previous_actions": torch.zeros(0, 4, dtype=torch.long),
            "hidden_tiles": torch.zeros(0, 5, 34),
            "action_label": torch.zeros(0, dtype=torch.long),
            "claim_label": torch.zeros(0, dtype=torch.long),
            "discard_label": torch.zeros(0, dtype=torch.long),
            "value_target": torch.zeros(0, dtype=torch.float32),
        }
        empty["stats"] = dict(sorted(stats.items()))
        return empty
    return {
        "visible_tiles": torch.stack([example.visible_tiles for example in examples], dim=0),
        "game_features": torch.stack([example.game_features for example in examples], dim=0),
        "rewards": torch.stack([example.rewards for example in examples], dim=0),
        "previous_actions": torch.stack([example.previous_actions for example in examples], dim=0),
        "sub_visible_tiles": torch.stack([example.sub_visible_tiles for example in examples], dim=0),
        "sub_game_features": torch.stack([example.sub_game_features for example in examples], dim=0),
        "sub_rewards": torch.stack([example.sub_rewards for example in examples], dim=0),
        "sub_previous_actions": torch.stack([example.sub_previous_actions for example in examples], dim=0),
        "hidden_tiles": torch.stack([example.hidden_tiles for example in examples], dim=0),
        "action_label": torch.tensor([example.action_label for example in examples], dtype=torch.long),
        "claim_label": torch.tensor([example.claim_label for example in examples], dtype=torch.long),
        "discard_label": torch.tensor([example.discard_label for example in examples], dtype=torch.long),
        "value_target": torch.tensor([example.value_target for example in examples], dtype=torch.float32),
        "metadata": {
            "match_id": [example.match_id for example in examples],
            "player": [example.player for example in examples],
            "turn_index": [example.turn_index for example in examples],
            "request": [example.request for example in examples],
            "response": [example.response for example in examples],
            "kind": [example.kind for example in examples],
        },
        "stats": dict(sorted(stats.items())),
    }


def _payload_from_examples(
    examples: list[EncodedExample],
    stats: Counter[str],
    *,
    compact_metadata: bool,
) -> dict[str, Any]:
    payload = _allocate_payload(len(examples), stats, compact_metadata=compact_metadata)
    for index, example in enumerate(examples):
        _write_example(payload, index, example)
    return payload


def _compact_supervised_payload(payload: dict[str, Any]) -> None:
    for key in (
        "visible_tiles",
        "game_features",
        "sub_visible_tiles",
        "sub_game_features",
        "hidden_tiles",
    ):
        payload[key] = _to_uint8_tensor(payload[key], key)
    for key in (
        "previous_actions",
        "sub_previous_actions",
        "action_label",
        "claim_label",
        "discard_label",
    ):
        payload[key] = _to_uint8_tensor(payload[key], key)
    payload.pop("rewards", None)
    payload.pop("sub_rewards", None)
    payload.pop("value_target", None)
    payload["compact_storage"] = True
    payload["storage_schema"] = {
        "format": "supervised_uint8_v1",
        "feature_dtype": "uint8",
        "label_dtype": "uint8",
        "omitted_zero_tensors": ["rewards", "sub_rewards", "value_target"],
    }


def _to_uint8_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.numel():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} contains non-finite values and cannot be compacted")
        min_value = float(tensor.min().item())
        max_value = float(tensor.max().item())
        if min_value < 0.0 or max_value > 255.0:
            raise ValueError(f"{name} has values outside uint8 range: min={min_value} max={max_value}")
    return tensor.to(dtype=torch.uint8).contiguous()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate_payload(total_examples: int, stats: Counter[str], *, compact_metadata: bool) -> dict[str, Any]:
    payload = {
        "visible_tiles": torch.empty(total_examples, 4, 22, 34, dtype=torch.float32),
        "game_features": torch.empty(total_examples, 4, 24, dtype=torch.float32),
        "rewards": torch.empty(total_examples, 4, dtype=torch.float32),
        "previous_actions": torch.empty(total_examples, 4, dtype=torch.long),
        "sub_visible_tiles": torch.empty(total_examples, 4, 22, 34, dtype=torch.float32),
        "sub_game_features": torch.empty(total_examples, 4, 24, dtype=torch.float32),
        "sub_rewards": torch.empty(total_examples, 4, dtype=torch.float32),
        "sub_previous_actions": torch.empty(total_examples, 4, dtype=torch.long),
        "hidden_tiles": torch.empty(total_examples, 5, 34, dtype=torch.float32),
        "action_label": torch.empty(total_examples, dtype=torch.long),
        "claim_label": torch.empty(total_examples, dtype=torch.long),
        "discard_label": torch.empty(total_examples, dtype=torch.long),
        "value_target": torch.empty(total_examples, dtype=torch.float32),
        "metadata": _allocate_metadata(total_examples, compact=compact_metadata),
        "stats": dict(sorted(stats.items())),
    }
    return payload


def _allocate_metadata(total_examples: int, *, compact: bool) -> dict[str, Any]:
    if compact:
        return {
            "format": "compact_v1",
            "match_id": [],
            "match_index": torch.empty(total_examples, dtype=torch.long),
            "player": torch.empty(total_examples, dtype=torch.long),
            "turn_index": torch.empty(total_examples, dtype=torch.long),
            "kind": [],
            "_match_id_to_index": {},
        }
    return {
        "match_id": [],
        "player": [],
        "turn_index": [],
        "request": [],
        "response": [],
        "kind": [],
    }


def _write_example(payload: dict[str, Any], index: int, example: EncodedExample) -> None:
    payload["visible_tiles"][index].copy_(example.visible_tiles)
    payload["game_features"][index].copy_(example.game_features)
    payload["rewards"][index].copy_(example.rewards)
    payload["previous_actions"][index].copy_(example.previous_actions)
    payload["sub_visible_tiles"][index].copy_(example.sub_visible_tiles)
    payload["sub_game_features"][index].copy_(example.sub_game_features)
    payload["sub_rewards"][index].copy_(example.sub_rewards)
    payload["sub_previous_actions"][index].copy_(example.sub_previous_actions)
    payload["hidden_tiles"][index].copy_(example.hidden_tiles)
    payload["action_label"][index] = int(example.action_label)
    payload["claim_label"][index] = int(example.claim_label)
    payload["discard_label"][index] = int(example.discard_label)
    payload["value_target"][index] = float(example.value_target)
    _write_metadata(payload["metadata"], index, example)


def _write_metadata(metadata: dict[str, Any], index: int, example: EncodedExample) -> None:
    if metadata.get("format") == "compact_v1":
        match_id_to_index = metadata["_match_id_to_index"]
        match_index = match_id_to_index.get(example.match_id)
        if match_index is None:
            match_index = len(metadata["match_id"])
            match_id_to_index[example.match_id] = match_index
            metadata["match_id"].append(example.match_id)
        metadata["match_index"][index] = int(match_index)
        metadata["player"][index] = int(example.player)
        metadata["turn_index"][index] = int(example.turn_index)
        return

    metadata["match_id"].append(example.match_id)
    metadata["player"].append(example.player)
    metadata["turn_index"].append(example.turn_index)
    metadata["request"].append(example.request)
    metadata["response"].append(example.response)
    metadata["kind"].append(example.kind)


def _finalize_metadata(payload: dict[str, Any]) -> None:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("format") == "compact_v1":
        metadata.pop("_match_id_to_index", None)


def metadata_column(metadata: dict[str, Any] | None, key: str) -> list[Any]:
    if not metadata:
        return []
    if key == "match_id" and "match_index" in metadata:
        match_ids = list(metadata.get("match_id") or metadata.get("match_ids") or [])
        match_index = metadata.get("match_index")
        indices = match_index.tolist() if torch.is_tensor(match_index) else list(match_index or [])
        return [str(match_ids[int(index)]) for index in indices]
    values = metadata.get(key)
    if values is None:
        return []
    if torch.is_tensor(values):
        return values.tolist()
    return list(values)


def tensor_encoding_schema() -> dict[str, Any]:
    return {
        "version": TENSOR_ENCODING_VERSION,
        "visible_tile_rows": 22,
        "game_features": 24,
        "hidden_tile_rows": list(HIDDEN_TILE_ROW_NAMES),
        "hidden_tile_shape": [5, 34],
    }


def _counter_to_tensor(counter: Counter[str]) -> torch.Tensor:
    tensor = torch.zeros(len(TILE_NAMES), dtype=torch.float32)
    for tile, count in counter.items():
        if tile in TILE_NAMES and count > 0:
            tensor[tile_id(tile)] += float(count)
    return tensor


def _counter_to_ids(counter: Counter[str]) -> list[int]:
    ids: list[int] = []
    for tile, count in counter.items():
        if tile in TILE_NAMES and count > 0:
            ids.extend([tile_id(tile)] * int(count))
    return ids


def _meld_tiles(counter: Counter[str], width: int) -> Counter[str]:
    expanded = Counter()
    for tile, count in counter.items():
        if count > 0:
            expanded[tile] += int(count) * int(width)
    return expanded


def _display_player(display: dict[str, Any]) -> int | None:
    try:
        player = int(display.get("player"))
    except (TypeError, ValueError):
        return None
    return player if 0 <= player < 4 else None


def _can_hu_value(value: Any) -> bool:
    try:
        return int(value) >= 8
    except (TypeError, ValueError):
        return False


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--shard-index-out", default=None)
    parser.add_argument("--shard-max-examples", type=int, default=50000)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--corpus-validation", default=None)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--memory-len", type=int, default=4)
    parser.add_argument("--include-single-action", action="store_true")
    parser.add_argument("--single-action-discard-stride", type=int, default=1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--compact-metadata", action="store_true")
    parser.add_argument("--compact-storage", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()
    if not args.out and not args.shard_dir:
        parser.error("one of --out or --shard-dir is required")
    summary = tensorize_file(
        Path(args.infile),
        Path(args.out) if args.out else None,
        summary_out=Path(args.summary_out) if args.summary_out else None,
        corpus_validation=Path(args.corpus_validation) if args.corpus_validation else None,
        max_matches=args.max_matches,
        memory_len=args.memory_len,
        include_single_action=args.include_single_action,
        single_action_discard_stride=args.single_action_discard_stride,
        streaming=args.streaming,
        compact_metadata=args.compact_metadata,
        compact_storage=args.compact_storage,
        shard_dir=Path(args.shard_dir) if args.shard_dir else None,
        shard_index_out=Path(args.shard_index_out) if args.shard_index_out else None,
        shard_max_examples=args.shard_max_examples,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
