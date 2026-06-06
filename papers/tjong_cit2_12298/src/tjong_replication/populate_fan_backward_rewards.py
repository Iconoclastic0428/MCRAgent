"""Populate Tjong PPO tensors with fan-backward rewards from game logs."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .actions import ACTION_TO_INDEX
from .encoding import VISIBLE_ROW_NAMES, tile_multiset
from .fan_attribution import FanAttributionError, attribute_display_fans
from .fan_backward import FanItem, calculate_score, winning_reward
from .tensorize_botzone import ReplayState, chow_sequence
from .tiles import TILE_NAMES, tile_id


DISCARD_ACTION = ACTION_TO_INDEX["DISCARD"]


@dataclass
class ClaimPenaltyEvent:
    turn_index: int
    claimer: int
    from_whom: int
    tiles: tuple[int, ...]


@dataclass
class TerminalContext:
    match_id: str
    terminal_action: str
    winner: int | None
    loser: int | None
    winning_tile: int | None
    hand_private: torch.Tensor
    hand_claim: torch.Tensor
    hand_hu: torch.Tensor
    fans: list[FanItem]
    fan_tile_source: str
    claim_events: list[ClaimPenaltyEvent]


def populate_file(
    *,
    tensor_pt: Path,
    raw_path: Path,
    out_path: Path,
    summary_out: Path | None = None,
    fan_items_jsonl: Path | None = None,
    allow_approximate_fans: bool = False,
    require_hu_reward: bool = False,
    huang_penalty: float = 0.0,
) -> dict[str, Any]:
    data = torch.load(tensor_pt, map_location="cpu")
    metadata = data.get("metadata") or {}
    match_ids = [str(value) for value in metadata.get("match_id") or []]
    players = [int(value) for value in metadata.get("player") or []]
    turn_indices = [int(value) for value in metadata.get("turn_index") or []]
    if not match_ids:
        raise ValueError("tensor file must contain metadata.match_id to populate fan-backward rewards")

    needed_matches = set(match_ids)
    explicit_fans = load_explicit_fan_items(fan_items_jsonl) if fan_items_jsonl else {}
    contexts: dict[str, TerminalContext] = {}
    for record in iter_jsonl(raw_path):
        match_id = str(record.get("match_id") or "")
        if match_id not in needed_matches:
            continue
        contexts[match_id] = terminal_context_from_record(
            record,
            explicit_fans=explicit_fans.get(match_id),
            allow_approximate_fans=allow_approximate_fans,
        )
        if len(contexts) == len(needed_matches):
            break

    missing = sorted(needed_matches - set(contexts))
    if missing:
        raise ValueError(f"raw logs missing matches from tensor metadata: {missing[:5]}")

    rewards = torch.zeros(len(match_ids), dtype=torch.float32)
    row_by_key: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    rows_by_match: dict[str, list[int]] = defaultdict(list)
    rows_by_match_player: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(match_ids, players, turn_indices)):
        match_id = str(key[0])
        player = int(key[1])
        row_by_key[(match_id, player, int(key[2]))].append(index)
        rows_by_match[match_id].append(index)
        rows_by_match_player[(match_id, player)].append(index)

    fan_sources: dict[str, int] = defaultdict(int)
    terminal_actions: dict[str, int] = defaultdict(int)
    hu_matches = 0
    hu_matches_with_fans = 0
    hu_matches_with_reward = 0
    hu_matches_without_reward: list[str] = []
    huang_penalized_matches = 0
    huang_penalized_player_terminals = 0
    for match_id, context in contexts.items():
        fan_sources[context.fan_tile_source] += 1
        terminal_actions[context.terminal_action] += 1
        if context.terminal_action == "HUANG":
            if huang_penalty > 0.0:
                penalized_this_match = 0
                for player in range(4):
                    rows = rows_by_match_player.get((match_id, player), [])
                    if not rows:
                        continue
                    target = max(rows, key=lambda idx: turn_indices[idx])
                    rewards[target] -= float(huang_penalty)
                    penalized_this_match += 1
                if penalized_this_match:
                    huang_penalized_matches += 1
                    huang_penalized_player_terminals += penalized_this_match
            continue
        if context.terminal_action != "HU" or context.winner is None:
            continue
        hu_matches += 1
        if context.fans:
            hu_matches_with_fans += 1
        match_reward_abs = 0.0
        score = calculate_score((context.hand_hu - context.hand_claim).clamp_min(0), context.hand_claim, context.fans)
        for index in rows_by_match_player.get((match_id, context.winner), []):
            current = current_hand_from_visible(data["visible_tiles"][index])
            if int(data["action_label"][index]) == DISCARD_ACTION:
                current = current - tile_multiset([int(data["discard_label"][index])])
            reward_value = float(winning_reward(context.hand_hu, current, score))
            rewards[index] += reward_value
            match_reward_abs += abs(reward_value)

        for claim in context.claim_events:
            if claim.claimer != context.winner:
                continue
            penalty = -float((tile_multiset(claim.tiles) * score).sum().item())
            for index in row_by_key.get((match_id, claim.from_whom, claim.turn_index), []):
                rewards[index] += penalty
                match_reward_abs += abs(penalty)

        if context.loser is not None and context.winning_tile is not None:
            penalty = -float((tile_multiset([context.winning_tile]) * score).sum().item())
            loser_rows = rows_by_match_player.get((match_id, context.loser), [])
            if loser_rows:
                rewards[max(loser_rows, key=lambda idx: turn_indices[idx])] += penalty
                match_reward_abs += abs(penalty)

        if match_reward_abs > 0.0:
            hu_matches_with_reward += 1
        elif context.fans:
            hu_matches_without_reward.append(match_id)

    if require_hu_reward and hu_matches == 0:
        raise ValueError("fan-backward paper mode requires at least one HU match")
    if require_hu_reward and hu_matches_without_reward:
        raise ValueError(
            "HU matches with attributed fan items produced zero fan-backward reward: "
            f"{hu_matches_without_reward[:5]}"
        )

    data["fan_backward_reward"] = rewards
    data["fan_backward_summary"] = {
        "tensor_pt": str(tensor_pt),
        "raw_path": str(raw_path),
        "fan_items_jsonl": str(fan_items_jsonl) if fan_items_jsonl else None,
        "allow_approximate_fans": allow_approximate_fans,
        "require_hu_reward": require_hu_reward,
        "huang_penalty": float(huang_penalty),
        "reward_row_index": "precomputed_by_match_player_turn",
        "indexed_matches": int(len(rows_by_match)),
        "examples": int(rewards.numel()),
        "nonzero_rewards": int((rewards != 0).sum().item()),
        "reward_mean": float(rewards.mean().item()) if rewards.numel() else None,
        "reward_std": float(rewards.std(unbiased=False).item()) if rewards.numel() > 1 else 0.0,
        "fan_tile_sources": dict(sorted(fan_sources.items())),
        "terminal_actions": dict(sorted(terminal_actions.items())),
        "hu_matches": int(hu_matches),
        "huang_matches": int(terminal_actions.get("HUANG", 0)),
        "hu_rate": float(hu_matches / len(contexts)) if contexts else None,
        "huang_rate": float(terminal_actions.get("HUANG", 0) / len(contexts)) if contexts else None,
        "hu_matches_with_fans": int(hu_matches_with_fans),
        "hu_matches_with_reward": int(hu_matches_with_reward),
        "hu_matches_without_reward": hu_matches_without_reward[:20],
        "huang_penalized_matches": int(huang_penalized_matches),
        "huang_penalized_player_terminals": int(huang_penalized_player_terminals),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    summary = {**data["fan_backward_summary"], "output": str(out_path)}
    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def terminal_context_from_record(
    record: dict[str, Any],
    *,
    explicit_fans: list[FanItem] | None,
    allow_approximate_fans: bool,
) -> TerminalContext:
    match_id = str(record.get("match_id") or "")
    state = ReplayState.from_record(record)
    claim_events: list[ClaimPenaltyEvent] = []
    terminal_display: dict[str, Any] = {}
    last_event_tile: str | None = None
    last_event_player: int | None = None
    last_event_action: str | None = None
    last_discard_turn: int | None = None
    last_discard_player: int | None = None
    last_discard_tile: str | None = None
    logs = record.get("logs") or []
    for index, item in enumerate(logs):
        if not isinstance(item, dict):
            continue
        output = item.get("output") or {}
        display = output.get("display") or {}
        if not isinstance(display, dict):
            continue
        action = str(display.get("action") or "").upper()
        if action in {"HU", "HUANG"}:
            terminal_display = display
            break
        turn_index = index // 2
        if action in {"CHI", "PENG", "GANG"} and state.claimable_discard and last_discard_turn is not None:
            claimer = _display_player(display)
            if claimer is not None:
                event_tile = state.claimable_discard[1]
                if action == "CHI" and display.get("tileCHI") in TILE_NAMES:
                    tiles = tuple(tile_id(tile) for tile in chow_sequence(str(display["tileCHI"])))
                elif event_tile in TILE_NAMES:
                    width = 4 if action == "GANG" else 3
                    tiles = tuple([tile_id(event_tile)] * width)
                else:
                    tiles = ()
                if tiles:
                    claim_events.append(
                        ClaimPenaltyEvent(
                            turn_index=int(last_discard_turn),
                            claimer=int(claimer),
                            from_whom=int(state.claimable_discard[0]),
                            tiles=tiles,
                        )
                    )
        state.apply_display(display)
        if action == "DRAW":
            last_event_tile = display.get("tile")
            last_event_player = _display_player(display)
            last_event_action = "DRAW"
        elif action == "PLAY":
            last_event_tile = display.get("tile")
            last_event_player = _display_player(display)
            last_event_action = "PLAY"
            last_discard_turn = turn_index
            last_discard_player = last_event_player
            last_discard_tile = last_event_tile

    terminal_action = str(terminal_display.get("action") or "UNKNOWN").upper()
    winner = _display_player(terminal_display) if terminal_action == "HU" else None
    scores = terminal_display.get("score")
    loser = infer_loser(scores, winner)
    if winner is None:
        zero = torch.zeros(len(TILE_NAMES), dtype=torch.float32)
        return TerminalContext(
            match_id=match_id,
            terminal_action=terminal_action,
            winner=None,
            loser=None,
            winning_tile=None,
            hand_private=zero,
            hand_claim=zero,
            hand_hu=zero,
            fans=[],
            fan_tile_source="none",
            claim_events=claim_events,
        )

    hand_private = counter_to_tensor(state.hands[winner])
    hand_claim = meld_tensor(state, winner)
    hand_hu = hand_private + hand_claim
    winning_tile_name = last_event_tile if last_event_tile in TILE_NAMES else None
    is_self_draw = last_event_action == "DRAW" and last_event_player == winner
    if winning_tile_name is not None and not is_self_draw:
        hand_hu = hand_hu + tile_multiset([tile_id(winning_tile_name)])
    winning_tile = tile_id(winning_tile_name) if winning_tile_name in TILE_NAMES else None

    if explicit_fans and all(fan.tiles for fan in explicit_fans):
        fans = explicit_fans
        fan_tile_source = "explicit_fan_items"
    elif explicit_fans is not None and any(not fan.tiles for fan in explicit_fans):
        fans, fan_tile_source = structural_fan_items(
            terminal_display=terminal_display,
            hand_hu=hand_hu,
            winner=winner,
            winning_tile=winning_tile,
            prevailing_wind=state.prevailing_wind,
            source="structural_fan_attribution_from_display_with_empty_explicit_items",
            match_id=match_id,
        )
    elif explicit_fans is not None:
        fans, fan_tile_source = structural_fan_items(
            terminal_display=terminal_display,
            hand_hu=hand_hu,
            winner=winner,
            winning_tile=winning_tile,
            prevailing_wind=state.prevailing_wind,
            source="structural_fan_attribution_from_display",
            match_id=match_id,
        )
    elif allow_approximate_fans:
        fan_total = display_fan_total(terminal_display)
        tiles = tuple(int(index) for index in torch.nonzero(hand_hu > 0, as_tuple=False).flatten().tolist())
        fans = [FanItem(score=float(fan_total), tiles=tiles)] if tiles and fan_total else []
        fan_tile_source = "final_hand_uniform_approximation"
    else:
        fans, fan_tile_source = structural_fan_items(
            terminal_display=terminal_display,
            hand_hu=hand_hu,
            winner=winner,
            winning_tile=winning_tile,
            prevailing_wind=state.prevailing_wind,
            source="structural_fan_attribution_from_display",
            match_id=match_id,
        )

    return TerminalContext(
        match_id=match_id,
        terminal_action=terminal_action,
        winner=winner,
        loser=loser,
        winning_tile=winning_tile,
        hand_private=hand_private,
        hand_claim=hand_claim,
        hand_hu=hand_hu,
        fans=fans,
        fan_tile_source=fan_tile_source,
        claim_events=claim_events,
    )


def structural_fan_items(
    *,
    terminal_display: dict[str, Any],
    hand_hu: torch.Tensor,
    winner: int,
    winning_tile: int | None,
    prevailing_wind: int,
    source: str,
    match_id: str,
) -> tuple[list[FanItem], str]:
    attributed = attribute_display_fans(
        terminal_display,
        hand_hu=hand_hu,
        winner=winner,
        winning_tile=winning_tile,
        prevailing_wind=prevailing_wind,
    )
    if attributed.unsupported:
        raise ValueError(
            f"match {match_id} has unsupported fan-tile attribution names: {sorted(attributed.unsupported)}"
        )
    if terminal_display.get("fan") and not attributed.fans:
        raise FanAttributionError(f"match {match_id} did not produce any structurally attributed fan items")
    return attributed.fans, source


def load_explicit_fan_items(path: Path) -> dict[str, list[FanItem]]:
    by_match: dict[str, list[FanItem]] = {}
    for row in iter_jsonl(path):
        match_id = str(row.get("match_id") or "")
        fans = []
        for item in row.get("fans") or []:
            raw_tiles = item.get("tiles") or []
            tiles = tuple(tile_id(tile) if isinstance(tile, str) else int(tile) for tile in raw_tiles)
            fans.append(FanItem(score=float(item.get("score", item.get("fan", 0.0))), tiles=tiles))
        if match_id:
            by_match[match_id] = fans
    return by_match


def current_hand_from_visible(visible_memory: torch.Tensor) -> torch.Tensor:
    current = visible_memory[-1].float()
    row = {name: index for index, name in enumerate(VISIBLE_ROW_NAMES)}
    return (
        current[row["hand_self"]]
        + current[row["peng_p0"]]
        + current[row["chow_p0"]]
        + current[row["kong_p0"]]
    )


def counter_to_tensor(counter) -> torch.Tensor:
    tensor = torch.zeros(len(TILE_NAMES), dtype=torch.float32)
    for tile, count in counter.items():
        if tile in TILE_NAMES and count > 0:
            tensor[tile_id(tile)] += float(count)
    return tensor


def meld_tensor(state: ReplayState, player: int) -> torch.Tensor:
    tensor = torch.zeros(len(TILE_NAMES), dtype=torch.float32)
    for counter, width in ((state.pengs[player], 3), (state.kongs[player], 4)):
        for tile, count in counter.items():
            if tile in TILE_NAMES:
                tensor[tile_id(tile)] += float(count) * float(width)
    for tile, count in state.chows[player].items():
        if tile in TILE_NAMES:
            tensor[tile_id(tile)] += float(count)
    return tensor


def display_fan_total(display: dict[str, Any]) -> float:
    total = 0.0
    for item in display.get("fan") or []:
        total += float(item.get("value", 0.0)) * float(item.get("cnt", 1.0))
    if total <= 0.0:
        total = float(display.get("fanCnt") or 0.0)
    return total


def infer_loser(scores: Any, winner: int | None) -> int | None:
    if winner is None or not isinstance(scores, list) or len(scores) != 4:
        return None
    loser_scores = [(index, float(score)) for index, score in enumerate(scores) if index != winner]
    if not loser_scores:
        return None
    min_score = min(score for _, score in loser_scores)
    min_players = [index for index, score in loser_scores if score == min_score]
    return min_players[0] if len(min_players) == 1 else None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def _display_player(display: dict[str, Any]) -> int | None:
    try:
        player = int(display.get("player"))
    except (TypeError, ValueError):
        return None
    return player if 0 <= player < 4 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-pt", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--fan-items-jsonl", default=None)
    parser.add_argument("--allow-approximate-fans", action="store_true")
    parser.add_argument("--require-hu-reward", action="store_true")
    parser.add_argument("--huang-penalty", "--no-hu-penalty", dest="huang_penalty", type=float, default=0.0)
    args = parser.parse_args()
    populate_file(
        tensor_pt=Path(args.tensor_pt),
        raw_path=Path(args.raw),
        out_path=Path(args.out),
        summary_out=Path(args.summary_out) if args.summary_out else None,
        fan_items_jsonl=Path(args.fan_items_jsonl) if args.fan_items_jsonl else None,
        allow_approximate_fans=args.allow_approximate_fans,
        require_hu_reward=args.require_hu_reward,
        huang_penalty=args.huang_penalty,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
