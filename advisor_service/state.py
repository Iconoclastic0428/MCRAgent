"""Normalized read-only game state for the Tziakcha advisor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tiles import display_name

ACTION_NAMES = {
    1: "flower",
    2: "discard",
    3: "chow",
    4: "pung",
    5: "kong",
    6: "hu",
    8: "pass",
    9: "waive",
}


@dataclass
class AdvisorState:
    seat: int | None = None
    turn: int | None = None
    prevalent_wind: int | None = None
    wall_count: int | None = None
    hand: list[int] = field(default_factory=list)
    melds: list[int] = field(default_factory=list)
    flowers: int = 0
    discards: list[dict[str, Any]] = field(default_factory=list)
    last_draw: dict[str, Any] | None = None
    available_actions: dict[str, list[int]] = field(default_factory=dict)
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    unknown_events: list[dict[str, Any]] = field(default_factory=list)

    def ingest(self, message: dict[str, Any]) -> None:
        self.raw_events.append(message)
        if message.get("m") == 4 and isinstance(message.get("i"), dict):
            self._apply_reconnect(message)
            return
        if message.get("m") != 2:
            return

        event = message.get("r")
        if event == 2:
            self._apply_deal(message)
        elif event == 3:
            self.turn = _optional_int(message.get("v"), self.turn)
            self._apply_prompt(message.get("a") or message.get("v") or {})
        elif event == 4:
            self._apply_flower(message)
        elif event == 6:
            self._apply_draw(message)
        elif event == 7:
            self._apply_discard(message)
        elif event == 8:
            self._apply_chow(message)
        elif event == 9:
            self._apply_pung(message)
        elif event == 10:
            self._apply_mkong(message)
        elif event == 11:
            self._apply_ckong(message)
        elif event == 14:
            self._apply_round_exchange(message)
        else:
            self.unknown_events.append(message)

    def reset(self) -> None:
        self.seat = None
        self.turn = None
        self.prevalent_wind = None
        self.wall_count = None
        self.hand.clear()
        self.melds.clear()
        self.flowers = 0
        self.discards.clear()
        self.last_draw = None
        self.available_actions.clear()
        self.raw_events.clear()
        self.unknown_events.clear()

    def _apply_reconnect(self, message: dict[str, Any]) -> None:
        info = message["i"]
        if "h" not in info and "w" not in info:
            self._apply_snapshot(info)
            if "v" in message:
                self.seat = _optional_int(message.get("v"), self.seat)
            return

        self.seat = _optional_int(message.get("v", info.get("v")), self.seat)
        self.turn = _optional_int(info.get("t"), self.turn)
        self.prevalent_wind = _deal_index_to_prevalent_wind(info.get("d"), self.prevalent_wind)
        wall = info.get("w") or {}
        if isinstance(wall, dict) and ("f" in wall or "b" in wall):
            self.wall_count = 144 - int(wall.get("f") or 0) - int(wall.get("b") or 0)
        else:
            self.wall_count = _optional_int(info.get("y"), self.wall_count)

        hands = info.get("h") or []
        if self.seat is not None and 0 <= self.seat < len(hands):
            hand_info = hands[self.seat] or {}
            self.hand = _visible_tiles(hand_info.get("s") or [])
            self.melds = [int(pack) for pack in hand_info.get("p") or []]
            self.flowers = _player_flower_count(message.get("u") or info.get("u"), self.seat, self.flowers)
        self._apply_prompt(info.get("a") or {})

    def _apply_snapshot(self, info: dict[str, Any]) -> None:
        self.seat = _optional_int(info.get("v"), self.seat)
        self.turn = _optional_int(info.get("t"), self.turn)
        self.prevalent_wind = _deal_index_to_prevalent_wind(info.get("d"), self.prevalent_wind)
        self.wall_count = _optional_int(info.get("y"), self.wall_count)
        players = info.get("p") or []
        if self.seat is not None and 0 <= self.seat < len(players):
            player = players[self.seat] or {}
            hand_info = player.get("h") or {}
            self.hand = _visible_tiles(hand_info.get("s") or [])
            self.melds = [int(pack) for pack in hand_info.get("p") or []]
            self.flowers = int(player.get("f") or 0)
        self._apply_prompt(info.get("a") or {})

    def _apply_prompt(self, prompt: Any) -> None:
        actions: dict[str, list[int]] = {}
        if isinstance(prompt, dict):
            for key, value in prompt.items():
                code = int(key)
                name = ACTION_NAMES.get(code, f"action_{code}")
                if isinstance(value, list):
                    actions[name] = [_normalize_action_value(code, v) for v in value]
                elif value is not None and value is not False:
                    actions[name] = [_normalize_action_value(code, value)]
        elif isinstance(prompt, list):
            for encoded in prompt:
                code = int(encoded) >> 16
                value = int(encoded) & 0xFFFF
                name = ACTION_NAMES.get(code, f"action_{code}")
                actions.setdefault(name, []).append(_normalize_action_value(code, value))
        self.available_actions = actions

    def _apply_deal(self, message: dict[str, Any]) -> None:
        tiles = message.get("v")
        self.prevalent_wind = _deal_index_to_prevalent_wind(message.get("h"), self.prevalent_wind)
        self.melds = []
        self.flowers = 0
        self.discards.clear()
        self.last_draw = None
        if isinstance(tiles, list):
            self.hand = _visible_tiles(tiles)
        self.available_actions = {}

    def _apply_round_exchange(self, message: dict[str, Any]) -> None:
        self.seat = _optional_int(message.get("v"), self.seat)
        self.turn = None
        self.prevalent_wind = None
        self.wall_count = None
        self.hand.clear()
        self.melds.clear()
        self.flowers = 0
        self.discards.clear()
        self.last_draw = None
        self.available_actions.clear()

    def _apply_flower(self, message: dict[str, Any]) -> None:
        player = _optional_int(message.get("v"), self.turn)
        self.turn = player
        self.wall_count = _optional_int(message.get("h"), self.wall_count)
        if player == self.seat:
            self.flowers += 1
            replacement_tile = _tile_value(message.get("t"))
            if replacement_tile is not None and replacement_tile not in self.hand:
                self.hand.append(replacement_tile)
                self.hand.sort()
            if replacement_tile is not None:
                self.last_draw = {"seat": player, "tile": replacement_tile, "display": display_name(replacement_tile)}
            self._apply_prompt(message.get("a") or {})
        else:
            self.available_actions = {}

    def _apply_draw(self, message: dict[str, Any]) -> None:
        player = _optional_int(message.get("v"), self.turn)
        self.turn = player
        self.wall_count = _optional_int(message.get("h"), self.wall_count)
        tile_value = _tile_value(message.get("t"))
        if player == self.seat and tile_value is not None:
            if tile_value not in self.hand:
                self.hand.append(tile_value)
                self.hand.sort()
            self.last_draw = {"seat": player, "tile": tile_value, "display": display_name(tile_value)}
        self._apply_prompt(message.get("a") or {})

    def _apply_discard(self, message: dict[str, Any]) -> None:
        seat = _optional_int(message.get("v"), None)
        tile = _tile_value(message.get("t"))
        if tile is None:
            self.unknown_events.append(message)
            return
        self.turn = seat
        self.discards.append({"seat": seat, "tile": tile, "display": display_name(tile)})
        if seat == self.seat:
            self._remove_one_from_hand(tile)
            self.last_draw = None
        self._apply_prompt(message.get("a") or {})

    def _apply_chow(self, message: dict[str, Any]) -> None:
        player = _optional_int(message.get("v"), self.turn)
        pack = _optional_int(message.get("p"), None)
        if not pack:
            self._apply_prompt(message.get("a") or {})
            return
        base = _pack_tile(pack)
        offer = _pack_offer(pack)
        if offer == 2:
            consumed = [base - 4, base + 4]
        elif offer == 3:
            consumed = [base - 4, base]
        else:
            consumed = [base, base + 4]
        self._apply_meld(player, pack, consumed)

    def _apply_pung(self, message: dict[str, Any]) -> None:
        player = _optional_int(message.get("v"), self.turn)
        pack = _optional_int(message.get("p"), None)
        if not pack:
            self._apply_prompt(message.get("a") or {})
            return
        base = _pack_tile(pack)
        self._apply_meld(player, pack, [base, base])

    def _apply_mkong(self, message: dict[str, Any]) -> None:
        player = _optional_int(message.get("v"), self.turn)
        pack = _optional_int(message.get("p"), None)
        if not pack:
            self._apply_prompt(message.get("a") or {})
            return
        base = _pack_tile(pack)
        consumed = [base] if _pack_type(pack) == 3 else [base, base, base]
        self._apply_meld(player, pack, consumed)

    def _apply_ckong(self, message: dict[str, Any]) -> None:
        player = _optional_int(message.get("v"), self.turn)
        pack = _optional_int(message.get("p"), None)
        if not pack:
            self._apply_prompt(message.get("a") or {})
            return
        base = _pack_tile(pack)
        self._apply_meld(player, pack, [base, base, base, base])

    def _apply_meld(self, player: int | None, pack: int, consumed_tiles: list[int]) -> None:
        self.turn = player
        if player == self.seat:
            for tile in consumed_tiles:
                self._remove_one_from_hand(tile)
            self._record_meld(pack)
        self.available_actions = {}

    def _record_meld(self, pack: int) -> None:
        if _pack_type(pack) == 3:
            pack_tile = _pack_tile(pack)
            for index, existing_pack in enumerate(self.melds):
                if _pack_type(existing_pack) == 1 and _pack_tile(existing_pack) == pack_tile:
                    self.melds[index] = pack
                    return
        self.melds.append(pack)

    def _remove_one_from_hand(self, tile: int) -> None:
        tile_kind = tile >> 2
        for index, owned_tile in enumerate(self.hand):
            if owned_tile >> 2 == tile_kind:
                del self.hand[index]
                return

    def snapshot(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "turn": self.turn,
            "prevalent_wind": self.prevalent_wind,
            "wall_count": self.wall_count,
            "hand": list(self.hand),
            "hand_display": [display_name(tile) for tile in self.hand],
            "melds": list(self.melds),
            "flowers": self.flowers,
            "last_draw": dict(self.last_draw) if self.last_draw else None,
            "available_actions": dict(self.available_actions),
            "last_discard": self.discards[-1] if self.discards else None,
            "last_discard_display": self.discards[-1]["display"] if self.discards else None,
            "event_count": len(self.raw_events),
            "unknown_event_count": len(self.unknown_events),
        }


def _optional_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _deal_index_to_prevalent_wind(value: Any, default: int | None) -> int | None:
    deal_index = _optional_int(value, None)
    if deal_index is None:
        return default
    return (deal_index >> 2) & 3


def _tile_value(value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    tile = value & 0xFF
    if _is_playable_tile(tile):
        return tile
    return None


def _visible_tiles(values: Any) -> list[int]:
    result: list[int] = []
    for value in values:
        if isinstance(value, int) and _is_playable_tile(value):
            result.append(value)
    return sorted(result)


def _is_playable_tile(value: int) -> bool:
    return 0 <= value <= 135


def _player_flower_count(players: Any, seat: int | None, default: int) -> int:
    if seat is None or not isinstance(players, list) or not 0 <= seat < len(players):
        return default
    player = players[seat]
    if not isinstance(player, dict):
        return default
    count = _optional_int(player.get("f"), default)
    return default if count is None else count


def _pack_type(pack: int) -> int:
    return (pack >> 8) & 3


def _pack_tile(pack: int) -> int:
    return (pack & 63) << 2


def _pack_offer(pack: int) -> int:
    return (pack >> 6) & 3


def _normalize_action_value(code: int, value: Any) -> int:
    normalized = int(value)
    if code in {3, 4, 5} and normalized > 143:
        return _pack_tile(normalized)
    return normalized
