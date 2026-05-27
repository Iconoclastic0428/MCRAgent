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

RESULT_EVENT_CODES = {12, 13}


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
    visible_counts: dict[int, int] = field(default_factory=dict)
    pending_discard: dict[str, Any] | None = None
    last_win_event: dict[str, Any] | None = None
    next_draw_about_kong: set[int] = field(default_factory=set)
    round_index: int = 0
    result_recorded_for_round: bool = False
    result_history: list[dict[str, Any]] = field(default_factory=list)
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
        result_applied = self._maybe_apply_result(message)
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
        elif result_applied:
            self.available_actions = {}
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
        self.visible_counts.clear()
        self.pending_discard = None
        self.last_win_event = None
        self.next_draw_about_kong.clear()
        self.round_index = 0
        self.result_recorded_for_round = False
        self.result_history.clear()
        self.available_actions.clear()
        self.raw_events.clear()
        self.unknown_events.clear()

    def _apply_reconnect(self, message: dict[str, Any]) -> None:
        info = message["i"]
        self._clear_public_context()
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
        self._clear_public_context()
        self.round_index += 1
        self.result_recorded_for_round = False
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
        self.visible_counts.clear()
        self.pending_discard = None
        self.last_win_event = None
        self.next_draw_about_kong.clear()
        self.available_actions.clear()

    def _apply_flower(self, message: dict[str, Any]) -> None:
        self._commit_pending_discard()
        player = _optional_int(message.get("v"), self.turn)
        self.turn = player
        self.wall_count = _optional_int(message.get("h"), self.wall_count)
        if player == self.seat:
            self.flowers += 1
            replacement_tile = _tile_value(message.get("t"))
            about_kong = player in self.next_draw_about_kong if player is not None else False
            if replacement_tile is not None and replacement_tile not in self.hand:
                self.hand.append(replacement_tile)
                self.hand.sort()
            if replacement_tile is not None:
                if player is not None:
                    self.next_draw_about_kong.discard(player)
                self.last_draw = {"seat": player, "tile": replacement_tile, "display": display_name(replacement_tile)}
                self._set_last_win_event(player, replacement_tile, "draw", True, about_kong)
            self._apply_prompt(message.get("a") or {})
        else:
            self.available_actions = {}

    def _apply_draw(self, message: dict[str, Any]) -> None:
        self._commit_pending_discard()
        player = _optional_int(message.get("v"), self.turn)
        self.turn = player
        self.wall_count = _optional_int(message.get("h"), self.wall_count)
        tile_value = _tile_value(message.get("t"))
        about_kong = player in self.next_draw_about_kong if player is not None else False
        if player is not None:
            self.next_draw_about_kong.discard(player)
        if player == self.seat and tile_value is not None:
            if tile_value not in self.hand:
                self.hand.append(tile_value)
                self.hand.sort()
            self.last_draw = {"seat": player, "tile": tile_value, "display": display_name(tile_value)}
        if player is not None and tile_value is not None:
            self._set_last_win_event(player, tile_value, "draw", True, about_kong)
        self._apply_prompt(message.get("a") or {})

    def _apply_discard(self, message: dict[str, Any]) -> None:
        self._commit_pending_discard()
        seat = _optional_int(message.get("v"), None)
        tile = _tile_value(message.get("t"))
        if tile is None:
            self.unknown_events.append(message)
            return
        self.turn = seat
        self.wall_count = _optional_int(message.get("h"), self.wall_count)
        self.discards.append({"seat": seat, "tile": tile, "display": display_name(tile)})
        self.pending_discard = {"seat": seat, "tile": tile}
        self._set_last_win_event(seat, tile, "discard", False, False)
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
        self._apply_claimed_meld_visibility(player, [base - 4, base, base + 4])
        self._apply_meld(player, pack, consumed, message.get("a") or {})

    def _apply_pung(self, message: dict[str, Any]) -> None:
        player = _optional_int(message.get("v"), self.turn)
        pack = _optional_int(message.get("p"), None)
        if not pack:
            self._apply_prompt(message.get("a") or {})
            return
        base = _pack_tile(pack)
        self._apply_claimed_meld_visibility(player, [base, base, base])
        self._apply_meld(player, pack, [base, base], message.get("a") or {})

    def _apply_mkong(self, message: dict[str, Any]) -> None:
        player = _optional_int(message.get("v"), self.turn)
        pack = _optional_int(message.get("p"), None)
        if not pack:
            self._apply_prompt(message.get("a") or {})
            return
        base = _pack_tile(pack)
        pack_type = _pack_type(pack)
        if pack_type == 3:
            self._commit_pending_discard()
            self._set_visible_count(base, 4)
            if player is not None:
                self.next_draw_about_kong.add(player)
            self._set_last_win_event(player, base, "bugang", False, True)
            consumed = [base]
        else:
            self._apply_claimed_meld_visibility(player, [base, base, base, base])
            if player is not None:
                self.next_draw_about_kong.add(player)
            consumed = [base, base, base]
        self._apply_meld(player, pack, consumed, message.get("a") or {})

    def _apply_ckong(self, message: dict[str, Any]) -> None:
        self._commit_pending_discard()
        player = _optional_int(message.get("v"), self.turn)
        pack = _optional_int(message.get("p"), None)
        if not pack:
            self._apply_prompt(message.get("a") or {})
            return
        base = _pack_tile(pack)
        if player is not None:
            self.next_draw_about_kong.add(player)
        self._apply_meld(player, pack, [base, base, base, base], message.get("a") or {})

    def _apply_meld(
        self, player: int | None, pack: int, consumed_tiles: list[int], prompt: Any = None
    ) -> None:
        self.turn = player
        if player == self.seat:
            for tile in consumed_tiles:
                self._remove_one_from_hand(tile)
            self._record_meld(pack)
        self._apply_prompt(prompt or {})

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

    def _clear_public_context(self) -> None:
        self.visible_counts.clear()
        self.pending_discard = None
        self.last_win_event = None
        self.next_draw_about_kong.clear()

    def _commit_pending_discard(self) -> None:
        if not self.pending_discard:
            return
        tile = self.pending_discard.get("tile")
        if isinstance(tile, int):
            self._add_visible_tiles([tile])
        self.pending_discard = None

    def _apply_claimed_meld_visibility(self, player: int | None, tiles: list[int]) -> None:
        if self.pending_discard and player != self.pending_discard.get("seat"):
            self.pending_discard = None
        else:
            self._commit_pending_discard()
        self._add_visible_tiles(tiles)

    def _add_visible_tiles(self, tiles: list[int]) -> None:
        for tile in tiles:
            if _is_playable_tile(tile):
                kind = tile >> 2
                self.visible_counts[kind] = min(4, self.visible_counts.get(kind, 0) + 1)

    def _set_visible_count(self, tile: int, count: int) -> None:
        if _is_playable_tile(tile):
            kind = tile >> 2
            self.visible_counts[kind] = max(self.visible_counts.get(kind, 0), min(4, count))

    def _set_last_win_event(
        self,
        seat: int | None,
        tile: int | None,
        source: str,
        is_self_draw: bool,
        is_about_kong: bool,
    ) -> None:
        if tile is None:
            self.last_win_event = None
            return
        self.last_win_event = {
            "seat": seat,
            "tile": tile,
            "source": source,
            "is_self_draw": is_self_draw,
            "is_about_kong": is_about_kong,
        }

    def _maybe_apply_result(self, message: dict[str, Any]) -> bool:
        if self.result_recorded_for_round:
            return False
        event = _optional_int(message.get("r"), None)
        if event not in RESULT_EVENT_CODES and not _has_result_fields(message):
            return False

        payload = _result_payload(message)
        scores = _result_scores(payload)
        flags = _optional_int(payload.get("b"), None)
        winner = _winner_from_flags(flags)
        discarder = _discarder_from_flags(flags)

        if winner is None:
            winner = _optional_int(payload.get("winner"), None)
        if winner is None and event in RESULT_EVENT_CODES:
            winner = _optional_int(payload.get("v"), None)
        if winner is None:
            winner = _winner_from_scores(scores)

        if discarder is None:
            discarder = _optional_int(payload.get("discarder"), None)
        if discarder is None and winner is not None and self.last_win_event:
            source = self.last_win_event.get("source")
            event_seat = _optional_int(self.last_win_event.get("seat"), None)
            if source in {"discard", "bugang"} and event_seat != winner:
                discarder = event_seat
        if discarder is None and winner is not None:
            discarder = _discarder_from_scores(scores, winner)

        is_draw = winner is None
        is_self_draw = False
        if not is_draw:
            if discarder is None or discarder == winner:
                is_self_draw = True
            if self.last_win_event:
                source = self.last_win_event.get("source")
                event_seat = _optional_int(self.last_win_event.get("seat"), None)
                if source == "draw" and event_seat == winner:
                    is_self_draw = True
                elif source in {"discard", "bugang"} and event_seat != winner:
                    is_self_draw = False

        seat = self.seat
        result = {
            "round_index": self.round_index,
            "seat": seat,
            "winner": winner,
            "discarder": discarder,
            "is_draw": is_draw,
            "is_self_draw": is_self_draw,
            "is_win": seat is not None and winner == seat,
            "is_deal_in": seat is not None and discarder == seat and winner != seat,
            "scores": scores,
            "score_delta": scores[seat] if seat is not None and scores is not None and 0 <= seat < len(scores) else None,
            "fan": _result_fan(payload),
            "source": "settlement",
            "event_count": len(self.raw_events),
        }
        self.result_history.append(result)
        self.result_recorded_for_round = True
        self.available_actions = {}
        return True

    def results_snapshot(self) -> dict[str, Any]:
        return {"stats": _result_stats(self.result_history), "history": list(self.result_history)}

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
            "visible_counts": dict(self.visible_counts),
            "last_win_event": dict(self.last_win_event) if self.last_win_event else None,
            "last_result": dict(self.result_history[-1]) if self.result_history else None,
            "result_stats": _result_stats(self.result_history),
            "result_history": list(self.result_history[-20:]),
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


def _has_result_fields(message: dict[str, Any]) -> bool:
    if any(key in message for key in ("s", "scores", "score", "b", "winner", "discarder")):
        return True
    value = message.get("v")
    return isinstance(value, dict) and any(
        key in value for key in ("s", "scores", "score", "b", "winner", "discarder")
    )


def _result_payload(message: dict[str, Any]) -> dict[str, Any]:
    payload = dict(message)
    value = message.get("v")
    if isinstance(value, dict):
        payload.update(value)
    return payload


def _result_scores(payload: dict[str, Any]) -> list[int] | None:
    for key in ("s", "scores", "score"):
        scores = payload.get(key)
        if isinstance(scores, list) and len(scores) >= 4:
            try:
                return [int(scores[index]) for index in range(4)]
            except (TypeError, ValueError):
                return None
        if isinstance(scores, dict):
            try:
                return [int(scores[str(index)]) for index in range(4)]
            except (KeyError, TypeError, ValueError):
                return None
    return None


def _winner_from_flags(flags: int | None) -> int | None:
    if flags is None:
        return None
    for seat in range(4):
        if flags & (1 << seat):
            return seat
    return None


def _discarder_from_flags(flags: int | None) -> int | None:
    if flags is None:
        return None
    for seat in range(4):
        if flags & (1 << (seat + 4)):
            return seat
    return None


def _winner_from_scores(scores: list[int] | None) -> int | None:
    if not scores:
        return None
    best = max(scores)
    if best <= 0 or scores.count(best) != 1:
        return None
    return scores.index(best)


def _discarder_from_scores(scores: list[int] | None, winner: int) -> int | None:
    if not scores:
        return None
    losers = [(seat, score) for seat, score in enumerate(scores[:4]) if seat != winner]
    if not losers:
        return None
    worst_score = min(score for _, score in losers)
    worst = [seat for seat, score in losers if score == worst_score]
    if len(worst) == 1 and len({score for _, score in losers}) > 1:
        return worst[0]
    return None


def _result_fan(payload: dict[str, Any]) -> int | None:
    for key in ("fan", "fanCnt", "h"):
        value = _optional_int(payload.get(key), None)
        if value is not None:
            return value
    wins = payload.get("y")
    if isinstance(wins, list):
        for item in wins:
            if isinstance(item, dict):
                value = _optional_int(item.get("fan") or item.get("fanCnt") or item.get("h"), None)
                if value is not None:
                    return value
    return None


def _result_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(results)
    wins = sum(1 for result in results if result.get("is_win"))
    deal_ins = sum(1 for result in results if result.get("is_deal_in"))
    draws = sum(1 for result in results if result.get("is_draw"))
    losses = games - wins - draws
    score_values = [result.get("score_delta") for result in results if result.get("score_delta") is not None]
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "deal_ins": deal_ins,
        "win_rate": wins / games if games else 0.0,
        "deal_in_rate": deal_ins / games if games else 0.0,
        "average_score_delta": sum(score_values) / len(score_values) if score_values else 0.0,
    }


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
