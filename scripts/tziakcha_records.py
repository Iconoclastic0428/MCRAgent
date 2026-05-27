#!/usr/bin/env python3
"""Convert decoded Tziakcha record JSON into Botzone-like raw JSONL."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path
from typing import Iterable


BOTZONE_SYMBOLS = (
    [f"W{rank}" for rank in range(1, 10)]
    + [f"B{rank}" for rank in range(1, 10)]
    + [f"T{rank}" for rank in range(1, 10)]
    + [f"F{rank}" for rank in range(1, 5)]
    + [f"J{rank}" for rank in range(1, 4)]
)


def tziakcha_kind_to_botzone_kind(kind: int) -> int:
    if 9 <= kind <= 17:
        return kind + 9
    if 18 <= kind <= 26:
        return kind - 9
    return kind


def tile_id_to_botzone_symbol(tile_id: int) -> str:
    if not isinstance(tile_id, int):
        raise TypeError("tile_id must be int")
    if tile_id < 0 or tile_id > 147:
        raise ValueError(f"tile_id out of range: {tile_id}")
    if tile_id >= 136:
        return f"H{tile_id - 135}"
    botzone_kind = tziakcha_kind_to_botzone_kind(tile_id >> 2)
    return BOTZONE_SYMBOLS[botzone_kind]


def wall_from_hex(wall_hex: str) -> list[int]:
    if len(wall_hex) % 2:
        raise ValueError("wall hex length must be even")
    return [int(wall_hex[index : index + 2], 16) for index in range(0, len(wall_hex), 2)]


def dice_from_int(value: int) -> list[int]:
    return [value & 15, (value >> 4) & 15, (value >> 8) & 15, (value >> 12) & 15]


def action_player(combined: int) -> int:
    return (combined >> 4) & 3


def action_type(combined: int) -> int:
    return combined & 15


def parsed_actions(step: dict) -> list[dict]:
    actions = []
    for raw in step.get("a") or []:
        if not isinstance(raw, list) or len(raw) < 3:
            continue
        combined = int(raw[0])
        actions.append(
            {
                "player": action_player(combined),
                "type": action_type(combined),
                "data": int(raw[1]),
                "time": int(raw[2]),
                "raw": raw,
            }
        )
    return actions


def decode_script(encoded: str) -> dict:
    if not encoded or encoded == "<Decoded>":
        raise ValueError("record script is not encoded")
    padded = encoded + ("=" * ((4 - len(encoded) % 4) % 4))
    try:
        compressed = base64.b64decode(padded)
        raw = zlib.decompress(compressed).replace(b"\x00", b"")
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to decode Tziakcha script: {exc}") from exc


def record_step(record: dict) -> dict:
    step = record.get("step")
    if isinstance(step, dict):
        return step
    script = record.get("script")
    if isinstance(script, str):
        return decode_script(script)
    raise ValueError("record must contain decoded step data or encoded script")


def setup_wall_and_deal(step: dict) -> tuple[list[list[int]], list[int]]:
    if "w" not in step:
        raise ValueError("decoded Tziakcha record is missing step.w wall data")
    wall_indices = wall_from_hex(str(step["w"]))
    if len(wall_indices) < 54:
        raise ValueError("decoded Tziakcha record step.w does not contain enough tiles")
    dice = dice_from_int(int(step.get("d", 0)))
    dealer = 0
    wall_break_pos = (dealer - (dice[0] + dice[1] - 1) + 12) % 4
    start_pos = (wall_break_pos * 36 + sum(dice) * 2) % len(wall_indices)
    wall = wall_indices[start_pos:] + wall_indices[:start_pos]

    hands = [[] for _ in range(4)]
    front = 0
    for _ in range(3):
        for offset in range(4):
            player = (dealer + offset) % 4
            hands[player].extend(wall[front : front + 4])
            front += 4
    for offset in range(4):
        player = (dealer + offset) % 4
        hands[player].append(wall[front])
        front += 1
    hands[dealer].append(wall[front])
    front += 1
    for hand in hands:
        hand.sort()
    return hands, wall


def remove_one_matching(hand: list[int], tile_id: int) -> int | None:
    try:
        hand.remove(tile_id)
        return tile_id
    except ValueError:
        pass
    tile_kind = tile_id >> 2
    for existing in list(hand):
        if existing >> 2 == tile_kind:
            hand.remove(existing)
            return existing
    return None


def remove_kind(hand: list[int], tile_id: int, count: int) -> None:
    for _ in range(count):
        if remove_one_matching(hand, tile_id) is None:
            return


def append_kind(hand: list[int], tile_id: int) -> None:
    hand.append(tile_id)
    hand.sort()


def hand_symbols(hand: Iterable[int]) -> list[str]:
    return [tile_id_to_botzone_symbol(tile_id) for tile_id in sorted(hand)]


def make_log(content: dict[str, str], responses: dict[str, str], display: dict | None = None) -> list[dict]:
    return [
        {
            "output": {
                "command": "request",
                "content": content,
                "display": display or {},
            }
        },
        {
            player: {"response": response, "raw": response, "verdict": "OK"}
            for player, response in responses.items()
        },
    ]


def scores_from_step(step: dict) -> dict[str, int]:
    scores = step.get("s")
    if isinstance(scores, list) and len(scores) >= 4:
        return {str(player): int(scores[player]) for player in range(4)}
    return {str(player): 0 for player in range(4)}


def round_wind(step: dict) -> int:
    return (int(step.get("i", 0)) // 4) % 4


def apply_flower(hands: list[list[int]], flower_counts: list[int], player: int, data: int) -> int:
    replacement = data & 0xFF
    flower = ((data >> 8) & 15) + 136
    remove_one_matching(hands[player], flower)
    append_kind(hands[player], replacement)
    flower_counts[player] += 1
    return replacement


def chi_tiles(data: int, offer_tile: int | None = None) -> list[int]:
    tile_val = (data & 0x3F) << 2
    if offer_tile is not None and tile_val - 4 + ((data >> 10) & 3) < 0:
        tile_val = offer_tile
    return [
        tile_val - 4 + ((data >> 10) & 3),
        tile_val + ((data >> 12) & 3),
        tile_val + 4 + ((data >> 14) & 3),
    ]


def meld_tile(data: int) -> int:
    return ((data & 0x3F) << 2) + ((data >> 10) & 3)


def is_promoted_gang(data: int) -> bool:
    return (data & 0x0300) == 0x0300


def event_request(event: dict) -> str:
    actor = event["actor"]
    action = event["action"]
    tile = tile_id_to_botzone_symbol(event["tile_id"])
    if action == "CHI":
        return f"3 {actor} CHI {event['middle']} {tile}"
    return f"3 {actor} {action} {tile}"


class TziakchaLogBuilder:
    def __init__(self, record: dict):
        self.record = record
        self.step = record_step(record)
        self.actions = parsed_actions(self.step)
        self.hands, _ = setup_wall_and_deal(self.step)
        self.flower_counts = [0, 0, 0, 0]
        self.logs: list[dict] = []
        self.pending_event: dict | None = None
        self.pending_draws: dict[int, int] = {}
        self.initial_dealer_draw: int | None = None

    def convert(self) -> dict:
        start_index = self._apply_prestart_actions()
        post_actions = self.actions[start_index:]
        self.initial_dealer_draw = self._choose_initial_dealer_draw(post_actions)
        self._emit_initial_logs()
        self._process_actions(post_actions)
        self._flush_event()
        return {
            "match_id": str(self.record.get("id") or self.step.get("r") or ""),
            "belongs": self.record.get("belongs"),
            "game": "Chinese-Standard-Mahjong",
            "source": "tziakcha_record_miner",
            "scores": scores_from_step(self.step),
            "log_count": len(self.logs),
            "turn_count": len(self.logs) // 2,
            "logs": self.logs,
        }

    def _apply_prestart_actions(self) -> int:
        for index, action in enumerate(self.actions):
            if action["type"] == 0:
                return index + 1
            if action["type"] == 1:
                apply_flower(self.hands, self.flower_counts, action["player"], action["data"])
        return 0

    def _choose_initial_dealer_draw(self, actions: list[dict]) -> int | None:
        if not self.hands[0]:
            return None
        for action in actions:
            if action["player"] == 0 and action["type"] == 2:
                tile = action["data"] & 0xFF
                if any(existing >> 2 == tile >> 2 for existing in self.hands[0]):
                    return tile
                break
            if action["type"] == 7 and action["player"] == 0:
                break
        return self.hands[0][-1]

    def _emit_initial_logs(self) -> None:
        wind = round_wind(self.step)
        init_content = {str(player): f"0 {player} {wind}" for player in range(4)}
        self.logs.extend(make_log(init_content, {str(player): "PASS" for player in range(4)}, {"action": "INIT"}))

        deal_content = {}
        for player in range(4):
            deal_hand = list(self.hands[player])
            if player == 0 and self.initial_dealer_draw is not None:
                remove_one_matching(deal_hand, self.initial_dealer_draw)
            tiles = " ".join(hand_symbols(deal_hand[:13]))
            prefix = " ".join(str(count) for count in self.flower_counts)
            deal_content[str(player)] = f"1 {prefix} {tiles}".strip()
        self.logs.extend(make_log(deal_content, {str(player): "PASS" for player in range(4)}, {"action": "DEAL"}))

    def _process_actions(self, actions: list[dict]) -> None:
        index = 0
        while index < len(actions):
            action = actions[index]
            player = action["player"]
            kind = action["type"]
            data = action["data"]

            if kind in {8, 9}:
                index += 1
                continue

            if self.pending_event and kind not in {3, 4, 5, 6}:
                self._flush_event()

            if kind == 1:
                replacement = apply_flower(self.hands, self.flower_counts, player, data)
                self.pending_draws[player] = replacement
                index += 1
            elif kind == 2:
                self._handle_discard(player, data & 0xFF)
                index += 1
            elif kind == 3:
                index = self._handle_chi(actions, index)
            elif kind == 4:
                index = self._handle_peng(actions, index)
            elif kind == 5:
                self._handle_gang(player, data)
                index += 1
            elif kind == 6:
                self._handle_hu(player, data >> 1)
                index += 1
            elif kind == 7:
                self._flush_event()
                tile = data & 0xFF
                append_kind(self.hands[player], tile)
                self.pending_draws[player] = tile
                index += 1
            else:
                index += 1

    def _emit_draw_response(self, player: int, draw_tile: int, response: str) -> None:
        tile = tile_id_to_botzone_symbol(draw_tile)
        content = {
            str(other): (f"2 {tile}" if other == player else f"3 {player} DRAW")
            for other in range(4)
        }
        responses = {str(other): "PASS" for other in range(4)}
        responses[str(player)] = response
        self.logs.extend(
            make_log(
                content,
                responses,
                {"action": "DRAW", "player": player, "tile": tile, "canHu": [-4, -4, -4, -4]},
            )
        )

    def _draw_tile_for_response(self, player: int) -> int | None:
        if player in self.pending_draws:
            return self.pending_draws.pop(player)
        if player == 0 and self.initial_dealer_draw is not None:
            tile = self.initial_dealer_draw
            self.initial_dealer_draw = None
            return tile
        return None

    def _handle_discard(self, player: int, tile_id: int) -> None:
        draw_tile = self._draw_tile_for_response(player)
        if draw_tile is not None:
            self._emit_draw_response(player, draw_tile, f"PLAY {tile_id_to_botzone_symbol(tile_id)}")
        remove_one_matching(self.hands[player], tile_id)
        self.pending_event = {"actor": player, "action": "PLAY", "tile_id": tile_id}

    def _handle_chi(self, actions: list[dict], index: int) -> int:
        action = actions[index]
        player = action["player"]
        if not self.pending_event:
            return index + 1
        follow = self._following_discard(actions, index, player)
        if follow is None:
            return index + 1
        discard_id = follow["data"] & 0xFF
        tiles = chi_tiles(action["data"], self.pending_event["tile_id"])
        middle = tile_id_to_botzone_symbol(tiles[1])
        response = f"CHI {middle} {tile_id_to_botzone_symbol(discard_id)}"
        event_tile = self.pending_event["tile_id"]
        self._flush_event(player, response)
        offered_kind = event_tile >> 2
        for tile in tiles:
            if tile >> 2 != offered_kind:
                remove_kind(self.hands[player], tile, 1)
        remove_one_matching(self.hands[player], discard_id)
        self.pending_event = {
            "actor": player,
            "action": "CHI",
            "tile_id": discard_id,
            "middle": middle,
        }
        return index + 2

    def _handle_peng(self, actions: list[dict], index: int) -> int:
        action = actions[index]
        player = action["player"]
        if not self.pending_event:
            return index + 1
        follow = self._following_discard(actions, index, player)
        if follow is None:
            return index + 1
        discard_id = follow["data"] & 0xFF
        response = f"PENG {tile_id_to_botzone_symbol(discard_id)}"
        event_tile = self.pending_event["tile_id"]
        self._flush_event(player, response)
        remove_kind(self.hands[player], event_tile, 2)
        remove_one_matching(self.hands[player], discard_id)
        self.pending_event = {"actor": player, "action": "PENG", "tile_id": discard_id}
        return index + 2

    def _handle_gang(self, player: int, data: int) -> None:
        tile = meld_tile(data)
        draw_tile = self._draw_tile_for_response(player)
        if draw_tile is not None:
            action_name = "BUGANG" if is_promoted_gang(data) else "GANG"
            self._emit_draw_response(player, draw_tile, f"{action_name} {tile_id_to_botzone_symbol(tile)}")
            remove_kind(self.hands[player], tile, 1 if action_name == "BUGANG" else 4)
            if action_name == "BUGANG":
                self.pending_event = {"actor": player, "action": "BUGANG", "tile_id": tile}
            return
        if self.pending_event:
            event_tile = self.pending_event["tile_id"]
            self._flush_event(player, "GANG")
            remove_kind(self.hands[player], event_tile, 3)
            return
        remove_kind(self.hands[player], tile, 4)

    def _handle_hu(self, player: int, fan: int) -> None:
        if fan < 8:
            draw_tile = self._draw_tile_for_response(player)
            if draw_tile is not None:
                self._emit_draw_response(player, draw_tile, "PASS")
            return
        draw_tile = self._draw_tile_for_response(player)
        if draw_tile is not None:
            self._emit_draw_response(player, draw_tile, "HU")
        elif self.pending_event:
            self._flush_event(player, "HU")

    def _following_discard(self, actions: list[dict], index: int, player: int) -> dict | None:
        if index + 1 >= len(actions):
            return None
        follow = actions[index + 1]
        if follow["player"] == player and follow["type"] == 2:
            return follow
        return None

    def _flush_event(self, response_player: int | None = None, response: str = "PASS") -> None:
        if not self.pending_event:
            return
        request = event_request(self.pending_event)
        content = {str(player): request for player in range(4)}
        responses = {str(player): "PASS" for player in range(4)}
        if response_player is not None:
            responses[str(response_player)] = response
        display = {
            "action": self.pending_event["action"],
            "player": self.pending_event["actor"],
            "tile": tile_id_to_botzone_symbol(self.pending_event["tile_id"]),
            "canHu": [-4, -4, -4, -4],
        }
        if self.pending_event["action"] == "CHI":
            display["tileCHI"] = self.pending_event["middle"]
        self.logs.extend(make_log(content, responses, display))
        self.pending_event = None


def convert_record(record: dict) -> dict:
    return TziakchaLogBuilder(record).convert()


def iter_input_records(path: Path) -> Iterable[dict]:
    if path.is_dir():
        for child in sorted(path.glob("*.json")):
            yield json.loads(child.read_text(encoding="utf-8"))
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    yield json.loads(line)
        return
    yield json.loads(path.read_text(encoding="utf-8"))


def convert_path(in_path: Path, out_path: Path, limit: int | None = None) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    written = 0
    errors: list[dict] = []
    with out_path.open("w", encoding="utf-8") as out:
        for record in iter_input_records(in_path):
            seen += 1
            if limit is not None and written >= limit:
                break
            try:
                converted = convert_record(record)
            except Exception as exc:
                errors.append({"index": seen - 1, "id": record.get("id"), "error": str(exc)})
                continue
            out.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    return {
        "input": str(in_path),
        "output": str(out_path),
        "records_seen": seen,
        "records_written": written,
        "error_count": len(errors),
        "errors": errors[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = convert_path(Path(args.in_path), Path(args.out), args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
