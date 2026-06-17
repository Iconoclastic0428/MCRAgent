#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import json
from collections import Counter


TUIBUDAO_TILES = set("B1 B2 B3 B4 B5 B8 B9 T2 T4 T5 T6 T8 T9 J3".split())
LVYISE_TILES = set("T2 T3 T4 T6 T8 J2".split())


def remove_tiles(hand, tiles):
    ok = True
    for tile in tiles:
        if hand[tile] <= 0:
            ok = False
        else:
            hand[tile] -= 1
    return ok


def hand_to_list(hand):
    tiles = []
    for tile, count in hand.items():
        tiles.extend([tile] * count)
    return tiles


def chi_tiles(center_tile):
    suit = center_tile[0]
    rank = int(center_tile[1])
    return [f"{suit}{rank - 1}", center_tile, f"{suit}{rank + 1}"]


def new_match(match_index, match_id):
    return {
        "index": match_index,
        "id": match_id,
        "hands": [Counter() for _ in range(4)],
        "packs": [[] for _ in range(4)],
        "last_discard": None,
        "last_discard_player": None,
        "last_event": None,
        "winner": None,
        "winning_tiles": [],
        "warnings": [],
    }


def full_winning_tiles(state, player):
    tiles = hand_to_list(state["hands"][player])
    for pack in state["packs"][player]:
        tiles.extend(pack)

    last_event = state["last_event"]
    if (
        state["last_discard"] is not None
        and state["last_discard_player"] != player
        and last_event == "Play"
    ):
        tiles.append(state["last_discard"])
    elif (
        last_event == "BuGang"
        and state["last_discard"] is not None
        and state["last_discard_player"] != player
    ):
        tiles.append(state["last_discard"])

    return tiles


def classify_tiles(tiles):
    tile_set = set(tiles)
    fans = []
    if tiles and tile_set <= LVYISE_TILES:
        fans.append("lvyise")
    if tiles and tile_set <= TUIBUDAO_TILES:
        fans.append("tuibudao")
    return fans


def parse_transcript(path):
    matches = []
    stats = Counter()
    state = None
    match_index = -1

    with open(path, "r", encoding="utf8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()

            if parts[0] == "Match":
                match_index += 1
                match_id = parts[1] if len(parts) > 1 else str(match_index)
                state = new_match(match_index, match_id)
                stats["matches_seen"] += 1
                continue

            if state is None:
                continue

            if parts[0] == "Wind":
                continue

            if parts[0] == "Score":
                if state["winner"] is not None:
                    fans = classify_tiles(state["winning_tiles"])
                    if fans:
                        matches.append(
                            {
                                "index": state["index"],
                                "id": state["id"],
                                "winner": state["winner"],
                                "fans": fans,
                                "winning_tiles": sorted(state["winning_tiles"]),
                                "line": line_number,
                            }
                        )
                        for fan in fans:
                            stats[f"{fan}_matches"] += 1
                    stats["hu_matches"] += 1
                else:
                    stats["huang_matches"] += 1
                state = None
                continue

            if parts[0] != "Player" or len(parts) < 3:
                continue

            player = int(parts[1])
            action = parts[2]
            hand = state["hands"][player]

            if action == "Deal":
                state["hands"][player] = Counter(parts[3:])
                state["last_event"] = "Deal"
            elif action == "Draw":
                if len(parts) >= 4:
                    hand[parts[3]] += 1
                    state["last_discard"] = parts[3]
                    state["last_discard_player"] = player
                state["last_event"] = "Draw"
            elif action == "Play":
                if len(parts) >= 4:
                    tile = parts[3]
                    if not remove_tiles(hand, [tile]):
                        state["warnings"].append(f"line {line_number}: missing play tile {tile}")
                    state["last_discard"] = tile
                    state["last_discard_player"] = player
                state["last_event"] = "Play"
            elif action == "Chi" and len(parts) >= 4:
                center = parts[3]
                sequence = chi_tiles(center)
                claimed = state["last_discard"]
                remove_from_hand = list(sequence)
                if claimed in remove_from_hand:
                    remove_from_hand.remove(claimed)
                if not remove_tiles(hand, remove_from_hand):
                    state["warnings"].append(f"line {line_number}: missing chi tiles {remove_from_hand}")
                state["packs"][player].append(sequence)
                state["last_discard"] = None
                state["last_discard_player"] = None
                state["last_event"] = "Chi"
            elif action == "Peng" and len(parts) >= 4:
                tile = parts[3]
                if not remove_tiles(hand, [tile, tile]):
                    state["warnings"].append(f"line {line_number}: missing peng tiles {tile}")
                state["packs"][player].append([tile, tile, tile])
                state["last_discard"] = None
                state["last_discard_player"] = None
                state["last_event"] = "Peng"
            elif action == "Gang" and len(parts) >= 4:
                tile = parts[3]
                if not remove_tiles(hand, [tile, tile, tile]):
                    state["warnings"].append(f"line {line_number}: missing gang tiles {tile}")
                state["packs"][player].append([tile, tile, tile, tile])
                state["last_discard"] = None
                state["last_discard_player"] = None
                state["last_event"] = "Gang"
            elif action == "AnGang" and len(parts) >= 4:
                tile = parts[3]
                if not remove_tiles(hand, [tile, tile, tile, tile]):
                    state["warnings"].append(f"line {line_number}: missing angang tiles {tile}")
                state["packs"][player].append([tile, tile, tile, tile])
                state["last_event"] = "AnGang"
            elif action == "BuGang" and len(parts) >= 4:
                tile = parts[3]
                if not remove_tiles(hand, [tile]):
                    state["warnings"].append(f"line {line_number}: missing bugang tile {tile}")
                upgraded = False
                for pack in state["packs"][player]:
                    if len(pack) == 3 and all(pack_tile == tile for pack_tile in pack):
                        pack.append(tile)
                        upgraded = True
                        break
                if not upgraded:
                    state["packs"][player].append([tile, tile, tile, tile])
                state["last_discard"] = tile
                state["last_discard_player"] = player
                state["last_event"] = "BuGang"
            elif action == "Hu":
                state["winner"] = player
                state["winning_tiles"] = full_winning_tiles(state, player)
                state["last_event"] = "Hu"

    return matches, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find feature-agent transcript matches whose winning hand is lvyise or tuibudao."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    matches, stats = parse_transcript(args.input)
    lvyise_indices = sorted(match["index"] for match in matches if "lvyise" in match["fans"])
    tuibudao_indices = sorted(match["index"] for match in matches if "tuibudao" in match["fans"])
    special_indices = sorted(set(lvyise_indices) | set(tuibudao_indices))
    payload = {
        "input": args.input,
        "matches_seen": stats["matches_seen"],
        "hu_matches": stats["hu_matches"],
        "huang_matches": stats["huang_matches"],
        "lvyise_match_indices": lvyise_indices,
        "tuibudao_match_indices": tuibudao_indices,
        "special_match_indices": special_indices,
        "special_match_count": len(special_indices),
        "special_matches": matches,
        "tile_sets": {
            "lvyise": sorted(LVYISE_TILES),
            "tuibudao": sorted(TUIBUDAO_TILES),
        },
    }
    with open(args.output, "w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
