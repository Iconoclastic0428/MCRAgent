#!/usr/bin/env python3
"""Legal Botzone response candidates for Chinese-Standard-Mahjong."""

from __future__ import annotations

from functools import lru_cache
from collections import Counter

from hand_features import TILE_ORDER, TILE_TO_INDEX, candidate_feature_text, min_shanten


HONORS = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"}
STANDARD_KNITTED_STRAIGHTS = (
    ("W1", "W4", "W7", "B2", "B5", "B8", "T3", "T6", "T9"),
    ("W1", "W4", "W7", "B3", "B6", "B9", "T2", "T5", "T8"),
    ("W2", "W5", "W8", "B1", "B4", "B7", "T3", "T6", "T9"),
    ("W2", "W5", "W8", "B3", "B6", "B9", "T1", "T4", "T7"),
    ("W3", "W6", "W9", "B1", "B4", "B7", "T2", "T5", "T8"),
    ("W3", "W6", "W9", "B2", "B5", "B8", "T1", "T4", "T7"),
)
STANDARD_KNITTED_STRAIGHT_SETS = tuple(frozenset(seq) for seq in STANDARD_KNITTED_STRAIGHTS)


def parse_request(request: str) -> list[str]:
    return request.strip().split()


def is_suited(tile: str) -> bool:
    return len(tile) == 2 and tile[0] in {"W", "B", "T"} and tile[1].isdigit()


def request_event_tile(tokens: list[str]) -> str | None:
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2] in {"PLAY", "PENG", "CHI"}:
        return tokens[-1]
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2] == "BUGANG":
        return tokens[3]
    return None


def can_complete_hand(hand: Counter[str], tile: str | None = None, meld_count: int = 0) -> bool:
    tiles = list(hand.elements())
    if tile:
        tiles.append(tile)
    if meld_count == 0 and min_shanten(tiles) <= -1:
        return True
    if meld_count == 0 and can_complete_honors_and_knitted_tiles(tiles):
        return True
    if can_complete_knitted_straight_hand(tiles, meld_count=meld_count):
        return True
    return can_complete_regular_hand(tiles, meld_count)


def can_complete_honors_and_knitted_tiles(tiles: list[str]) -> bool:
    """Return true for 全不靠 / 七星不靠 special Hu shapes."""
    if len(tiles) != 14:
        return False
    counts = Counter(tiles)
    if any(count != 1 for count in counts.values()):
        return False
    suited = {tile for tile in counts if is_suited(tile)}
    honors = set(counts) - suited
    if not honors <= HONORS:
        return False
    if not 7 <= len(suited) <= 9:
        return False
    if not any(suited <= seq for seq in STANDARD_KNITTED_STRAIGHT_SETS):
        return False
    if len(suited) == 7:
        return honors == HONORS
    return honors <= HONORS


def can_complete_knitted_straight_hand(tiles: list[str], meld_count: int = 0) -> bool:
    """Return true for 组合龙 plus the remaining required set/pair shape."""
    if meld_count < 0 or meld_count > 1:
        return False
    expected_len = (1 - meld_count) * 3 + 2 + 9
    if len(tiles) != expected_len:
        return False
    counts = Counter(tiles)
    for seq in STANDARD_KNITTED_STRAIGHTS:
        if all(counts[tile] > 0 for tile in seq):
            remaining = counts.copy()
            for tile in seq:
                remaining[tile] -= 1
                if remaining[tile] <= 0:
                    del remaining[tile]
            if can_complete_regular_hand(list(remaining.elements()), meld_count=3 + meld_count):
                return True
    return False


def can_complete_regular_hand(tiles: list[str], meld_count: int = 0) -> bool:
    if meld_count < 0 or meld_count > 4:
        return False
    sets_needed = 4 - meld_count
    if len(tiles) != sets_needed * 3 + 2:
        return False
    counts = [0] * len(TILE_ORDER)
    for tile in tiles:
        index = TILE_TO_INDEX.get(tile)
        if index is None:
            return False
        counts[index] += 1
    return _can_complete_regular_from_counts(tuple(counts), sets_needed, False)


@lru_cache(maxsize=200_000)
def _can_complete_regular_from_counts(
    counts: tuple[int, ...], sets_needed: int, pair_used: bool
) -> bool:
    try:
        index = next(i for i, count in enumerate(counts) if count)
    except StopIteration:
        return sets_needed == 0 and pair_used

    counts_list = list(counts)
    if not pair_used and counts_list[index] >= 2:
        counts_list[index] -= 2
        if _can_complete_regular_from_counts(tuple(counts_list), sets_needed, True):
            return True
        counts_list[index] += 2

    if sets_needed > 0 and counts_list[index] >= 3:
        counts_list[index] -= 3
        if _can_complete_regular_from_counts(tuple(counts_list), sets_needed - 1, pair_used):
            return True
        counts_list[index] += 3

    if (
        sets_needed > 0
        and 0 <= index < 27
        and index % 9 <= 6
        and counts_list[index + 1] > 0
        and counts_list[index + 2] > 0
    ):
        counts_list[index] -= 1
        counts_list[index + 1] -= 1
        counts_list[index + 2] -= 1
        if _can_complete_regular_from_counts(tuple(counts_list), sets_needed - 1, pair_used):
            return True
    return False


def generate_legal_responses(
    player_id: int | None, request: str, hand: Counter[str], meld_count: int = 0
) -> list[str]:
    tokens = parse_request(request)
    if not tokens:
        return ["PASS"]
    if tokens[0] in {"0", "1"}:
        return ["PASS"]
    if tokens[0] == "2" and len(tokens) >= 2:
        return generate_draw_responses(hand, meld_count=meld_count)
    if tokens[0] == "3" and len(tokens) >= 3:
        return generate_reaction_responses(player_id, tokens, hand, meld_count=meld_count)
    return ["PASS"]


def generate_draw_responses(hand: Counter[str], meld_count: int = 0) -> list[str]:
    responses = ["PASS"] if not hand else []
    if can_complete_hand(hand, meld_count=meld_count):
        responses.append("HU")
    for tile in sorted(tile for tile, count in hand.items() if count > 0):
        responses.append(f"PLAY {tile}")
    for tile in sorted(tile for tile, count in hand.items() if count >= 4):
        responses.append(f"GANG {tile}")
    return dedupe(responses)


def generate_reaction_responses(
    player_id: int | None, tokens: list[str], hand: Counter[str], meld_count: int = 0
) -> list[str]:
    responses = ["PASS"]
    event_tile = request_event_tile(tokens)
    if not event_tile:
        return responses
    if player_id is not None and len(tokens) >= 2 and player_id == int(tokens[1]):
        return responses

    if can_complete_hand(hand, event_tile, meld_count=meld_count):
        responses.append("HU")

    if tokens[2] in {"PLAY", "PENG", "CHI"}:
        if hand[event_tile] >= 3:
            responses.append("GANG")
        if hand[event_tile] >= 2:
            after_peng = Counter(hand)
            after_peng[event_tile] -= 2
            cleanup(after_peng)
            for discard in sorted(tile for tile, count in after_peng.items() if count > 0):
                responses.append(f"PENG {discard}")
        if can_chi(player_id, int(tokens[1]), event_tile):
            responses.extend(generate_chi_responses(event_tile, hand))
    return dedupe(responses)


def can_chi(player_id: int | None, actor_id: int, event_tile: str) -> bool:
    return player_id is not None and player_id == (actor_id + 1) % 4 and is_suited(event_tile)


def generate_chi_responses(event_tile: str, hand: Counter[str]) -> list[str]:
    suit = event_tile[0]
    rank = int(event_tile[1])
    responses: list[str] = []
    for middle_rank in range(rank - 1, rank + 2):
        if middle_rank < 2 or middle_rank > 8:
            continue
        sequence = [f"{suit}{middle_rank - 1}", f"{suit}{middle_rank}", f"{suit}{middle_rank + 1}"]
        needed = Counter(sequence)
        needed[event_tile] -= 1
        if all(hand[tile] >= count for tile, count in needed.items() if count > 0):
            after_chi = Counter(hand)
            for tile, count in needed.items():
                after_chi[tile] -= count
            cleanup(after_chi)
            for discard in sorted(tile for tile, count in after_chi.items() if count > 0):
                responses.append(f"CHI {suit}{middle_rank} {discard}")
    return responses


def cleanup(hand: Counter[str]) -> None:
    for tile in list(hand):
        if hand[tile] <= 0:
            del hand[tile]


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def apply_response(hand: Counter[str], request: str, response: str) -> None:
    tokens = parse_request(request)
    parts = response.strip().split()
    if not tokens or not parts:
        return
    action = parts[0].upper()
    if tokens[0] == "2" and len(tokens) >= 2:
        if action == "PLAY" and len(parts) == 2:
            hand[parts[1]] -= 1
        elif action == "GANG" and len(parts) == 2:
            hand[parts[1]] -= 4
        elif action == "BUGANG" and len(parts) == 2:
            hand[parts[1]] -= 1
    elif tokens[0] == "3":
        event_tile = request_event_tile(tokens)
        if action == "PENG" and event_tile and len(parts) == 2:
            hand[event_tile] -= 2
            hand[parts[1]] -= 1
        elif action == "GANG" and event_tile:
            hand[event_tile] -= 3
        elif action == "CHI" and event_tile and len(parts) == 3:
            middle = parts[1]
            sequence = [f"{middle[0]}{int(middle[1]) - 1}", middle, f"{middle[0]}{int(middle[1]) + 1}"]
            needed = Counter(sequence)
            needed[event_tile] -= 1
            for tile, count in needed.items():
                hand[tile] -= count
            hand[parts[2]] -= 1
    cleanup(hand)


def response_candidate_text(input_text: str, response: str, hand: Counter[str], request: str) -> str:
    parts = response.split()
    action = parts[0] if parts else "PASS"
    discard = response_discard_tile(response)
    features = [
        input_text,
        f"RESP {response}",
        f"ACTION {action}",
    ]
    if discard:
        drawn = parse_request(request)[1] if request.startswith("2 ") else None
        features.append(candidate_feature_text(list(hand.elements()), discard, drawn_tile=drawn))
    else:
        features.append(f"HAND_MIN_SHANTEN {min_shanten(hand.elements())}")
    return "\n".join(features)


def response_discard_tile(response: str) -> str | None:
    parts = response.split()
    if len(parts) == 2 and parts[0] in {"PLAY", "PENG"}:
        return parts[1]
    if len(parts) == 3 and parts[0] == "CHI":
        return parts[2]
    return None
