#!/usr/bin/env python3
"""Numeric legal-candidate features for Botzone MCR rankers."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Iterable

import numpy as np

from hand_features import (
    TILE_ORDER,
    min_shanten,
    regular_shanten,
    remove_one_tile,
    seven_pairs_shanten,
    thirteen_orphans_shanten,
)
from legal_actions import parse_request, request_event_tile, response_discard_tile


ACTIONS = ("PASS", "HU", "PLAY", "GANG", "BUGANG", "PENG", "CHI")
REQUEST_TYPES = ("init", "deal", "draw", "reaction", "other")
EVENT_ACTIONS = ("NONE", "PLAY", "PENG", "CHI", "GANG", "BUGANG")


@lru_cache(maxsize=1)
def feature_names() -> list[str]:
    return [
        *[f"request_{name}" for name in REQUEST_TYPES],
        *[f"event_{name}" for name in EVENT_ACTIONS],
        *[f"action_{name}" for name in ACTIONS],
        *[f"hand_{tile}" for tile in TILE_ORDER],
        *[f"discard_{tile}" for tile in TILE_ORDER],
        *[f"claim_{tile}" for tile in TILE_ORDER],
        *[f"drawn_{tile}" for tile in TILE_ORDER],
        "hand_total",
        "candidate_tile_count",
        "current_regular_shanten",
        "current_seven_shanten",
        "current_orphan_shanten",
        "current_min_shanten",
        "after_regular_shanten",
        "after_seven_shanten",
        "after_orphan_shanten",
        "after_min_shanten",
        "delta_min_shanten",
        "candidate_is_drawn",
        "legal_count",
    ]


def response_action(response: str) -> str:
    parts = response.strip().split()
    action = parts[0].upper() if parts else "PASS"
    return action if action in ACTIONS else "PASS"


def request_type(tokens: list[str]) -> str:
    if not tokens:
        return "other"
    return {"0": "init", "1": "deal", "2": "draw", "3": "reaction"}.get(tokens[0], "other")


def event_action(tokens: list[str]) -> str:
    if len(tokens) >= 3 and tokens[0] == "3":
        action = tokens[2].upper()
        return action if action in EVENT_ACTIONS else "NONE"
    return "NONE"


def as_counter(hand: Counter[str] | dict[str, int] | Iterable[str]) -> Counter[str]:
    if isinstance(hand, Counter):
        return Counter(hand)
    if isinstance(hand, dict):
        return Counter({tile: int(count) for tile, count in hand.items()})
    return Counter(hand)


def set_one_hot(values: list[float], names: list[str], prefix: str, key: str | None) -> None:
    if key is None:
        return
    name = f"{prefix}_{key}"
    try:
        values[names.index(name)] = 1.0
    except ValueError:
        return


def hand_shanten_values(tiles: list[str]) -> tuple[float, float, float, float]:
    regular = float(regular_shanten(tiles))
    seven = float(seven_pairs_shanten(tiles))
    orphan = float(thirteen_orphans_shanten(tiles))
    return regular, seven, orphan, float(min(regular, seven, orphan))


def featurize_response(
    input_text: str,
    request: str,
    response: str,
    hand: Counter[str] | dict[str, int] | Iterable[str],
    legal_count: int | None = None,
) -> list[float]:
    names = feature_names()
    values = [0.0] * len(names)
    hand_counts = as_counter(hand)
    tokens = parse_request(request)
    action = response_action(response)
    discard = response_discard_tile(response)
    claim_tile = request_event_tile(tokens)
    drawn_tile = tokens[1] if len(tokens) >= 2 and tokens[0] == "2" else None

    set_one_hot(values, names, "request", request_type(tokens))
    set_one_hot(values, names, "event", event_action(tokens))
    set_one_hot(values, names, "action", action)
    set_one_hot(values, names, "discard", discard)
    set_one_hot(values, names, "claim", claim_tile)
    set_one_hot(values, names, "drawn", drawn_tile)

    for tile, count in hand_counts.items():
        if tile in TILE_ORDER:
            values[names.index(f"hand_{tile}")] = min(float(count), 4.0) / 4.0

    hand_tiles = list(hand_counts.elements())
    current_regular, current_seven, current_orphan, current_min = hand_shanten_values(hand_tiles)
    if discard:
        after_tiles = remove_one_tile(hand_tiles, discard)
    else:
        after_tiles = hand_tiles
    after_regular, after_seven, after_orphan, after_min = hand_shanten_values(after_tiles)

    scalar_values = {
        "hand_total": min(float(sum(hand_counts.values())), 14.0) / 14.0,
        "candidate_tile_count": min(float(hand_counts[discard]), 4.0) / 4.0 if discard else 0.0,
        "current_regular_shanten": current_regular,
        "current_seven_shanten": current_seven,
        "current_orphan_shanten": current_orphan,
        "current_min_shanten": current_min,
        "after_regular_shanten": after_regular,
        "after_seven_shanten": after_seven,
        "after_orphan_shanten": after_orphan,
        "after_min_shanten": after_min,
        "delta_min_shanten": after_min - current_min,
        "candidate_is_drawn": 1.0 if discard and discard == drawn_tile else 0.0,
        "legal_count": min(float(legal_count or 0), 32.0) / 32.0,
    }
    for name, value in scalar_values.items():
        values[names.index(name)] = value
    return values


def featurize_candidate(item: dict, legal_count: int | None = None) -> list[float]:
    return featurize_response(
        input_text=str(item.get("input_text", "")),
        request=str(item.get("request", "")),
        response=str(item.get("candidate_response", "")),
        hand=item.get("hand", Counter()),
        legal_count=legal_count,
    )


def featurize_candidates(items: list[dict]) -> np.ndarray:
    group_sizes: dict[tuple[str, int, int], int] = {}
    for item in items:
        key = (str(item.get("match_id")), int(item.get("player", 0)), int(item.get("turn_index", 0)))
        group_sizes[key] = group_sizes.get(key, 0) + 1
    rows = []
    for item in items:
        key = (str(item.get("match_id")), int(item.get("player", 0)), int(item.get("turn_index", 0)))
        rows.append(featurize_candidate(item, legal_count=group_sizes[key]))
    return np.asarray(rows, dtype=np.float32)


def featurize_legal_responses(
    input_text: str,
    request: str,
    hand: Counter[str],
    responses: list[str],
) -> np.ndarray:
    rows = [
        featurize_response(input_text, request, response, hand, legal_count=len(responses))
        for response in responses
    ]
    return np.asarray(rows, dtype=np.float32)
