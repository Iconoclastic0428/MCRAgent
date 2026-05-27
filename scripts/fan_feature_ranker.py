#!/usr/bin/env python3
"""Fan-potential numeric features for legal MCR response ranking."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Iterable

import numpy as np

from feature_ranker import (
    as_counter,
    feature_names,
    featurize_response,
)
from hand_features import (
    DRAGONS,
    ORPHANS,
    SUITS,
    TILE_ORDER,
    TILE_TO_INDEX,
    WINDS,
    regular_shanten,
    remove_one_tile,
    seven_pairs_shanten,
    thirteen_orphans_shanten,
)
from legal_actions import response_discard_tile


FAN_FEATURES = [
    "fan_current_max_suit_fraction",
    "fan_after_max_suit_fraction",
    "fan_delta_max_suit_fraction",
    "fan_current_suit_diversity",
    "fan_after_suit_diversity",
    "fan_after_honor_fraction",
    "fan_after_terminal_honor_fraction",
    "fan_after_orphan_unique_fraction",
    "fan_after_pair_count",
    "fan_after_triplet_count",
    "fan_after_dragon_count_fraction",
    "fan_after_wind_count_fraction",
    "fan_after_regular_progress",
    "fan_after_seven_pair_progress",
    "fan_after_orphan_progress",
]


@lru_cache(maxsize=1)
def fan_feature_names() -> list[str]:
    return [*feature_names(), *FAN_FEATURES]


def _suit_counts(tiles: Iterable[str]) -> dict[str, int]:
    counts = {suit: 0 for suit in SUITS}
    for tile in tiles:
        if tile and tile[0] in counts:
            counts[tile[0]] += 1
    return counts


def _is_terminal(tile: str) -> bool:
    return len(tile) >= 2 and tile[0] in SUITS and tile[1:] in {"1", "9"}


def _is_honor(tile: str) -> bool:
    return tile in WINDS or tile in DRAGONS


def _is_orphan(tile: str) -> bool:
    index = TILE_TO_INDEX.get(tile)
    return index in ORPHANS if index is not None else False


def _max_suit_fraction(tiles: list[str]) -> float:
    if not tiles:
        return 0.0
    counts = _suit_counts(tiles)
    return max(counts.values()) / len(tiles)


def _suit_diversity(tiles: list[str]) -> float:
    counts = _suit_counts(tiles)
    return sum(1 for count in counts.values() if count > 0) / len(SUITS)


def _progress(shanten: int, maximum: int) -> float:
    return max(0.0, min(1.0, 1.0 - max(float(shanten), 0.0) / float(maximum)))


def _fan_stats(tiles: list[str]) -> dict[str, float]:
    total = float(len(tiles) or 1)
    counts = Counter(tiles)
    honor_count = sum(count for tile, count in counts.items() if _is_honor(tile))
    terminal_honor_count = sum(
        count for tile, count in counts.items() if _is_honor(tile) or _is_terminal(tile)
    )
    orphan_unique = sum(1 for tile in counts if _is_orphan(tile))
    pair_count = sum(1 for count in counts.values() if count >= 2)
    triplet_count = sum(1 for count in counts.values() if count >= 3)
    dragon_count = sum(counts[tile] for tile in DRAGONS)
    wind_count = sum(counts[tile] for tile in WINDS)
    return {
        "max_suit_fraction": _max_suit_fraction(tiles),
        "suit_diversity": _suit_diversity(tiles),
        "honor_fraction": honor_count / total,
        "terminal_honor_fraction": terminal_honor_count / total,
        "orphan_unique_fraction": orphan_unique / 13.0,
        "pair_count": min(pair_count, 7) / 7.0,
        "triplet_count": min(triplet_count, 4) / 4.0,
        "dragon_count_fraction": dragon_count / total,
        "wind_count_fraction": wind_count / total,
        "regular_progress": _progress(regular_shanten(tiles), 8),
        "seven_pair_progress": _progress(seven_pairs_shanten(tiles), 6),
        "orphan_progress": _progress(thirteen_orphans_shanten(tiles), 13),
    }


def _fan_feature_values(
    response: str,
    hand: Counter[str] | dict[str, int] | Iterable[str],
) -> list[float]:
    hand_counts = as_counter(hand)
    current_tiles = list(hand_counts.elements())
    discard = response_discard_tile(response)
    after_tiles = remove_one_tile(current_tiles, discard) if discard else current_tiles
    current = _fan_stats(current_tiles)
    after = _fan_stats(after_tiles)
    return [
        current["max_suit_fraction"],
        after["max_suit_fraction"],
        after["max_suit_fraction"] - current["max_suit_fraction"],
        current["suit_diversity"],
        after["suit_diversity"],
        after["honor_fraction"],
        after["terminal_honor_fraction"],
        after["orphan_unique_fraction"],
        after["pair_count"],
        after["triplet_count"],
        after["dragon_count_fraction"],
        after["wind_count_fraction"],
        after["regular_progress"],
        after["seven_pair_progress"],
        after["orphan_progress"],
    ]


def featurize_fan_response(
    input_text: str,
    request: str,
    response: str,
    hand: Counter[str] | dict[str, int] | Iterable[str],
    legal_count: int | None = None,
) -> list[float]:
    return [
        *featurize_response(input_text, request, response, hand, legal_count=legal_count),
        *_fan_feature_values(response, hand),
    ]


def featurize_fan_candidate(item: dict, legal_count: int | None = None) -> list[float]:
    return featurize_fan_response(
        input_text=str(item.get("input_text", "")),
        request=str(item.get("request", "")),
        response=str(item.get("candidate_response", "")),
        hand=item.get("hand", Counter()),
        legal_count=legal_count,
    )


def featurize_fan_candidates(items: list[dict]) -> np.ndarray:
    group_sizes: dict[tuple[str, int, int], int] = {}
    for item in items:
        key = (str(item.get("match_id")), int(item.get("player", 0)), int(item.get("turn_index", 0)))
        group_sizes[key] = group_sizes.get(key, 0) + 1
    rows = []
    for item in items:
        key = (str(item.get("match_id")), int(item.get("player", 0)), int(item.get("turn_index", 0)))
        rows.append(featurize_fan_candidate(item, legal_count=group_sizes[key]))
    return np.asarray(rows, dtype=np.float32)


def featurize_fan_legal_responses(
    input_text: str,
    request: str,
    hand: Counter[str] | dict[str, int] | Iterable[str],
    responses: list[str],
) -> np.ndarray:
    rows = [
        featurize_fan_response(input_text, request, response, hand, legal_count=len(responses))
        for response in responses
    ]
    return np.asarray(rows, dtype=np.float32)
