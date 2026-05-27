#!/usr/bin/env python3
"""Pure-Python hand-quality features for MCR discard ranking.

This is intentionally small and dependency-free. It is not a replacement for
the official fan calculator, but it provides standard shanten-style signals for
legal discard ranking when PyMahjongGB is unavailable on the host.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Iterable


SUITS = ("W", "B", "T")
WINDS = ("F1", "F2", "F3", "F4")
DRAGONS = ("J1", "J2", "J3")
TILE_ORDER = [
    *[f"{suit}{rank}" for suit in SUITS for rank in range(1, 10)],
    *WINDS,
    *DRAGONS,
]
TILE_TO_INDEX = {tile: index for index, tile in enumerate(TILE_ORDER)}
INDEX_TO_TILE = {index: tile for tile, index in TILE_TO_INDEX.items()}
ORPHANS = {
    TILE_TO_INDEX[tile]
    for tile in [
        "W1",
        "W9",
        "B1",
        "B9",
        "T1",
        "T9",
        "F1",
        "F2",
        "F3",
        "F4",
        "J1",
        "J2",
        "J3",
    ]
}


def tile_counts(tiles: Iterable[str]) -> tuple[int, ...]:
    counts = [0] * len(TILE_ORDER)
    for tile in tiles:
        if tile in TILE_TO_INDEX:
            counts[TILE_TO_INDEX[tile]] += 1
    return tuple(counts)


def is_suited_index(index: int) -> bool:
    return 0 <= index < 27


def same_suit_sequence_start(index: int) -> bool:
    return is_suited_index(index) and index % 9 <= 6


def regular_shanten(tiles: Iterable[str]) -> int:
    return _regular_shanten_from_counts(tile_counts(tiles))


@lru_cache(maxsize=200_000)
def _regular_shanten_from_counts(counts: tuple[int, ...]) -> int:
    @lru_cache(maxsize=None)
    def search(state: tuple[int, ...], melds: int, taatsu: int, pair: int) -> int:
        try:
            index = next(i for i, count in enumerate(state) if count)
        except StopIteration:
            usable_taatsu = min(taatsu, 4 - melds)
            return 8 - 2 * melds - usable_taatsu - pair

        best = 8
        counts_list = list(state)

        if counts_list[index] >= 3:
            counts_list[index] -= 3
            best = min(best, search(tuple(counts_list), melds + 1, taatsu, pair))
            counts_list[index] += 3

        if (
            same_suit_sequence_start(index)
            and counts_list[index + 1] > 0
            and counts_list[index + 2] > 0
        ):
            counts_list[index] -= 1
            counts_list[index + 1] -= 1
            counts_list[index + 2] -= 1
            best = min(best, search(tuple(counts_list), melds + 1, taatsu, pair))
            counts_list[index] += 1
            counts_list[index + 1] += 1
            counts_list[index + 2] += 1

        if counts_list[index] >= 2:
            counts_list[index] -= 2
            if pair == 0:
                best = min(best, search(tuple(counts_list), melds, taatsu, 1))
            best = min(best, search(tuple(counts_list), melds, taatsu + 1, pair))
            counts_list[index] += 2

        if same_suit_sequence_start(index) and counts_list[index + 1] > 0:
            counts_list[index] -= 1
            counts_list[index + 1] -= 1
            best = min(best, search(tuple(counts_list), melds, taatsu + 1, pair))
            counts_list[index] += 1
            counts_list[index + 1] += 1

        if is_suited_index(index) and index % 9 <= 6 and counts_list[index + 2] > 0:
            counts_list[index] -= 1
            counts_list[index + 2] -= 1
            best = min(best, search(tuple(counts_list), melds, taatsu + 1, pair))
            counts_list[index] += 1
            counts_list[index + 2] += 1

        counts_list[index] -= 1
        best = min(best, search(tuple(counts_list), melds, taatsu, pair))
        return best

    return search(counts, 0, 0, 0)


def seven_pairs_shanten(tiles: Iterable[str]) -> int:
    return _seven_pairs_shanten_from_counts(tile_counts(tiles))


@lru_cache(maxsize=200_000)
def _seven_pairs_shanten_from_counts(counts: tuple[int, ...]) -> int:
    pairs = sum(1 for count in counts if count >= 2)
    distinct = sum(1 for count in counts if count > 0)
    return 6 - pairs + max(0, 7 - distinct)


def thirteen_orphans_shanten(tiles: Iterable[str]) -> int:
    return _thirteen_orphans_shanten_from_counts(tile_counts(tiles))


@lru_cache(maxsize=200_000)
def _thirteen_orphans_shanten_from_counts(counts: tuple[int, ...]) -> int:
    unique = sum(1 for index in ORPHANS if counts[index] > 0)
    has_pair = any(counts[index] >= 2 for index in ORPHANS)
    return 13 - unique - (1 if has_pair else 0)


def min_shanten(tiles: Iterable[str]) -> int:
    tile_list = list(tiles)
    return min(
        regular_shanten(tile_list),
        seven_pairs_shanten(tile_list),
        thirteen_orphans_shanten(tile_list),
    )


def remove_one_tile(tiles: Iterable[str], tile: str) -> list[str]:
    remaining = list(tiles)
    try:
        remaining.remove(tile)
    except ValueError:
        pass
    return remaining


def hand_feature_text(hand_tiles: Iterable[str], candidate_tile: str) -> str:
    hand = list(hand_tiles)
    after = remove_one_tile(hand, candidate_tile)
    counts = Counter(after)
    reg = regular_shanten(after)
    seven = seven_pairs_shanten(after)
    orphan = thirteen_orphans_shanten(after)
    minimum = min(reg, seven, orphan)
    tile_count_features = " ".join(
        f"CNT_{tile}_{counts[tile]}" for tile in sorted(counts) if counts[tile]
    )
    return "\n".join(
        [
            f"CAND {candidate_tile}",
            f"REG_SHANTEN {reg}",
            f"SEVEN_SHANTEN {seven}",
            f"ORPHAN_SHANTEN {orphan}",
            f"MIN_SHANTEN {minimum}",
            f"PAIR_COUNT {sum(1 for count in counts.values() if count >= 2)}",
            f"TRIPLE_COUNT {sum(1 for count in counts.values() if count >= 3)}",
            tile_count_features,
        ]
    ).strip()


def candidate_feature_text(
    hand_tiles: Iterable[str], candidate_tile: str, drawn_tile: str | None = None
) -> str:
    hand = list(hand_tiles)
    base = hand_feature_text(hand, candidate_tile)
    if drawn_tile is None:
        return base
    candidate_min = min_shanten(remove_one_tile(hand, candidate_tile))
    drawn_min = min_shanten(remove_one_tile(hand, drawn_tile))
    return "\n".join(
        [
            base,
            f"DRAWN_TILE {drawn_tile}",
            f"CAND_IS_DRAWN {1 if candidate_tile == drawn_tile else 0}",
            f"DRAWN_MIN_SHANTEN {drawn_min}",
            f"SHANTEN_DELTA_VS_DRAWN {candidate_min - drawn_min}",
        ]
    )
