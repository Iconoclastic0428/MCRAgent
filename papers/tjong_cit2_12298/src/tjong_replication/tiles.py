"""Tile vocabulary used by Botzone MCR logs and the Tjong tensors."""

from __future__ import annotations

TILE_NAMES = (
    *tuple(f"W{rank}" for rank in range(1, 10)),
    *tuple(f"T{rank}" for rank in range(1, 10)),
    *tuple(f"B{rank}" for rank in range(1, 10)),
    *tuple(f"F{rank}" for rank in range(1, 5)),
    *tuple(f"J{rank}" for rank in range(1, 4)),
)

TILE_TO_ID = {tile: index for index, tile in enumerate(TILE_NAMES)}
ID_TO_TILE = {index: tile for tile, index in TILE_TO_ID.items()}
SUIT_ORDER = ("W", "T", "B")
SUIT_TO_ID = {suit: index for index, suit in enumerate(SUIT_ORDER)}


def is_tile_name(value: str) -> bool:
    return value in TILE_TO_ID


def tile_id(tile: str) -> int:
    try:
        return TILE_TO_ID[tile]
    except KeyError as exc:
        raise ValueError(f"unknown tile name: {tile}") from exc


def tile_name(index: int) -> str:
    try:
        return ID_TO_TILE[int(index)]
    except KeyError as exc:
        raise ValueError(f"unknown tile id: {index}") from exc


def tiles_to_ids(tiles: list[str] | tuple[str, ...]) -> list[int]:
    return [tile_id(tile) for tile in tiles]


def is_suited(tile: str) -> bool:
    return (
        len(tile) == 2
        and tile[0] in SUIT_TO_ID
        and tile[1].isdigit()
        and 1 <= int(tile[1]) <= 9
    )


def suited_rank(tile: str) -> int:
    if not is_suited(tile):
        raise ValueError(f"tile is not suited: {tile}")
    rank = int(tile[1])
    if rank < 1 or rank > 9:
        raise ValueError(f"suited tile rank out of range: {tile}")
    return rank
