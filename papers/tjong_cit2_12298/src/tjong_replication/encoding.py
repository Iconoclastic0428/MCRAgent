"""Mahjong feature encoding described by the Tjong paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

TILE_TYPES = 34
VISIBLE_TILE_ROWS = 22
GAME_FEATURES = 24
HIDDEN_TILE_ROWS = 5

VISIBLE_ROW_NAMES = (
    "hand_self",
    "discard_p0",
    "discard_p1",
    "discard_p2",
    "discard_p3",
    "peng_p0",
    "peng_p1",
    "peng_p2",
    "peng_p3",
    "chow_p0",
    "chow_p1",
    "chow_p2",
    "chow_p3",
    "kong_p0",
    "kong_p1",
    "kong_p2",
    "kong_p3",
    "available_tile_mask",
    "remaining_tiles_p0",
    "remaining_tiles_p1",
    "remaining_tiles_p2",
    "remaining_tiles_p3",
)


@dataclass
class EncodedMahjongState:
    visible_tiles: torch.Tensor
    game_features: torch.Tensor
    hidden_tiles: torch.Tensor | None = None

    def validate(self) -> None:
        if tuple(self.visible_tiles.shape[-2:]) != (VISIBLE_TILE_ROWS, TILE_TYPES):
            raise ValueError(f"visible tile features must end with {(VISIBLE_TILE_ROWS, TILE_TYPES)}")
        if self.game_features.shape[-1] != GAME_FEATURES:
            raise ValueError(f"game features must end with {GAME_FEATURES}")
        if self.hidden_tiles is not None and tuple(self.hidden_tiles.shape[-2:]) != (HIDDEN_TILE_ROWS, TILE_TYPES):
            raise ValueError(f"hidden tile features must end with {(HIDDEN_TILE_ROWS, TILE_TYPES)}")


def tile_multiset(tile_ids: Iterable[int], *, device: torch.device | None = None) -> torch.Tensor:
    counts = torch.zeros(TILE_TYPES, dtype=torch.float32, device=device)
    for tile_id in tile_ids:
        tile = int(tile_id)
        if tile < 0 or tile >= TILE_TYPES:
            raise ValueError(f"tile id must be in [0, 33], got {tile_id}")
        counts[tile] += 1.0
    return counts


def stack_visible_tile_features(rows: dict[str, Sequence[int] | torch.Tensor]) -> torch.Tensor:
    encoded = []
    for name in VISIBLE_ROW_NAMES:
        value = rows.get(name, [])
        if isinstance(value, torch.Tensor):
            tensor = value.float()
            if tensor.numel() != TILE_TYPES:
                raise ValueError(f"{name} must have {TILE_TYPES} entries")
            encoded.append(tensor.reshape(TILE_TYPES))
        else:
            encoded.append(tile_multiset(value))
    return torch.stack(encoded, dim=0)


def build_game_features(
    *,
    prevailing_wind: int,
    seat_wind: int,
    opponent_concealed_kongs: Sequence[float],
    remaining_tile_counts: Sequence[float],
    hand_tile_counts: Sequence[float],
    action_mask: Sequence[float],
) -> torch.Tensor:
    if not 0 <= int(prevailing_wind) < 4:
        raise ValueError("prevailing_wind must be in [0, 3]")
    if not 0 <= int(seat_wind) < 4:
        raise ValueError("seat_wind must be in [0, 3]")
    if len(opponent_concealed_kongs) != 3:
        raise ValueError("opponent_concealed_kongs must have length 3")
    if len(remaining_tile_counts) != 4:
        raise ValueError("remaining_tile_counts must have length 4")
    if len(hand_tile_counts) != 4:
        raise ValueError("hand_tile_counts must have length 4")
    if len(action_mask) != 8:
        raise ValueError("action_mask must have length 8")

    features = torch.zeros(GAME_FEATURES, dtype=torch.float32)
    cursor = 0
    features[cursor] = float(prevailing_wind)
    cursor += 1
    features[cursor + int(seat_wind)] = 1.0
    cursor += 4
    features[cursor : cursor + 3] = torch.tensor(opponent_concealed_kongs, dtype=torch.float32)
    cursor += 3
    features[cursor : cursor + 4] = torch.tensor(remaining_tile_counts, dtype=torch.float32)
    cursor += 4
    features[cursor : cursor + 4] = torch.tensor(hand_tile_counts, dtype=torch.float32)
    cursor += 4
    features[cursor : cursor + 8] = torch.tensor(action_mask, dtype=torch.float32)
    return features


def stack_hidden_tile_features(rows: Sequence[Sequence[int] | torch.Tensor]) -> torch.Tensor:
    if len(rows) != HIDDEN_TILE_ROWS:
        raise ValueError(f"hidden tile features require {HIDDEN_TILE_ROWS} rows")
    encoded = []
    for row in rows:
        if isinstance(row, torch.Tensor):
            tensor = row.float()
            if tensor.numel() != TILE_TYPES:
                raise ValueError(f"hidden tile row must have {TILE_TYPES} entries")
            encoded.append(tensor.reshape(TILE_TYPES))
        else:
            encoded.append(tile_multiset(row))
    return torch.stack(encoded, dim=0)
