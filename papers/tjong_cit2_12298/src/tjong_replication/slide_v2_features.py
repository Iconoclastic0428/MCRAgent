"""Search/shanten feature planes for the slide v2 side model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Iterable

import torch

from .encoding import GAME_FEATURES, TILE_TYPES, VISIBLE_ROW_NAMES
from .slide_resnet import SLIDE_GRID_HEIGHT, SLIDE_GRID_WIDTH, SLIDE_SEARCH_CHANNELS, tile_vectors_to_grid
from .tiles import TILE_NAMES, is_suited, tile_id

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if _SCRIPTS_DIR.exists() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover - import availability is environment-specific.
    from effective_tiles import EffectiveTileEvaluator
    from hand_features import (
        min_shanten,
        regular_shanten,
        seven_pairs_shanten,
        thirteen_orphans_shanten,
    )
except Exception:  # pragma: no cover
    EffectiveTileEvaluator = None
    min_shanten = regular_shanten = seven_pairs_shanten = thirteen_orphans_shanten = None


@dataclass(frozen=True)
class SlideV2SearchContext:
    hand: Counter[str]
    live_counts: Counter[str]
    visible_counts: Counter[str]
    packs: tuple[dict, ...] = ()
    flower_count: int = 0
    seat_wind: int = 0
    prevalent_wind: int = 0
    player: int = 0


def build_batch_search_features_from_tensors(
    visible_tiles: torch.Tensor,
    game_features: torch.Tensor,
    *,
    fan_checker=None,
    levels: int = 3,
) -> torch.Tensor:
    if visible_tiles.ndim != 4:
        raise ValueError("visible_tiles must have shape [batch, memory, 22, 34]")
    if game_features.ndim != 3 or game_features.shape[:2] != visible_tiles.shape[:2]:
        raise ValueError("game_features must align with visible_tiles")
    features = []
    visible_cpu = visible_tiles.detach().cpu()
    game_cpu = game_features.detach().cpu()
    for visible, game in zip(visible_cpu[:, -1], game_cpu[:, -1]):
        context = context_from_encoded_current_frame(visible, game)
        features.append(build_search_feature_planes(context, fan_checker=fan_checker, levels=levels))
    return torch.stack(features, dim=0).to(device=visible_tiles.device, dtype=visible_tiles.dtype)


def context_from_encoded_current_frame(
    visible_current: torch.Tensor,
    game_current: torch.Tensor,
) -> SlideV2SearchContext:
    if visible_current.shape != (len(VISIBLE_ROW_NAMES), TILE_TYPES):
        raise ValueError(f"visible_current must have shape {(len(VISIBLE_ROW_NAMES), TILE_TYPES)}")
    if game_current.shape[-1] != GAME_FEATURES:
        raise ValueError(f"game_current must end with {GAME_FEATURES}")
    hand = _counter_from_vector(visible_current[VISIBLE_ROW_NAMES.index("hand_self")])
    live = _counter_from_vector(visible_current[VISIBLE_ROW_NAMES.index("remaining_tiles_p0")])
    visible_counts = Counter()
    for tile in TILE_NAMES:
        visible_counts[tile] = max(0, 4 - int(hand[tile]) - int(live[tile]))
    packs = tuple(_packs_from_visible_current(visible_current))
    prevalent_wind = int(round(float(game_current[0].item())))
    seat_slice = game_current[1:5]
    seat_wind = int(torch.argmax(seat_slice).item()) if seat_slice.numel() else 0
    return SlideV2SearchContext(
        hand=hand,
        live_counts=live,
        visible_counts=visible_counts,
        packs=packs,
        seat_wind=seat_wind,
        prevalent_wind=prevalent_wind,
        player=seat_wind,
    )


def build_search_feature_planes(
    context: SlideV2SearchContext,
    *,
    fan_checker=None,
    levels: int = 3,
) -> torch.Tensor:
    """Return 30 v2 feature planes.

    This is an implementation inference for the slide's unspecified "30 channels
    based on shanten search": it combines tile-local live/wait/effective masks
    with broadcast hand-level shanten and fan-valid wait summaries.
    """

    hand = Counter(context.hand)
    live = Counter(context.live_counts)
    visible = Counter(context.visible_counts)
    hand_tiles = list(hand.elements())
    current_min = _safe_shanten(min_shanten, hand_tiles)
    current_regular = _safe_shanten(regular_shanten, hand_tiles)
    current_seven = _safe_shanten(seven_pairs_shanten, hand_tiles)
    current_orphan = _safe_shanten(thirteen_orphans_shanten, hand_tiles)
    pair_count = sum(1 for count in hand.values() if count >= 2)
    triplet_count = sum(1 for count in hand.values() if count >= 3)

    profile = None
    fan8_wait_tiles: set[str] = set()
    first_tiles: set[str] = set()
    second_tiles: set[str] = set()
    third_tiles: set[str] = set()
    fan_by_tile: dict[str, int] = {}
    if EffectiveTileEvaluator is not None:
        evaluator = EffectiveTileEvaluator(
            fan_checker,
            packs=list(context.packs),
            visible_counts=visible,
            flower_count=context.flower_count,
            seat_wind=context.seat_wind,
            prevalent_wind=context.prevalent_wind,
            player=context.player,
            levels=levels,
        )
        profile = evaluator.profile(hand, visible, levels=levels)
        try:
            _, _, _, fan8 = evaluator._fan8_wait_counts(hand, visible)
            fan8_wait_tiles = set(fan8)
        except Exception:
            fan8_wait_tiles = set()
        for tile in fan8_wait_tiles:
            fan_by_tile[tile] = _fan_for_tile(
                fan_checker,
                context=context,
                hand=hand,
                tile=tile,
            )
        for attr, target in (
            ("_first_effective_counts", "first"),
            ("_second_effective_counts", "second"),
            ("_third_effective_counts", "third"),
        ):
            if levels < {"first": 1, "second": 2, "third": 3}[target]:
                continue
            try:
                _, _, tiles = getattr(evaluator, attr)(hand, visible)
            except Exception:
                tiles = frozenset()
            if target == "first":
                first_tiles = set(tiles)
            elif target == "second":
                second_tiles = set(tiles)
            else:
                third_tiles = set(tiles)

    channels = torch.zeros(SLIDE_SEARCH_CHANNELS, TILE_TYPES, dtype=torch.float32)
    channels[0] = _vector_from_counter(hand, scale=4.0)
    channels[1] = _vector_from_counter(live, scale=4.0)
    channels[2] = _mask_from_tiles(fan8_wait_tiles)
    channels[3] = _live_mask_from_tiles(fan8_wait_tiles, live, scale=4.0)
    channels[4] = _fan_vector(fan_by_tile, scale=88.0)
    channels[5] = _mask_from_tiles(first_tiles)
    channels[6] = _live_mask_from_tiles(first_tiles, live, scale=4.0)
    channels[7] = _mask_from_tiles(second_tiles)
    channels[8] = _live_mask_from_tiles(second_tiles, live, scale=4.0)
    channels[9] = _mask_from_tiles(third_tiles)
    channels[10] = _live_mask_from_tiles(third_tiles, live, scale=4.0)
    _fill_draw_search_channels(channels, hand=hand, live=live)
    channels[18].fill_(_normalize_shanten(current_min))
    channels[19].fill_(_normalize_shanten(current_regular))
    channels[20].fill_(_normalize_shanten(current_seven))
    channels[21].fill_(_normalize_shanten(current_orphan))
    channels[22].fill_(min(pair_count, 7) / 7.0)
    channels[23].fill_(min(triplet_count, 4) / 4.0)
    if profile is not None:
        channels[24].fill_(min(float(profile.fan8_wait_types), 34.0) / 34.0)
        channels[25].fill_(min(float(profile.fan8_wait_tiles), 136.0) / 136.0)
        channels[26].fill_(min(float(profile.max_fan), 88.0) / 88.0)
        channels[27].fill_(min(float(profile.first_effective_tiles), 136.0) / 136.0)
        channels[28].fill_(min(float(profile.second_effective_tiles), 136.0) / 136.0)
        channels[29].fill_(min(float(profile.third_effective_tiles), 136.0) / 136.0)
    return tile_vectors_to_grid(channels)


def load_default_official_fan_checker():
    try:
        from official_fan import OfficialFanChecker
        return OfficialFanChecker.default()
    except Exception:
        return None


def _packs_from_visible_current(visible_current: torch.Tensor) -> list[dict]:
    packs: list[dict] = []
    peng_row = visible_current[VISIBLE_ROW_NAMES.index("peng_p0")]
    kong_row = visible_current[VISIBLE_ROW_NAMES.index("kong_p0")]
    chow_row = visible_current[VISIBLE_ROW_NAMES.index("chow_p0")]
    for tile, count in _counter_from_vector(peng_row).items():
        for _ in range(int(count) // 3):
            packs.append({"type": "PENG", "tile": tile, "offer": 0})
    for tile, count in _counter_from_vector(kong_row).items():
        for _ in range(int(count) // 4):
            packs.append({"type": "GANG", "tile": tile, "offer": 0})
    for middle in _decompose_chows(_counter_from_vector(chow_row)):
        packs.append({"type": "CHI", "tile": middle, "offer": 1})
    return packs


def _decompose_chows(counts: Counter[str]) -> list[str]:
    result: list[str] = []
    working = Counter(counts)
    for suit in ("W", "T", "B"):
        changed = True
        while changed:
            changed = False
            for start in range(1, 8):
                seq = [f"{suit}{start}", f"{suit}{start + 1}", f"{suit}{start + 2}"]
                if all(working[tile] > 0 for tile in seq):
                    for tile in seq:
                        working[tile] -= 1
                    result.append(f"{suit}{start + 1}")
                    changed = True
                    break
    return result


def _fill_draw_search_channels(channels: torch.Tensor, *, hand: Counter[str], live: Counter[str]) -> None:
    if min_shanten is None:
        return
    base_min = _safe_shanten(min_shanten, hand.elements())
    for tile in TILE_NAMES:
        if live[tile] <= 0:
            continue
        drawn = Counter(hand)
        drawn[tile] += 1
        best_min = 99
        best_regular = 99
        best_seven = 99
        best_orphan = 99
        for discard, count in list(drawn.items()):
            if count <= 0:
                continue
            after = Counter(drawn)
            after[discard] -= 1
            _cleanup(after)
            tiles = list(after.elements())
            best_min = min(best_min, _safe_shanten(min_shanten, tiles))
            best_regular = min(best_regular, _safe_shanten(regular_shanten, tiles))
            best_seven = min(best_seven, _safe_shanten(seven_pairs_shanten, tiles))
            best_orphan = min(best_orphan, _safe_shanten(thirteen_orphans_shanten, tiles))
        index = tile_id(tile)
        channels[11, index] = _normalize_shanten(best_min)
        channels[12, index] = _normalize_shanten(best_regular)
        channels[13, index] = _normalize_shanten(best_seven)
        channels[14, index] = _normalize_shanten(best_orphan)
        channels[15, index] = 1.0 if best_min < base_min else 0.0
        channels[16, index] = min(float(best_min), 8.0) / 8.0 if best_min < 99 else 0.0
        channels[17, index] = 1.0 if is_suited(tile) else 0.0


def _fan_for_tile(fan_checker, *, context: SlideV2SearchContext, hand: Counter[str], tile: str) -> int:
    if fan_checker is None:
        return 0
    try:
        result = fan_checker.evaluate(
            packs=list(context.packs),
            hand=list(hand.elements()),
            win_tile=tile,
            flower_count=context.flower_count,
            is_self_draw=True,
            is_4th_tile=False,
            is_about_kong=False,
            is_last=False,
            seat_wind=context.seat_wind,
            prevalent_wind=context.prevalent_wind,
            player=context.player,
        )
    except Exception:
        return 0
    if not result.get("can_hu", False):
        return 0
    return int(result.get("fan", result.get("fanCnt", 0)) or 0)


def _counter_from_vector(vector: torch.Tensor) -> Counter[str]:
    return Counter(
        {
            tile: int(round(float(vector[index].item())))
            for index, tile in enumerate(TILE_NAMES)
            if int(round(float(vector[index].item()))) > 0
        }
    )


def _vector_from_counter(counter: Counter[str], *, scale: float) -> torch.Tensor:
    vector = torch.zeros(TILE_TYPES, dtype=torch.float32)
    for tile, count in counter.items():
        if tile in TILE_NAMES:
            vector[tile_id(tile)] = min(float(count) / float(scale), 1.0)
    return vector


def _mask_from_tiles(tiles: Iterable[str]) -> torch.Tensor:
    vector = torch.zeros(TILE_TYPES, dtype=torch.float32)
    for tile in tiles:
        if tile in TILE_NAMES:
            vector[tile_id(tile)] = 1.0
    return vector


def _live_mask_from_tiles(tiles: Iterable[str], live: Counter[str], *, scale: float) -> torch.Tensor:
    vector = torch.zeros(TILE_TYPES, dtype=torch.float32)
    for tile in tiles:
        if tile in TILE_NAMES:
            vector[tile_id(tile)] = min(float(live[tile]) / float(scale), 1.0)
    return vector


def _fan_vector(fan_by_tile: dict[str, int], *, scale: float) -> torch.Tensor:
    vector = torch.zeros(TILE_TYPES, dtype=torch.float32)
    for tile, fan in fan_by_tile.items():
        if tile in TILE_NAMES:
            vector[tile_id(tile)] = min(float(fan) / float(scale), 1.0)
    return vector


def _safe_shanten(fn, tiles) -> int:
    if fn is None:
        return 99
    try:
        return int(fn(list(tiles)))
    except Exception:
        return 99


def _normalize_shanten(value: int) -> float:
    if value >= 99:
        return 0.0
    return max(0.0, min(1.0, (8.0 - float(value)) / 9.0))


def _cleanup(counter: Counter[str]) -> None:
    for tile in list(counter):
        if counter[tile] <= 0:
            del counter[tile]
