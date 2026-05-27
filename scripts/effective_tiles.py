#!/usr/bin/env python3
"""Fan-valid effective-tile scoring for MCR draw decisions.

The rule encoded here follows the local training objective:
prefer stable >=8-fan hand structures with the most live waits. Incidental fan
sources are intentionally disabled when estimating future structure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from hand_features import TILE_ORDER, min_shanten, remove_one_tile
from legal_actions import can_complete_hand, response_discard_tile


@dataclass(frozen=True)
class EffectiveTileProfile:
    min_shanten: int
    fan8_wait_types: int
    fan8_wait_tiles: int
    max_fan: int
    first_effective_types: int
    first_effective_tiles: int
    second_effective_types: int
    second_effective_tiles: int
    third_effective_types: int
    third_effective_tiles: int


@dataclass(frozen=True)
class DiscardCandidateScore:
    response: str
    discard: str | None
    base_score: float
    guideline_score: float
    final_score: float
    profile: EffectiveTileProfile


EMPTY_PROFILE = EffectiveTileProfile(
    min_shanten=99,
    fan8_wait_types=0,
    fan8_wait_tiles=0,
    max_fan=0,
    first_effective_types=0,
    first_effective_tiles=0,
    second_effective_types=0,
    second_effective_tiles=0,
    third_effective_types=0,
    third_effective_tiles=0,
)


class EffectiveTileEvaluator:
    """Evaluate fan-valid waits and effective-tile classes for one player state."""

    def __init__(
        self,
        fan_checker,
        *,
        packs: list[dict] | None = None,
        visible_counts: Counter[str] | dict[str, int] | None = None,
        flower_count: int = 0,
        seat_wind: int = 0,
        prevalent_wind: int = 0,
        player: int = 0,
        min_fan: int = 8,
        levels: int = 1,
    ):
        self.fan_checker = fan_checker
        self.packs = list(packs or [])
        self.visible_counts = _as_counter(visible_counts or {})
        self.flower_count = int(flower_count)
        self.seat_wind = int(seat_wind)
        self.prevalent_wind = int(prevalent_wind)
        self.player = int(player)
        self.min_fan = int(min_fan)
        self.levels = max(0, min(3, int(levels)))
        self._wait_cache: dict[tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]], tuple[int, int, int, frozenset[str]]] = {}
        self._first_cache: dict[tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]], tuple[int, int, frozenset[str]]] = {}
        self._second_cache: dict[tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]], tuple[int, int, frozenset[str]]] = {}
        self._third_cache: dict[tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]], tuple[int, int, frozenset[str]]] = {}
        self._profile_cache: dict[tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...], int], EffectiveTileProfile] = {}

    def profile(
        self,
        hand: Counter[str] | dict[str, int] | Iterable[str],
        visible_counts: Counter[str] | dict[str, int] | None = None,
        *,
        levels: int | None = None,
    ) -> EffectiveTileProfile:
        hand_counts = _as_counter(hand)
        visible = _as_counter(visible_counts) if visible_counts is not None else Counter(self.visible_counts)
        profile_levels = self.levels if levels is None else max(0, min(3, int(levels)))
        key = (*self._state_key(hand_counts, visible), profile_levels)
        cached = self._profile_cache.get(key)
        if cached is not None:
            return cached

        waits = self._fan8_wait_counts(hand_counts, visible)
        first_types = first_tiles = 0
        second_types = second_tiles = third_types = third_tiles = 0
        if profile_levels >= 1:
            first_types, first_tiles, _ = self._first_effective_counts(hand_counts, visible)
        if profile_levels >= 2:
            second_types, second_tiles, _ = self._second_effective_counts(hand_counts, visible)
        if profile_levels >= 3:
            third_types, third_tiles, _ = self._third_effective_counts(hand_counts, visible)
        profile = EffectiveTileProfile(
            min_shanten=min_shanten(hand_counts.elements()),
            fan8_wait_types=waits[0],
            fan8_wait_tiles=waits[1],
            max_fan=waits[2],
            first_effective_types=first_types,
            first_effective_tiles=first_tiles,
            second_effective_types=second_types,
            second_effective_tiles=second_tiles,
            third_effective_types=third_types,
            third_effective_tiles=third_tiles,
        )
        self._profile_cache[key] = profile
        return profile

    def _fan8_wait_counts(
        self,
        hand: Counter[str],
        visible: Counter[str],
    ) -> tuple[int, int, int, frozenset[str]]:
        key = self._state_key(hand, visible)
        cached = self._wait_cache.get(key)
        if cached is not None:
            return cached
        wait_tiles: list[str] = []
        wait_live_count = 0
        max_fan = 0
        for tile in TILE_ORDER:
            live = _live_count(tile, hand, visible)
            if live <= 0:
                continue
            if not can_complete_hand(hand, tile, meld_count=len(self.packs)):
                continue
            fan = self._fan_for_draw(hand, tile)
            if fan >= self.min_fan:
                wait_tiles.append(tile)
                wait_live_count += live
                max_fan = max(max_fan, fan)
        result = (len(wait_tiles), wait_live_count, max_fan, frozenset(wait_tiles))
        self._wait_cache[key] = result
        return result

    def _fan_for_draw(self, hand: Counter[str], win_tile: str) -> int:
        if self.fan_checker is None:
            return 0
        try:
            result = self.fan_checker.evaluate(
                packs=list(self.packs),
                hand=list(hand.elements()),
                win_tile=win_tile,
                flower_count=self.flower_count,
                is_self_draw=True,
                is_4th_tile=False,
                is_about_kong=False,
                is_last=False,
                seat_wind=self.seat_wind,
                prevalent_wind=self.prevalent_wind,
                player=self.player,
            )
        except Exception:
            return 0
        if not result.get("can_hu", False):
            return 0
        return int(result.get("fan", result.get("fanCnt", 0)) or 0)

    def _first_effective_counts(
        self, hand: Counter[str], visible: Counter[str]
    ) -> tuple[int, int, frozenset[str]]:
        key = self._state_key(hand, visible)
        cached = self._first_cache.get(key)
        if cached is not None:
            return cached
        base_shanten = min_shanten(hand.elements())
        effective: set[str] = set()
        live_total = 0
        for draw_tile, live in self._live_draws(hand, visible):
            if self._draw_can_reduce_shanten_with_fan8_path(hand, visible, draw_tile, base_shanten):
                effective.add(draw_tile)
                live_total += live
        result = (len(effective), live_total, frozenset(effective))
        self._first_cache[key] = result
        return result

    def _second_effective_counts(
        self, hand: Counter[str], visible: Counter[str]
    ) -> tuple[int, int, frozenset[str]]:
        key = self._state_key(hand, visible)
        cached = self._second_cache.get(key)
        if cached is not None:
            return cached
        base_shanten = min_shanten(hand.elements())
        _, base_first_live, first_tiles = self._first_effective_counts(hand, visible)
        effective: set[str] = set()
        live_total = 0
        for draw_tile, live in self._live_draws(hand, visible):
            if draw_tile in first_tiles:
                continue
            if self._draw_can_increase_lower_class(
                hand,
                visible,
                draw_tile,
                base_shanten,
                base_count=base_first_live,
                lower_class="first",
            ):
                effective.add(draw_tile)
                live_total += live
        result = (len(effective), live_total, frozenset(effective))
        self._second_cache[key] = result
        return result

    def _third_effective_counts(
        self, hand: Counter[str], visible: Counter[str]
    ) -> tuple[int, int, frozenset[str]]:
        key = self._state_key(hand, visible)
        cached = self._third_cache.get(key)
        if cached is not None:
            return cached
        base_shanten = min_shanten(hand.elements())
        _, _, first_tiles = self._first_effective_counts(hand, visible)
        _, base_second_live, second_tiles = self._second_effective_counts(hand, visible)
        effective: set[str] = set()
        live_total = 0
        for draw_tile, live in self._live_draws(hand, visible):
            if draw_tile in first_tiles or draw_tile in second_tiles:
                continue
            if self._draw_can_increase_lower_class(
                hand,
                visible,
                draw_tile,
                base_shanten,
                base_count=base_second_live,
                lower_class="second",
            ):
                effective.add(draw_tile)
                live_total += live
        result = (len(effective), live_total, frozenset(effective))
        self._third_cache[key] = result
        return result

    def _draw_can_reduce_shanten_with_fan8_path(
        self,
        hand: Counter[str],
        visible: Counter[str],
        draw_tile: str,
        base_shanten: int,
    ) -> bool:
        drawn = Counter(hand)
        drawn[draw_tile] += 1
        for discard in sorted(drawn):
            after, after_visible = self._after_discard(drawn, visible, discard)
            if min_shanten(after.elements()) >= base_shanten:
                continue
            if self._fan8_wait_counts(after, after_visible)[1] > 0:
                return True
        return False

    def _draw_can_increase_lower_class(
        self,
        hand: Counter[str],
        visible: Counter[str],
        draw_tile: str,
        base_shanten: int,
        *,
        base_count: int,
        lower_class: str,
    ) -> bool:
        drawn = Counter(hand)
        drawn[draw_tile] += 1
        for discard in sorted(drawn):
            after, after_visible = self._after_discard(drawn, visible, discard)
            if min_shanten(after.elements()) > base_shanten:
                continue
            if lower_class == "first":
                _, count, _ = self._first_effective_counts(after, after_visible)
            elif lower_class == "second":
                _, count, _ = self._second_effective_counts(after, after_visible)
            else:
                raise ValueError(f"unknown lower class {lower_class!r}")
            if count > base_count:
                return True
        return False

    def _live_draws(self, hand: Counter[str], visible: Counter[str]) -> list[tuple[str, int]]:
        return [
            (tile, live)
            for tile in TILE_ORDER
            if (live := _live_count(tile, hand, visible)) > 0
        ]

    def _after_discard(
        self, hand: Counter[str], visible: Counter[str], discard: str
    ) -> tuple[Counter[str], Counter[str]]:
        after = Counter(hand)
        after[discard] -= 1
        _cleanup(after)
        after_visible = Counter(visible)
        after_visible[discard] += 1
        return after, after_visible

    def _state_key(
        self, hand: Counter[str], visible: Counter[str]
    ) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
        return _counter_key(hand), _counter_key(visible)


def evaluate_discard_candidates(
    *,
    hand: Counter[str] | dict[str, int] | Iterable[str],
    responses: list[str],
    fan_checker,
    base_scores: list[float] | None = None,
    packs: list[dict] | None = None,
    visible_counts: Counter[str] | dict[str, int] | None = None,
    flower_count: int = 0,
    seat_wind: int = 0,
    prevalent_wind: int = 0,
    player: int = 0,
    levels: int = 1,
    base_score_weight: float = 0.05,
    guideline_weight: float = 1.0,
) -> list[DiscardCandidateScore]:
    hand_counts = _as_counter(hand)
    visible = _as_counter(visible_counts or {})
    scores = base_scores or [0.0] * len(responses)
    if len(scores) != len(responses):
        raise ValueError("base_scores length must match responses")
    evaluator = EffectiveTileEvaluator(
        fan_checker,
        packs=packs,
        visible_counts=visible,
        flower_count=flower_count,
        seat_wind=seat_wind,
        prevalent_wind=prevalent_wind,
        player=player,
        levels=levels,
    )
    result: list[DiscardCandidateScore] = []
    for response, base_score in zip(responses, scores):
        discard = response_discard_tile(response)
        if discard is None or hand_counts[discard] <= 0:
            profile = EMPTY_PROFILE
            guideline_score = -1_000_000.0
        else:
            after_hand = Counter(hand_counts)
            after_hand[discard] -= 1
            _cleanup(after_hand)
            after_visible = Counter(visible)
            after_visible[discard] += 1
            profile = evaluator.profile(after_hand, after_visible, levels=levels)
            guideline_score = _guideline_score(profile)
        final = float(guideline_weight) * guideline_score + float(base_score_weight) * float(base_score)
        result.append(
            DiscardCandidateScore(
                response=response,
                discard=discard,
                base_score=float(base_score),
                guideline_score=guideline_score,
                final_score=final,
                profile=profile,
            )
        )
    return sorted(result, key=lambda item: (item.final_score, item.response), reverse=True)


def choose_effective_discard(scores: list[DiscardCandidateScore]) -> DiscardCandidateScore:
    if not scores:
        raise ValueError("no discard scores to choose from")
    return max(scores, key=lambda item: (item.final_score, item.response))


def _guideline_score(profile: EffectiveTileProfile) -> float:
    return (
        10_000.0 * profile.fan8_wait_tiles
        + 1_000.0 * profile.fan8_wait_types
        - 100.0 * profile.min_shanten
        + 40.0 * profile.first_effective_tiles
        + 10.0 * profile.first_effective_types
        + 8.0 * profile.second_effective_tiles
        + 2.0 * profile.second_effective_types
        + 1.0 * profile.third_effective_tiles
        + 0.25 * profile.third_effective_types
        + 0.05 * profile.max_fan
    )


def _as_counter(values: Counter[str] | dict[str, int] | Iterable[str]) -> Counter[str]:
    if isinstance(values, Counter):
        return Counter(values)
    if isinstance(values, dict):
        return Counter({tile: int(count) for tile, count in values.items() if int(count) > 0})
    return Counter(values)


def _counter_key(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((tile, int(count)) for tile, count in counter.items() if count > 0))


def _live_count(tile: str, hand: Counter[str], visible: Counter[str]) -> int:
    return max(0, 4 - int(hand[tile]) - int(visible[tile]))


def _cleanup(hand: Counter[str]) -> None:
    for tile in list(hand):
        if hand[tile] <= 0:
            del hand[tile]
