#!/usr/bin/env python3
"""Lawlorentz-based MCR policy with an effective-tile draw rule.

This module uses the upstream `lawlorentz/Chinese-Standard-Mahjong-DRL`
FeatureAgent/action mask as the state and action interface.  The draw decision
is deterministic and rule-first: preserve or improve an >=8 fan structure with
the most live waits before considering lower-class effective tiles.
"""

from __future__ import annotations

import copy
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from MahjongGB import MahjongFanCalculator, MahjongShanten


LAWLORENTZ_DIR = Path(__file__).resolve().parents[1] / "external" / "Chinese-Standard-Mahjong-DRL"
if str(LAWLORENTZ_DIR) not in sys.path:
    sys.path.insert(0, str(LAWLORENTZ_DIR))

from feature import FeatureAgent  # noqa: E402
from model import CNNModel  # noqa: E402


TILE_LIST = list(FeatureAgent.TILE_LIST)
TILE_ORDER = {tile: index for index, tile in enumerate(TILE_LIST)}


@dataclass(frozen=True)
class EffectiveProfile:
    fan8_wait_tiles: int
    fan8_wait_types: int
    min_shanten: int
    first_effective_tiles: int
    first_effective_types: int
    second_effective_tiles: int
    second_effective_types: int
    third_effective_tiles: int
    third_effective_types: int
    max_fan: int

    def key(self) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        return (
            self.fan8_wait_tiles,
            self.fan8_wait_types,
            -self.min_shanten,
            self.first_effective_tiles,
            self.first_effective_types,
            self.second_effective_tiles,
            self.second_effective_types,
            self.third_effective_tiles,
            self.third_effective_types,
            self.max_fan,
        )


class LawlorentzEffectiveScorer:
    def __init__(
        self,
        *,
        packs: Iterable[tuple] = (),
        shown_tiles: Counter[str] | dict[str, int] | None = None,
        seat_wind: int = 0,
        prevalent_wind: int = 0,
        levels: int = 3,
    ):
        self.packs = tuple(packs)
        self.shown_tiles = Counter(shown_tiles or {})
        self.seat_wind = int(seat_wind)
        self.prevalent_wind = int(prevalent_wind)
        self.levels = max(0, min(3, int(levels)))
        self._profile_cache: dict[tuple, EffectiveProfile] = {}
        self._wait_cache: dict[tuple, tuple[int, int, int, frozenset[str]]] = {}
        self._first_cache: dict[tuple, tuple[int, int, frozenset[str]]] = {}
        self._second_cache: dict[tuple, tuple[int, int, frozenset[str]]] = {}
        self._third_cache: dict[tuple, tuple[int, int, frozenset[str]]] = {}

    def discard_key(self, hand: Iterable[str], discard: str) -> tuple:
        after = list(hand)
        try:
            after.remove(discard)
        except ValueError:
            return (-1_000_000, discard)
        shown = Counter(self.shown_tiles)
        shown[discard] += 1
        profile = self.profile(after, shown)
        return (*profile.key(), -TILE_ORDER.get(discard, 999), discard)

    def profile(
        self,
        hand: Iterable[str],
        shown_tiles: Counter[str] | dict[str, int] | None = None,
    ) -> EffectiveProfile:
        hand_tuple = tuple(sorted(hand, key=_tile_sort_key))
        shown = Counter(self.shown_tiles if shown_tiles is None else shown_tiles)
        key = (hand_tuple, _counter_key(shown), self.packs, self.levels)
        cached = self._profile_cache.get(key)
        if cached is not None:
            return cached

        waits = self._fan8_waits(hand_tuple, shown)
        first_types = first_tiles = 0
        second_types = second_tiles = 0
        third_types = third_tiles = 0
        if self.levels >= 1:
            first_types, first_tiles, _ = self._first_effective(hand_tuple, shown)
        if self.levels >= 2:
            second_types, second_tiles, _ = self._second_effective(hand_tuple, shown)
        if self.levels >= 3:
            third_types, third_tiles, _ = self._third_effective(hand_tuple, shown)

        profile = EffectiveProfile(
            fan8_wait_tiles=waits[1],
            fan8_wait_types=waits[0],
            min_shanten=_shanten(self.packs, hand_tuple),
            first_effective_tiles=first_tiles,
            first_effective_types=first_types,
            second_effective_tiles=second_tiles,
            second_effective_types=second_types,
            third_effective_tiles=third_tiles,
            third_effective_types=third_types,
            max_fan=waits[2],
        )
        self._profile_cache[key] = profile
        return profile

    def _fan8_waits(
        self, hand: tuple[str, ...], shown: Counter[str]
    ) -> tuple[int, int, int, frozenset[str]]:
        key = (hand, _counter_key(shown), self.packs)
        cached = self._wait_cache.get(key)
        if cached is not None:
            return cached
        waits: set[str] = set()
        live_total = 0
        max_fan = 0
        hand_counts = Counter(hand)
        for tile in TILE_LIST:
            live = max(0, 4 - hand_counts[tile] - shown[tile])
            if live <= 0:
                continue
            fan = _structural_fan(
                self.packs,
                hand,
                tile,
                seat_wind=self.seat_wind,
                prevalent_wind=self.prevalent_wind,
            )
            if fan >= 8:
                waits.add(tile)
                live_total += live
                max_fan = max(max_fan, fan)
        result = (len(waits), live_total, max_fan, frozenset(waits))
        self._wait_cache[key] = result
        return result

    def _first_effective(
        self, hand: tuple[str, ...], shown: Counter[str]
    ) -> tuple[int, int, frozenset[str]]:
        key = (hand, _counter_key(shown), self.packs)
        cached = self._first_cache.get(key)
        if cached is not None:
            return cached
        base_shanten = _shanten(self.packs, hand)
        effective: set[str] = set()
        live_total = 0
        for draw, live in self._live_draws(hand, shown):
            drawn = list(hand)
            drawn.append(draw)
            for discard in sorted(set(drawn), key=_tile_sort_key):
                after = list(drawn)
                after.remove(discard)
                if _shanten(self.packs, tuple(after)) >= base_shanten:
                    continue
                after_shown = Counter(shown)
                after_shown[discard] += 1
                if self._fan8_waits(tuple(sorted(after, key=_tile_sort_key)), after_shown)[1] > 0:
                    effective.add(draw)
                    live_total += live
                    break
        result = (len(effective), live_total, frozenset(effective))
        self._first_cache[key] = result
        return result

    def _second_effective(
        self, hand: tuple[str, ...], shown: Counter[str]
    ) -> tuple[int, int, frozenset[str]]:
        return self._higher_class_effective(hand, shown, lower="first")

    def _third_effective(
        self, hand: tuple[str, ...], shown: Counter[str]
    ) -> tuple[int, int, frozenset[str]]:
        return self._higher_class_effective(hand, shown, lower="second")

    def _higher_class_effective(
        self,
        hand: tuple[str, ...],
        shown: Counter[str],
        *,
        lower: str,
    ) -> tuple[int, int, frozenset[str]]:
        cache = self._second_cache if lower == "first" else self._third_cache
        key = (hand, _counter_key(shown), self.packs)
        cached = cache.get(key)
        if cached is not None:
            return cached
        base_shanten = _shanten(self.packs, hand)
        if lower == "first":
            _, base_lower_live, lower_tiles = self._first_effective(hand, shown)
        else:
            _, base_lower_live, lower_tiles = self._second_effective(hand, shown)
        effective: set[str] = set()
        live_total = 0
        for draw, live in self._live_draws(hand, shown):
            if draw in lower_tiles:
                continue
            drawn = list(hand)
            drawn.append(draw)
            for discard in sorted(set(drawn), key=_tile_sort_key):
                after = list(drawn)
                after.remove(discard)
                if _shanten(self.packs, tuple(after)) > base_shanten:
                    continue
                after_shown = Counter(shown)
                after_shown[discard] += 1
                sorted_after = tuple(sorted(after, key=_tile_sort_key))
                if lower == "first":
                    _, lower_live, _ = self._first_effective(sorted_after, after_shown)
                else:
                    _, lower_live, _ = self._second_effective(sorted_after, after_shown)
                if lower_live > base_lower_live:
                    effective.add(draw)
                    live_total += live
                    break
        result = (len(effective), live_total, frozenset(effective))
        cache[key] = result
        return result

    def _live_draws(self, hand: tuple[str, ...], shown: Counter[str]) -> list[tuple[str, int]]:
        hand_counts = Counter(hand)
        return [
            (tile, live)
            for tile in TILE_LIST
            if (live := max(0, 4 - hand_counts[tile] - shown[tile])) > 0
        ]


class LawlorentzEffectivePolicy:
    """Botzone text-protocol policy using Lawlorentz state updates."""

    def __init__(self, *, levels: int = 1):
        self.levels = levels
        self.agent: FeatureAgent | None = None
        self.seat_wind = 0
        self.prevalent_wind = 0
        self.zimo = False
        self.angang: str | None = None
        self.stats: Counter[str] = Counter()
        self.last_response: str | None = None

    def respond(self, request: str) -> str:
        tokens = request.strip().split()
        if not tokens:
            return "PASS"
        try:
            response = self._respond(tokens)
        except Exception:
            self.stats["errors"] += 1
            response = "PASS"
        self.last_response = response
        return response

    def _respond(self, tokens: list[str]) -> str:
        if tokens[0] == "0":
            self.seat_wind = int(tokens[1])
            self.prevalent_wind = int(tokens[2]) if len(tokens) >= 3 else 0
            self.agent = FeatureAgent(self.seat_wind)
            self.agent.request2obs(f"Wind {self.prevalent_wind}")
            return "PASS"
        self._ensure_agent()
        assert self.agent is not None

        if tokens[0] == "1":
            self.agent.request2obs(" ".join(["Deal", *tokens[5:]]))
            return "PASS"
        if tokens[0] == "2":
            obs = self.agent.request2obs(f"Draw {tokens[1]}")
            response = self._choose_from_obs(obs, allow_meld=False)
            parts = response.split()
            if parts and parts[0] == "GANG" and len(parts) == 2:
                self.angang = parts[1]
            return response
        if tokens[0] != "3" or len(tokens) < 3:
            return "PASS"

        actor = int(tokens[1])
        if tokens[2] == "DRAW":
            self.agent.request2obs(f"Player {actor} Draw")
            self.zimo = True
            return "PASS"
        if tokens[2] == "GANG":
            if actor == self.seat_wind and self.angang:
                self.agent.request2obs(f"Player {actor} AnGang {self.angang}")
            elif self.zimo:
                self.agent.request2obs(f"Player {actor} AnGang")
            else:
                self.agent.request2obs(f"Player {actor} Gang")
            self.angang = None
            return "PASS"
        if tokens[2] == "BUGANG" and len(tokens) >= 4:
            obs = self.agent.request2obs(f"Player {actor} BuGang {tokens[3]}")
            if actor == self.seat_wind:
                return "PASS"
            return self._choose_reaction(obs)

        self.zimo = False
        if tokens[2] == "CHI":
            self.agent.request2obs(f"Player {actor} Chi {tokens[3]}")
        elif tokens[2] == "PENG":
            self.agent.request2obs(f"Player {actor} Peng")
        obs = self.agent.request2obs(f"Player {actor} Play {tokens[-1]}")
        if actor == self.seat_wind:
            return "PASS"
        return self._choose_reaction(obs)

    def _choose_reaction(self, obs: dict) -> str:
        actions = self._valid_actions(obs)
        responses = [self.agent.action2response(action) for action in actions]  # type: ignore[union-attr]
        if "Hu" in responses:
            self.stats["hu_taken"] += 1
            return "HU"
        best_claim: tuple[tuple, str] | None = None
        pass_profile = self._current_profile_key()
        for response in responses:
            head = response.split()[0]
            if head not in {"Peng", "Chi"}:
                continue
            trial = copy.deepcopy(self.agent)
            claim_obs = trial.request2obs(f"Player {self.seat_wind} {response}")
            discard, key = self._choose_play_with_agent(trial, claim_obs)
            if discard is None:
                continue
            if key <= pass_profile:
                continue
            botzone = f"{head.upper()} {response.split()[1]} {discard}" if head == "Chi" else f"PENG {discard}"
            candidate = (key, botzone)
            if best_claim is None or candidate > best_claim:
                best_claim = candidate
        if best_claim is None:
            return "PASS"
        self.stats["meld_taken"] += 1
        return best_claim[1]

    def _choose_from_obs(self, obs: dict, *, allow_meld: bool) -> str:
        actions = self._valid_actions(obs)
        responses = [self.agent.action2response(action) for action in actions]  # type: ignore[union-attr]
        if "Hu" in responses:
            self.stats["hu_taken"] += 1
            return "HU"
        discard, _ = self._choose_play_with_agent(self.agent, obs)  # type: ignore[arg-type]
        if discard is not None:
            self.stats["draw_discards"] += 1
            return f"PLAY {discard}"
        if allow_meld:
            for response in responses:
                if response.startswith("Gang "):
                    return response.upper()
                if response.startswith("BuGang "):
                    return response.replace("BuGang", "BUGANG")
        return "PASS"

    def _choose_play_with_agent(self, agent: FeatureAgent, obs: dict) -> tuple[str | None, tuple]:
        actions = np.flatnonzero(obs["action_mask"] > 0)
        play_tiles = []
        for action in actions:
            response = agent.action2response(int(action))
            if response.startswith("Play "):
                play_tiles.append(response.split()[1])
        if not play_tiles:
            return None, (-1_000_000,)
        scorer = LawlorentzEffectiveScorer(
            packs=tuple(agent.packs[0]),
            shown_tiles=Counter(agent.shownTiles),
            seat_wind=self.seat_wind,
            prevalent_wind=self.prevalent_wind,
            levels=self.levels,
        )
        return max(
            ((tile, scorer.discard_key(agent.hand, tile)) for tile in play_tiles),
            key=lambda item: item[1],
        )

    def _current_profile_key(self) -> tuple:
        assert self.agent is not None
        scorer = LawlorentzEffectiveScorer(
            packs=tuple(self.agent.packs[0]),
            shown_tiles=Counter(self.agent.shownTiles),
            seat_wind=self.seat_wind,
            prevalent_wind=self.prevalent_wind,
            levels=self.levels,
        )
        return scorer.profile(tuple(self.agent.hand)).key()

    def _valid_actions(self, obs: dict) -> list[int]:
        return [int(action) for action in np.flatnonzero(obs["action_mask"] > 0)]

    def _ensure_agent(self) -> None:
        if self.agent is None:
            self.agent = FeatureAgent(self.seat_wind)
            self.agent.request2obs(f"Wind {self.prevalent_wind}")

    def diagnostics(self) -> dict[str, int | str | None]:
        data = {key: int(value) for key, value in sorted(self.stats.items())}
        data["kind"] = "lawlorentz_effective"
        data["last_response"] = self.last_response
        return data


class LawlorentzModelPolicy(LawlorentzEffectivePolicy):
    """Botzone policy backed by an upstream Lawlorentz CNNModel checkpoint."""

    def __init__(self, model_path: str | Path, *, device: str | None = None):
        super().__init__(levels=0)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CNNModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model_path = str(model_path)

    def _choose_from_obs(self, obs: dict, *, allow_meld: bool) -> str:
        response = self._model_lawlorentz_response(obs, self.agent)
        parts = response.split()
        if not parts:
            return "PASS"
        if parts[0] == "Hu":
            self.stats["hu_taken"] += 1
            return "HU"
        if parts[0] == "Play" and len(parts) >= 2:
            self.stats["draw_discards"] += 1
            return f"PLAY {parts[1]}"
        if parts[0] == "Gang" and len(parts) >= 2:
            return f"GANG {parts[1]}"
        if parts[0] == "BuGang" and len(parts) >= 2:
            return f"BUGANG {parts[1]}"
        return "PASS"

    def _choose_reaction(self, obs: dict) -> str:
        response = self._model_lawlorentz_response(obs, self.agent)
        parts = response.split()
        if not parts:
            return "PASS"
        if parts[0] == "Hu":
            self.stats["hu_taken"] += 1
            return "HU"
        if parts[0] == "Pass":
            return "PASS"
        if parts[0] == "Gang":
            return "GANG"
        if parts[0] in {"Peng", "Chi"}:
            assert self.agent is not None
            trial = copy.deepcopy(self.agent)
            claim_request = f"Player {self.seat_wind} {parts[0]}"
            if parts[0] == "Chi" and len(parts) >= 2:
                claim_request += f" {parts[1]}"
            claim_obs = trial.request2obs(claim_request)
            discard = self._model_discard(claim_obs, trial)
            if discard is None:
                return "PASS"
            self.stats["meld_taken"] += 1
            if parts[0] == "Chi":
                return f"CHI {parts[1]} {discard}"
            return f"PENG {discard}"
        return "PASS"

    def _model_lawlorentz_response(self, obs: dict, agent: FeatureAgent | None) -> str:
        if agent is None:
            return "Pass"
        with torch.no_grad():
            logits, _ = self.model(
                {
                    "observation": torch.from_numpy(np.expand_dims(obs["observation"], 0)).to(self.device),
                    "action_mask": torch.from_numpy(np.expand_dims(obs["action_mask"], 0)).to(self.device),
                }
            )
        self.stats["model_calls"] += 1
        action = int(torch.argmax(logits, dim=1).item())
        return agent.action2response(action)

    def _model_discard(self, obs: dict, agent: FeatureAgent) -> str | None:
        response = self._model_lawlorentz_response(obs, agent)
        if response.startswith("Play "):
            return response.split()[1]
        for action in np.flatnonzero(obs["action_mask"] > 0):
            fallback = agent.action2response(int(action))
            if fallback.startswith("Play "):
                return fallback.split()[1]
        return None

    def diagnostics(self) -> dict[str, int | str | None]:
        data = {key: int(value) for key, value in sorted(self.stats.items())}
        data["kind"] = "lawlorentz_model"
        data["model_path"] = self.model_path
        data["last_response"] = self.last_response
        return data


@lru_cache(maxsize=500_000)
def _structural_fan(
    packs: tuple,
    hand: tuple[str, ...],
    win_tile: str,
    *,
    seat_wind: int,
    prevalent_wind: int,
) -> int:
    try:
        fans = MahjongFanCalculator(
            pack=tuple(packs),
            hand=tuple(hand),
            winTile=win_tile,
            flowerCount=0,
            isSelfDrawn=True,
            is4thTile=False,
            isAboutKong=False,
            isWallLast=False,
            seatWind=seat_wind,
            prevalentWind=prevalent_wind,
            verbose=True,
        )
    except Exception:
        return 0
    return int(sum(fan_point * count for fan_point, count, _name, _name_en in fans))


@lru_cache(maxsize=200_000)
def _shanten(packs: tuple, hand: tuple[str, ...]) -> int:
    try:
        return int(MahjongShanten(pack=tuple(packs), hand=tuple(hand)))
    except Exception:
        return 99


def _counter_key(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((tile, int(count)) for tile, count in counter.items() if count > 0))


def _tile_sort_key(tile: str) -> int:
    return TILE_ORDER.get(tile, 999)
