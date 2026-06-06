"""Deterministic tile attribution for MCR fan-backward rewards.

The paper's fan-backward algorithm requires each fan item to carry the tiles
that produced that fan. The public MahjongGB/Botzone fan table reports fan
names, counts, and scores, but not the tile sets, so this module reconstructs
only the fan types whose tile support is structurally unambiguous from the
final state. Unsupported names raise instead of silently creating approximate
fan-backward rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .fan_backward import FanItem
from .tiles import is_suited, suited_rank, tile_id, tile_name


class FanAttributionError(ValueError):
    pass


@dataclass(frozen=True)
class AttributedFanItems:
    fans: list[FanItem]
    unsupported: tuple[str, ...] = ()


WINNING_TILE_FANS = {
    "自摸",
    "妙手回春",
    "海底捞月",
    "杠上开花",
    "抢杠和",
    "和绝张",
    "边张",
    "嵌张",
    "单钓将",
    "selfdrawn",
    "lasttiledraw",
    "lasttileclaim",
    "outwithreplacementtile",
    "robbingthekong",
    "lasttile",
    "edgewait",
    "closedwait",
    "singlewait",
}


WHOLE_HAND_FANS = {
    "大四喜",
    "大三元",
    "绿一色",
    "九莲宝灯",
    "四杠",
    "连七对",
    "十三幺",
    "清幺九",
    "小四喜",
    "小三元",
    "字一色",
    "四暗刻",
    "一色双龙会",
    "一色四同顺",
    "一色四节高",
    "一色四步高",
    "混幺九",
    "七对",
    "七星不靠",
    "全双刻",
    "清一色",
    "一色三同顺",
    "一色三节高",
    "全大",
    "全中",
    "全小",
    "全带五",
    "三暗刻",
    "全不靠",
    "组合龙",
    "大于五",
    "小于五",
    "推不倒",
    "三色三同顺",
    "三色三节高",
    "无番和",
    "碰碰和",
    "混一色",
    "三色三步高",
    "五门齐",
    "全求人",
    "双暗杠",
    "双箭刻",
    "全带幺",
    "不求人",
    "双明杠",
    "门前清",
    "平和",
    "断幺",
    "缺一门",
    "无字",
    "bigfourwinds",
    "bigthreedragons",
    "allgreen",
    "ninegates",
    "fourkongs",
    "sevenshiftedpairs",
    "thirteenorphans",
    "allterminals",
    "littlefourwinds",
    "littlethreedragons",
    "allhonors",
    "fourconcealedpungs",
    "pureterminalchows",
    "quadruplechow",
    "fourpureshiftedpungs",
    "fourpureshiftedchows",
    "allterminalsandhonors",
    "sevenpairs",
    "greaterhonorsandknittedtiles",
    "allevenpungs",
    "fullflush",
    "puretriplechow",
    "pureshiftedpungs",
    "uppertiles",
    "middletiles",
    "lowertiles",
    "allfive",
    "threeconcealedpungs",
    "lesserhonorsandknittedtiles",
    "knittedstraight",
    "upperfour",
    "lowerfour",
    "reversibletiles",
    "mixedtriplechow",
    "mixedshiftedpungs",
    "chickenhand",
    "allpungs",
    "halfflush",
    "mixedshiftedchows",
    "alltypes",
    "meldedhand",
    "twoconcealedkongs",
    "twodragonspungs",
    "outsidehand",
    "fullyconcealedhand",
    "twomeldedkongs",
    "concealedhand",
    "allchows",
    "allsimples",
    "onevoidedsuit",
    "nohonors",
}


FLOWER_FANS = {"花牌", "flowertiles"}


def attribute_display_fans(
    display: dict[str, Any],
    *,
    hand_hu: torch.Tensor,
    winner: int,
    winning_tile: int | None,
    prevailing_wind: int,
) -> AttributedFanItems:
    fans: list[FanItem] = []
    unsupported: list[str] = []
    hand_tiles = tuple(tile_id for tile_id in torch.nonzero(hand_hu > 0, as_tuple=False).flatten().tolist())
    pung_tiles = tuple(index for index in hand_tiles if float(hand_hu[index].item()) >= 3.0)
    kong_tiles = tuple(index for index in hand_tiles if float(hand_hu[index].item()) >= 4.0)
    fan_rows = display.get("fan") or []
    for row in fan_rows:
        name = str(row.get("name") or "")
        count = max(1, int(row.get("cnt", row.get("count", 1)) or 1))
        value = float(row.get("value", row.get("fan", row.get("score", 0.0))) or 0.0)
        total = value * float(count)
        normalized = normalize_fan_name(name)
        if normalized in FLOWER_FANS:
            unsupported.append(name)
            continue
        tile_groups = tile_groups_for_fan(
            normalized,
            hand_tiles=hand_tiles,
            pung_tiles=pung_tiles,
            kong_tiles=kong_tiles,
            winning_tile=winning_tile,
            winner=winner,
            prevailing_wind=prevailing_wind,
        )
        if not tile_groups:
            unsupported.append(name)
            continue
        expanded_groups = repeat_or_trim(tile_groups, count)
        score_per_group = total / float(len(expanded_groups))
        fans.extend(FanItem(score=score_per_group, tiles=tuple(group)) for group in expanded_groups if group)
    return AttributedFanItems(fans=fans, unsupported=tuple(unsupported))


def tile_groups_for_fan(
    normalized: str,
    *,
    hand_tiles: tuple[int, ...],
    pung_tiles: tuple[int, ...],
    kong_tiles: tuple[int, ...],
    winning_tile: int | None,
    winner: int,
    prevailing_wind: int,
) -> list[tuple[int, ...]]:
    if normalized in WHOLE_HAND_FANS:
        return [hand_tiles]
    if normalized in WINNING_TILE_FANS:
        return [(int(winning_tile),)] if winning_tile is not None else []
    if normalized in {"箭刻", "dragonpung"}:
        return [(tile,) for tile in pung_tiles if tile_name(tile).startswith("J")]
    if normalized in {"圈风刻", "prevalentwind"}:
        tile = tile_id(f"F{int(prevailing_wind) + 1}")
        return [(tile,)] if tile in pung_tiles else []
    if normalized in {"门风刻", "seatwind"}:
        tile = tile_id(f"F{int(winner) + 1}")
        return [(tile,)] if tile in pung_tiles else []
    if normalized in {"幺九刻", "pungofterminalsorhonors"}:
        return [(tile,) for tile in pung_tiles if is_terminal_or_honor(tile)]
    if normalized in {"明杠", "暗杠", "明暗杠", "meldedkong", "concealedkong", "concealedkongandmeldedkong"}:
        return [(tile,) for tile in kong_tiles]
    if normalized in {"清龙", "purestraight"}:
        return pure_straight_groups(hand_tiles)
    if normalized in {"花龙", "mixedstraight"}:
        return mixed_straight_groups(hand_tiles)
    if normalized in {"三同刻", "triplepung"}:
        return same_rank_pung_groups(pung_tiles, size=3)
    if normalized in {"双同刻", "doublepung"}:
        return same_rank_pung_groups(pung_tiles, size=2)
    return []


def repeat_or_trim(groups: list[tuple[int, ...]], count: int) -> list[tuple[int, ...]]:
    if not groups:
        return []
    if len(groups) >= count:
        return groups[:count]
    if len(groups) == 1:
        return groups * count
    return groups


def normalize_fan_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def is_terminal_or_honor(index: int) -> bool:
    tile = tile_name(index)
    if not is_suited(tile):
        return True
    return suited_rank(tile) in {1, 9}


def pure_straight_groups(hand_tiles: tuple[int, ...]) -> list[tuple[int, ...]]:
    present = set(hand_tiles)
    groups = []
    for suit in ("W", "T", "B"):
        group = tuple(tile_id(f"{suit}{rank}") for rank in range(1, 10))
        if all(tile in present for tile in group):
            groups.append(group)
    return groups


def mixed_straight_groups(hand_tiles: tuple[int, ...]) -> list[tuple[int, ...]]:
    present = set(hand_tiles)
    groups = []
    suits = ("W", "T", "B")
    for low in suits:
        for middle in suits:
            for high in suits:
                if len({low, middle, high}) != 3:
                    continue
                group = tuple(
                    [*(tile_id(f"{low}{rank}") for rank in range(1, 4))]
                    + [*(tile_id(f"{middle}{rank}") for rank in range(4, 7))]
                    + [*(tile_id(f"{high}{rank}") for rank in range(7, 10))]
                )
                if all(tile in present for tile in group):
                    groups.append(group)
    return groups


def same_rank_pung_groups(pung_tiles: tuple[int, ...], *, size: int) -> list[tuple[int, ...]]:
    by_rank: dict[int, list[int]] = {}
    for tile in pung_tiles:
        name = tile_name(tile)
        if is_suited(name):
            by_rank.setdefault(suited_rank(name), []).append(tile)
    return [tuple(sorted(tiles)[:size]) for tiles in by_rank.values() if len(tiles) >= size]
