#!/usr/bin/env python3
"""Python bridge for the official MahjongGB fan calculator helper."""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

from official_judge_match import judge_env


DEFAULT_FAN_CHECK = Path("build/official_judge/mcr_fan_check.exe")


class OfficialFanChecker:
    def __init__(self, exe_path: Path | str = DEFAULT_FAN_CHECK):
        self.exe_path = Path(exe_path)

    @classmethod
    def default(cls):
        if DEFAULT_FAN_CHECK.exists():
            return cls(DEFAULT_FAN_CHECK)
        return None

    def can_hu(self, **kwargs) -> bool:
        return bool(self.evaluate(**kwargs)["can_hu"])

    def evaluate(
        self,
        *,
        packs: list[dict],
        hand: list[str],
        win_tile: str | None,
        flower_count: int,
        is_self_draw: bool,
        is_4th_tile: bool,
        is_about_kong: bool,
        is_last: bool,
        seat_wind: int,
        prevalent_wind: int,
        player: int,
    ) -> dict:
        if not win_tile:
            return {"fan": -3, "can_hu": False}
        payload = {
            "packs": packs,
            "hand": hand,
            "win_tile": win_tile,
            "flower_count": flower_count,
            "is_self_draw": is_self_draw,
            "is_4th_tile": is_4th_tile,
            "is_about_kong": is_about_kong,
            "is_last": is_last,
            "seat_wind": seat_wind,
            "prevalent_wind": prevalent_wind,
            "player": player,
        }
        result = dict(self._evaluate_cached(json.dumps(payload, sort_keys=True)))
        total_fan = int(result.get("fan", -3))
        if int(flower_count) == 0:
            base_result = result
        else:
            base_payload = dict(payload)
            base_payload["flower_count"] = 0
            base_result = dict(self._evaluate_cached(json.dumps(base_payload, sort_keys=True)))
        base_fan = int(base_result.get("fan", -3))
        result["fan"] = total_fan
        result["base_fan"] = base_fan
        result["can_hu"] = base_fan >= 8
        return result

    @lru_cache(maxsize=200000)
    def _evaluate_cached(self, payload_text: str) -> dict:
        proc = subprocess.run(
            [str(self.exe_path)],
            input=payload_text,
            text=True,
            capture_output=True,
            env=judge_env(),
            timeout=10,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"fan checker failed rc={proc.returncode}: {proc.stderr[:500]}")
        return json.loads(proc.stdout)
