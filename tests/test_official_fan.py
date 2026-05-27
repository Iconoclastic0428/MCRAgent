import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from official_fan import OfficialFanChecker


def test_official_fan_checker_accepts_high_fan_hand():
    checker = OfficialFanChecker(Path("build/official_judge/mcr_fan_check.exe"))

    result = checker.evaluate(
        packs=[],
        hand="W1 W1 W1 W2 W2 W2 W3 W3 W3 W4 W4 W4 W5".split(),
        win_tile="W5",
        flower_count=0,
        is_self_draw=False,
        is_4th_tile=False,
        is_about_kong=False,
        is_last=False,
        seat_wind=0,
        prevalent_wind=0,
        player=0,
    )

    assert result["fan"] >= 8
    assert result["can_hu"]


def test_official_fan_checker_rejects_non_winning_hand():
    checker = OfficialFanChecker(Path("build/official_judge/mcr_fan_check.exe"))

    result = checker.evaluate(
        packs=[],
        hand="W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 F1 F2".split(),
        win_tile="J1",
        flower_count=0,
        is_self_draw=False,
        is_4th_tile=False,
        is_about_kong=False,
        is_last=False,
        seat_wind=0,
        prevalent_wind=0,
        player=0,
    )

    assert result["fan"] < 8
    assert not result["can_hu"]
