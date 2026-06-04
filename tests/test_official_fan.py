import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from official_fan import OfficialFanChecker


def test_default_fan_checker_prefers_flat_build_path(tmp_path, monkeypatch):
    flat = tmp_path / "build" / "official_judge" / "mcr_fan_check.exe"
    nested = tmp_path / "build" / "official_judge" / "official_judge" / "mcr_fan_check.exe"
    flat.parent.mkdir(parents=True)
    flat.write_text("", encoding="utf-8")
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    checker = OfficialFanChecker.default()

    assert checker is not None
    assert checker.exe_path == Path("build/official_judge/mcr_fan_check.exe")


def test_default_fan_checker_falls_back_to_nested_build_path(tmp_path, monkeypatch):
    nested = tmp_path / "build" / "official_judge" / "official_judge" / "mcr_fan_check.exe"
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    checker = OfficialFanChecker.default()

    assert checker is not None
    assert checker.exe_path == Path("build/official_judge/official_judge/mcr_fan_check.exe")


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
    assert any(item["name"] == "四暗刻" and item["fan"] == 64 for item in result["fan_items"])
    assert result["base_fan_items"] == result["fan_items"]


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


def test_official_fan_checker_excludes_flowers_from_minimum_hu_gate():
    checker = OfficialFanChecker(Path("build/official_judge/mcr_fan_check.exe"))

    result = checker.evaluate(
        packs=[],
        hand="W1 W2 W3 W2 W3 W4 B2 B3 B4 T5 T6 T7 F1".split(),
        win_tile="F1",
        flower_count=4,
        is_self_draw=False,
        is_4th_tile=False,
        is_about_kong=False,
        is_last=False,
        seat_wind=0,
        prevalent_wind=1,
        player=0,
    )

    assert result["fan"] >= 8
    assert result["base_fan"] == 4
    assert not result["can_hu"]
    assert any(item["name"] == "花牌" and item["total"] == 4 for item in result["fan_items"])
    assert all(item["name"] != "花牌" for item in result["base_fan_items"])


def test_official_fan_checker_requires_base_calculator_acceptance(monkeypatch):
    checker = OfficialFanChecker(Path("unused.exe"))

    def fake_evaluate_cached(payload_text):
        return {"fan": 8, "can_hu": False}

    monkeypatch.setattr(checker, "_evaluate_cached", fake_evaluate_cached)

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

    assert result["fan"] == 8
    assert result["base_fan"] == 8
    assert not result["can_hu"]
