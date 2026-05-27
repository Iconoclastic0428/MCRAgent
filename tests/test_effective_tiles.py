import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from effective_tiles import (
    EffectiveTileEvaluator,
    choose_effective_discard,
    evaluate_discard_candidates,
)


class PatternFanChecker:
    def __init__(self, winning_tiles=None, required_hand_tiles=None, forbidden_hand_tiles=None):
        self.winning_tiles = set(winning_tiles or [])
        self.required_hand_tiles = set(required_hand_tiles or [])
        self.forbidden_hand_tiles = set(forbidden_hand_tiles or [])
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        hand_tiles = set(kwargs["hand"])
        if (
            kwargs["win_tile"] in self.winning_tiles
            and self.required_hand_tiles.issubset(hand_tiles)
            and not self.forbidden_hand_tiles.intersection(hand_tiles)
        ):
            return {"can_hu": True, "fan": 8}
        return {"can_hu": False, "fan": 0}


def test_fan8_waits_ignore_incidental_fan_flags():
    checker = PatternFanChecker({"W5"})
    evaluator = EffectiveTileEvaluator(
        checker,
        seat_wind=1,
        prevalent_wind=2,
        player=1,
    )

    profile = evaluator.profile(
        "W1 W1 W1 W2 W2 W2 W3 W3 W3 W4 W4 W4 W5".split(),
    )

    assert profile.fan8_wait_types == 1
    assert profile.fan8_wait_tiles == 3
    assert checker.calls
    assert all(call["is_last"] is False for call in checker.calls)
    assert all(call["is_about_kong"] is False for call in checker.calls)


def test_guideline_prefers_discard_with_more_fan8_waits_over_base_score():
    checker = PatternFanChecker({"W5"}, required_hand_tiles={"W5"}, forbidden_hand_tiles={"J1"})

    scored = evaluate_discard_candidates(
        hand="W1 W1 W1 W2 W2 W2 W3 W3 W3 W4 W4 W4 W5 J1".split(),
        responses=["PLAY W5", "PLAY J1"],
        fan_checker=checker,
        base_scores=[0.99, 0.01],
        base_score_weight=0.0,
    )

    assert scored[0].response == "PLAY J1"
    assert scored[0].profile.fan8_wait_tiles == 3
    assert choose_effective_discard(scored).response == "PLAY J1"


def test_profile_counts_first_class_effective_tiles_when_shanten_improves():
    checker = PatternFanChecker({"W5"})
    evaluator = EffectiveTileEvaluator(checker)

    profile = evaluator.profile(
        "W1 W1 W1 W2 W2 W2 W3 W3 W3 W4 W4 W5 J1".split(),
    )

    assert profile.min_shanten == 1
    assert profile.first_effective_tiles > 0
    assert profile.first_effective_types > 0
