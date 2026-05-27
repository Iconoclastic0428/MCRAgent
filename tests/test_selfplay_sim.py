import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from policy_bot import BotzonePolicy
from selfplay_sim import (
    choose_reaction_winner,
    make_policy_factory,
    reaction_priority,
    run_game,
)


def test_reaction_priority_orders_hu_before_gang_before_peng_before_chi():
    assert reaction_priority("HU") > reaction_priority("GANG")
    assert reaction_priority("GANG") > reaction_priority("PENG W1")
    assert reaction_priority("PENG W1") > reaction_priority("CHI W2 B1")
    assert reaction_priority("CHI W2 B1") > reaction_priority("PASS")


def test_choose_reaction_winner_uses_priority_then_turn_order():
    responses = {0: "PASS", 1: "PENG W1", 2: "GANG", 3: "PENG B1"}

    assert choose_reaction_winner(actor=0, responses=responses) == (2, "GANG")


def test_run_game_produces_trajectory_and_rewards():
    result = run_game(
        [make_policy_factory("fallback") for _ in range(4)],
        seed=3,
        max_turns=12,
    )

    assert result["turns"] > 0
    assert len(result["rewards"]) == 4
    assert result["terminal_reason"] in {"hu", "wall", "turn_limit"}
    assert result["trajectory"]
    assert {"player", "request", "response"}.issubset(result["trajectory"][0])
