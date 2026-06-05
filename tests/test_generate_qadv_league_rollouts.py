import argparse
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_qadv_league_rollouts as rollouts  # noqa: E402


class _EchoPolicy:
    def __init__(self, name):
        self.name = name

    def respond(self, request):
        return "PASS"


def test_parse_policy_spec_accepts_transformer_qadv_variant():
    spec = rollouts.parse_policy_spec(
        "name=qadv005,policy=transformer,model=models/base.pt,"
        "qadv_model=models/qadv.pt,qadv_lambda=0.05"
    )

    assert spec.name == "qadv005"
    assert spec.policy == "transformer"
    assert spec.model == "models/base.pt"
    assert spec.qadv_model == "models/qadv.pt"
    assert spec.qadv_lambda == 0.05


def test_run_rollout_set_writes_rows_with_policy_pool_and_terminal_result(monkeypatch, tmp_path):
    made = []

    def fake_make_policy(kind, model=None, **kwargs):
        made.append((kind, model, kwargs.get("qadv_model"), kwargs.get("qadv_lambda")))
        return _EchoPolicy(kind)

    def fake_run_match(policies, initdata, **kwargs):
        return {
            "terminal_reason": "finish",
            "turns": 2,
            "scores": [16, -8, -4, -4],
            "final_output": {"display": {"action": "HU", "player": 0, "fanCnt": 8, "baseFanCnt": 8}},
            "policy_diagnostics": [
                {"legal_hu_seen": 2, "fan_check_accepts": 1, "last_fallback_used": False},
                {"legal_hu_seen": 3, "fan_check_rejects": 2},
                {"legal_hu_seen": 5, "illegal_predictions": 1},
                {"draw_turns": 7},
            ],
            "log": [
                {"output": {"content": {"0": "2 W1", "2": "3 0 PLAY W1"}}},
                {"0": {"response": "PLAY W1"}, "2": {"response": "PASS"}},
            ],
        }

    monkeypatch.setattr(rollouts, "load_initdata", lambda raw, limit, offset: [{"seed": 1}])
    monkeypatch.setattr(rollouts, "make_policy", fake_make_policy)
    monkeypatch.setattr(rollouts, "run_match", fake_run_match)

    out_jsonl = tmp_path / "rollouts.jsonl"
    summary_out = tmp_path / "summary.json"
    args = argparse.Namespace(
        policy_spec=[
            "name=base,policy=transformer,model=models/base.pt",
            "name=qadv005,policy=transformer,model=models/base.pt,qadv_model=models/qadv.pt,qadv_lambda=0.05",
        ],
        raw="data/raw/example.jsonl",
        games=1,
        offset=0,
        max_turns=10,
        judge="judge.exe",
        aleo_exe="aleo.exe",
        sample_exe="sample.exe",
        lawlorentz_levels=1,
        out_jsonl=str(out_jsonl),
        summary_out=str(summary_out),
        min_games=1,
        min_rows=1,
        min_nonzero_score_rate=0.0,
        min_return_std=0.0,
        require_policy_pool_size=2,
        fail_on_gate=True,
    )

    summary = rollouts.run_rollout_set(args)

    rows = [rollouts.loads_jsonl_line(line) for line in out_jsonl.read_text().splitlines()]
    assert summary["rows"] == 2
    assert summary["policy_pool_names"] == ["base", "qadv005"]
    assert summary["terminal_action_counts"] == {"HU": 1}
    assert summary["hu_rate"] == 1.0
    assert summary["average_placement_rewards_4_2_1_0"] == [4.0, 0.0, 1.5, 1.5]
    assert summary["policy_seat_games"] == {"base": 2, "qadv005": 2}
    assert summary["policy_hu_counts"] == {"base": 1, "qadv005": 0}
    assert summary["policy_hu_rates"] == {"base": 0.5, "qadv005": 0.0}
    assert summary["policy_average_hu_turns"] == {"base": 2.0, "qadv005": None}
    assert summary["policy_average_final_score_rewards_4_2_1_0"] == {
        "base": 2.75,
        "qadv005": 0.75,
    }
    assert summary["policy_diagnostics_totals"]["base"] == {
        "legal_hu_seen": 7,
        "fan_check_accepts": 1,
        "illegal_predictions": 1,
    }
    assert summary["policy_diagnostics_totals"]["qadv005"] == {
        "legal_hu_seen": 3,
        "fan_check_rejects": 2,
        "draw_turns": 7,
    }
    assert summary["seat_policy_diagnostics_totals"][0] == {
        "legal_hu_seen": 2,
        "fan_check_accepts": 1,
    }
    assert rows[0]["schema_version"] == rollouts.QADV_ROLLOUT_SCHEMA
    assert rows[0]["policy_pool"][0]["name"] == "base"
    assert rows[0]["seat_policy_names"] == ["base", "qadv005", "base", "qadv005"]
    assert rows[0]["terminal_result"]["winner"] == 0
    assert rows[0]["terminal_result"]["scores"] == [16, -8, -4, -4]
    assert rows[0]["terminal_result"]["base_fan_count"] == 8
    assert rows[0]["safety"]["illegal_hu"] is False
    assert made[1] == ("transformer", "models/base.pt", "models/qadv.pt", 0.05)


def test_rollout_gate_rejects_all_zero_terminal_signal():
    summary = rollouts.summarize_games(
        [
            {
                "scores": [0, 0, 0, 0],
                "terminal_result": {"action": "HUANG", "fan_count": None},
                "rows": [],
                "seat_policy_names": ["base", "base", "base", "base"],
            }
        ],
        policy_specs=[rollouts.PolicySpec(name="base", policy="transformer")],
        min_games=1,
        min_rows=0,
        min_nonzero_score_rate=0.05,
        min_return_std=0.03,
        require_policy_pool_size=1,
    )

    assert summary["gate_passed"] is False
    assert summary["gate_failures"]["nonzero_score_rate"] == 0.0
    assert summary["gate_failures"]["return_std"] == 0.0


def test_rollout_gate_rejects_low_fan_hu():
    summary = rollouts.summarize_games(
        [
            {
                "scores": [8, -8, 0, 0],
                "terminal_result": {"action": "HU", "winner": 0, "fan_count": 8, "base_fan_count": 4},
                "rows": [],
                "seat_policy_names": ["base", "base", "base", "base"],
            }
        ],
        policy_specs=[rollouts.PolicySpec(name="base", policy="transformer")],
        min_games=1,
        min_rows=0,
        min_nonzero_score_rate=0.0,
        min_return_std=0.0,
        require_policy_pool_size=1,
    )

    assert summary["gate_passed"] is False
    assert summary["low_fan_hu_count"] == 1
    assert summary["illegal_hu_count"] == 1
    assert summary["gate_failures"]["illegal_hu_count"] == 1


def test_rollout_gate_rejects_wrong_hu_terminal():
    summary = rollouts.summarize_games(
        [
            {
                "scores": [10, 10, -30, 10],
                "terminal_result": {"action": "WH", "winner": 2, "fan_count": 5},
                "rows": [],
                "seat_policy_names": ["base", "base", "base", "base"],
            }
        ],
        policy_specs=[rollouts.PolicySpec(name="base", policy="transformer")],
        min_games=1,
        min_rows=0,
        min_nonzero_score_rate=0.0,
        min_return_std=0.0,
        require_policy_pool_size=1,
    )

    assert summary["gate_passed"] is False
    assert summary["wrong_hu_count"] == 1
    assert summary["low_fan_hu_count"] == 1
    assert summary["illegal_hu_count"] == 1
    assert summary["gate_failures"]["illegal_hu_count"] == 1
