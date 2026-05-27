import sys
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from official_trajectories import game_from_official_result, run_trajectory_set


def test_game_from_official_result_extracts_requested_responses_and_scores():
    result = {
        "scores": [32, -8, -8, -16],
        "log": [
            {"output": {"content": {"0": "0 0 1", "1": "0 1 1"}}},
            {
                "0": {"response": "PASS", "verdict": "OK"},
                "1": {"response": "PASS", "verdict": "OK"},
            },
            {"output": {"content": {"0": "2 W1", "1": "3 0 DRAW"}}},
            {
                "0": {"response": "PLAY W1", "verdict": "OK"},
                "1": {"response": "PASS", "verdict": "OK"},
            },
        ],
    }

    game = game_from_official_result(result, match_id="g1")

    assert game["match_id"] == "g1"
    assert game["scores"] == [32, -8, -8, -16]
    assert game["rewards"] == [1.0, 0.0, 0.0, 0.0]
    assert game["trajectory"] == [
        {"player": 0, "request": "0 0 1", "response": "PASS"},
        {"player": 1, "request": "0 1 1", "response": "PASS"},
        {"player": 0, "request": "2 W1", "response": "PLAY W1"},
        {"player": 1, "request": "3 0 DRAW", "response": "PASS"},
    ]


def test_game_from_official_result_preserves_finish_fan_metadata():
    result = {
        "terminal_reason": "finish",
        "turns": 42,
        "scores": [-8, -8, 40, -24],
        "final_output": {
            "command": "finish",
            "content": {"0": -8, "1": -8, "2": 40, "3": -24},
            "display": {
                "action": "HU",
                "player": 2,
                "fanCnt": 16,
                "score": [-8, -8, 40, -24],
            },
        },
        "log": [],
    }

    game = game_from_official_result(result, match_id="fan")

    assert game["finish"] == {
        "action": "HU",
        "winner": 2,
        "fan_count": 16,
        "score": [-8, -8, 40, -24],
    }


def test_game_from_official_result_scales_negative_player_rewards_to_floor():
    result = {
        "scores": [-30, 10, 10, 10],
        "log": [
            {"output": {"content": {"0": "2 W1"}}},
            {"0": {"response": "HU", "verdict": "OK"}},
        ],
    }

    game = game_from_official_result(result, match_id="bad")

    assert game["rewards"][0] == -1.0
    assert game["rewards"][1:] == [1 / 3, 1 / 3, 1 / 3]


def test_run_trajectory_set_can_create_external_policy_kinds(monkeypatch, tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"initdata": {"seed": 1}}\n', encoding="utf-8")
    calls = []

    def fake_make_policy(kind, model=None, aleo_exe="default-aleo", sample_exe="default-sample"):
        calls.append((kind, model, aleo_exe, sample_exe))
        return {"kind": kind}

    def fake_run_match(policies, initdata, exe_path=None, max_turns=500):
        return {
            "terminal_reason": "finish",
            "turns": 1,
            "scores": [0, 0, 0, 0],
            "log": [],
        }

    monkeypatch.setattr("official_trajectories.make_policy", fake_make_policy)
    monkeypatch.setattr("official_trajectories.run_match", fake_run_match)

    summary = run_trajectory_set(
        Namespace(
            policy="aleo",
            model=None,
            opponent="sample",
            opponent_model=None,
            raw=str(raw),
            games=1,
            offset=0,
            max_turns=10,
            judge="judge.exe",
            aleo_exe="custom-aleo.exe",
            sample_exe="custom-sample.exe",
        )
    )

    assert summary["games"] == 1
    assert calls == [
        ("aleo", None, "custom-aleo.exe", "custom-sample.exe"),
        ("sample", None, "custom-aleo.exe", "custom-sample.exe"),
        ("sample", None, "custom-aleo.exe", "custom-sample.exe"),
        ("sample", None, "custom-aleo.exe", "custom-sample.exe"),
    ]
