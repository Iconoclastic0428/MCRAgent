import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_lawlorentz_dataset import build_dataset, iter_lawlorentz_examples
from lawlorentz_policy import FeatureAgent


def all_pass(players=range(4)):
    return {str(player): {"response": "PASS", "raw": "PASS", "verdict": "OK"} for player in players}


def base_record(extra_logs):
    hands = {
        "0": "W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 B1",
        "1": "W1 W1 B1 B2 B3 B4 B5 B6 B7 T1 T2 T3 J1",
        "2": "W2 W3 W4 W5 W6 W7 W8 W9 T4 T5 T6 B8 B9",
        "3": "B1 B2 B3 B4 B5 B6 B7 B8 B9 F1 F2 F3 J2",
    }
    logs = [
        {"output": {"content": {str(player): f"0 {player} 0" for player in range(4)}, "display": {"action": "INIT"}}},
        all_pass(),
        {
            "output": {
                "content": {
                    player: f"1 0 0 0 0 {tiles}"
                    for player, tiles in hands.items()
                },
                "display": {"action": "DEAL"},
            }
        },
        all_pass(),
    ]
    logs.extend(extra_logs)
    return {"match_id": "unit", "game": "Chinese-Standard-Mahjong", "logs": logs, "scores": {"0": 0, "1": 0, "2": 0, "3": 0}}


def test_iter_lawlorentz_examples_maps_draw_discard_to_feature_agent_action():
    record = base_record(
        [
            {
                "output": {
                    "content": {"0": "2 B2", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                    "display": {"action": "DRAW", "player": 0, "tile": "B2"},
                }
            },
            {
                "0": {"response": "PLAY B2", "raw": "PLAY B2", "verdict": "OK"},
                **all_pass(players=[1, 2, 3]),
            },
        ]
    )

    examples, stats = iter_lawlorentz_examples(record)

    draw = [example for example in examples if example.player == 0 and example.kind == "primary"][0]
    agent = FeatureAgent(0)
    assert agent.action2response(draw.act) == "Play B2"
    assert draw.obs.shape == (71, 4, 9)
    assert draw.mask.shape == (235,)
    assert stats["examples"] >= 1


def test_iter_lawlorentz_examples_adds_claim_discard_training_label():
    record = base_record(
        [
            {
                "output": {
                    "content": {"0": "2 W1", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                    "display": {"action": "DRAW", "player": 0, "tile": "W1"},
                }
            },
            {
                "0": {"response": "PLAY W1", "raw": "PLAY W1", "verdict": "OK"},
                **all_pass(players=[1, 2, 3]),
            },
            {
                "output": {
                    "content": {str(player): "3 0 PLAY W1" for player in range(4)},
                    "display": {"action": "PLAY", "player": 0, "tile": "W1"},
                }
            },
            {
                "0": {"response": "PASS", "raw": "PASS", "verdict": "OK"},
                "1": {"response": "PENG B1", "raw": "PENG B1", "verdict": "OK"},
                "2": {"response": "PASS", "raw": "PASS", "verdict": "OK"},
                "3": {"response": "PASS", "raw": "PASS", "verdict": "OK"},
            },
        ]
    )

    examples, stats = iter_lawlorentz_examples(record)

    player_one = [example.response for example in examples if example.player == 1]
    assert "Peng" in player_one
    assert "Play B1" in player_one
    assert stats["claim_discard_examples"] == 1


def test_iter_lawlorentz_examples_honors_train_players_without_breaking_context():
    record = base_record(
        [
            {
                "output": {
                    "content": {"0": "2 B2", "1": "2 B4"},
                    "display": {"action": "DRAW", "player": 0, "tile": "B2"},
                }
            },
            {
                "0": {"response": "PLAY B2", "raw": "PLAY B2", "verdict": "OK"},
                "1": {"response": "PLAY B4", "raw": "PLAY B4", "verdict": "OK"},
            },
        ]
    )
    record["train_players"] = ["1"]

    examples, stats = iter_lawlorentz_examples(record)

    assert examples
    assert {example.player for example in examples} == {1}
    assert stats["filtered_train_player_examples"] >= 1


def test_build_dataset_writes_lawlorentz_npz_shards_and_manifest(tmp_path):
    record = base_record(
        [
            {
                "output": {
                    "content": {"0": "2 B2", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                    "display": {"action": "DRAW", "player": 0, "tile": "B2"},
                }
            },
            {
                "0": {"response": "PLAY B2", "raw": "PLAY B2", "verdict": "OK"},
                **all_pass(players=[1, 2, 3]),
            },
        ]
    )
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(record) + "\n", encoding="utf-8")

    manifest = build_dataset([raw], tmp_path / "lawlorentz", shard_size=1)

    assert manifest["format"] == "lawlorentz_cooked_npz_v1"
    assert manifest["examples"] >= 1
    counts = json.loads((tmp_path / "lawlorentz" / "count.json").read_text(encoding="utf-8"))
    assert sum(counts) == manifest["examples"]
    shard = np.load(tmp_path / "lawlorentz" / "cooked_data_without0" / "0.npz")
    assert shard["obs"].shape[1:] == (71, 4, 9)
    assert shard["mask"].shape[1:] == (235,)
    assert shard["act"].ndim == 1
