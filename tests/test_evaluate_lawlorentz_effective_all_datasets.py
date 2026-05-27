import json
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_lawlorentz_effective_all_datasets as eval_all


def test_discover_datasets_keeps_initdata_and_replay_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data" / "raw"
    data.mkdir(parents=True)
    good = data / "good.jsonl"
    replay = data / "replay.jsonl"
    bad = data / "bad.jsonl"
    good.write_text('{"initdata": {"srand": 1}}\n', encoding="utf-8")
    replay.write_text('{"logs": []}\n', encoding="utf-8")
    bad.write_text('{"records": []}\n', encoding="utf-8")

    assert eval_all.discover_datasets(["data/raw/*.jsonl"]) == [good, replay]
    assert eval_all.count_initdata_records(good) == 1


def test_evaluate_replay_dataset_records_action_type_metrics(tmp_path):
    raw = tmp_path / "replay.jsonl"
    raw.write_text(
        json.dumps(
            {
                "logs": [
                    {"output": {"content": {"0": "0 0 0"}}},
                    {"0": {"response": "PASS", "raw": "PASS"}},
                    {
                        "output": {
                            "content": {
                                "0": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T1 T1 B1"
                            }
                        }
                    },
                    {"0": {"response": "PASS", "raw": "PASS"}},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = eval_all.evaluate_replay_dataset(raw, limit=1, levels=0)

    assert metrics["matches"] == 1
    assert metrics["examples"] == 2
    assert metrics["exact_accuracy"] == 1.0


def test_evaluate_replay_dataset_records_hu_rate_and_average_turn(tmp_path, monkeypatch):
    raw = tmp_path / "replay_hu.jsonl"
    raw.write_text(
        json.dumps(
            {
                "logs": [
                    {"output": {"content": {"0": "0 0 0"}}},
                    {"0": {"response": "PASS", "raw": "PASS"}},
                    {"output": {"content": {"0": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T1 T1 B1"}}},
                    {"0": {"response": "PASS", "raw": "PASS"}},
                    {"output": {"content": {"0": "3 1 PLAY W1"}}},
                    {"0": {"response": "HU", "raw": "HU"}},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakePolicy:
        def respond(self, request):
            return "HU" if request == "3 1 PLAY W1" else "PASS"

    monkeypatch.setattr(eval_all, "_make_replay_policy", lambda *args, **kwargs: FakePolicy())

    metrics = eval_all.evaluate_replay_dataset(raw, limit=1, levels=0)

    assert metrics["actual_hu_matches"] == 1
    assert metrics["actual_hu_rate"] == 1.0
    assert metrics["average_actual_hu_turn"] == 3.0
    assert metrics["predicted_hu_matches"] == 1
    assert metrics["predicted_hu_rate"] == 1.0
    assert metrics["average_predicted_hu_turn"] == 3.0


def test_run_all_records_dataset_errors(monkeypatch, tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"initdata": {"srand": 1}}\n', encoding="utf-8")
    monkeypatch.setattr(eval_all, "discover_datasets", lambda patterns: [raw])
    monkeypatch.setattr(eval_all, "dependency_status", lambda: {"PyMahjongGB": True})

    def fake_evaluate(path, args):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(eval_all, "evaluate_dataset", fake_evaluate)
    report = eval_all.run_all(
        Namespace(
            dataset_pattern=["data/raw/*.jsonl"],
            games_per_dataset=1,
            lawlorentz_levels=1,
        )
    )

    assert report["dependencies"] == {"PyMahjongGB": True}
    assert report["results"] == [{"dataset": str(raw), "error": "judge unavailable"}]
