import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_chaga_replay as chaga


class FakePolicy:
    def __init__(self, calls):
        self.calls = calls

    def respond(self, request):
        self.calls.append(request)
        if request == "2 W1":
            return "PLAY W2"
        if request == "3 1 PLAY B1":
            return "HU"
        return "PASS"


def test_selected_players_from_raw_record_keeps_chaga02_to_chaga08_only():
    record = {
        "step": {
            "p": [
                {"n": "CHAGA01"},
                {"n": "CHAGA02"},
                {"n": "CHAGA08"},
                {"n": "Human"},
            ]
        }
    }

    assert chaga.selected_players_from_raw_record(record, chaga.DEFAULT_PLAYER_RE) == {
        "1": "CHAGA02",
        "2": "CHAGA08",
    }


def test_chaga_filtered_replay_counts_hu_and_play_match_rates():
    calls = []
    converted = {
        "logs": [
            {"output": {"content": {"0": "0 0 0", "1": "0 1 0"}}},
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}},
            {"output": {"content": {"0": "2 W1", "1": "2 B1"}}},
            {"0": {"raw": "PLAY W1"}, "1": {"raw": "PLAY B1"}},
            {"output": {"content": {"0": "3 1 PLAY B1", "1": "3 0 PLAY W1"}}},
            {"0": {"raw": "HU"}, "1": {"raw": "PASS"}},
        ]
    }

    metrics = chaga.empty_metrics()
    chaga.evaluate_converted_record_for_players(
        converted,
        {"0": "CHAGA02"},
        policy_factory=lambda: FakePolicy(calls),
        metrics=metrics,
    )
    chaga.add_rates(metrics)

    assert "2 B1" in calls
    assert metrics["selected_decisions"] == 3
    assert metrics["selected_actual_play_decisions"] == 1
    assert metrics["selected_actual_play_exact_matches"] == 0
    assert metrics["selected_actual_play_action_type_matches"] == 1
    assert metrics["selected_play_exact_match_rate"] == 0.0
    assert metrics["selected_play_action_type_match_rate"] == 1.0
    assert metrics["selected_actual_hu_decisions"] == 1
    assert metrics["selected_actual_hu_exact_matches"] == 1
    assert metrics["selected_hu_exact_match_rate"] == 1.0
    assert metrics["selected_player_decisions"] == {"CHAGA02": 3}
    assert metrics["per_player"]["CHAGA02"]["selected_play_exact_match_rate"] == 0.0
    assert metrics["per_player"]["CHAGA02"]["selected_hu_exact_match_rate"] == 1.0
    assert metrics["records_with_selected_actual_hu"] == 1
    assert metrics["average_selected_actual_hu_turn"] == 3.0


def test_make_policy_factory_can_wrap_botzone_model(monkeypatch, tmp_path):
    calls = []

    class FakePredictor:
        def __init__(self, path):
            self.path = path

    class FakePolicy:
        def __init__(self, predictor):
            self.predictor = predictor

        def respond(self, request):
            calls.append((self.predictor.path, request))
            return "PASS"

    monkeypatch.setattr(chaga, "SklearnPredictor", FakePredictor)
    monkeypatch.setattr(chaga, "BotzonePolicy", FakePolicy)
    model_path = tmp_path / "model.pkl"
    factory = chaga.make_policy_factory("botzone_model", levels=0, model=str(model_path))

    policy = factory()

    assert policy.respond("0 0 0") == "PASS"
    assert calls == [(model_path, "0 0 0")]


def test_evaluate_prepared_file_uses_embedded_train_player_names(tmp_path):
    raw = tmp_path / "prepared.jsonl"
    raw.write_text(
        '{"train_player_names":{"0":"CHAGA02"},"logs":[{"output":{"content":{"0":"0 0 0","1":"0 1 0"}}},{"0":{"raw":"PASS"},"1":{"raw":"PASS"}},{"output":{"content":{"0":"2 W1","1":"2 B1"}}},{"0":{"raw":"PLAY W1"},"1":{"raw":"PLAY B1"}}]}\n',
        encoding="utf-8",
    )

    metrics = chaga.evaluate_prepared_file(
        raw,
        policy_factory=lambda: FakePolicy([]),
    )

    assert metrics["records_seen"] == 1
    assert metrics["records_evaluated"] == 1
    assert metrics["selected_decisions"] == 2
    assert metrics["selected_actual_play_decisions"] == 1
