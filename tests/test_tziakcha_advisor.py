import sys
from pathlib import Path

from advisor_service.advisor import recommend
from advisor_service.model_advisor import TziakchaModelAdvisor
from advisor_service.state import AdvisorState
from advisor_service.tiles import tile_id_from_botzone_symbol
import torch

TJONG_SRC = Path(__file__).resolve().parents[1] / "papers" / "tjong_cit2_12298" / "src"
if str(TJONG_SRC) not in sys.path:
    sys.path.insert(0, str(TJONG_SRC))

from tjong_replication.policy_bot import respond_json_with_predictor  # noqa: E402


class FixedPredictor:
    def __init__(self, response):
        self.response = response

    def predict_legal_response(self, input_text, hand, player_id, request, candidates):
        return self.response


class PreferHuPredictor(FixedPredictor):
    prefer_hu = True


class RejectingFanChecker:
    def evaluate(self, **kwargs):
        return {"fan": 4, "can_hu": False}


class AcceptingFanChecker:
    def evaluate(self, **kwargs):
        return {"fan": 8, "can_hu": True}


class FanBreakdownChecker:
    def evaluate(self, **kwargs):
        return {
            "fan": 10,
            "base_fan": 10,
            "can_hu": True,
            "fan_items": [
                {"name": "花龙", "fan": 8, "count": 1, "total": 8},
                {"name": "门前清", "fan": 2, "count": 1, "total": 2},
            ],
            "base_fan_items": [
                {"name": "花龙", "fan": 8, "count": 1, "total": 8},
                {"name": "门前清", "fan": 2, "count": 1, "total": 2},
            ],
        }


class FlowerOnlyFanChecker:
    def evaluate(self, **kwargs):
        return {"fan": 8, "base_fan": 4, "can_hu": True}


class RecordingFanChecker:
    def __init__(self, fan=8, can_hu=True):
        self.fan = fan
        self.can_hu = can_hu
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return {"fan": self.fan, "can_hu": self.can_hu}


def test_recommend_does_not_hu_without_fan_checker():
    rec = recommend({"available_actions": {"hu": [0], "pass": [0]}, "hand": [0]})
    assert rec["action"] == "pass"
    assert rec["text"] == "Pass"


def test_recommend_legal_hu_includes_fan_count(monkeypatch):
    monkeypatch.setattr("advisor_service.advisor.calculate_hu_fan", lambda snapshot: {"fan": 8, "source": "aleo"})

    rec = recommend({"available_actions": {"hu": [0]}, "hand": [0], "seat": 0}, use_aleo=True)

    assert rec["action"] == "hu"
    assert rec["fan"] == 8
    assert rec["text"] == "Hu (8 fan)"


def test_recommend_sub_8_hu_passes_instead(monkeypatch):
    monkeypatch.setattr("advisor_service.advisor.calculate_hu_fan", lambda snapshot: {"fan": 4, "source": "aleo"})

    rec = recommend({"available_actions": {"hu": [0], "pass": [0]}, "hand": [0], "seat": 0}, use_aleo=True)

    assert rec["action"] == "pass"
    assert rec["fan"] == 4
    assert "below 8" in rec["text"]


def test_recommend_rejects_aleo_hu_when_only_flowers_reach_8(monkeypatch):
    monkeypatch.setattr(
        "advisor_service.advisor.calculate_hu_fan",
        lambda snapshot: {"fan": 8, "base_fan": 4, "source": "aleo"},
    )

    rec = recommend({"available_actions": {"hu": [0], "pass": [0]}, "hand": [0], "seat": 0}, use_aleo=True)

    assert rec["action"] == "pass"
    assert rec["fan"] == 8
    assert rec["base_fan"] == 4
    assert "base fan" in rec["text"]


def test_recommend_discard_uses_display_name():
    rec = recommend({"available_actions": {"discard": [36]}, "hand": [0, 36]})
    assert rec["action"] == "discard"
    assert rec["tile"] == 36
    assert rec["text"] == "Discard 1s"


def test_recommend_discard_from_fourteen_tile_hand():
    rec = recommend(
        {
            "seat": 0,
            "turn": 0,
            "available_actions": {},
            "hand": [0, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76, 80, 124, 132],
        }
    )
    assert rec["action"] == "discard"
    assert rec["tile"] in {124, 132}
    assert rec["text"].startswith("Discard ")


def test_model_advisor_uses_model_draw_discard():
    advisor = TziakchaModelAdvisor(
        predictor=FixedPredictor("PLAY T2"),
        fan_checker=RejectingFanChecker(),
    )

    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 0,
            "available_actions": {"discard": [40]},
            "hand": [0, 4, 36, 40, 44, 72, 76, 80, 124, 132, 108, 112, 116, 120],
            "last_draw": {"seat": 0, "tile": 40},
        }
    )

    assert rec["action"] == "discard"
    assert rec["tile_symbol"] == "T2"
    assert rec["text"] == "Discard 2s"


def test_model_advisor_chi_recommendation_includes_shape_and_filters_offered_middle():
    seen_candidates = []

    class RecordingPredictor:
        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            seen_candidates.extend(candidates)
            return "CHI W4 W1"

    advisor = TziakchaModelAdvisor(
        predictor=RecordingPredictor(),
        fan_checker=RejectingFanChecker(),
    )

    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 3,
            "available_actions": {"chow": [12], "pass": [0]},
            "hand": [0, 4, 12, 16, 72],
            "last_discard": {"seat": 3, "tile": 8},
        }
    )

    assert seen_candidates
    assert all(candidate == "PASS" or candidate.startswith("CHI W4 ") for candidate in seen_candidates)
    assert rec["action"] == "chow"
    assert rec["meld_tile"] == "W4"
    assert rec["meld_shape"] == ["W3", "W4", "W5"]
    assert rec["meld_shape_display"] == ["3m", "4m", "5m"]
    assert rec["text"] == "Chi W4 (3m 4m 5m); discard 1m"


def test_model_advisor_passes_botzone_history_to_predictor():
    seen = {}

    class RecordingPredictor:
        requires_botzone_history = True

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            seen["input_text"] = input_text
            seen["hand"] = hand.copy()
            seen["player_id"] = player_id
            seen["request"] = request
            seen["candidates"] = list(candidates)
            return "CHI W4 W1"

    advisor = TziakchaModelAdvisor(
        predictor=RecordingPredictor(),
        fan_checker=RejectingFanChecker(),
    )

    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 3,
            "available_actions": {"chow": [12], "pass": [0]},
            "hand": [0, 4, 12, 16, 72],
            "last_discard": {"seat": 3, "tile": 8},
            "requests": [
                "0 0 1",
                "1 0 0 0 0 W1 W2 W4 W5 B1 T1 T2 T3 F1 F2 J1 J2 J3",
                "3 1 PLAY F2",
                "3 3 PLAY W3",
            ],
            "responses": ["PASS", "PASS", "PASS"],
        }
    )

    assert rec["action"] == "chow"
    assert seen["request"] == "3 3 PLAY W3"
    assert seen["input_text"].startswith("REQ 0 0 1\nRES PASS")
    assert "REQ 3 1 PLAY F2\nRES PASS" in seen["input_text"]
    assert seen["input_text"].endswith("REQ 3 3 PLAY W3")
    assert seen["input_text"].count("REQ 3 3 PLAY W3") == 1
    assert "CHI W4 W1" in seen["candidates"]


def test_model_advisor_uses_live_state_botzone_history_for_tjong_predictor():
    seen = {}

    class RecordingPredictor:
        requires_botzone_history = True

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            seen["input_text"] = input_text
            seen["request"] = request
            seen["candidates"] = list(candidates)
            return "CHI W4 W1"

    state = AdvisorState()
    for message in (
        {"m": 4, "v": 0, "i": {"t": 0}},
        {
            "m": 2,
            "r": 2,
            "v": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 72],
        },
        {"m": 2, "r": 6, "v": 0, "t": 76, "h": 70, "a": {"2": [76]}},
        {"m": 2, "r": 7, "v": 0, "t": 76},
        {"m": 2, "r": 7, "v": 3, "t": 8, "a": {"3": [12], "8": [0]}},
    ):
        state.ingest(message)

    rec = TziakchaModelAdvisor(
        predictor=RecordingPredictor(),
        fan_checker=RejectingFanChecker(),
    ).recommend(state.snapshot())

    assert rec["action"] == "chow"
    assert seen["request"] == "3 3 PLAY W3"
    assert "REQ 2 B2\nRES PLAY B2" in seen["input_text"]
    assert seen["input_text"].endswith("REQ 3 3 PLAY W3")
    assert seen["input_text"].count("REQ 3 3 PLAY W3") == 1
    assert "CHI W4 W1" in seen["candidates"]


def test_tziakcha_complete_history_matches_botzone_json_wrapper():
    class CrossPlatformPredictor:
        requires_botzone_history = True
        kind = "legal_action_ranker"

        def __init__(self):
            self.calls = []

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            self.calls.append(
                {
                    "input_text": input_text,
                    "hand": hand.copy(),
                    "player_id": player_id,
                    "request": request,
                    "candidates": list(candidates),
                }
            )
            return "PLAY W5"

    predictor = CrossPlatformPredictor()
    initial = "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
    payload = {
        "requests": ["0 0 1", initial, "2 W5"],
        "responses": ["PASS", "PASS"],
    }

    botzone_response = respond_json_with_predictor(payload, predictor)
    snapshot = {
        "seat": 0,
        "turn": 0,
        "botzone_history_complete": True,
        "botzone_history": [
            "REQ 0 0 1",
            "RES PASS",
            f"REQ {initial}",
            "RES PASS",
        ],
        "available_actions": {"discard": [tile_id_from_botzone_symbol("W5")]},
        "hand": [
            tile_id_from_botzone_symbol(tile)
            for tile in "W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 W5".split()
        ],
        "last_draw": {"seat": 0, "tile": tile_id_from_botzone_symbol("W5")},
    }
    tziakcha_rec = TziakchaModelAdvisor(
        predictor=predictor,
        fan_checker=RejectingFanChecker(),
    ).recommend(snapshot)

    assert botzone_response == tziakcha_rec["raw_response"] == "PLAY W5"
    assert len(predictor.calls) == 2
    botzone_call, tziakcha_call = predictor.calls
    assert tziakcha_call["input_text"] == botzone_call["input_text"]
    assert tziakcha_call["request"] == botzone_call["request"] == "2 W5"
    assert tziakcha_call["player_id"] == botzone_call["player_id"] == 0
    assert tziakcha_call["hand"] == botzone_call["hand"]
    assert tziakcha_call["candidates"] == botzone_call["candidates"]


def test_tziakcha_illegal_draw_prediction_falls_back_like_botzone_json_wrapper():
    class IllegalDrawPredictor:
        requires_botzone_history = True
        kind = "legal_action_ranker"

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            return "PASS"

    predictor = IllegalDrawPredictor()
    initial = "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
    payload = {
        "requests": ["0 0 1", initial, "2 W5"],
        "responses": ["PASS", "PASS"],
    }

    botzone_response = respond_json_with_predictor(payload, predictor)
    snapshot = {
        "seat": 0,
        "turn": 0,
        "botzone_history_complete": True,
        "botzone_history": [
            "REQ 0 0 1",
            "RES PASS",
            f"REQ {initial}",
            "RES PASS",
        ],
        "available_actions": {"discard": [tile_id_from_botzone_symbol("W5")]},
        "hand": [
            tile_id_from_botzone_symbol(tile)
            for tile in "W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 W5".split()
        ],
        "last_draw": {"seat": 0, "tile": tile_id_from_botzone_symbol("W5")},
    }
    tziakcha_rec = TziakchaModelAdvisor(
        predictor=predictor,
        fan_checker=RejectingFanChecker(),
    ).recommend(snapshot)

    assert botzone_response == tziakcha_rec["raw_response"] == "PLAY W5"


def test_tziakcha_illegal_reaction_prediction_falls_back_like_botzone_json_wrapper():
    class IllegalReactionPredictor:
        requires_botzone_history = True
        kind = "legal_action_ranker"

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            return "PENG W1"

    predictor = IllegalReactionPredictor()
    initial = "1 0 0 0 0 W2 W4 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3"
    payload = {
        "requests": ["0 0 1", initial, "3 3 PLAY W3"],
        "responses": ["PASS", "PASS"],
    }

    botzone_response = respond_json_with_predictor(payload, predictor)
    snapshot = {
        "seat": 0,
        "turn": 3,
        "botzone_history_complete": True,
        "botzone_history": [
            "REQ 0 0 1",
            "RES PASS",
            f"REQ {initial}",
            "RES PASS",
        ],
        "available_actions": {"chow": [tile_id_from_botzone_symbol("W4")], "pass": [0]},
        "hand": [
            tile_id_from_botzone_symbol(tile)
            for tile in "W2 W4 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3".split()
        ],
        "last_discard": {"seat": 3, "tile": tile_id_from_botzone_symbol("W3")},
    }
    tziakcha_rec = TziakchaModelAdvisor(
        predictor=predictor,
        fan_checker=RejectingFanChecker(),
    ).recommend(snapshot)

    assert botzone_response == tziakcha_rec["raw_response"] == "PASS"


def test_model_advisor_skips_tjong_predictor_when_live_history_is_incomplete():
    class HistoryRequiredPredictor:
        requires_botzone_history = True

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            raise AssertionError("Tjong predictor should not run with incomplete live history")

    advisor = TziakchaModelAdvisor(
        predictor=HistoryRequiredPredictor(),
        fan_checker=RejectingFanChecker(),
    )

    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 0,
            "botzone_history_complete": False,
            "botzone_history": [],
            "available_actions": {"discard": [40]},
            "hand": [0, 4, 36, 40, 44, 72, 76, 80, 124, 132, 108, 112, 116, 120],
            "last_draw": {"seat": 0, "tile": 40},
        }
    )

    assert rec["action"] == "discard"
    assert rec["source"] == "local-model-fallback"
    assert rec["model_skipped_reason"] == "incomplete_botzone_history"


def test_model_advisor_autoloads_tjong_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "tjong.pt"
    torch.save(
        {
            "model": {},
            "tensor_encoding_version": "tjong_cit2_12298_v3_hidden_concealed_kong",
            "encoding_schema": {},
            "config": {},
        },
        checkpoint,
    )
    loaded = []

    class FakeTjongPredictor:
        kind = "legal_action_ranker"

        def __init__(self, path):
            loaded.append(path)

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            return "PLAY T2"

    def fail_transformer(*args, **kwargs):
        raise AssertionError("Tjong checkpoints must not use the transformer-candidate loader")

    monkeypatch.setattr("advisor_service.model_advisor._build_tjong_predictor", FakeTjongPredictor)
    monkeypatch.setattr("advisor_service.model_advisor._build_transformer_predictor", fail_transformer)

    advisor = TziakchaModelAdvisor(
        model_path=checkpoint,
        qadv_path=None,
        qadv_lambda=0.0,
        fan_checker=RejectingFanChecker(),
    )
    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 0,
            "available_actions": {"discard": [40]},
            "hand": [0, 4, 36, 40, 44, 72, 76, 80, 124, 132, 108, 112, 116, 120],
            "last_draw": {"seat": 0, "tile": 40},
        }
    )

    assert loaded == [checkpoint]
    assert rec["action"] == "discard"
    assert rec["tile_symbol"] == "T2"


def test_model_advisor_prefer_hu_requires_official_fan_acceptance():
    snapshot = {
        "seat": 0,
        "turn": 1,
        "available_actions": {"hu": [0], "pass": [0]},
        "hand": [0, 4, 12, 16, 20, 76, 80, 84, 112, 116, 120, 124, 125],
        "last_discard": {"seat": 1, "tile": 8},
    }
    rejecting = TziakchaModelAdvisor(
        predictor=PreferHuPredictor("PASS"),
        fan_checker=RejectingFanChecker(),
    )
    accepting = TziakchaModelAdvisor(
        predictor=PreferHuPredictor("PASS"),
        fan_checker=AcceptingFanChecker(),
    )

    assert rejecting.recommend(snapshot)["action"] == "pass"
    rec = accepting.recommend(snapshot)
    assert rec["action"] == "hu"
    assert rec["fan"] == 8


def test_model_advisor_hu_text_includes_fan_breakdown():
    advisor = TziakchaModelAdvisor(
        predictor=PreferHuPredictor("PASS"),
        fan_checker=FanBreakdownChecker(),
    )

    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 1,
            "available_actions": {"hu": [0], "pass": [0]},
            "hand": [0, 4, 12, 16, 20, 76, 80, 84, 112, 116, 120, 124, 125],
            "last_discard": {"seat": 1, "tile": 8},
        }
    )

    assert rec["action"] == "hu"
    assert rec["fan_items"][0]["name"] == "花龙"
    assert rec["fan_text"] == "花龙 8 + 门前清 2"
    assert rec["text"] == "Hu (10 fan: 花龙 8 + 门前清 2)"


def test_model_advisor_rejects_hu_when_only_flowers_reach_8_fan():
    advisor = TziakchaModelAdvisor(
        predictor=PreferHuPredictor("PASS"),
        fan_checker=FlowerOnlyFanChecker(),
    )

    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 0,
            "available_actions": {"hu": [0], "discard": [120]},
            "hand": [0, 4, 8, 12, 16, 20, 76, 80, 84, 112, 116, 120, 120, 120],
            "last_draw": {"seat": 0, "tile": 120},
            "flowers": 4,
        }
    )

    assert rec["action"] == "discard"
    assert rec["fan"] == 8
    assert rec["base_fan"] == 4
    assert "base fan" in rec["note"]


def test_model_advisor_passes_last_tile_and_self_draw_flags_to_fan_checker():
    checker = RecordingFanChecker(fan=8, can_hu=True)
    advisor = TziakchaModelAdvisor(
        predictor=PreferHuPredictor("PASS"),
        fan_checker=checker,
    )
    snapshot = {
        "seat": 0,
        "turn": 0,
        "available_actions": {"hu": [0], "discard": [120]},
        "hand": [0, 4, 8, 12, 16, 20, 76, 80, 84, 112, 116, 120, 120, 120],
        "last_draw": {"seat": 0, "tile": 120},
        "last_win_event": {
            "seat": 0,
            "tile": 120,
            "source": "draw",
            "is_self_draw": True,
            "is_about_kong": False,
        },
        "visible_counts": {120 >> 2: 3},
        "wall_count": 0,
    }

    rec = advisor.recommend(snapshot)

    assert rec["action"] == "hu"
    assert checker.calls[0]["is_self_draw"] is True
    assert checker.calls[0]["is_last"] is True
    assert checker.calls[0]["is_4th_tile"] is True
    assert checker.calls[0]["is_about_kong"] is False


def test_model_advisor_passes_after_kong_and_robbing_kong_flags_to_fan_checker():
    after_kong = RecordingFanChecker(fan=8, can_hu=True)
    TziakchaModelAdvisor(
        predictor=PreferHuPredictor("PASS"),
        fan_checker=after_kong,
    ).recommend(
        {
            "seat": 0,
            "turn": 0,
            "available_actions": {"hu": [0], "discard": [120]},
            "hand": [0, 4, 8, 12, 16, 20, 76, 80, 84, 112, 116, 120, 120, 120],
            "last_draw": {"seat": 0, "tile": 120},
            "last_win_event": {
                "seat": 0,
                "tile": 120,
                "source": "draw",
                "is_self_draw": True,
                "is_about_kong": True,
            },
            "visible_counts": {},
            "wall_count": 10,
        }
    )

    assert after_kong.calls[0]["is_self_draw"] is True
    assert after_kong.calls[0]["is_about_kong"] is True

    rob_kong = RecordingFanChecker(fan=8, can_hu=True)
    rec = TziakchaModelAdvisor(
        predictor=PreferHuPredictor("PASS"),
        fan_checker=rob_kong,
    ).recommend(
        {
            "seat": 0,
            "turn": 1,
            "available_actions": {"hu": [0], "pass": [0]},
            "hand": [0, 4, 8, 12, 16, 20, 76, 80, 84, 112, 116, 120, 120],
            "last_win_event": {
                "seat": 1,
                "tile": 120,
                "source": "bugang",
                "is_self_draw": False,
                "is_about_kong": True,
            },
            "visible_counts": {},
            "wall_count": 10,
        }
    )

    assert rec["action"] == "hu"
    assert rob_kong.calls[0]["is_self_draw"] is False
    assert rob_kong.calls[0]["is_about_kong"] is True
    assert rob_kong.calls[0]["win_tile"] == "F4"


def test_model_advisor_keeps_hu_suppressed_when_fan_checker_rejects_special_context():
    advisor = TziakchaModelAdvisor(
        predictor=PreferHuPredictor("PASS"),
        fan_checker=RecordingFanChecker(fan=7, can_hu=False),
    )

    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 0,
            "available_actions": {"hu": [0], "discard": [120]},
            "hand": [0, 4, 8, 12, 16, 20, 76, 80, 84, 112, 116, 120, 120, 120],
            "last_draw": {"seat": 0, "tile": 120},
            "last_win_event": {
                "seat": 0,
                "tile": 120,
                "source": "draw",
                "is_self_draw": True,
                "is_about_kong": True,
            },
            "last_discard": {"seat": 1, "tile": 36},
            "visible_counts": {120 >> 2: 3},
            "wall_count": 0,
        }
    )

    assert rec["action"] == "discard"
    assert rec["fan"] == 7


def test_recommend_skips_aleo_when_no_decision_is_pending(monkeypatch):
    def fake_aleo(snapshot):
        raise AssertionError("Aleo should not be called without a pending decision")

    monkeypatch.setattr("advisor_service.advisor.recommend_with_aleo", fake_aleo)
    rec = recommend({"seat": 0, "turn": 1, "available_actions": {}, "hand": [0, 36]}, use_aleo=True)
    assert rec["action"] == "wait"
    assert rec["source"] == "local-advisor"
