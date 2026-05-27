from advisor_service.advisor import recommend
from advisor_service.model_advisor import TziakchaModelAdvisor


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


def test_recommend_skips_aleo_when_no_decision_is_pending(monkeypatch):
    def fake_aleo(snapshot):
        raise AssertionError("Aleo should not be called without a pending decision")

    monkeypatch.setattr("advisor_service.advisor.recommend_with_aleo", fake_aleo)
    rec = recommend({"seat": 0, "turn": 1, "available_actions": {}, "hand": [0, 36]}, use_aleo=True)
    assert rec["action"] == "wait"
    assert rec["source"] == "local-advisor"
