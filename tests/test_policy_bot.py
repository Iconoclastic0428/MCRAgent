import sys
import pickle
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from feature_ranker import feature_names
from fan_feature_ranker import fan_feature_names
from policy_bot import BotzonePolicy, ShantenHeuristicPredictor, SklearnPredictor


class FixedPredictor:
    def __init__(self, response):
        self.response = response

    def predict_response(self, input_text):
        return self.response


class PreferHuPredictor(FixedPredictor):
    prefer_hu = True


class FeatureScoreModel:
    def predict_proba(self, features):
        names = feature_names()
        w2_index = names.index("discard_W2")
        scores = np.asarray(features)[:, w2_index]
        return np.column_stack([1.0 - scores, scores])


class FeatureScoreW3Model:
    def predict_proba(self, features):
        names = feature_names()
        w3_index = names.index("discard_W3")
        scores = np.asarray(features)[:, w3_index]
        return np.column_stack([1.0 - scores, scores])


class FanFeatureFlushModel:
    def predict_proba(self, features):
        names = fan_feature_names()
        index = names.index("fan_after_max_suit_fraction")
        scores = np.asarray(features)[:, index]
        return np.column_stack([1.0 - scores, scores])


class LowMarginReactionModel:
    classes_ = [0, 1]

    def predict_proba(self, texts):
        scores = []
        for text in texts:
            if "RESP PASS" in text:
                scores.append(0.49)
            elif "RESP PENG" in text:
                scores.append(0.51)
            else:
                scores.append(0.1)
        return np.asarray([[1.0 - score, score] for score in scores])


class HuScoreModel:
    classes_ = [0, 1]

    def predict_proba(self, texts):
        scores = []
        for text in texts:
            if "RESP HU" in text:
                scores.append(0.99)
            elif "RESP PENG" in text:
                scores.append(0.75)
            else:
                scores.append(0.1)
        return np.asarray([[1.0 - score, score] for score in scores])


class GangScoreModel:
    classes_ = [0, 1]

    def predict_proba(self, texts):
        scores = []
        for text in texts:
            if "RESP GANG" in text:
                scores.append(0.99)
            elif "RESP PLAY" in text:
                scores.append(0.75)
            else:
                scores.append(0.1)
        return np.asarray([[1.0 - score, score] for score in scores])


class RejectingFanChecker:
    def __init__(self):
        self.calls = []

    def can_hu(self, **kwargs):
        self.calls.append(kwargs)
        return False


class AcceptingFanChecker:
    def can_hu(self, **kwargs):
        return True


def test_policy_passes_initialization_requests():
    policy = BotzonePolicy()

    assert policy.respond("0 2 3") == "PASS"
    assert policy.respond("1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2") == "PASS"


def test_policy_accepts_legal_model_draw_discard_and_updates_hand():
    policy = BotzonePolicy(FixedPredictor("PLAY W2"))
    policy.respond("1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2")

    response = policy.respond("2 W4")

    assert response == "PLAY W2"
    assert "W2" not in policy.hand
    assert "W4" in policy.hand


def test_policy_concealed_gang_removes_all_four_tiles_from_hand():
    policy = BotzonePolicy(FixedPredictor("GANG W1"))
    policy.respond("1 0 0 0 0 W1 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2")

    response = policy.respond("2 W1")

    assert response == "GANG W1"
    assert policy.hand["W1"] == 0


def test_policy_falls_back_to_drawn_tile_when_model_discard_is_illegal():
    policy = BotzonePolicy(FixedPredictor("PLAY W9"))
    policy.respond("1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2")

    response = policy.respond("2 W4")

    assert response == "PLAY W4"
    assert "W4" not in policy.hand


def test_policy_passes_on_other_players_events_by_default():
    policy = BotzonePolicy(FixedPredictor("PENG W1"))

    assert policy.respond("3 0 PLAY W1") == "PASS"


def test_policy_accepts_legal_peng_reaction_and_updates_hand():
    policy = BotzonePolicy(FixedPredictor("PENG B1"))
    policy.respond("0 2 3")
    policy.respond("1 0 0 0 0 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3")

    response = policy.respond("3 1 PLAY W1")

    assert response == "PENG B1"
    assert policy.hand["W1"] == 0
    assert policy.hand["B1"] == 0


def test_policy_rejects_illegal_peng_reaction_and_passes():
    policy = BotzonePolicy(FixedPredictor("PENG B1"))
    policy.respond("0 2 3")
    policy.respond("1 0 0 0 0 W1 W2 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3")

    response = policy.respond("3 1 PLAY W1")

    assert response == "PASS"
    assert policy.last_illegal_prediction


def test_policy_tracks_wall_counts_from_initial_flowers_and_public_draws():
    policy = BotzonePolicy()
    policy.respond("0 0 2")
    policy.respond("1 1 0 2 3 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2")

    assert policy.wall_counts == [20, 21, 19, 18]
    assert policy.flower_counts == [1, 0, 2, 3]

    policy.respond("2 W4")
    policy.respond("3 1 DRAW")
    policy.respond("3 2 BUHUA H1")

    assert policy.wall_counts == [19, 20, 18, 18]
    assert policy.flower_counts == [1, 0, 3, 3]


def test_policy_rejects_draw_gang_when_replacement_wall_is_empty():
    policy = BotzonePolicy(FixedPredictor("GANG W1"))
    policy.respond("0 0 2")
    policy.respond("1 0 0 0 0 W1 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2")
    policy.wall_counts = [1, 1, 1, 1]

    response = policy.respond("2 W1")

    assert response == "PLAY W1"
    assert policy.last_illegal_prediction


def test_shanten_heuristic_prefers_candidate_that_reaches_tenpai():
    predictor = ShantenHeuristicPredictor()
    input_text = "REQ 2 F2"
    hand = {
        tile: 1
        for tile in "W1 W2 W3 W4 W5 W6 B1 B2 B3 T1 T2 T3 F1 F2".split()
    }

    response = predictor.predict_draw_response(input_text, hand)

    assert response == "PLAY F2"


def test_sklearn_predictor_ranks_legal_responses_from_feature_payload(tmp_path):
    model_path = tmp_path / "feature.pkl"
    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "feature_action_ranker",
                "feature_mode": "numeric_v1",
                "model": FeatureScoreModel(),
                "feature_names": feature_names(),
            },
            out,
        )
    predictor = SklearnPredictor(model_path)
    policy = BotzonePolicy(predictor)
    policy.respond("0 0 1")
    policy.respond("1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2")

    response = policy.respond("2 W4")

    assert response == "PLAY W2"


def test_sklearn_predictor_dispatches_numeric_fan_feature_payload(tmp_path):
    model_path = tmp_path / "fan_feature.pkl"
    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "feature_action_ranker",
                "feature_mode": "numeric_fan_v1",
                "model": FanFeatureFlushModel(),
                "feature_names": fan_feature_names(),
            },
            out,
        )
    predictor = SklearnPredictor(model_path)
    policy = BotzonePolicy(predictor)
    policy.respond("0 0 1")
    policy.respond("1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 F1 F2 J1 J2")

    response = policy.respond("2 W9")

    assert response == "PLAY F1"


def test_draw_ensemble_composite_averages_weighted_feature_payload_scores(tmp_path):
    model_path = tmp_path / "draw_ensemble.pkl"
    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "draw_ensemble_composite_policy",
                "draw_payloads": [
                    {
                        "kind": "feature_action_ranker",
                        "model": FeatureScoreModel(),
                        "feature_names": feature_names(),
                    },
                    {
                        "kind": "feature_action_ranker",
                        "model": FeatureScoreW3Model(),
                        "feature_names": feature_names(),
                    },
                ],
                "draw_weights": [0.25, 0.75],
                "reaction_payload": {
                    "kind": "legal_action_ranker",
                    "pipeline": LowMarginReactionModel(),
                },
            },
            out,
        )
    predictor = SklearnPredictor(model_path)
    policy = BotzonePolicy(predictor)
    policy.respond("0 0 1")
    policy.respond("1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2")

    response = policy.respond("2 W4")

    assert response == "PLAY W3"


def test_draw_ensemble_composite_routes_reactions_to_reaction_payload(tmp_path):
    model_path = tmp_path / "draw_ensemble_reaction.pkl"
    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "draw_ensemble_composite_policy",
                "draw_payloads": [
                    {
                        "kind": "feature_action_ranker",
                        "model": FeatureScoreModel(),
                        "feature_names": feature_names(),
                    }
                ],
                "draw_weights": [1.0],
                "reaction_payload": {
                    "kind": "legal_action_ranker",
                    "pipeline": LowMarginReactionModel(),
                },
            },
            out,
        )
    predictor = SklearnPredictor(model_path)
    policy = BotzonePolicy(predictor)
    policy.respond("0 2 3")
    policy.respond("1 0 0 0 0 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3")

    response = policy.respond("3 1 PLAY W1")

    assert response == "PENG B1"


def test_composite_reaction_threshold_keeps_low_margin_claim_as_pass(tmp_path):
    model_path = tmp_path / "threshold.pkl"
    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "composite_policy",
                "draw_payload": {
                    "kind": "feature_action_ranker",
                    "model": FeatureScoreModel(),
                    "feature_names": feature_names(),
                },
                "reaction_payload": {
                    "kind": "legal_action_ranker",
                    "pipeline": LowMarginReactionModel(),
                },
                "reaction_thresholds": {"min_margin": 0.05},
            },
            out,
        )
    predictor = SklearnPredictor(model_path)
    policy = BotzonePolicy(predictor)
    policy.respond("0 2 3")
    policy.respond("1 0 0 0 0 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3")

    response = policy.respond("3 1 PLAY W1")

    assert response == "PASS"


def test_sklearn_predictor_can_suppress_hu_candidates_for_official_safe_policy(tmp_path):
    model_path = tmp_path / "suppress_hu.pkl"
    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "legal_action_ranker",
                "pipeline": HuScoreModel(),
                "suppress_hu": True,
            },
            out,
        )
    predictor = SklearnPredictor(model_path)
    policy = BotzonePolicy(predictor)
    policy.respond("0 0 3")
    policy.respond("1 0 0 0 0 W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 F1 F1")

    response = policy.respond("3 1 PLAY W1")

    assert response.startswith("PENG")
    assert response != "HU"


def test_policy_filters_hu_when_official_fan_checker_rejects_claim(tmp_path):
    model_path = tmp_path / "fan_filter.pkl"
    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "legal_action_ranker",
                "pipeline": HuScoreModel(),
            },
            out,
        )
    checker = RejectingFanChecker()
    predictor = SklearnPredictor(model_path)
    policy = BotzonePolicy(predictor, fan_checker=checker)
    policy.respond("0 0 3")
    policy.respond("1 0 0 0 0 W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 F1 F1")

    response = policy.respond("3 1 PLAY W1")

    assert response.startswith("PENG")
    assert response != "HU"
    assert checker.calls
    assert checker.calls[0]["win_tile"] == "W1"


def test_policy_prefer_hu_takes_only_official_fan_checked_hu():
    accepting_policy = BotzonePolicy(PreferHuPredictor("PLAY W1"), fan_checker=AcceptingFanChecker())
    accepting_policy.respond("0 0 0")
    accepting_policy.hand.update(
        [
            "W1",
            "W2",
            "W4",
            "W5",
            "W6",
            "B2",
            "B3",
            "B4",
            "T2",
            "T3",
            "T4",
            "F1",
            "F1",
        ]
    )

    assert accepting_policy.respond("3 1 PLAY W3") == "HU"

    rejecting_policy = BotzonePolicy(PreferHuPredictor("PLAY W1"), fan_checker=RejectingFanChecker())
    rejecting_policy.respond("0 0 0")
    rejecting_policy.hand.update(
        [
            "W1",
            "W2",
            "W4",
            "W5",
            "W6",
            "B2",
            "B3",
            "B4",
            "T2",
            "T3",
            "T4",
            "F1",
            "F1",
        ]
    )

    assert rejecting_policy.respond("3 1 PLAY W3") != "HU"


def test_policy_suppresses_hu_when_fan_checker_unavailable():
    policy = BotzonePolicy(PreferHuPredictor("PLAY W1"))
    policy.fan_checker = None
    policy.respond("0 0 0")
    policy.hand.update(
        [
            "W1",
            "W2",
            "W4",
            "W5",
            "W6",
            "B2",
            "B3",
            "B4",
            "T2",
            "T3",
            "T4",
            "F1",
            "F1",
        ]
    )

    assert policy.respond("3 1 PLAY W3") != "HU"


def test_sklearn_predictor_can_suppress_gang_candidates_for_official_safe_policy(tmp_path):
    model_path = tmp_path / "suppress_gang.pkl"
    with model_path.open("wb") as out:
        pickle.dump(
            {
                "kind": "legal_action_ranker",
                "pipeline": GangScoreModel(),
                "suppress_actions": ["GANG"],
            },
            out,
        )
    predictor = SklearnPredictor(model_path)
    policy = BotzonePolicy(predictor)
    policy.respond("0 0 3")
    policy.respond("1 0 0 0 0 W1 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2")

    response = policy.respond("2 W1")

    assert response != "GANG W1"


def test_sklearn_predictor_reads_prefer_hu_payload_flag(tmp_path):
    model_path = tmp_path / "prefer_hu.pkl"
    with model_path.open("wb") as out:
        pickle.dump({"kind": "response_classifier", "pipeline": None, "prefer_hu": True}, out)

    predictor = SklearnPredictor(model_path)

    assert predictor.prefer_hu
