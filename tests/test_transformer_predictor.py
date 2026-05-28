from collections import Counter
from pathlib import Path

import torch

from advisor_service.model_advisor import TziakchaModelAdvisor
from advisor_service.transformer_predictor import TransformerCheckpointPredictor


class FixedActionModel(torch.nn.Module):
    def __init__(self, preferred_action: int):
        super().__init__()
        self.preferred_action = preferred_action

    def forward(self, batch):
        actions = batch["candidate_actions"]
        logits = torch.full(actions.shape, -20.0, dtype=torch.float32, device=actions.device)
        logits = logits.masked_fill(actions.eq(self.preferred_action), 20.0)
        return logits.masked_fill(~batch["candidate_mask"], float("-inf")), torch.zeros(actions.shape[0])


class RejectingFanChecker:
    def evaluate(self, **kwargs):
        return {"fan": 4, "can_hu": False}


def test_transformer_predictor_ranks_mapped_discard_candidate():
    predictor = TransformerCheckpointPredictor(
        model=FixedActionModel(12),
        config={"history_len": 4, "max_candidates": 8},
        device="cpu",
    )

    response = predictor.predict_legal_response(
        "REQ 2 T2",
        Counter({"W1": 1, "T2": 1}),
        0,
        "2 T2",
        ["PLAY W1", "PLAY T2"],
    )

    assert response == "PLAY T2"


def test_transformer_predictor_maps_reaction_claims_from_offered_tile():
    predictor = TransformerCheckpointPredictor(
        model=FixedActionModel(100),
        config={"history_len": 4, "max_candidates": 8},
        device="cpu",
    )

    response = predictor.predict_legal_response(
        "REQ 3 1 PLAY W2",
        Counter({"W2": 2, "T2": 1, "F1": 1}),
        0,
        "3 1 PLAY W2",
        ["PASS", "PENG T2", "PENG F1"],
    )

    assert response.startswith("PENG ")
    assert response in {"PENG T2", "PENG F1"}


def test_model_advisor_autoloads_pt_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    loaded_paths: list[Path] = []

    class FakePredictor:
        def __init__(self, path):
            loaded_paths.append(Path(path))

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            return "PLAY T2"

        def info(self):
            return {"type": "fake-transformer"}

    monkeypatch.setattr("advisor_service.model_advisor._build_transformer_predictor", FakePredictor)
    advisor = TziakchaModelAdvisor(model_path=checkpoint, fan_checker=RejectingFanChecker())

    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 0,
            "available_actions": {"discard": [40]},
            "hand": [0, 4, 36, 40, 44, 72, 76, 80, 124, 132, 108, 112, 116, 120],
            "last_draw": {"seat": 0, "tile": 40},
        }
    )

    assert loaded_paths == [checkpoint]
    assert rec["action"] == "discard"
    assert rec["tile_symbol"] == "T2"


def test_model_advisor_exposes_chi_peng_gang_candidates_without_low_fan_hu():
    seen_candidates = []

    class RecordingPredictor:
        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            seen_candidates.extend(candidates)
            return "CHI W2 T2"

    advisor = TziakchaModelAdvisor(
        predictor=RecordingPredictor(),
        fan_checker=RejectingFanChecker(),
    )

    rec = advisor.recommend(
        {
            "seat": 0,
            "turn": 3,
            "available_actions": {"chow": [0], "pung": [0], "kong": [0], "hu": [0], "pass": [0]},
            "hand": [0, 8, 4, 5, 6, 7, 40],
            "last_discard": {"seat": 3, "tile": 4},
            "visible_counts": {},
            "wall_count": 20,
        }
    )

    assert "PASS" in seen_candidates
    assert "HU" not in seen_candidates
    assert "GANG" in seen_candidates
    assert any(candidate.startswith("PENG ") for candidate in seen_candidates)
    assert any(candidate.startswith("CHI ") for candidate in seen_candidates)
    assert rec["action"] == "chow"
    assert rec["raw_response"] == "CHI W2 T2"
