import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lawlorentz_policy import CNNModel, LawlorentzEffectivePolicy, LawlorentzEffectiveScorer, LawlorentzModelPolicy


def test_lawlorentz_dependency_and_feature_agent_are_available():
    from MahjongGB import MahjongShanten
    from lawlorentz_policy import FeatureAgent

    agent = FeatureAgent(0)
    agent.request2obs("Wind 0")
    agent.request2obs("Deal W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T1 T1 B1")
    assert MahjongShanten(pack=(), hand=tuple(agent.hand)) >= 0

    obs = agent.request2obs("Draw B1")

    assert obs["observation"].shape == (71, 4, 9)
    assert obs["action_mask"].shape == (235,)


def test_effective_scorer_uses_pymahjonggb_fan_waits():
    scorer = LawlorentzEffectiveScorer(
        packs=(),
        shown_tiles={},
        seat_wind=0,
        prevalent_wind=0,
        levels=0,
    )
    profile = scorer.profile("W1 W1 W1 W2 W2 W2 W3 W3 W3 W4 W4 W4 W5".split())

    assert profile.fan8_wait_tiles >= 1
    assert profile.max_fan >= 8


def test_policy_uses_lawlorentz_protocol_and_returns_legal_draw_response():
    policy = LawlorentzEffectivePolicy(levels=0)

    assert policy.respond("0 0 0") == "PASS"
    assert policy.respond("1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T1 T1 B1") == "PASS"
    response = policy.respond("2 B1")

    assert response == "HU" or response.startswith("PLAY ")
    assert policy.diagnostics()["kind"] == "lawlorentz_effective"


def test_lawlorentz_model_policy_loads_checkpoint_and_uses_feature_mask(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(CNNModel().state_dict(), checkpoint)
    policy = LawlorentzModelPolicy(checkpoint, device="cpu")

    assert policy.respond("0 0 0") == "PASS"
    assert policy.respond("1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T1 T1 B1") == "PASS"
    response = policy.respond("2 B1")

    assert response == "HU" or response.startswith("PLAY ") or response.startswith("GANG ")
    assert policy.diagnostics()["kind"] == "lawlorentz_model"
