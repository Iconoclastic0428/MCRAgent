#!/usr/bin/env python3
"""Botzone JSON/text wrapper for the promoted feature-agent checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from feature_repo_json_runtime import ReplayFeatureJsonBot, WORKSPACE_ROOT, main_from_factory


FEATURE_AGENT_DIR = Path(
    os.environ.get(
        "MCR_FEATURE_AGENT_DIR",
        str(WORKSPACE_ROOT / "feature-agent"),
    )
)
DEFAULT_CHECKPOINT = (
    WORKSPACE_ROOT
    / "models"
    / "feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a"
    / "16.pkl"
)

if str(FEATURE_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_AGENT_DIR))

from feature import FeatureAgent  # noqa: E402
from model import SelfVecModel  # noqa: E402


class FeatureAgentCheckpointJsonBot(ReplayFeatureJsonBot):
    def __init__(self) -> None:
        checkpoint = Path(os.environ.get("MCR_FEATURE_AGENT_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
        use_river = os.environ.get("MCR_FEATURE_AGENT_RIVER", "1").lower() not in {"0", "false", "no"}
        kwargs = {"obs_dim": FeatureAgent.OBS_SIZE, "vec_dim": FeatureAgent.VEC_SIZE}
        if use_river:
            kwargs.update({"mixed_kernel_input": True, "dueling_head": True})
        model = SelfVecModel(**kwargs)
        state = torch.load(checkpoint, map_location=torch.device("cpu"))
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state)
        model.eval()
        self.FeatureAgent = FeatureAgent
        super().__init__(model, obs_mode="vec")

    def _new_agent(self, seat_wind: int):
        return self.FeatureAgent(seat_wind)


if __name__ == "__main__":
    raise SystemExit(main_from_factory(FeatureAgentCheckpointJsonBot))
