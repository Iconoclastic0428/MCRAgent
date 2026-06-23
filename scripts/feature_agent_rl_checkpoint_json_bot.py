#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from feature_repo_json_runtime import ReplayFeatureJsonBot, WORKSPACE_ROOT, main_from_factory


FEATURE_AGENT_DIR = Path(
    os.environ.get(
        "MCR_FEATURE_AGENT_DIR",
        str(
            WORKSPACE_ROOT
            / "third_party"
            / "mahjong-agent-2025-aug12-excludespecial-3pkl"
            / "feature-agent"
        ),
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
from rl_model_utils import load_selfvec_model  # noqa: E402


class FeatureAgentRLCheckpointJsonBot(ReplayFeatureJsonBot):
    def __init__(self) -> None:
        checkpoint = Path(os.environ.get("MCR_FEATURE_AGENT_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
        legal_mean = os.environ.get("MCR_FEATURE_AGENT_LEGAL_DUELING_MEAN", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self.legal_dueling_mean = legal_mean
        self.FeatureAgent = FeatureAgent
        model = load_selfvec_model(checkpoint, device="cpu", mixed_kernel_input=True, dueling_head=True)
        model.eval()
        super().__init__(model, obs_mode="vec")

    def _new_agent(self, seat_wind: int) -> Any:
        return self.FeatureAgent(seat_wind)

    def _predict_response(self, obs: dict[str, Any]) -> str:
        obs_tensor = torch.from_numpy(
            np.expand_dims(np.asarray(obs["observation"], dtype=np.float32), 0)
        )
        mask_tensor = torch.from_numpy(
            np.expand_dims(np.asarray(obs["action_mask"], dtype=np.float32), 0)
        )
        vec_tensor = torch.from_numpy(np.expand_dims(np.asarray(obs["vec"], dtype=np.float32), 0))
        with torch.no_grad():
            out = self.model(
                {
                    "is_training": False,
                    "return_raw_logits": True,
                    "legal_dueling_mean": self.legal_dueling_mean,
                    "obs": {
                        "observation": obs_tensor,
                        "vec": vec_tensor,
                        "action_mask": mask_tensor,
                    },
                }
            )
        action = int(out["masked_q"].detach().cpu().numpy().reshape(-1).argmax())
        assert self.agent is not None
        return str(self.agent.action2response(action))


if __name__ == "__main__":
    raise SystemExit(main_from_factory(FeatureAgentRLCheckpointJsonBot))
