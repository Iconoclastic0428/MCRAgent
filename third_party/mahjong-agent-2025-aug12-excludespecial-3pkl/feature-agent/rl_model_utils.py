#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from model import SelfVecModel

try:
    from feature import FeatureAgent

    OBS_SIZE = FeatureAgent.OBS_SIZE
    VEC_SIZE = FeatureAgent.VEC_SIZE
except ModuleNotFoundError as exc:
    if exc.name != "MahjongGB":
        raise
    OBS_SIZE = 185
    VEC_SIZE = 117


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if any(str(key).startswith("module.") for key in state):
        return {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def load_selfvec_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    mixed_kernel_input: bool = True,
    dueling_head: bool = True,
) -> SelfVecModel:
    model = SelfVecModel(
        obs_dim=OBS_SIZE,
        vec_dim=VEC_SIZE,
        mixed_kernel_input=mixed_kernel_input,
        dueling_head=dueling_head,
    )
    state = torch.load(Path(checkpoint), map_location=torch.device(device))
    model.load_state_dict(strip_module_prefix(state))
    return model.to(device)


def save_state_dict(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, path)


def freeze_for_mode(model: torch.nn.Module, mode: str) -> list[str]:
    mode = mode.lower()
    if mode not in {"head", "fc", "none"}:
        raise ValueError(f"unknown freeze mode: {mode}")
    for param in model.parameters():
        param.requires_grad_(mode == "none")
    trainable: list[str] = []
    if mode == "none":
        return [name for name, param in model.named_parameters() if param.requires_grad]
    prefixes = ["_output_layer"]
    if mode == "fc":
        prefixes.append("_projection")
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(True)
            trainable.append(name)
    return trainable


def write_config(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
