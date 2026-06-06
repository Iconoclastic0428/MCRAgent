"""Deterministic supervised evaluation for Tjong paper metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from .model import TjongConfig, TjongNetwork
from .paper_metrics import PAPER_NAME, PAPER_REPORTED_SUPERVISED_METRICS
from .train_supervised import evaluate_model, load_tensor_dataset


PAPER_CHECKPOINT_CONFIG_KEYS = (
    "tile_rows",
    "hidden_tile_rows",
    "tile_types",
    "game_features",
    "action_size",
    "claim_size",
    "discard_size",
    "memory_len",
    "d_model",
    "n_heads",
    "inner_layers",
    "outer_layers",
    "ffn_dim",
    "action_embedding_size",
)


def configure_deterministic_eval(seed: int = 0, *, strict: bool = False) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def checkpoint_encoding_version(payload: dict) -> str | None:
    metrics = payload.get("metrics") or {}
    schema = payload.get("encoding_schema") or metrics.get("encoding_schema") or {}
    return (
        payload.get("tensor_encoding_version")
        or schema.get("version")
        or metrics.get("checkpoint_encoding_version")
        or metrics.get("required_encoding_version")
        or metrics.get("paper_tensor_encoding_version")
    )


def validate_checkpoint_provenance(
    payload: object,
    *,
    expected_encoding_version: str | None = None,
    require_paper_config: bool = False,
    path: Path | None = None,
) -> None:
    location = f" in {path}" if path is not None else ""
    if expected_encoding_version is None and not require_paper_config:
        return
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint metadata is missing{location}; faithful paper runs require structured checkpoints")
    if expected_encoding_version is not None:
        observed = checkpoint_encoding_version(payload)
        if observed != expected_encoding_version:
            raise ValueError(
                f"checkpoint tensor encoding version mismatch{location}: "
                f"expected {expected_encoding_version!r}, observed {observed!r}"
            )
    if require_paper_config:
        raw_config = payload.get("config")
        if not isinstance(raw_config, dict):
            raise ValueError(f"checkpoint config is missing{location}; faithful paper runs require paper model config")
        expected = TjongConfig()
        mismatches = {
            key: {"expected": getattr(expected, key), "observed": raw_config.get(key)}
            for key in PAPER_CHECKPOINT_CONFIG_KEYS
            if raw_config.get(key) != getattr(expected, key)
        }
        if mismatches:
            raise ValueError(f"checkpoint config mismatch{location}: {json.dumps(mismatches, sort_keys=True)}")


def load_model(
    checkpoint: Path,
    device: torch.device,
    *,
    expected_encoding_version: str | None = None,
    require_paper_config: bool = False,
) -> TjongNetwork:
    payload = torch.load(checkpoint, map_location="cpu")
    validate_checkpoint_provenance(
        payload,
        expected_encoding_version=expected_encoding_version,
        require_paper_config=require_paper_config,
        path=checkpoint,
    )
    raw_config = payload.get("config", {}) if isinstance(payload, dict) else {}
    allowed = {field.name for field in fields(TjongConfig)}
    config = TjongConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    model = TjongNetwork(config)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def evaluate(args: argparse.Namespace) -> dict:
    configure_deterministic_eval(getattr(args, "seed", 0), strict=getattr(args, "strict_deterministic", False))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = load_tensor_dataset(Path(args.eval_pt), expected_encoding_version=args.require_encoding_version)
    model = load_model(
        Path(args.checkpoint),
        device,
        expected_encoding_version=args.require_encoding_version,
        require_paper_config=args.require_paper_config,
    )
    metrics = evaluate_model(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    output = {
        "paper": PAPER_NAME,
        "checkpoint": args.checkpoint,
        "eval_pt": args.eval_pt,
        "deterministic_eval": True,
        "deterministic_seed": getattr(args, "seed", 0),
        "strict_deterministic": getattr(args, "strict_deterministic", False),
        "device": str(device),
        "required_encoding_version": args.require_encoding_version,
        "require_paper_config": args.require_paper_config,
        **metrics,
        "paper_reported": dict(PAPER_REPORTED_SUPERVISED_METRICS),
    }
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, sort_keys=True), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-pt", required=True)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--require-paper-config", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict-deterministic", action="store_true")
    args = parser.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
