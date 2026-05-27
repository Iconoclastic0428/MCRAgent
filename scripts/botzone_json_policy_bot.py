#!/usr/bin/env python3
"""Botzone JSON-protocol wrapper for the local MCR policy."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from policy_bot import BotzonePolicy, SklearnPredictor


DEFAULT_MODEL = "models/ensemble_draw_public1000_2026_050_050_reaction1000.pkl"


def make_policy() -> BotzonePolicy:
    model_path = os.environ.get("MCR_MODEL", DEFAULT_MODEL)
    if model_path:
        path = Path(model_path)
        if path.exists():
            return BotzonePolicy(SklearnPredictor(path))
    return BotzonePolicy()


def respond(payload: dict) -> str:
    requests = [str(item) for item in payload.get("requests", [])]
    responses = [str(item) for item in payload.get("responses", [])]
    policy = make_policy()

    for request, expected_response in zip(requests[:-1], responses):
        actual_response = policy.respond(request)
        if actual_response != expected_response:
            # Botzone replays our own prior responses. If the local reconstruction
            # diverges, keep state internally consistent with the deterministic policy.
            continue
    if not requests:
        return "PASS"
    return policy.respond(requests[-1])


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    print(json.dumps({"response": respond(payload)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
