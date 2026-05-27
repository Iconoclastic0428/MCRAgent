#!/usr/bin/env python3
"""Create a composite policy with weighted ensemble draw rankers."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def load_pickle(path: Path) -> dict:
    with path.open("rb") as src:
        return pickle.load(src)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw-model", action="append", required=True)
    parser.add_argument("--draw-weight", action="append", type=float, default=None)
    parser.add_argument("--reaction-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reaction-min-margin", type=float, default=None)
    parser.add_argument("--reaction-min-score", type=float, default=None)
    parser.add_argument("--prefer-hu", action="store_true")
    args = parser.parse_args()

    draw_weights = args.draw_weight or [1.0] * len(args.draw_model)
    if len(draw_weights) != len(args.draw_model):
        raise ValueError("--draw-weight count must match --draw-model count")

    reaction_thresholds = {}
    if args.reaction_min_margin is not None:
        reaction_thresholds["min_margin"] = args.reaction_min_margin
    if args.reaction_min_score is not None:
        reaction_thresholds["min_score"] = args.reaction_min_score

    payload = {
        "kind": "draw_ensemble_composite_policy",
        "draw_payloads": [load_pickle(Path(path)) for path in args.draw_model],
        "draw_weights": [float(weight) for weight in draw_weights],
        "reaction_payload": load_pickle(Path(args.reaction_model)),
        "components": {
            "draw_models": list(args.draw_model),
            "reaction_model": args.reaction_model,
        },
    }
    if reaction_thresholds:
        payload["reaction_thresholds"] = reaction_thresholds
    if args.prefer_hu:
        payload["prefer_hu"] = True

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        pickle.dump(payload, out)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
