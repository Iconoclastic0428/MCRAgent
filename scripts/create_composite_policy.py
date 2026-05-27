#!/usr/bin/env python3
"""Combine a draw model and a reaction model into one policy artifact."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def load_pickle(path: Path) -> dict:
    with path.open("rb") as src:
        return pickle.load(src)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw-model", required=True)
    parser.add_argument("--reaction-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reaction-min-margin", type=float, default=None)
    parser.add_argument("--reaction-min-score", type=float, default=None)
    parser.add_argument("--suppress-hu", action="store_true")
    parser.add_argument("--suppress-gang", action="store_true")
    args = parser.parse_args()

    reaction_thresholds = {}
    if args.reaction_min_margin is not None:
        reaction_thresholds["min_margin"] = args.reaction_min_margin
    if args.reaction_min_score is not None:
        reaction_thresholds["min_score"] = args.reaction_min_score

    payload = {
        "kind": "composite_policy",
        "draw_payload": load_pickle(Path(args.draw_model)),
        "reaction_payload": load_pickle(Path(args.reaction_model)),
        "components": {
            "draw_model": args.draw_model,
            "reaction_model": args.reaction_model,
        },
    }
    if reaction_thresholds:
        payload["reaction_thresholds"] = reaction_thresholds
    if args.suppress_hu:
        payload["suppress_hu"] = True
    suppress_actions = []
    if args.suppress_gang:
        suppress_actions.extend(["GANG", "BUGANG"])
    if suppress_actions:
        payload["suppress_actions"] = suppress_actions
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        pickle.dump(payload, out)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
