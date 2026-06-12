#!/usr/bin/env python3
"""Botzone JSON-protocol wrapper for a trained Tjong checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TJONG_SRC = WORKSPACE_ROOT / "papers" / "tjong_cit2_12298" / "src"
if str(TJONG_SRC) not in sys.path:
    sys.path.insert(0, str(TJONG_SRC))

from tjong_replication.policy_bot import respond_json  # noqa: E402

DEFAULT_ENCODING_VERSION = "tjong_cit2_12298_v3_hidden_concealed_kong"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def respond(
    payload: dict,
    *,
    checkpoint: str,
    device: str | None = None,
    require_encoding_version: str | None = DEFAULT_ENCODING_VERSION,
    require_paper_config: bool = False,
) -> str:
    return respond_json(
        payload,
        checkpoint,
        device=device,
        require_encoding_version=require_encoding_version,
        require_paper_config=require_paper_config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get("TJONG_CHECKPOINT"))
    parser.add_argument("--device", default=os.environ.get("TJONG_DEVICE"))
    parser.add_argument(
        "--require-encoding-version",
        default=os.environ.get("TJONG_REQUIRE_ENCODING_VERSION", DEFAULT_ENCODING_VERSION),
    )
    parser.add_argument(
        "--require-paper-config",
        action="store_true",
        default=env_bool("TJONG_REQUIRE_PAPER_CONFIG", False),
    )
    args = parser.parse_args(argv)
    if not args.checkpoint:
        raise SystemExit("missing --checkpoint or TJONG_CHECKPOINT")
    payload = json.loads(sys.stdin.read() or "{}")
    response = respond(
        payload,
        checkpoint=args.checkpoint,
        device=args.device,
        require_encoding_version=args.require_encoding_version or None,
        require_paper_config=bool(args.require_paper_config),
    )
    print(json.dumps({"response": response}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
