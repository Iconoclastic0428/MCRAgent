#!/usr/bin/env python3
"""Merge parallel feature-agent vector conversion part indexes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


COUNTER_FIELDS = (
    "label_outside_mask",
    "action_seen",
    "action_kept",
    "response_seen",
    "unsupported_requests",
    "errors",
)

COUNT_FIELDS = (
    "records_seen",
    "records_written",
    "records_skipped",
    "examples_seen",
    "examples_kept",
    "single_action_skipped",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_part_indexes(root: Path, *, expect_parts: int | None = None) -> dict[str, Any]:
    part_indexes = sorted(root.glob("part_*/index.json"))
    if expect_parts is not None and len(part_indexes) != int(expect_parts):
        raise ValueError(f"expected {expect_parts} part indexes under {root}, found {len(part_indexes)}")
    if not part_indexes:
        raise ValueError(f"no part indexes found under {root}")

    totals: dict[str, int] = {field: 0 for field in COUNT_FIELDS}
    counters = {field: Counter() for field in COUNTER_FIELDS}
    shards: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    sources: set[str] = set()
    samples_per_shard: set[int] = set()

    for index_path in part_indexes:
        part_dir = index_path.parent
        index = load_json(index_path)
        if index.get("format") != "feature_agent_vec_shards_v1":
            raise ValueError(f"unexpected format in {index_path}: {index.get('format')}")
        if not index.get("sharded") or not index.get("compact_storage"):
            raise ValueError(f"part is not compact sharded: {index_path}")

        for field in COUNT_FIELDS:
            totals[field] += int(index.get(field, 0))
        for field in COUNTER_FIELDS:
            counters[field].update(index.get(field) or {})
        if index.get("source"):
            sources.add(str(index["source"]))
        if index.get("samples_per_shard"):
            samples_per_shard.add(int(index["samples_per_shard"]))

        part_examples = 0
        for shard in index.get("shards") or []:
            shard_path = part_dir / str(shard["path"])
            if not shard_path.exists():
                raise FileNotFoundError(shard_path)
            examples = int(shard["examples"])
            part_examples += examples
            shards.append(
                {
                    "path": str(shard_path.relative_to(root).as_posix()),
                    "examples": examples,
                }
            )
        if part_examples != int(index.get("examples", 0)):
            raise ValueError(f"part example mismatch in {index_path}: shards={part_examples}, index={index.get('examples')}")
        parts.append(
            {
                "path": str(index_path.relative_to(root).as_posix()),
                "examples": part_examples,
                "records_seen": int(index.get("records_seen", 0)),
                "records_skipped": int(index.get("records_skipped", 0)),
            }
        )

    examples = sum(int(shard["examples"]) for shard in shards)
    if examples != totals["examples_kept"]:
        raise ValueError(f"merged example mismatch: shards={examples}, examples_kept={totals['examples_kept']}")

    return {
        "format": "feature_agent_vec_shards_v1",
        **totals,
        **{field: dict(sorted(counter.items())) for field, counter in counters.items()},
        "shard_examples": [int(shard["examples"]) for shard in shards],
        "source": sorted(sources),
        "sharded": True,
        "compact_storage": True,
        "samples_per_shard": sorted(samples_per_shard),
        "examples": examples,
        "shards": shards,
        "parts": parts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-index", default=None)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--expect-parts", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    merged = merge_part_indexes(root, expect_parts=args.expect_parts)
    out_index = Path(args.out_index) if args.out_index else root / "index.json"
    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_index.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary_out:
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
