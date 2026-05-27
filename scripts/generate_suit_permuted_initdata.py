#!/usr/bin/env python3
"""Generate suit-permuted initdata JSONL for official-judge evaluation."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Iterable


SUITS = ("W", "T", "B")
TILE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9/])([WTB])([1-9])(?![A-Za-z0-9])")
METADATA_STRING_KEYS = {
    "match_id",
    "source_match_id",
    "url",
    "game",
    "source",
    "belongs",
}


def permutation_mapping(permutation: tuple[str, str, str]) -> dict[str, str]:
    if sorted(permutation) != sorted(SUITS):
        raise ValueError(f"permutation must contain exactly {SUITS}: {permutation}")
    return dict(zip(SUITS, permutation))


def permutation_tag(permutation: tuple[str, str, str]) -> str:
    return "".join(permutation)


def all_suit_permutations(include_identity: bool = True) -> list[tuple[str, str, str]]:
    permutations = list(itertools.permutations(SUITS))
    if include_identity:
        return permutations
    return [permutation for permutation in permutations if permutation != SUITS]


def permute_tile_symbol(tile: str, mapping: dict[str, str]) -> str:
    if len(tile) >= 2 and tile[0] in SUITS:
        return f"{mapping[tile[0]]}{tile[1:]}"
    return tile


def permute_walltiles(walltiles: str, mapping: dict[str, str]) -> str:
    return " ".join(permute_tile_symbol(tile, mapping) for tile in walltiles.split())


def permute_text_tiles(text: str, mapping: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        return f"{mapping[match.group(1)]}{match.group(2)}"

    return TILE_TOKEN_RE.sub(replace, text)


def _permute_value(value, mapping: dict[str, str], key: str | None = None):
    if isinstance(value, dict):
        return {
            child_key: _permute_value(child_value, mapping, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_permute_value(item, mapping, key) for item in value]
    if isinstance(value, str):
        if key in METADATA_STRING_KEYS:
            return value
        return permute_text_tiles(value, mapping)
    return value


def permute_record_tiles(record: dict, mapping: dict[str, str]) -> dict:
    return _permute_value(record, mapping)


def generate_permuted_records(
    raw_path: Path | str,
    *,
    include_identity: bool = True,
    limit: int | None = None,
    offset: int = 0,
    permutations: Iterable[tuple[str, str, str]] | None = None,
) -> Iterable[dict]:
    selected_permutations = list(
        permutations if permutations is not None else all_suit_permutations(include_identity)
    )
    with Path(raw_path).open("r", encoding="utf-8") as src:
        seen = 0
        emitted_sources = 0
        for line in src:
            if not line.strip():
                continue
            if seen < offset:
                seen += 1
                continue
            if limit is not None and emitted_sources >= limit:
                break
            record = json.loads(line)
            initdata = dict(record["initdata"])
            source_match_id = str(record.get("match_id", seen))
            for permutation in selected_permutations:
                mapping = permutation_mapping(tuple(permutation))
                permuted_initdata = dict(initdata)
                permuted_initdata["walltiles"] = permute_walltiles(
                    str(initdata.get("walltiles", "")),
                    mapping,
                )
                tag = permutation_tag(tuple(permutation))
                yield {
                    "match_id": f"{source_match_id}__suit_{tag}",
                    "source_match_id": source_match_id,
                    "source_index": seen,
                    "suit_permutation": mapping,
                    "initdata": permuted_initdata,
                }
            seen += 1
            emitted_sources += 1


def generate_permuted_full_records(
    raw_path: Path | str,
    *,
    include_identity: bool = True,
    limit: int | None = None,
    offset: int = 0,
    permutations: Iterable[tuple[str, str, str]] | None = None,
) -> Iterable[dict]:
    selected_permutations = list(
        permutations if permutations is not None else all_suit_permutations(include_identity)
    )
    with Path(raw_path).open("r", encoding="utf-8") as src:
        seen = 0
        emitted_sources = 0
        for line in src:
            if not line.strip():
                continue
            if seen < offset:
                seen += 1
                continue
            if limit is not None and emitted_sources >= limit:
                break
            record = json.loads(line)
            source_match_id = str(record.get("match_id", seen))
            for permutation in selected_permutations:
                mapping = permutation_mapping(tuple(permutation))
                tag = permutation_tag(tuple(permutation))
                permuted = permute_record_tiles(record, mapping)
                permuted["match_id"] = f"{source_match_id}__suit_{tag}"
                permuted["source_match_id"] = source_match_id
                permuted["source_index"] = seen
                permuted["suit_permutation"] = mapping
                yield permuted
            seen += 1
            emitted_sources += 1


def parse_permutation(text: str) -> tuple[str, str, str]:
    stripped = text.strip().upper()
    if "," in stripped:
        parts = tuple(part.strip() for part in stripped.split(",") if part.strip())
    else:
        parts = tuple(stripped)
    if len(parts) != 3:
        raise ValueError(f"expected three suit letters, got {text!r}")
    return tuple(parts)  # type: ignore[return-value]


def write_records(records: Iterable[dict], out_path: Path | str) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="\n") as dst:
        for record in records:
            dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/botzone_mcr_1000.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--permutation", action="append", default=[])
    parser.add_argument("--full-record", action="store_true")
    args = parser.parse_args()

    permutations = [parse_permutation(text) for text in args.permutation] or None
    generator = generate_permuted_full_records if args.full_record else generate_permuted_records
    count = write_records(
        generator(
            args.raw,
            include_identity=args.include_identity,
            limit=args.limit,
            offset=args.offset,
            permutations=permutations,
        ),
        args.out,
    )
    print(json.dumps({"out": args.out, "records": count}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
