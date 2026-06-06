"""Hierarchical action spaces from the Tjong paper."""

from __future__ import annotations

from dataclasses import dataclass

ACTION_NAMES = (
    "PASS",
    "HU",
    "DISCARD",
    "CHOW",
    "PONG",
    "MINGKONG",
    "BUKONG",
    "ANKONG",
)

CLAIM_ACTION_NAMES = ("CHOW", "PONG", "MINGKONG", "BUKONG", "ANKONG")

ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
CLAIM_ACTION_TO_INDEX = {name: index for index, name in enumerate(CLAIM_ACTION_NAMES)}

DISCARD_SIZE = 34

# Table 1 gives Chow as 63 and the other claim families as 34 each.
CLAIM_GROUP_SIZES = {
    "CHOW": 63,
    "PONG": 34,
    "MINGKONG": 34,
    "BUKONG": 34,
    "ANKONG": 34,
}
CLAIM_SIZE = sum(CLAIM_GROUP_SIZES.values())


@dataclass(frozen=True)
class ClaimIndex:
    family: str
    local_index: int
    flat_index: int


def claim_offsets() -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for name in CLAIM_ACTION_NAMES:
        offsets[name] = cursor
        cursor += CLAIM_GROUP_SIZES[name]
    return offsets


CLAIM_OFFSETS = claim_offsets()


def flatten_claim(family: str, local_index: int) -> int:
    family = family.upper()
    if family not in CLAIM_GROUP_SIZES:
        raise ValueError(f"unknown claim family: {family}")
    if local_index < 0 or local_index >= CLAIM_GROUP_SIZES[family]:
        raise ValueError(f"claim index out of range for {family}: {local_index}")
    return CLAIM_OFFSETS[family] + int(local_index)


def unflatten_claim(flat_index: int) -> ClaimIndex:
    if flat_index < 0 or flat_index >= CLAIM_SIZE:
        raise ValueError(f"claim index out of range: {flat_index}")
    for family in CLAIM_ACTION_NAMES:
        start = CLAIM_OFFSETS[family]
        end = start + CLAIM_GROUP_SIZES[family]
        if start <= flat_index < end:
            return ClaimIndex(family=family, local_index=flat_index - start, flat_index=flat_index)
    raise AssertionError("unreachable claim index state")
