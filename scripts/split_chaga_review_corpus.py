#!/usr/bin/env python3
"""Build session-disjoint raw/audit splits for CHAGA-reviewed training."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "val", "test")


def split_review_corpus(
    *,
    raw_path: Path,
    audit_path: Path,
    out_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict:
    if train_ratio < 0.0 or val_ratio < 0.0 or train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to at most 1")

    raw_rows = list(iter_jsonl(raw_path))
    audit_rows = list(iter_jsonl(audit_path))
    if not audit_rows:
        raise ValueError("audit corpus is empty")

    raw_by_session: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        session_id = str(row.get("belongs") or "")
        if session_id:
            raw_by_session[session_id].append(row)

    audit_by_session: dict[str, list[dict]] = defaultdict(list)
    for row in audit_rows:
        session_id = str(row.get("session_id") or "")
        if session_id:
            audit_by_session[session_id].append(row)

    audit_sessions = set(audit_by_session)
    raw_sessions = set(raw_by_session)
    missing_raw = sorted(audit_sessions - raw_sessions)
    if missing_raw:
        raise ValueError(f"audit sessions missing raw records: {missing_raw[:10]}")

    split_sessions = assign_session_splits(sorted(audit_sessions), seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    assert_session_disjoint(split_sessions)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "mcr_chaga_review_session_split_v1",
        "raw_path": str(raw_path),
        "audit_path": str(audit_path),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "raw_records": len(raw_rows),
        "audit_rows": len(audit_rows),
        "sessions_with_review": len(audit_sessions),
        "dropped_raw_records_without_review": sum(
            len(rows)
            for session_id, rows in raw_by_session.items()
            if session_id not in audit_sessions
        ),
        "splits": {},
    }

    for split in SPLITS:
        sessions = split_sessions[split]
        raw_split = [row for session in sessions for row in raw_by_session.get(session, [])]
        audit_split = [row for session in sessions for row in audit_by_session.get(session, [])]
        write_jsonl(out_dir / f"{split}.raw.jsonl", raw_split)
        write_jsonl(out_dir / f"{split}.audit.jsonl", audit_split)
        summary["splits"][split] = {
            "sessions": sessions,
            "session_count": len(sessions),
            "raw_records": len(raw_split),
            "reviewed_states": len(audit_split),
        }

    verify_split_outputs(summary, raw_by_session, audit_by_session)
    (out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def assign_session_splits(
    sessions: list[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[str]]:
    shuffled = list(sessions)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    counts = allocate_split_counts(total, train_ratio=train_ratio, val_ratio=val_ratio)
    train_count = counts["train"]
    val_count = counts["val"]
    train_sessions = sorted(shuffled[:train_count])
    val_sessions = sorted(shuffled[train_count : train_count + val_count])
    test_sessions = sorted(shuffled[train_count + val_count :])
    return {"train": train_sessions, "val": val_sessions, "test": test_sessions}


def allocate_split_counts(total: int, *, train_ratio: float, val_ratio: float) -> dict[str, int]:
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    quotas = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: int(quotas[split]) for split in SPLITS}
    remaining = total - sum(counts.values())
    for split in sorted(SPLITS, key=lambda item: quotas[item] - counts[item], reverse=True):
        if remaining <= 0:
            break
        counts[split] += 1
        remaining -= 1
    positive_splits = [split for split in SPLITS if ratios[split] > 0.0]
    if total >= len(positive_splits):
        for split in positive_splits:
            if counts[split] > 0:
                continue
            donor = max(
                (candidate for candidate in positive_splits if counts[candidate] > 1),
                key=lambda candidate: counts[candidate],
                default=None,
            )
            if donor is None:
                break
            counts[donor] -= 1
            counts[split] += 1
    return counts


def verify_split_outputs(
    summary: dict,
    raw_by_session: dict[str, list[dict]],
    audit_by_session: dict[str, list[dict]],
) -> None:
    split_sets = {split: set(summary["splits"][split]["sessions"]) for split in SPLITS}
    assert_session_disjoint({split: sorted(sessions) for split, sessions in split_sets.items()})
    assigned = set().union(*split_sets.values())
    if assigned != set(audit_by_session):
        missing = sorted(set(audit_by_session) - assigned)
        extra = sorted(assigned - set(audit_by_session))
        raise ValueError(f"split sessions do not match audit sessions: missing={missing[:10]} extra={extra[:10]}")
    for split in SPLITS:
        sessions = split_sets[split]
        expected_raw = sum(len(raw_by_session[session]) for session in sessions)
        expected_audit = sum(len(audit_by_session[session]) for session in sessions)
        if summary["splits"][split]["raw_records"] != expected_raw:
            raise ValueError(f"{split} raw count mismatch")
        if summary["splits"][split]["reviewed_states"] != expected_audit:
            raise ValueError(f"{split} audit count mismatch")


def assert_session_disjoint(split_sessions: dict[str, list[str]]) -> None:
    seen: Counter[str] = Counter()
    for sessions in split_sessions.values():
        seen.update(sessions)
    repeated = sorted(session for session, count in seen.items() if count > 1)
    if repeated:
        raise ValueError(f"sessions appear in multiple splits: {repeated[:10]}")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    summary = split_review_corpus(
        raw_path=Path(args.raw),
        audit_path=Path(args.audit),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
