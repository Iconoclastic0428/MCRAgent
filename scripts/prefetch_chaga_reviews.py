#!/usr/bin/env python3
"""Parallel-prefetch CHAGA review JSON files for an audit raw corpus."""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from audit_chaga_review_alignment import (
    build_session_api_seat_maps,
    fetch_review,
    selected_players_for_audit,
)
from evaluate_chaga_replay import DEFAULT_PLAYER_RE


@dataclass(frozen=True, order=True)
class ReviewTarget:
    session_id: str
    api_seat: int
    player_name: str


def collect_review_targets(
    raw_path: Path,
    *,
    player_pattern: re.Pattern[str] = DEFAULT_PLAYER_RE,
    use_train_players: bool = False,
) -> list[ReviewTarget]:
    raw_records = list(_iter_jsonl(raw_path))
    session_api_seats = build_session_api_seat_maps(raw_records)
    targets: set[ReviewTarget] = set()
    for raw_record in raw_records:
        session_id = str(raw_record.get("belongs") or "")
        if not session_id:
            continue
        selected = selected_players_for_audit(
            raw_record,
            player_pattern=player_pattern,
            use_train_players=use_train_players,
        )
        for player_text, player_name in selected.items():
            try:
                seat = int(player_text)
            except (TypeError, ValueError):
                continue
            api_seat = session_api_seats.get(session_id, {}).get(player_name, seat)
            targets.add(ReviewTarget(session_id=session_id, api_seat=api_seat, player_name=player_name))
    return sorted(targets)


def prefetch_reviews(
    targets: list[ReviewTarget],
    *,
    cache_dir: Path,
    workers: int,
    force: bool = False,
    progress_every: int = 100,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [target for target in targets if force or not _cache_path(cache_dir, target).exists()]
    summary = {
        "targets": len(targets),
        "already_cached": len(targets) - len(missing),
        "requested": len(missing),
        "fetched": 0,
        "errors": [],
    }
    if not missing:
        return summary

    def fetch_one(target: ReviewTarget) -> tuple[ReviewTarget, int]:
        rows = fetch_review(target.session_id, target.api_seat, cache_dir)
        return target, len(rows)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch_one, target) for target in missing]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                target, rows = future.result()
                summary["fetched"] += 1
                if progress_every and (index == 1 or index % progress_every == 0 or index == len(futures)):
                    print(
                        f"prefetch {index}/{len(futures)} fetched={summary['fetched']} "
                        f"last={target.session_id}_seat{target.api_seat} rows={rows}",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as exc:
                summary["errors"].append(str(exc))
                if progress_every and (index == 1 or index % progress_every == 0 or index == len(futures)):
                    print(f"prefetch {index}/{len(futures)} error={exc}", file=sys.stderr, flush=True)
    return summary


def _cache_path(cache_dir: Path, target: ReviewTarget) -> Path:
    return cache_dir / f"{target.session_id}_seat{target.api_seat}.json"


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--cache-dir", default="data/raw/chaga_reviews")
    parser.add_argument("--player-regex", default=DEFAULT_PLAYER_RE.pattern)
    parser.add_argument("--use-train-players", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--summary-out")
    args = parser.parse_args()

    targets = collect_review_targets(
        Path(args.raw),
        player_pattern=re.compile(args.player_regex),
        use_train_players=args.use_train_players,
    )
    summary = prefetch_reviews(
        targets,
        cache_dir=Path(args.cache_dir),
        workers=args.workers,
        force=args.force,
        progress_every=args.progress_every,
    )
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
