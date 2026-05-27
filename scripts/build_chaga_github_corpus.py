#!/usr/bin/env python3
"""Build CHAGA02-08 high-ELO corpora from the public tziakcha_records archive."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from tziakcha_records import convert_record, record_step


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/tziakcha-stats/tziakcha_records/main"
GITHUB_API_BASE = "https://api.github.com/repos/tziakcha-stats/tziakcha_records/contents"
DEFAULT_PLAYER_PATTERN = r"^CHAGA0[2-8]$"


@dataclass(frozen=True)
class TargetSession:
    session_id: str
    players: dict[str, str]
    start_time: int | None = None
    finish_time: int | None = None


@dataclass(frozen=True)
class RecordLocation:
    session_id: str
    record_id: str
    period: str


def load_player_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        for row in csv.DictReader(src):
            name = str(row.get("Player Name") or "").strip()
            if not name:
                continue
            try:
                scores[name] = float(row.get("Value") or 0.0)
            except (TypeError, ValueError):
                continue
    return scores


def select_target_sessions(
    history: Iterable[dict],
    *,
    player_scores: dict[str, float],
    min_elo: float,
    player_pattern: str = DEFAULT_PLAYER_PATTERN,
) -> list[TargetSession]:
    pattern = re.compile(player_pattern)
    selected: list[TargetSession] = []
    for session in history:
        players: dict[str, str] = {}
        for index, player in enumerate(session.get("players") or []):
            if not isinstance(player, dict):
                continue
            name = str(player.get("n") or "").strip()
            if not pattern.search(name):
                continue
            current_elo = player_scores.get(name)
            if current_elo is None or current_elo <= min_elo:
                continue
            try:
                session_elo = float(player.get("e") if player.get("e") is not None else current_elo)
            except (TypeError, ValueError):
                session_elo = current_elo
            if session_elo <= min_elo:
                continue
            players[str(index)] = name
        if players:
            selected.append(
                TargetSession(
                    session_id=str(session.get("id") or ""),
                    players=players,
                    start_time=_optional_int(session.get("start_time")),
                    finish_time=_optional_int(session.get("finish_time")),
                )
            )
    return [session for session in selected if session.session_id]


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_history(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_session_files(session_dir: Path) -> dict[str, list[dict]]:
    session_files: dict[str, list[dict]] = {}
    for path in sorted(session_dir.glob("*.json")):
        session_files[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return session_files


def plan_record_locations(
    selected_session_ids: set[str],
    session_files: dict[str, list[dict]],
) -> list[RecordLocation]:
    locations: list[RecordLocation] = []
    seen_records: set[str] = set()
    for period, sessions in sorted(session_files.items()):
        for session in sessions:
            session_id = str(session.get("session_id") or session.get("id") or "")
            if session_id not in selected_session_ids:
                continue
            for record_id in _record_ids_from_session_entry(session):
                if record_id in seen_records:
                    continue
                seen_records.add(record_id)
                locations.append(RecordLocation(session_id=session_id, record_id=record_id, period=period))
    return locations


def _record_ids_from_session_entry(session: dict) -> list[str]:
    ids: list[str] = []
    for item in session.get("records") or []:
        if isinstance(item, dict):
            record_id = item.get("i") or item.get("id")
        else:
            record_id = item
        if record_id:
            ids.append(str(record_id))
    return ids


def prepare_record(raw_record: dict, converted_record: dict, selected_players: dict[str, str]) -> dict | None:
    if not selected_players:
        return None
    prepared = dict(converted_record)
    prepared["source_record_id"] = str(raw_record.get("id") or raw_record.get("match_id") or "")
    prepared["source_players"] = player_names_from_record(raw_record)
    prepared["train_players"] = sorted(selected_players)
    prepared["train_player_names"] = {player: selected_players[player] for player in sorted(selected_players)}
    return prepared


def player_names_from_record(raw_record: dict) -> list[str]:
    try:
        step = record_step(raw_record)
    except Exception:
        return []
    names: list[str] = []
    for player in step.get("p") or []:
        if isinstance(player, dict):
            names.append(str(player.get("n", "")).strip())
        else:
            names.append(str(player).strip())
    return names


def build_github_chaga_corpus(
    *,
    history_path: Path,
    session_dir: Path,
    elo_csv: Path,
    raw_out: Path,
    prepared_out: Path,
    summary_out: Path,
    min_elo: float,
    player_pattern: str = DEFAULT_PLAYER_PATTERN,
    fetch_record: Callable[[RecordLocation], dict] | None = None,
    converter: Callable[[dict], dict] = convert_record,
    max_sessions: int | None = None,
    max_records: int | None = None,
    workers: int = 1,
) -> dict:
    player_scores = load_player_scores(elo_csv)
    target_sessions = select_target_sessions(
        load_history(history_path),
        player_scores=player_scores,
        min_elo=min_elo,
        player_pattern=player_pattern,
    )
    if max_sessions is not None:
        target_sessions = target_sessions[:max_sessions]
    selected_by_session = {session.session_id: session.players for session in target_sessions}
    locations = plan_record_locations(set(selected_by_session), load_session_files(session_dir))
    if max_records is not None:
        locations = locations[:max_records]
    fetch_record = fetch_record or github_record_fetcher(Path("data/raw/github_tziakcha_records"))

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    prepared_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "mcr_chaga_github_corpus_v1",
        "history_path": str(history_path),
        "session_dir": str(session_dir),
        "elo_csv": str(elo_csv),
        "min_elo": min_elo,
        "player_pattern": player_pattern,
        "sessions_selected": len(target_sessions),
        "record_locations": len(locations),
        "records_fetched": 0,
        "records_prepared": 0,
        "fetch_errors": [],
        "convert_errors": [],
        "prepare_drops": 0,
        "selected_player_record_counts": {},
        "raw_out": str(raw_out),
        "prepared_out": str(prepared_out),
        "summary_out": str(summary_out),
    }

    with raw_out.open("w", encoding="utf-8") as raw_file, prepared_out.open("w", encoding="utf-8") as prepared_file:
        for location, raw_record, fetch_error in iter_fetched_records(locations, fetch_record, workers=max(1, workers)):
            if fetch_error is not None:
                summary["fetch_errors"].append({"record": location.record_id, "period": location.period, "error": str(fetch_error)})
                continue
            summary["records_fetched"] += 1
            raw_file.write(json.dumps(raw_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            try:
                converted = converter(raw_record)
            except Exception as exc:
                summary["convert_errors"].append({"record": location.record_id, "period": location.period, "error": str(exc)})
                continue
            prepared = prepare_record(raw_record, converted, selected_by_session.get(location.session_id, {}))
            if prepared is None:
                summary["prepare_drops"] += 1
                continue
            prepared_file.write(json.dumps(prepared, ensure_ascii=False, separators=(",", ":")) + "\n")
            summary["records_prepared"] += 1
            for name in prepared["train_player_names"].values():
                counts = summary["selected_player_record_counts"]
                counts[name] = counts.get(name, 0) + 1

    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def iter_fetched_records(
    locations: list[RecordLocation],
    fetch_record: Callable[[RecordLocation], dict],
    *,
    workers: int,
) -> Iterable[tuple[RecordLocation, dict | None, Exception | None]]:
    def fetch_one(location: RecordLocation) -> tuple[RecordLocation, dict | None, Exception | None]:
        try:
            return location, fetch_record(location), None
        except Exception as exc:
            return location, None, exc

    if workers <= 1:
        for location in locations:
            yield fetch_one(location)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(fetch_one, locations)


def github_record_fetcher(cache_dir: Path, *, refresh: bool = False, delay: float = 0.0) -> Callable[[RecordLocation], dict]:
    def fetch(location: RecordLocation) -> dict:
        cache_path = cache_dir / location.period / f"{location.record_id}.json"
        if not refresh and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        url = f"{GITHUB_RAW_BASE}/records/{location.period}/{location.record_id}.json"
        data = fetch_json_url(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        if delay > 0:
            time.sleep(delay)
        return data

    return fetch


def local_archive_record_fetcher(archive_root: Path) -> Callable[[RecordLocation], dict]:
    """Read records from a local clone/extract of tziakcha-stats/tziakcha_records."""

    def fetch(location: RecordLocation) -> dict:
        path = archive_root / "records" / location.period / f"{location.record_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    return fetch


def fetch_json_url(url: str) -> dict | list:
    request = urllib.request.Request(url, headers={"User-Agent": "Codex"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_github_metadata(history_path: Path, session_dir: Path, *, refresh: bool = False) -> dict:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    downloaded_sessions = 0
    if refresh or not history_path.exists():
        history = fetch_json_url(f"{GITHUB_RAW_BASE}/history/history_util0103.json")
        history_path.write_text(json.dumps(history, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    listing = fetch_json_url(f"{GITHUB_API_BASE}/session?ref=main")
    if not isinstance(listing, list):
        raise ValueError("unexpected GitHub session listing response")
    for item in listing:
        name = str(item.get("name") or "")
        if not name.endswith(".json"):
            continue
        out = session_dir / name
        if not refresh and out.exists():
            continue
        download_url = item.get("download_url")
        if not download_url:
            continue
        data = fetch_json_url(str(download_url))
        out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        downloaded_sessions += 1
    return {
        "history_path": str(history_path),
        "session_dir": str(session_dir),
        "session_files": len(list(session_dir.glob("*.json"))),
        "downloaded_session_files": downloaded_sessions,
    }


def ensure_archive_metadata(archive_root: Path, history_path: Path, session_dir: Path, *, refresh: bool = False) -> dict:
    archive_history = archive_root / "history" / "history_util0103.json"
    archive_session_dir = archive_root / "session"
    if not archive_history.exists():
        raise FileNotFoundError(f"missing archive history file: {archive_history}")
    if not archive_session_dir.exists():
        raise FileNotFoundError(f"missing archive session directory: {archive_session_dir}")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    copied_sessions = 0
    if refresh or not history_path.exists():
        shutil.copy2(archive_history, history_path)
    for src in sorted(archive_session_dir.glob("*.json")):
        dst = session_dir / src.name
        if refresh or not dst.exists():
            shutil.copy2(src, dst)
            copied_sessions += 1
    return {
        "archive_root": str(archive_root),
        "history_path": str(history_path),
        "session_dir": str(session_dir),
        "session_files": len(list(session_dir.glob("*.json"))),
        "copied_session_files": copied_sessions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-path", default="data/raw/github_tziakcha_history_util0103.json")
    parser.add_argument("--session-dir", default="data/raw/github_tziakcha_sessions")
    parser.add_argument("--elo-csv", default="data/raw/tziakcha_current_elo.csv")
    parser.add_argument("--raw-out", default="data/raw/github_chaga0208_elo2300_records.jsonl")
    parser.add_argument("--prepared-out", default="data/processed/github_chaga0208_elo2300/chaga0208_elo2300_all.jsonl")
    parser.add_argument("--summary-out", default="data/processed/github_chaga0208_elo2300/chaga0208_elo2300_summary.json")
    parser.add_argument("--archive-root", default=None, help="local tziakcha_records clone/extract root; avoids GitHub raw rate limits")
    parser.add_argument("--record-cache-dir", default="data/raw/github_tziakcha_records")
    parser.add_argument("--min-elo", type=float, default=2300.0)
    parser.add_argument("--player-pattern", default=DEFAULT_PLAYER_PATTERN)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--refresh-records", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    if args.archive_root:
        archive_root = Path(args.archive_root)
        metadata_summary = ensure_archive_metadata(
            archive_root,
            Path(args.history_path),
            Path(args.session_dir),
            refresh=args.refresh_metadata,
        )
        fetch_record = local_archive_record_fetcher(archive_root)
    else:
        metadata_summary = ensure_github_metadata(
            Path(args.history_path),
            Path(args.session_dir),
            refresh=args.refresh_metadata,
        )
        fetch_record = github_record_fetcher(Path(args.record_cache_dir), refresh=args.refresh_records, delay=args.delay)
    summary = build_github_chaga_corpus(
        history_path=Path(args.history_path),
        session_dir=Path(args.session_dir),
        elo_csv=Path(args.elo_csv),
        raw_out=Path(args.raw_out),
        prepared_out=Path(args.prepared_out),
        summary_out=Path(args.summary_out),
        min_elo=args.min_elo,
        player_pattern=args.player_pattern,
        fetch_record=fetch_record,
        max_sessions=args.max_sessions,
        max_records=args.max_records,
        workers=args.workers,
    )
    summary["metadata_summary"] = metadata_summary
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["fetch_errors"] and not summary["convert_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
