#!/usr/bin/env python3
"""Fetch live tziakcha sessions and build high-ELO training rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_chaga_github_corpus import annotate_audit_raw_record, prepare_record, selected_players_for_record
from tziakcha_records import convert_record, record_step


class TziakchaClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_json(self, path: str, body: bytes = b"") -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def fetch_live_high_elo(
    *,
    session_ids_path: Path,
    session_out: Path,
    raw_out: Path,
    prepared_out: Path,
    summary_out: Path,
    min_elo: float,
    workers: int,
    base_url: str = "https://tziakcha.net",
    timeout: float = 30.0,
    max_sessions: int | None = None,
    progress_every: int = 100,
) -> dict:
    session_ids = [line.strip() for line in session_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_sessions is not None:
        session_ids = session_ids[:max_sessions]

    session_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    prepared_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    summary = Counter(
        {
            "sessions_requested": len(session_ids),
            "sessions_fetched": 0,
            "sessions_with_high_elo": 0,
            "record_ids_seen": 0,
            "records_fetched": 0,
            "records_converted": 0,
            "records_prepared": 0,
            "audit_raw_records_written": 0,
            "prepare_drops": 0,
            "high_elo_train_player_slots": 0,
        }
    )
    fetch_errors: list[dict] = []
    convert_errors: list[dict] = []
    selected_player_record_counts: Counter[str] = Counter()

    def fetch_session_records(session_id: str) -> dict:
        client = TziakchaClient(base_url, timeout)
        session = client.post_json(f"/_qry/game/?id={urllib.parse.quote(session_id, safe='')}", b"")
        target_names = high_elo_names_from_session(session, min_elo=min_elo)
        records: list[tuple[dict, dict | None, str | None]] = []
        if target_names:
            for record_id in record_ids_from_session(session):
                try:
                    raw_record = client.post_json("/_qry/record/", urllib.parse.urlencode({"id": record_id}).encode("ascii"))
                    try:
                        converted = convert_record(raw_record)
                        records.append((raw_record, converted, None))
                    except Exception as exc:
                        records.append((raw_record, None, str(exc)))
                except Exception as exc:
                    records.append(({"id": record_id, "belongs": session_id}, None, str(exc)))
        return {"session_id": session_id, "session": session, "target_names": sorted(target_names), "records": records}

    with session_out.open("w", encoding="utf-8") as session_file, raw_out.open("w", encoding="utf-8") as raw_file, prepared_out.open(
        "w", encoding="utf-8"
    ) as prepared_file:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for index, result in enumerate(pool.map(_safe_fetch_session(fetch_session_records), session_ids), start=1):
                if result.get("error"):
                    fetch_errors.append({"session": result.get("session_id"), "error": result["error"]})
                    continue
                session = result["session"]
                target_names = set(result["target_names"])
                session_file.write(
                    json.dumps(
                        {"session_id": result["session_id"], "high_elo_player_names": sorted(target_names), "session": session},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                summary["sessions_fetched"] += 1
                if target_names:
                    summary["sessions_with_high_elo"] += 1
                records = result["records"]
                summary["record_ids_seen"] += len(records)
                for raw_record, converted, error in records:
                    if error and converted is None:
                        record_id = str(raw_record.get("id") or raw_record.get("match_id") or "")
                        if "logs" in raw_record or "step" in raw_record:
                            convert_errors.append({"record": record_id, "error": error})
                        else:
                            fetch_errors.append({"record": record_id, "session": result["session_id"], "error": error})
                        continue
                    summary["records_fetched"] += 1
                    summary["records_converted"] += 1
                    selected_players = selected_players_for_record(raw_record, target_names, min_elo=min_elo)
                    prepared = prepare_record(raw_record, converted, selected_players)
                    if prepared is None:
                        summary["prepare_drops"] += 1
                        continue
                    audit_raw = annotate_audit_raw_record(raw_record, selected_players)
                    raw_file.write(json.dumps(audit_raw, ensure_ascii=False, separators=(",", ":")) + "\n")
                    prepared_file.write(json.dumps(prepared, ensure_ascii=False, separators=(",", ":")) + "\n")
                    summary["records_prepared"] += 1
                    summary["audit_raw_records_written"] += 1
                    summary["high_elo_train_player_slots"] += len(selected_players)
                    selected_player_record_counts.update(prepared["train_player_names"].values())
                if progress_every and (index == 1 or index % progress_every == 0 or index == len(session_ids)):
                    print(
                        f"live {index}/{len(session_ids)} sessions_fetched={summary['sessions_fetched']} "
                        f"high_elo_sessions={summary['sessions_with_high_elo']} records_prepared={summary['records_prepared']} "
                        f"fetch_errors={len(fetch_errors)} convert_errors={len(convert_errors)}",
                        file=sys.stderr,
                        flush=True,
                    )

    out = dict(summary)
    out.update(
        {
            "session_ids_path": str(session_ids_path),
            "session_out": str(session_out),
            "raw_out": str(raw_out),
            "prepared_out": str(prepared_out),
            "summary_out": str(summary_out),
            "min_elo": min_elo,
            "workers": workers,
            "fetch_errors": fetch_errors,
            "convert_errors": convert_errors,
            "selected_player_record_counts": dict(selected_player_record_counts),
        }
    )
    summary_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def high_elo_names_from_session(session: dict, *, min_elo: float) -> set[str]:
    names: set[str] = set()
    for player in session.get("players") or []:
        if not isinstance(player, dict):
            continue
        try:
            elo = float(player.get("e") if player.get("e") is not None else 0.0)
        except (TypeError, ValueError):
            continue
        name = str(player.get("n") or "").strip()
        if name and elo > min_elo:
            names.add(name)
    return names


def record_ids_from_session(session: dict) -> list[str]:
    ids: list[str] = []
    for record in session.get("records") or []:
        if isinstance(record, dict):
            record_id = record.get("i") or record.get("id")
        else:
            record_id = record
        if record_id:
            ids.append(str(record_id))
    return ids


def _safe_fetch_session(fetch):
    def wrapped(session_id: str) -> dict:
        try:
            return fetch(session_id)
        except Exception as exc:
            return {"session_id": session_id, "error": str(exc)}

    return wrapped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-ids", required=True)
    parser.add_argument("--session-out", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--prepared-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--min-elo", type=float, default=2300.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="https://tziakcha.net")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    summary = fetch_live_high_elo(
        session_ids_path=Path(args.session_ids),
        session_out=Path(args.session_out),
        raw_out=Path(args.raw_out),
        prepared_out=Path(args.prepared_out),
        summary_out=Path(args.summary_out),
        min_elo=args.min_elo,
        workers=args.workers,
        base_url=args.base_url,
        timeout=args.timeout,
        max_sessions=args.max_sessions,
        progress_every=args.progress_every,
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"fetch_errors", "convert_errors", "selected_player_record_counts"}}, ensure_ascii=True, indent=2))
    return 0 if not summary["fetch_errors"] and not summary["convert_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
