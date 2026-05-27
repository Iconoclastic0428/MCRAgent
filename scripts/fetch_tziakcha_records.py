#!/usr/bin/env python3
"""Fetch public Tziakcha records and convert them to Botzone-like raw JSONL."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tziakcha_records import convert_record


@dataclass
class TziakchaFetchConfig:
    target_records: int
    raw_out: Path
    converted_out: Path
    summary_out: Path | None = None
    session_ids: tuple[str, ...] = ()
    max_pages: int = 1
    include_bots: bool = False
    base_url: str = "https://tziakcha.net"
    timeout: float = 20.0
    delay: float = 0.0


class TziakchaHttpClient:
    def __init__(self, base_url: str = "https://tziakcha.net", timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_json(self, path: str, body: bytes = b"") -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"User-Agent": "Mozilla/5.0"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def history_page_body(page: int) -> bytes:
    if page <= 0:
        return b""
    return f"p={page}".encode("ascii")


def record_body(record_id: str) -> bytes:
    return urllib.parse.urlencode({"id": record_id}).encode("ascii")


def player_names(item: dict) -> list[str]:
    names = []
    for player in item.get("players") or []:
        if isinstance(player, dict):
            names.append(str(player.get("n", "")))
        else:
            names.append(str(player))
    return names


def is_obvious_bot_session(item: dict) -> bool:
    return any(name.strip().lower().startswith("chaga") for name in player_names(item))


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


def session_summary(session: dict, record_count: int) -> dict:
    return {
        "id": str(session.get("id", "")),
        "title": session.get("title"),
        "players": player_names(session),
        "record_count": record_count,
    }


def console_summary_json(summary: dict) -> str:
    return json.dumps(summary, ensure_ascii=True, indent=2)


def fetch_and_convert(
    config: TziakchaFetchConfig,
    client: TziakchaHttpClient | None = None,
    converter: Callable[[dict], dict] = convert_record,
) -> dict:
    if config.target_records <= 0:
        raise ValueError("target_records must be positive")
    if config.max_pages <= 0:
        raise ValueError("max_pages must be positive")

    client = client or TziakchaHttpClient(config.base_url, config.timeout)
    config.raw_out.parent.mkdir(parents=True, exist_ok=True)
    config.converted_out.parent.mkdir(parents=True, exist_ok=True)
    if config.summary_out:
        config.summary_out.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "target_records": config.target_records,
        "history_pages_seen": 0,
        "history_games_seen": 0,
        "sessions_seen": 0,
        "sessions_skipped": [],
        "sessions_used": [],
        "records_attempted": 0,
        "records_written": 0,
        "converted_written": 0,
        "fetch_errors": [],
        "convert_errors": [],
        "raw_out": str(config.raw_out),
        "converted_out": str(config.converted_out),
        "summary_out": str(config.summary_out) if config.summary_out else None,
        "include_bots": config.include_bots,
        "max_pages": config.max_pages,
    }
    seen_records: set[str] = set()

    with config.raw_out.open("w", encoding="utf-8") as raw, config.converted_out.open(
        "w", encoding="utf-8"
    ) as converted_out:
        def process_session(session_id: str, game: dict | None = None) -> None:
            if summary["converted_written"] >= config.target_records:
                return
            if not session_id:
                return
            if game and not config.include_bots and is_obvious_bot_session(game):
                summary["sessions_skipped"].append(
                    {
                        "id": session_id,
                        "reason": "obvious_bot_player",
                        "players": player_names(game),
                    }
                )
                return

            summary["sessions_seen"] += 1
            try:
                session = client.post_json(
                    f"/_qry/game/?id={urllib.parse.quote(session_id, safe='')}", b""
                )
            except Exception as exc:
                summary["fetch_errors"].append({"session": session_id, "error": str(exc)})
                return
            if not config.include_bots and is_obvious_bot_session(session):
                summary["sessions_skipped"].append(
                    {
                        "id": session_id,
                        "reason": "obvious_bot_player",
                        "players": player_names(session),
                    }
                )
                return

            record_ids = record_ids_from_session(session)
            summary["sessions_used"].append(session_summary(session, len(record_ids)))
            for record_id in record_ids:
                if summary["converted_written"] >= config.target_records:
                    break
                if record_id in seen_records:
                    continue
                seen_records.add(record_id)
                summary["records_attempted"] += 1

                try:
                    record = client.post_json("/_qry/record/", record_body(record_id))
                except Exception as exc:
                    summary["fetch_errors"].append({"record": record_id, "error": str(exc)})
                    continue

                raw.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                summary["records_written"] += 1
                try:
                    converted = converter(record)
                except Exception as exc:
                    summary["convert_errors"].append({"record": record_id, "error": str(exc)})
                    continue

                converted_out.write(
                    json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                summary["converted_written"] += 1
                if config.delay > 0:
                    time.sleep(config.delay)

        if config.session_ids:
            for session_id in config.session_ids:
                process_session(str(session_id))
                if summary["converted_written"] >= config.target_records:
                    break
        else:
            for page in range(config.max_pages):
                if summary["converted_written"] >= config.target_records:
                    break
                try:
                    history = client.post_json("/_qry/history/", history_page_body(page))
                except Exception as exc:
                    summary["fetch_errors"].append({"history_page": page, "error": str(exc)})
                    continue

                games = history.get("games") or []
                summary["history_pages_seen"] += 1
                summary["history_games_seen"] += len(games)

                for game in games:
                    if summary["converted_written"] >= config.target_records:
                        break
                    process_session(str(game.get("id") or ""), game)

    if config.summary_out:
        config.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-records", type=int, required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--converted-out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--session-ids-file", default=None)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--include-bots", action="store_true")
    parser.add_argument("--base-url", default="https://tziakcha.net")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    session_ids = list(args.session_id)
    if args.session_ids_file:
        session_ids.extend(
            line.strip()
            for line in Path(args.session_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    summary = fetch_and_convert(
        TziakchaFetchConfig(
            target_records=args.target_records,
            raw_out=Path(args.raw_out),
            converted_out=Path(args.converted_out),
            summary_out=Path(args.summary_out) if args.summary_out else None,
            session_ids=tuple(session_ids),
            max_pages=args.max_pages,
            include_bots=args.include_bots,
            base_url=args.base_url,
            timeout=args.timeout,
            delay=args.delay,
        )
    )
    print(console_summary_json(summary))
    return 0 if summary["converted_written"] >= summary["target_records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
