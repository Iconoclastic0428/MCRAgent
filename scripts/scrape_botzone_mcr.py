#!/usr/bin/env python3
"""Scrape public Botzone Chinese-Standard-Mahjong replay logs.

Botzone's monthly matchpack endpoint is the preferred large dataset source, but
the public replay pages also embed the full raw log JSON. This scraper follows
the global match-list pagination and stores one JSON object per match.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterable


BASE_URL = "https://en.botzone.org.cn"
GAME_ID = "5e37dcf74019f43051e53201"
GAME_NAME = "Chinese-Standard-Mahjong"
USER_AGENT = "MCR-Agent research scraper (public replay pages; contact: local research)"


MATCH_RE = re.compile(r"/match/([0-9a-f]{24})")
NEXT_RE = re.compile(
    r"globalmatchlist\?startid=([0-9a-f]{24})(?:&amp;|&)game=" + re.escape(GAME_ID)
)
RAW_LOG_RE = re.compile(r'var\s+_rawLogJSON\s*=\s*("(?:\\.|[^"\\])*")\s*;')
INITDATA_RE = re.compile(r'initdata\s*=\s*("(?:\\.|[^"\\])*")\s*;')
GLOBALDATA_RE = re.compile(r"globaldata\s*=\s*(\[[^\n;]*\])\s*;")


def fetch_text(url: str, timeout: float = 45.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8-sig", errors="replace")


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def extract_match_ids(list_html: str) -> list[str]:
    return unique_in_order(MATCH_RE.findall(list_html))


def extract_next_url(list_html: str) -> str | None:
    matches = list(NEXT_RE.finditer(list_html))
    if not matches:
        return None
    startid = matches[-1].group(1)
    return f"{BASE_URL}/globalmatchlist?startid={startid}&game={GAME_ID}"


def parse_js_string_literal(source: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(source)
    if not match:
        return None
    return json.loads(match.group(1))


def parse_replay_page(match_id: str, page_html: str) -> dict:
    raw_log_json = parse_js_string_literal(page_html, RAW_LOG_RE)
    if raw_log_json is None:
        raise ValueError(f"match {match_id}: _rawLogJSON not found")

    initdata = parse_js_string_literal(page_html, INITDATA_RE)
    globaldata_match = GLOBALDATA_RE.search(page_html)
    globaldata = json.loads(globaldata_match.group(1)) if globaldata_match else None

    logs = json.loads(raw_log_json)
    finish = logs[-1].get("output", {}) if logs else {}
    scores = finish.get("content") if finish.get("command") == "finish" else None

    return {
        "match_id": match_id,
        "url": f"{BASE_URL}/match/{match_id}",
        "game": GAME_NAME,
        "log_count": len(logs),
        "turn_count": max(0, (len(logs) - 1) // 2),
        "scores": scores,
        "initdata": json.loads(initdata) if initdata else None,
        "globaldata": globaldata,
        "logs": logs,
    }


def fetch_match_records(
    match_ids: list[str],
    sleep: float = 0.0,
    workers: int = 1,
    fetcher: Callable[[str], str] = fetch_text,
    verbose: bool = False,
    progress_every: int = 25,
) -> tuple[list[dict], list[dict]]:
    records: list[dict | None] = [None] * len(match_ids)
    failed: list[dict] = []
    max_workers = max(1, workers)

    def fetch_one(match_id: str) -> dict:
        url = f"{BASE_URL}/match/{match_id}"
        return parse_replay_page(match_id, fetcher(url))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        for index, match_id in enumerate(match_ids):
            future = executor.submit(fetch_one, match_id)
            future_to_index[future] = index
            if sleep and index < len(match_ids) - 1:
                time.sleep(sleep)

        completed = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            match_id = match_ids[index]
            url = f"{BASE_URL}/match/{match_id}"
            try:
                records[index] = future.result()
                completed += 1
                if verbose and (completed == 1 or completed % progress_every == 0):
                    print(
                        f"fetched {completed}/{len(match_ids)} match_id={match_id}",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as exc:
                failed.append({"match_id": match_id, "url": url, "error": str(exc)})

    return [record for record in records if record is not None], failed


def scrape(args: argparse.Namespace) -> dict:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_url = f"{BASE_URL}/globalmatchlist?game={GAME_ID}"
    all_match_ids: list[str] = []
    pages_seen = 0

    while list_url and pages_seen < args.pages and len(all_match_ids) < args.max_matches:
        pages_seen += 1
        list_html = fetch_text(list_url)
        page_ids = extract_match_ids(list_html)
        for match_id in page_ids:
            if match_id not in all_match_ids:
                all_match_ids.append(match_id)
                if len(all_match_ids) >= args.max_matches:
                    break
        list_url = extract_next_url(list_html)
        if args.verbose:
            print(
                f"page {pages_seen}: ids={len(all_match_ids)} next={bool(list_url)}",
                file=sys.stderr,
                flush=True,
            )
        if args.sleep:
            time.sleep(args.sleep)

    records, failed = fetch_match_records(
        all_match_ids,
        sleep=args.sleep,
        workers=args.workers,
        fetcher=fetch_text,
        verbose=args.verbose,
        progress_every=args.progress_every,
    )
    with out_path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "game": GAME_NAME,
        "source": f"{BASE_URL}/globalmatchlist?game={GAME_ID}",
        "pages_seen": pages_seen,
        "match_ids_found": len(all_match_ids),
        "matches_written": len(records),
        "failed": failed,
        "out": str(out_path),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1, help="match-list pages to follow")
    parser.add_argument("--max-matches", type=int, default=20, help="maximum replay pages to scrape")
    parser.add_argument("--sleep", type=float, default=0.2, help="seconds between HTTP requests")
    parser.add_argument("--out", default="data/raw/botzone_mcr_sample.jsonl")
    parser.add_argument("--verbose", action="store_true", help="print progress to stderr")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1, help="parallel replay fetch workers")
    args = parser.parse_args()

    if args.pages < 1 or args.max_matches < 1:
        parser.error("--pages and --max-matches must be positive")

    summary = scrape(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
