import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_tziakcha_records import (
    TziakchaFetchConfig,
    console_summary_json,
    fetch_and_convert,
    history_page_body,
)


class FakeClient:
    def __init__(self, responses=None, failures=None):
        self.responses = responses or {}
        self.failures = failures or {}
        self.calls = []

    def post_json(self, path, body=b""):
        self.calls.append((path, body))
        key = (path, body)
        if key in self.failures:
            raise self.failures[key]
        return self.responses[key]


def line_json(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fake_converter(record):
    if record["id"] == "badconvert":
        raise ValueError("cannot convert")
    return {"match_id": record["id"], "logs": [], "scores": {"0": 0, "1": 0, "2": 0, "3": 0}}


def test_history_page_body_matches_tziakcha_browser_post_shape():
    assert history_page_body(0) == b""
    assert history_page_body(1) == b"p=1"
    assert history_page_body(7) == b"p=7"


def test_fetch_and_convert_filters_chaga_sessions_and_dedupes_records(tmp_path):
    raw_out = tmp_path / "raw.jsonl"
    converted_out = tmp_path / "converted.jsonl"
    summary_out = tmp_path / "summary.json"
    client = FakeClient(
        {
            ("/_qry/history/", b""): {
                "games": [
                    {
                        "id": "human1",
                        "title": "h1",
                        "players": [{"n": "Alice"}, {"n": "Bob"}, {"n": "Carol"}, {"n": "Dan"}],
                    },
                    {
                        "id": "bot1",
                        "title": "bot",
                        "players": [{"n": "CHAGA-v1"}, {"n": "Bob"}, {"n": "Carol"}, {"n": "Dan"}],
                    },
                    {
                        "id": "human2",
                        "title": "h2",
                        "players": [{"n": "Eve"}, {"n": "Frank"}, {"n": "Grace"}, {"n": "Heidi"}],
                    },
                ],
                "p": 0,
            },
            ("/_qry/game/?id=human1", b""): {
                "id": "human1",
                "title": "h1",
                "players": [{"n": "Alice"}, {"n": "Bob"}, {"n": "Carol"}, {"n": "Dan"}],
                "records": [{"i": "r1"}, {"i": "r2"}, {"i": "r1"}],
            },
            ("/_qry/game/?id=human2", b""): {
                "id": "human2",
                "title": "h2",
                "players": [{"n": "Eve"}, {"n": "Frank"}, {"n": "Grace"}, {"n": "Heidi"}],
                "records": [{"i": "r2"}, {"i": "r3"}],
            },
            ("/_qry/record/", b"id=r1"): {"id": "r1", "script": "encoded"},
            ("/_qry/record/", b"id=r2"): {"id": "r2", "script": "encoded"},
            ("/_qry/record/", b"id=r3"): {"id": "r3", "script": "encoded"},
        }
    )

    summary = fetch_and_convert(
        TziakchaFetchConfig(
            target_records=3,
            raw_out=raw_out,
            converted_out=converted_out,
            summary_out=summary_out,
            max_pages=1,
        ),
        client=client,
        converter=fake_converter,
    )

    assert summary["records_written"] == 3
    assert summary["converted_written"] == 3
    assert summary["sessions_skipped"][0]["id"] == "bot1"
    assert ("/_qry/game/?id=bot1", b"") not in client.calls
    assert client.calls.count(("/_qry/record/", b"id=r1")) == 1
    assert client.calls.count(("/_qry/record/", b"id=r2")) == 1
    assert [record["id"] for record in line_json(raw_out)] == ["r1", "r2", "r3"]
    assert [record["match_id"] for record in line_json(converted_out)] == ["r1", "r2", "r3"]
    assert json.loads(summary_out.read_text(encoding="utf-8"))["converted_written"] == 3


def test_fetch_and_convert_continues_after_fetch_and_convert_errors(tmp_path):
    raw_out = tmp_path / "raw.jsonl"
    converted_out = tmp_path / "converted.jsonl"
    client = FakeClient(
        {
            ("/_qry/history/", b""): {
                "games": [
                    {
                        "id": "human1",
                        "title": "h1",
                        "players": [{"n": "Alice"}, {"n": "Bob"}, {"n": "Carol"}, {"n": "Dan"}],
                    }
                ],
            },
            ("/_qry/game/?id=human1", b""): {
                "id": "human1",
                "title": "h1",
                "players": [{"n": "Alice"}, {"n": "Bob"}, {"n": "Carol"}, {"n": "Dan"}],
                "records": [{"i": "badfetch"}, {"i": "badconvert"}, {"i": "good1"}, {"i": "good2"}],
            },
            ("/_qry/record/", b"id=badconvert"): {"id": "badconvert", "script": "encoded"},
            ("/_qry/record/", b"id=good1"): {"id": "good1", "script": "encoded"},
            ("/_qry/record/", b"id=good2"): {"id": "good2", "script": "encoded"},
        },
        failures={("/_qry/record/", b"id=badfetch"): RuntimeError("network down")},
    )

    summary = fetch_and_convert(
        TziakchaFetchConfig(
            target_records=2,
            raw_out=raw_out,
            converted_out=converted_out,
            max_pages=1,
        ),
        client=client,
        converter=fake_converter,
    )

    assert summary["records_attempted"] == 4
    assert summary["records_written"] == 3
    assert summary["converted_written"] == 2
    assert summary["fetch_errors"][0]["record"] == "badfetch"
    assert summary["convert_errors"][0]["record"] == "badconvert"
    assert [record["id"] for record in line_json(raw_out)] == ["badconvert", "good1", "good2"]
    assert [record["match_id"] for record in line_json(converted_out)] == ["good1", "good2"]


def test_fetch_and_convert_can_advance_history_pages(tmp_path):
    raw_out = tmp_path / "raw.jsonl"
    converted_out = tmp_path / "converted.jsonl"
    client = FakeClient(
        {
            ("/_qry/history/", b""): {
                "games": [
                    {
                        "id": "page0",
                        "title": "p0",
                        "players": [{"n": "A"}, {"n": "B"}, {"n": "C"}, {"n": "D"}],
                    }
                ],
            },
            ("/_qry/history/", b"p=1"): {
                "games": [
                    {
                        "id": "page1",
                        "title": "p1",
                        "players": [{"n": "E"}, {"n": "F"}, {"n": "G"}, {"n": "H"}],
                    }
                ],
            },
            ("/_qry/game/?id=page0", b""): {
                "id": "page0",
                "title": "p0",
                "players": [{"n": "A"}, {"n": "B"}, {"n": "C"}, {"n": "D"}],
                "records": [{"i": "r0"}],
            },
            ("/_qry/game/?id=page1", b""): {
                "id": "page1",
                "title": "p1",
                "players": [{"n": "E"}, {"n": "F"}, {"n": "G"}, {"n": "H"}],
                "records": [{"i": "r1"}],
            },
            ("/_qry/record/", b"id=r0"): {"id": "r0", "script": "encoded"},
            ("/_qry/record/", b"id=r1"): {"id": "r1", "script": "encoded"},
        }
    )

    summary = fetch_and_convert(
        TziakchaFetchConfig(
            target_records=2,
            raw_out=raw_out,
            converted_out=converted_out,
            max_pages=2,
        ),
        client=client,
        converter=fake_converter,
    )

    assert summary["history_pages_seen"] == 2
    assert client.calls[0] == ("/_qry/history/", b"")
    assert ("/_qry/history/", b"p=1") in client.calls
    assert summary["converted_written"] == 2


def test_fetch_and_convert_can_fetch_explicit_session_ids_without_history(tmp_path):
    raw_out = tmp_path / "raw.jsonl"
    converted_out = tmp_path / "converted.jsonl"
    client = FakeClient(
        {
            ("/_qry/game/?id=manual1", b""): {
                "id": "manual1",
                "title": "manual",
                "players": [{"n": "A"}, {"n": "B"}, {"n": "C"}, {"n": "D"}],
                "records": [{"i": "r1"}, {"i": "r2"}],
            },
            ("/_qry/record/", b"id=r1"): {"id": "r1", "script": "encoded"},
            ("/_qry/record/", b"id=r2"): {"id": "r2", "script": "encoded"},
        }
    )

    summary = fetch_and_convert(
        TziakchaFetchConfig(
            target_records=2,
            raw_out=raw_out,
            converted_out=converted_out,
            session_ids=("manual1",),
        ),
        client=client,
        converter=fake_converter,
    )

    assert summary["history_pages_seen"] == 0
    assert ("/_qry/history/", b"") not in client.calls
    assert summary["sessions_used"][0]["id"] == "manual1"
    assert summary["converted_written"] == 2


def test_console_summary_json_is_ascii_safe_for_windows_console():
    text = console_summary_json({"title": "🀚 逍遥宫月赛", "players": ["千咲"]})

    text.encode("ascii")
    assert "\\ud83c\\udc1a" in text
    assert "\\u900d\\u9065" in text
