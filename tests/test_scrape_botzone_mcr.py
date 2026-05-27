import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import json

from scrape_botzone_mcr import extract_next_url, fetch_match_records


def test_extract_next_url_uses_last_startid_pagination_link():
    html = """
    <a href="globalmatchlist?startid=oldcurrent&amp;game=5e37dcf74019f43051e53201">current</a>
    <a href="globalmatchlist?startid=1234567890abcdef12345678&game=5e37dcf74019f43051e53201">next</a>
    """

    assert extract_next_url(html).endswith("startid=1234567890abcdef12345678&game=5e37dcf74019f43051e53201")


def replay_html(score):
    logs = [{"output": {"command": "finish", "content": {"0": score}}}]
    raw_log_literal = json.dumps(json.dumps(logs))
    return f"var _rawLogJSON = {raw_log_literal};"


def test_fetch_match_records_preserves_match_order_with_workers():
    first = "111111111111111111111111"
    second = "222222222222222222222222"

    def fake_fetch(url):
        if first in url:
            return replay_html(1)
        if second in url:
            return replay_html(2)
        raise AssertionError(url)

    records, failed = fetch_match_records(
        [first, second],
        sleep=0,
        workers=2,
        fetcher=fake_fetch,
    )

    assert failed == []
    assert [record["match_id"] for record in records] == [first, second]
    assert [record["scores"]["0"] for record in records] == [1, 2]
