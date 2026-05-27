import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_tziakcha_high_elo_live import high_elo_names_from_session, record_ids_from_session  # noqa: E402


def test_high_elo_names_from_session_uses_strict_threshold():
    session = {
        "players": [
            {"n": "Below", "e": 2300},
            {"n": "Above", "e": 2301},
            {"n": "Bad", "e": "not-a-number"},
        ]
    }

    assert high_elo_names_from_session(session, min_elo=2300) == {"Above"}


def test_record_ids_from_session_handles_string_and_object_records():
    session = {"records": ["r1", {"i": "r2"}, {"id": "r3"}, {}, None]}

    assert record_ids_from_session(session) == ["r1", "r2", "r3"]
