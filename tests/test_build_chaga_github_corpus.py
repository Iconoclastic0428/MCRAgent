import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_chaga_github_corpus import (  # noqa: E402
    RecordLocation,
    build_github_chaga_corpus,
    local_archive_record_fetcher,
    plan_record_locations,
    select_target_sessions,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_select_target_sessions_requires_chaga0208_and_min_elo():
    history = [
        {
            "id": "s1",
            "players": [
                {"n": "CHAGA01", "e": 2800},
                {"n": "CHAGA02", "e": 2401},
                {"n": "CHAGA06", "e": 2299},
                {"n": "Human", "e": 2600},
            ],
        },
        {"id": "s2", "players": [{"n": "CHAGA08", "e": 2500}]},
    ]

    selected = select_target_sessions(history, player_scores={"CHAGA02": 2806, "CHAGA08": 2479}, min_elo=2300)

    assert [session.session_id for session in selected] == ["s1", "s2"]
    assert selected[0].players == {"1": "CHAGA02"}
    assert selected[1].players == {"0": "CHAGA08"}


def test_plan_record_locations_uses_period_directory_for_selected_sessions():
    session_files = {
        "history_a": [
            {"session_id": "s1", "records": ["r1", "r2"]},
            {"session_id": "skip", "records": ["x"]},
        ],
        "history_b": [
            {"session_id": "s2", "records": [{"i": "r3"}]},
        ],
    }

    locations = plan_record_locations({"s1", "s2"}, session_files)

    assert locations == [
        RecordLocation(session_id="s1", record_id="r1", period="history_a"),
        RecordLocation(session_id="s1", record_id="r2", period="history_a"),
        RecordLocation(session_id="s2", record_id="r3", period="history_b"),
    ]


def test_build_github_chaga_corpus_writes_only_eligible_train_players(tmp_path):
    history_path = tmp_path / "history.json"
    elo_path = tmp_path / "elo.csv"
    session_dir = tmp_path / "session"
    out_raw = tmp_path / "raw.jsonl"
    out_prepared = tmp_path / "prepared.jsonl"
    summary_out = tmp_path / "summary.json"
    write_json(
        history_path,
        [
            {
                "id": "s1",
                "players": [
                    {"n": "CHAGA01", "e": 2800},
                    {"n": "CHAGA02", "e": 2400},
                    {"n": "CHAGA06", "e": 2299},
                    {"n": "CHAGA08", "e": 2500},
                ],
            }
        ],
    )
    write_csv(
        elo_path,
        "Rank,Value,Player Name,Rounds,Player ID\n"
        "1,2800,CHAGA01,1,a\n"
        "2,2700,CHAGA02,1,b\n"
        "3,2600,CHAGA08,1,c\n",
    )
    write_json(session_dir / "history_a.json", [{"session_id": "s1", "records": ["r1"]}])

    record = {
        "id": "r1",
        "belongs": "s1",
        "step": {
            "p": [{"n": "CHAGA01"}, {"n": "CHAGA02"}, {"n": "CHAGA06"}, {"n": "CHAGA08"}],
        },
    }

    def fetch_record(location: RecordLocation) -> dict:
        assert location.record_id == "r1"
        return dict(record)

    def converter(raw_record: dict) -> dict:
        return {
            "match_id": raw_record["id"],
            "belongs": raw_record["belongs"],
            "logs": [],
            "scores": {"0": 0, "1": 0, "2": 0, "3": 0},
        }

    summary = build_github_chaga_corpus(
        history_path=history_path,
        session_dir=session_dir,
        elo_csv=elo_path,
        raw_out=out_raw,
        prepared_out=out_prepared,
        summary_out=summary_out,
        min_elo=2300,
        fetch_record=fetch_record,
        converter=converter,
        workers=2,
    )

    prepared = [json.loads(line) for line in out_prepared.read_text(encoding="utf-8").splitlines()]
    raw = [json.loads(line) for line in out_raw.read_text(encoding="utf-8").splitlines()]

    assert summary["sessions_selected"] == 1
    assert summary["records_prepared"] == 1
    assert summary["audit_raw_records_written"] == 1
    assert raw[0]["id"] == "r1"
    assert raw[0]["train_players"] == ["1", "3"]
    assert raw[0]["train_player_names"] == {"1": "CHAGA02", "3": "CHAGA08"}
    assert raw[0]["source_record_id"] == "r1"
    assert prepared[0]["train_players"] == ["1", "3"]
    assert prepared[0]["train_player_names"] == {"1": "CHAGA02", "3": "CHAGA08"}


def test_local_archive_record_fetcher_reads_period_record(tmp_path):
    archive_root = tmp_path / "tziakcha_records"
    write_json(archive_root / "records" / "history_a" / "r1.json", {"id": "r1", "belongs": "s1"})

    fetch_record = local_archive_record_fetcher(archive_root)

    assert fetch_record(RecordLocation(session_id="s1", record_id="r1", period="history_a")) == {
        "id": "r1",
        "belongs": "s1",
    }


def test_build_github_chaga_corpus_writes_raw_only_after_prepare_success(tmp_path):
    history_path = tmp_path / "history.json"
    elo_path = tmp_path / "elo.csv"
    session_dir = tmp_path / "session"
    out_raw = tmp_path / "raw.jsonl"
    out_prepared = tmp_path / "prepared.jsonl"
    summary_out = tmp_path / "summary.json"
    write_json(
        history_path,
        [{"id": "s1", "players": [{"n": "CHAGA02", "e": 2400}]}],
    )
    write_csv(elo_path, "Rank,Value,Player Name,Rounds,Player ID\n1,2700,CHAGA02,1,b\n")
    write_json(session_dir / "history_a.json", [{"session_id": "s1", "records": ["r1"]}])

    def fetch_record(location: RecordLocation) -> dict:
        return {"id": location.record_id, "belongs": location.session_id}

    def converter(raw_record: dict) -> dict:
        raise ValueError("synthetic convert failure")

    summary = build_github_chaga_corpus(
        history_path=history_path,
        session_dir=session_dir,
        elo_csv=elo_path,
        raw_out=out_raw,
        prepared_out=out_prepared,
        summary_out=summary_out,
        min_elo=2300,
        fetch_record=fetch_record,
        converter=converter,
    )

    assert summary["records_fetched"] == 1
    assert summary["records_converted"] == 0
    assert summary["records_prepared"] == 0
    assert summary["audit_raw_records_written"] == 0
    assert out_raw.read_text(encoding="utf-8") == ""
    assert out_prepared.read_text(encoding="utf-8") == ""
