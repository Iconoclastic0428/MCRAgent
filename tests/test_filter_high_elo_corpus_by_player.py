import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_high_elo_corpus_by_player import filter_corpus_by_player, selected_record_players  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_selected_record_players_filters_by_name_and_strict_elo():
    step = {
        "p": [
            {"n": "CHAGA01", "e": 2600},
            {"n": "CHAGA02", "e": 2301},
            {"n": "CHAGA06", "e": 2300},
            {"n": "Human", "e": 2700},
        ]
    }

    selected = selected_record_players(step, player_pattern=re.compile(r"^CHAGA0[2-8]$"), min_elo=2300)

    assert selected == {"1": "CHAGA02"}


def test_filter_corpus_by_player_rewrites_train_players_without_changing_context(tmp_path):
    raw = tmp_path / "raw.jsonl"
    prepared = tmp_path / "prepared.jsonl"
    raw_out = tmp_path / "out.raw.jsonl"
    prepared_out = tmp_path / "out.prepared.jsonl"
    summary_out = tmp_path / "summary.json"
    raw_record = {
        "id": "r1",
        "belongs": "s1",
        "step": {
            "p": [
                {"n": "CHAGA01", "e": 2500},
                {"n": "CHAGA02", "e": 2401},
                {"n": "CHAGA07", "e": 2299},
                {"n": "HumanPro", "e": 2600},
            ]
        },
        "logs": [{"keep": "full-context"}],
        "train_players": ["0", "1", "3"],
        "train_player_names": {"0": "CHAGA01", "1": "CHAGA02", "3": "HumanPro"},
    }
    prepared_record = {
        "source_record_id": "r1",
        "logs": [{"keep": "converted-context"}],
        "train_players": ["0", "1", "3"],
        "train_player_names": {"0": "CHAGA01", "1": "CHAGA02", "3": "HumanPro"},
    }
    write_jsonl(raw, [raw_record, {**raw_record, "id": "r2", "step": {"p": [{"n": "HumanPro", "e": 2600}]}}])
    write_jsonl(prepared, [prepared_record, {**prepared_record, "source_record_id": "r2"}])

    summary = filter_corpus_by_player(
        raw_path=raw,
        prepared_path=prepared,
        raw_out=raw_out,
        prepared_out=prepared_out,
        summary_out=summary_out,
        player_pattern=re.compile(r"^CHAGA0[2-8]$"),
        min_elo=2300,
    )

    assert summary["records_seen"] == 2
    assert summary["records_written"] == 1
    assert summary["train_player_slots"] == 1
    raw_filtered = read_jsonl(raw_out)
    prepared_filtered = read_jsonl(prepared_out)
    assert raw_filtered[0]["logs"] == [{"keep": "full-context"}]
    assert prepared_filtered[0]["logs"] == [{"keep": "converted-context"}]
    assert raw_filtered[0]["train_players"] == ["1"]
    assert raw_filtered[0]["train_player_names"] == {"1": "CHAGA02"}
    assert prepared_filtered[0]["train_players"] == ["1"]
    assert prepared_filtered[0]["train_player_names"] == {"1": "CHAGA02"}
