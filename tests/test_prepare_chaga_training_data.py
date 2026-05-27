import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_chaga_training_data as prep


def test_prepare_converted_record_marks_only_requested_chaga_players():
    raw_record = {
        "id": "r1",
        "step": {
            "p": [
                {"n": "CHAGA01"},
                {"n": "CHAGA02"},
                {"n": "Human"},
                {"n": "CHAGA08"},
            ]
        },
    }
    converted = {"match_id": "m1", "logs": [], "scores": {}}

    prepared = prep.prepare_converted_record(raw_record, converted)

    assert prepared["train_players"] == ["1", "3"]
    assert prepared["train_player_names"] == {"1": "CHAGA02", "3": "CHAGA08"}
    assert prepared["source_players"] == ["CHAGA01", "CHAGA02", "Human", "CHAGA08"]
    assert prepared["source_record_id"] == "r1"


def test_prepare_converted_record_can_filter_by_elo_threshold():
    raw_record = {
        "id": "r1",
        "step": {
            "p": [
                {"n": "CHAGA02"},
                {"n": "LowRank"},
                {"n": "HumanPro"},
                {"n": "Unknown"},
            ]
        },
    }
    converted = {"match_id": "m1", "logs": [], "scores": {}}

    prepared = prep.prepare_converted_record(
        raw_record,
        converted,
        player_scores={"CHAGA02": 2806.0, "LowRank": 2200.0, "HumanPro": 2401.0},
        min_score=2300.0,
        player_pattern=r".*",
    )

    assert prepared["train_players"] == ["0", "2"]
    assert prepared["train_player_names"] == {"0": "CHAGA02", "2": "HumanPro"}


def test_load_player_scores_reads_tziakcha_elo_csv(tmp_path):
    path = tmp_path / "elo.csv"
    path.write_text(
        'Rank,Value,Player Name,Rounds,Player ID\n'
        '1,2840.0,"CHAGA07",34011,"WySi0ZjP"\n'
        '2,2299.0,"Low",10,"x"\n',
        encoding="utf-8",
    )

    scores = prep.load_player_scores(path)

    assert scores == {"CHAGA07": 2840.0, "Low": 2299.0}


def test_build_split_writes_train_eval_and_summary(tmp_path):
    raw = tmp_path / "raw.jsonl"
    records = [
        {"id": "r1", "step": {"p": [{"n": "CHAGA02"}]}},
        {"id": "r2", "step": {"p": [{"n": "Human"}]}},
        {"id": "r3", "step": {"p": [{"n": "CHAGA03"}]}},
        {"id": "r4", "step": {"p": [{"n": "CHAGA04"}]}},
    ]
    raw.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    def fake_converter(record):
        return {"match_id": record["id"], "logs": [], "scores": {}}

    summary = prep.build_split(
        raw,
        tmp_path,
        prefix="unit",
        eval_fraction=0.34,
        seed=7,
        converter=fake_converter,
    )

    train_lines = (tmp_path / "unit_train.jsonl").read_text(encoding="utf-8").splitlines()
    eval_lines = (tmp_path / "unit_eval.jsonl").read_text(encoding="utf-8").splitlines()
    all_lines = (tmp_path / "unit_all.jsonl").read_text(encoding="utf-8").splitlines()
    saved_summary = json.loads((tmp_path / "unit_summary.json").read_text(encoding="utf-8"))

    assert summary["records_seen"] == 4
    assert summary["records_with_selected_players"] == 3
    assert summary["train_records"] == 1
    assert summary["eval_records"] == 2
    assert len(train_lines) == 1
    assert len(eval_lines) == 2
    assert len(all_lines) == 3
    assert saved_summary["selected_player_record_counts"] == {
        "CHAGA02": 1,
        "CHAGA03": 1,
        "CHAGA04": 1,
    }
