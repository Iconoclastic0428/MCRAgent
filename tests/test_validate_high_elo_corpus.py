import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_high_elo_corpus import validate_high_elo_corpus  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_validate_high_elo_corpus_accepts_only_record_local_high_elo_seats(tmp_path):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(
        raw,
        [
            {
                "id": "r1",
                "step": {
                    "p": [
                        {"n": "Strong", "e": 2401},
                        {"n": "Weak", "e": 2299},
                        {"n": "AlsoStrong", "e": 2600},
                    ]
                },
                "train_players": ["0", "2"],
            }
        ],
    )

    summary = validate_high_elo_corpus(raw, min_elo=2300)

    assert summary["records"] == 1
    assert summary["train_player_slots"] == 2
    assert summary["train_player_mismatches"] == 0
    assert summary["below_threshold_train_player_slots"] == 0


def test_validate_high_elo_corpus_rejects_below_threshold_train_seat(tmp_path):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(
        raw,
        [
            {
                "id": "r1",
                "step": {"p": [{"n": "Strong", "e": 2401}, {"n": "Weak", "e": 2299}]},
                "train_players": ["0", "1"],
            }
        ],
    )

    summary = validate_high_elo_corpus(raw, min_elo=2300)

    assert summary["train_player_mismatches"] == 1
    assert summary["below_threshold_train_player_slots"] == 1


def test_validate_high_elo_corpus_checks_prepared_train_players(tmp_path):
    raw = tmp_path / "raw.jsonl"
    prepared = tmp_path / "prepared.jsonl"
    write_jsonl(
        raw,
        [
            {
                "id": "r1",
                "step": {"p": [{"n": "Strong", "e": 2401}, {"n": "Weak", "e": 2299}]},
                "train_players": ["0"],
                "train_player_names": {"0": "Strong"},
            }
        ],
    )
    write_jsonl(prepared, [{"source_record_id": "r1", "train_players": ["0"], "train_player_names": {"0": "Strong"}}])

    summary = validate_high_elo_corpus(raw, min_elo=2300, prepared_path=prepared)

    assert summary["prepared_records"] == 1
    assert summary["prepared_train_player_mismatches"] == 0
    assert summary["prepared_source_record_mismatches"] == 0


def test_validate_high_elo_corpus_rejects_prepared_low_elo_leak(tmp_path):
    raw = tmp_path / "raw.jsonl"
    prepared = tmp_path / "prepared.jsonl"
    write_jsonl(
        raw,
        [
            {
                "id": "r1",
                "step": {"p": [{"n": "Strong", "e": 2401}, {"n": "Weak", "e": 2299}]},
                "train_players": ["0"],
            }
        ],
    )
    write_jsonl(prepared, [{"source_record_id": "r1", "train_players": ["0", "1"]}])

    summary = validate_high_elo_corpus(raw, min_elo=2300, prepared_path=prepared)

    assert summary["prepared_train_player_mismatches"] == 1
    assert summary["failures_preview"][0]["prepared_actual"] == ["0", "1"]
