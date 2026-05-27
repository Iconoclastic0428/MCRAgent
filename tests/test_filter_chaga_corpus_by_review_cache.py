import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_chaga_corpus_by_review_cache import filter_by_review_cache  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_filter_by_review_cache_keeps_only_cached_train_player_seats(tmp_path):
    raw = tmp_path / "raw.jsonl"
    prepared = tmp_path / "prepared.jsonl"
    cache = tmp_path / "cache"
    raw_out = tmp_path / "filtered.raw.jsonl"
    prepared_out = tmp_path / "filtered.prepared.jsonl"
    summary_out = tmp_path / "summary.json"
    cache.mkdir()
    (cache / "s1_seat1.json").write_text("[]", encoding="utf-8")
    (cache / "s1_seat2.json").write_text(json.dumps({"code": 404, "message": "missing"}), encoding="utf-8")
    write_jsonl(
        raw,
        [
            {
                "id": "r0",
                "belongs": "s1",
                "step": {
                    "i": 0,
                    "p": [{"n": "Human"}, {"n": "CHAGA02"}, {"n": "CHAGA03"}, {"n": "CHAGA04"}],
                },
                "train_players": ["1", "2"],
                "train_player_names": {"1": "CHAGA02", "2": "CHAGA03"},
            },
            {
                "id": "r1",
                "belongs": "s1",
                "step": {
                    "i": 1,
                    "p": [{"n": "CHAGA03"}, {"n": "Human"}, {"n": "CHAGA02"}, {"n": "CHAGA04"}],
                },
                "train_players": ["0", "2"],
                "train_player_names": {"0": "CHAGA03", "2": "CHAGA02"},
            },
        ],
    )
    write_jsonl(
        prepared,
        [
            {"source_record_id": "r0", "train_players": ["1", "2"], "train_player_names": {"1": "CHAGA02", "2": "CHAGA03"}},
            {"source_record_id": "r1", "train_players": ["0", "2"], "train_player_names": {"0": "CHAGA03", "2": "CHAGA02"}},
        ],
    )

    summary = filter_by_review_cache(
        raw_path=raw,
        prepared_path=prepared,
        cache_dir=cache,
        raw_out=raw_out,
        prepared_out=prepared_out,
        summary_out=summary_out,
        player_pattern=re.compile(r"^CHAGA0[2-8]$"),
    )

    assert summary["records_written"] == 2
    assert summary["train_player_slots"] == 2
    assert summary["invalid_cached_review"] == 2
    assert [row["train_players"] for row in read_jsonl(raw_out)] == [["1"], ["2"]]
    assert [row["train_player_names"] for row in read_jsonl(prepared_out)] == [{"1": "CHAGA02"}, {"2": "CHAGA02"}]
