import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from split_chaga_review_corpus import split_review_corpus  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_split_chaga_review_corpus_is_session_disjoint(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    out_dir = tmp_path / "split"
    raw_rows = [
        {"id": "r1", "belongs": "s1"},
        {"id": "r2", "belongs": "s1"},
        {"id": "r3", "belongs": "s2"},
        {"id": "r4", "belongs": "s3"},
    ]
    audit_rows = [
        {"record_id": "r1", "session_id": "s1"},
        {"record_id": "r2", "session_id": "s1"},
        {"record_id": "r3", "session_id": "s2"},
        {"record_id": "r4", "session_id": "s3"},
    ]
    write_jsonl(raw_path, raw_rows)
    write_jsonl(audit_path, audit_rows)

    summary = split_review_corpus(
        raw_path=raw_path,
        audit_path=audit_path,
        out_dir=out_dir,
        seed=7,
        train_ratio=0.34,
        val_ratio=0.33,
    )

    split_sessions = {
        split: set(summary["splits"][split]["sessions"])
        for split in ("train", "val", "test")
    }
    assert split_sessions["train"].isdisjoint(split_sessions["val"])
    assert split_sessions["train"].isdisjoint(split_sessions["test"])
    assert split_sessions["val"].isdisjoint(split_sessions["test"])
    assert sum("s1" in sessions for sessions in split_sessions.values()) == 1


def test_split_chaga_review_corpus_keeps_raw_and_audit_consistent(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    out_dir = tmp_path / "split"
    write_jsonl(
        raw_path,
        [
            {"id": "r1", "belongs": "s1"},
            {"id": "r2", "belongs": "s2"},
            {"id": "unused", "belongs": "s_without_review"},
        ],
    )
    write_jsonl(
        audit_path,
        [
            {"record_id": "r1", "session_id": "s1"},
            {"record_id": "r2", "session_id": "s2"},
        ],
    )

    summary = split_review_corpus(
        raw_path=raw_path,
        audit_path=audit_path,
        out_dir=out_dir,
        seed=1,
        train_ratio=0.5,
        val_ratio=0.0,
    )

    for split in ("train", "val", "test"):
        sessions = set(summary["splits"][split]["sessions"])
        raw_sessions = {row["belongs"] for row in read_jsonl(out_dir / f"{split}.raw.jsonl")}
        audit_sessions = {row["session_id"] for row in read_jsonl(out_dir / f"{split}.audit.jsonl")}
        assert raw_sessions <= sessions
        assert audit_sessions <= sessions
    assert summary["dropped_raw_records_without_review"] == 1


def test_split_chaga_review_corpus_splits_prepared_records_when_provided(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    prepared_path = tmp_path / "prepared.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    out_dir = tmp_path / "split"
    write_jsonl(
        raw_path,
        [
            {"id": "r1", "belongs": "s1"},
            {"id": "r2", "belongs": "s2"},
            {"id": "unused", "belongs": "s_without_review"},
        ],
    )
    write_jsonl(
        prepared_path,
        [
            {"source_record_id": "r1", "belongs": "s1", "logs": [{"keep": "r1"}]},
            {"source_record_id": "r2", "belongs": "s2", "logs": [{"keep": "r2"}]},
            {"source_record_id": "unused", "belongs": "s_without_review", "logs": [{"keep": "unused"}]},
        ],
    )
    write_jsonl(
        audit_path,
        [
            {"record_id": "r1", "session_id": "s1"},
            {"record_id": "r2", "session_id": "s2"},
        ],
    )

    summary = split_review_corpus(
        raw_path=raw_path,
        prepared_path=prepared_path,
        audit_path=audit_path,
        out_dir=out_dir,
        seed=1,
        train_ratio=0.5,
        val_ratio=0.0,
    )

    written_prepared = []
    for split in ("train", "val", "test"):
        rows = read_jsonl(out_dir / f"{split}.prepared.jsonl")
        written_prepared.extend(rows)
        assert len(rows) == summary["splits"][split]["prepared_records"]
        assert {row["belongs"] for row in rows} <= set(summary["splits"][split]["sessions"])
    assert {row["source_record_id"] for row in written_prepared} == {"r1", "r2"}
    assert all(row.get("logs") for row in written_prepared)
    assert summary["prepared_records"] == 3
    assert summary["dropped_prepared_records_without_review"] == 1


def test_split_chaga_review_corpus_is_deterministic(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    write_jsonl(raw_path, [{"id": f"r{i}", "belongs": f"s{i}"} for i in range(10)])
    write_jsonl(audit_path, [{"record_id": f"r{i}", "session_id": f"s{i}"} for i in range(10)])

    first = split_review_corpus(
        raw_path=raw_path,
        audit_path=audit_path,
        out_dir=tmp_path / "first",
        seed=20260527,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    second = split_review_corpus(
        raw_path=raw_path,
        audit_path=audit_path,
        out_dir=tmp_path / "second",
        seed=20260527,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    assert first["splits"] == second["splits"]


def test_split_chaga_review_corpus_preserves_small_positive_test_split(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    write_jsonl(raw_path, [{"id": f"r{i}", "belongs": f"s{i}"} for i in range(6)])
    write_jsonl(audit_path, [{"record_id": f"r{i}", "session_id": f"s{i}"} for i in range(6)])

    summary = split_review_corpus(
        raw_path=raw_path,
        audit_path=audit_path,
        out_dir=tmp_path / "split",
        seed=20260527,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    assert summary["splits"]["train"]["session_count"] == 4
    assert summary["splits"]["val"]["session_count"] == 1
    assert summary["splits"]["test"]["session_count"] == 1
