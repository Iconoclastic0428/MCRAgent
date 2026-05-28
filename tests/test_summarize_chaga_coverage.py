import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_chaga_coverage import summarize_coverage_rows  # noqa: E402


def test_summarize_coverage_rows_flags_sparse_train_play_slice():
    train_rows = [
        {"teacher_family": "PLAY", "turn": 7, "candidate_count": 30, "relaxed_region": "first_six_play"},
        {"teacher_family": "CHI", "turn": 18, "candidate_count": 3, "relaxed_region": "other"},
    ]
    val_rows = [
        {"teacher_family": "PLAY", "turn": 8, "candidate_count": 32, "relaxed_region": "first_six_play"},
        {"teacher_family": "PLAY", "turn": 9, "candidate_count": 34, "relaxed_region": "first_six_play"},
    ]

    summary = summarize_coverage_rows(train_rows, val_rows, min_train_to_val_ratio=5)

    assert summary["train_reviewed_states"] == 2
    assert summary["val_reviewed_states"] == 2
    assert summary["gates"]["overall_train_to_val_ratio"]["passed"] is False
    assert summary["gates"]["slice_train_to_val_ratio"]["passed"] is False
    assert "family=PLAY" in summary["gates"]["slice_train_to_val_ratio"]["worst_slice"]
