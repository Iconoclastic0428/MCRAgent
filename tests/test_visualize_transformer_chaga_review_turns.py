import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from visualize_transformer_chaga_review_turns import aggregate_by_turn, render_svg  # noqa: E402


def test_aggregate_by_turn_counts_top1_top3_and_relaxed_mismatches():
    rows = [
        {"turn": 1, "top1_match": True, "top3_match": True, "relaxed_match": True},
        {"turn": 1, "top1_match": False, "top3_match": True, "relaxed_match": True},
        {"turn": 2, "top1_match": False, "top3_match": True, "relaxed_match": False},
        {"turn": 2, "top1_match": False, "top3_match": False, "relaxed_match": False},
    ]

    stats = aggregate_by_turn(rows)

    assert stats[1]["total"] == 2
    assert stats[1]["top1_mismatch"] == 1
    assert stats[1]["relaxed_mismatch"] == 0
    assert stats[2]["top3_match"] == 1
    assert stats[2]["relaxed_mismatch_rate"] == 1.0


def test_render_svg_contains_turn_labels_and_title():
    stats = {
        1: {"total": 2, "relaxed_mismatch": 1, "relaxed_mismatch_rate": 0.5},
        2: {"total": 4, "relaxed_mismatch": 3, "relaxed_mismatch_rate": 0.75},
    }

    svg = render_svg(stats, title="Model vs CHAGA by turn")

    assert svg.startswith("<svg")
    assert "Model vs CHAGA by turn" in svg
    assert "turn 1" in svg
    assert "turn 2" in svg
