import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_transformer_chaga_review import select_reviewed_examples  # noqa: E402
from evaluate_transformer_chaga_review import (  # noqa: E402
    aggregate_original_candidate_metrics,
    load_evaluation_examples,
    resolve_eval_max_candidates,
    score_original_chaga_match,
)


def test_select_reviewed_examples_keeps_original_candidate_metadata_without_distribution():
    examples = [
        SimpleNamespace(teacher_action_distribution=[1.0, 0.0], teacher_candidate_norms=()),
        SimpleNamespace(teacher_action_distribution=None, teacher_candidate_norms=("PLAY W1", "PLAY W2")),
        SimpleNamespace(teacher_action_distribution=None, teacher_candidate_norms=()),
    ]

    assert select_reviewed_examples(examples) == [examples[1]]


def test_score_original_chaga_match_relaxes_only_with_accept_flag():
    accepted = score_original_chaga_match(
        "PLAY W3",
        teacher_top1_norm="PLAY W1",
        teacher_top3_norms=("PLAY W1", "PLAY W2", "PLAY W3"),
        accept_top3=True,
    )
    rejected = score_original_chaga_match(
        "PLAY W3",
        teacher_top1_norm="PLAY W1",
        teacher_top3_norms=("PLAY W1", "PLAY W2", "PLAY W3"),
        accept_top3=False,
    )

    assert accepted["top3_match"]
    assert accepted["relaxed_match"]
    assert rejected["top3_match"]
    assert not rejected["relaxed_match"]


def test_score_original_chaga_match_requires_top1_outside_relaxed_region():
    result = score_original_chaga_match(
        "PENG",
        teacher_top1_norm="PASS",
        teacher_top3_norms=("PASS", "PENG", "CHI W3"),
        accept_top3=False,
    )

    assert result["top3_match"]
    assert not result["relaxed_match"]


def test_resolve_eval_max_candidates_uses_cli_width_not_checkpoint_width():
    assert resolve_eval_max_candidates({"max_candidates": 96}, 235) == 235


def test_aggregate_original_candidate_metrics_uses_original_rows_not_distribution_proxy():
    rows = [
        {
            "top1_match": False,
            "top3_match": True,
            "relaxed_match": True,
            "chaga_top1_action": "PLAY W1",
            "has_teacher_distribution": False,
        },
        {
            "top1_match": True,
            "top3_match": True,
            "relaxed_match": True,
            "chaga_top1_action": "PENG",
            "has_teacher_distribution": True,
        },
    ]

    metrics = aggregate_original_candidate_metrics(rows)

    assert metrics["original_samples"] == 2
    assert metrics["original_relaxed_accuracy"] == 1.0
    assert metrics["original_top1_accuracy"] == 0.5
    assert metrics["original_play_samples"] == 1
    assert metrics["reviewed_without_teacher_distribution"] == 1


def test_load_evaluation_examples_can_disable_rule_feature_materialization(monkeypatch):
    captured = {}

    def fake_load_examples(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return [], {"examples": 0}

    monkeypatch.setattr("evaluate_transformer_chaga_review.load_review_target_lookup", lambda path: "lookup")
    monkeypatch.setattr("evaluate_transformer_chaga_review.load_examples", fake_load_examples)
    args = SimpleNamespace(
        raw=["eval.jsonl"],
        review_audit_jsonl="audit.jsonl",
        history_len=80,
        max_records_per_source=None,
        max_examples=None,
        teacher_temperature=1.0,
        no_rule_features=True,
    )

    examples, summary = load_evaluation_examples(args, {"history_len": 12})

    assert examples == []
    assert summary == {"examples": 0}
    assert captured["history_len"] == 12
    assert captured["teacher_lookup"] == "lookup"
    assert captured["compute_rule_features"] is False
