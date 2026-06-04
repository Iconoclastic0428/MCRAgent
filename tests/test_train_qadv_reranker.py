import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_qadv_reranker import evaluate  # noqa: E402
from build_qadv_terminal_trajectories import build_qadv_terminal_row  # noqa: E402
from mine_chaga_hard_examples import build_qadv_hard_example, qadv_candidate_digest  # noqa: E402
from train_qadv_reranker import collate_qadv_rows, load_qadv_rows, train  # noqa: E402


def _qadv_row(example_key: str, *, accepted_top1: bool) -> dict:
    return build_qadv_hard_example(
        {
            "example_key": example_key,
            "record_id": "rec",
            "session_id": "sess",
            "turn": 2,
            "player": 0,
            "request": "2 W1",
            "response": "PLAY W1",
            "candidate_actions": [10, 11, 12],
            "candidate_norms": ["PLAY W1", "PLAY W2", "PASS"],
            "candidate_rule_features": [[0.0] * 7 for _ in range(3)],
            "base_logits": [2.0, 0.1, 0.0] if accepted_top1 else [0.1, 2.0, 0.0],
            "base_ranks": [0.0, 1.0, 2.0] if accepted_top1 else [1.0, 0.0, 2.0],
            "scalar_features": [0.0, 3.0 / 235.0, 0.0],
            "teacher_candidate_norms": ["PLAY W1", "PLAY W2"],
            "teacher_target_dist": [0.9, 0.1, 0.0],
            "teacher_accept_top3": False,
            "allow_hu": False,
            "has_teacher_distribution": True,
        }
    )


def _terminal_row(example_key: str, *, discounted_return: float) -> dict:
    row = build_qadv_terminal_row(
        {
            "example_key": example_key,
            "record_id": "rec",
            "session_id": "sess",
            "turn": 2,
            "player": 0,
            "request": "2 W1",
            "response": "PLAY W1",
            "candidate_actions": [10, 11, 12],
            "candidate_norms": ["PLAY W1", "PLAY W2", "PASS"],
            "candidate_rule_features": [[0.0] * 7 for _ in range(3)],
            "base_logits": [0.4, 1.2, 0.0],
            "base_ranks": [1.0, 0.0, 2.0],
            "scalar_features": [0.0, 3.0 / 235.0, 0.0],
            "accepted_norms": ["PLAY W1"],
            "accepted_action_ids": [10],
            "teacher_candidate_norms": ["PLAY W1", "PLAY W2"],
            "teacher_target_dist": [0.9, 0.1, 0.0],
            "teacher_accept_top3": False,
            "allow_hu": False,
            "has_teacher_distribution": True,
        },
        {
            "result_type": "HU",
            "winner": 0 if discounted_return > 0 else 1,
            "discarder": None if discounted_return > 0 else 0,
            "is_self_draw": discounted_return > 0,
            "final_turn": 20,
            "hu_turn": 20,
            "player_won": discounted_return > 0,
            "player_dealt_in": discounted_return < 0,
            "score_delta": 16 if discounted_return > 0 else -16,
            "placement_delta": 1 if discounted_return > 0 else -1,
            "end_wait": False,
            "wait_when_deal_in": discounted_return < 0,
            "fan": 16,
            "base_fan": 16,
            "low_fan_hu": False,
        },
    )
    row["return"]["discounted_return"] = float(discounted_return)
    row["discounted_return"] = float(discounted_return)
    return row


def test_train_and_evaluate_qadv_reranker_smoke(tmp_path):
    hard_path = tmp_path / "hard.jsonl"
    rows = [
        _qadv_row("a", accepted_top1=True),
        _qadv_row("b", accepted_top1=False),
        _qadv_row("c", accepted_top1=True),
        _qadv_row("d", accepted_top1=False),
    ]
    hard_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    model_path = tmp_path / "qadv.pt"
    metrics_path = tmp_path / "metrics.json"

    train_metrics = train(
        argparse.Namespace(
            hard_jsonl=[str(hard_path)],
            eval_hard_jsonl=[],
            model_out=str(model_path),
            best_model_out=None,
            metrics_out=str(metrics_path),
            monitor_metric="val_accepted_accuracy",
            min_delta=0.0,
            epochs=1,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            dropout=0.0,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            val_ratio=0.5,
            seed=7,
            max_rows=None,
            weighted_sampler=False,
            train_lambda=1.0,
            eval_lambda=0.1,
            margin=0.5,
            accept_weight=1.0,
            pair_weight=0.5,
            cql_weight=0.03,
            soft_weight=0.05,
            q_l2_weight=0.0001,
            device="cpu",
        )
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert train_metrics["format"] == "mcr_qadv_reranker_train_v1"

    eval_metrics = evaluate(
        argparse.Namespace(
            hard_jsonl=[str(hard_path)],
            q_checkpoint=str(model_path),
            metrics_out=None,
            batch_size=2,
            max_rows=None,
            lambdas="0.00,0.10",
            device="cpu",
        )
    )

    assert eval_metrics["format"] == "mcr_qadv_reranker_eval_v1"
    assert eval_metrics["lambda_metrics"]["0.00"]["lambda_zero_reproduction_mismatches"] == 0


def test_train_qadv_uses_external_eval_hard_jsonl_not_random_split(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_rows = [_qadv_row(f"train-{index}", accepted_top1=index % 2 == 0) for index in range(4)]
    eval_rows = [_qadv_row(f"eval-{index}", accepted_top1=True) for index in range(3)]
    train_path.write_text("\n".join(json.dumps(row) for row in train_rows), encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(row) for row in eval_rows), encoding="utf-8")

    metrics = train(
        argparse.Namespace(
            hard_jsonl=[str(train_path)],
            eval_hard_jsonl=[str(eval_path)],
            model_out=str(tmp_path / "final.pt"),
            best_model_out=None,
            metrics_out=None,
            monitor_metric="val_accepted_accuracy",
            min_delta=0.0,
            epochs=1,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            dropout=0.0,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            val_ratio=0.5,
            seed=7,
            max_rows=None,
            weighted_sampler=False,
            train_lambda=1.0,
            eval_lambda=0.1,
            margin=0.5,
            accept_weight=1.0,
            pair_weight=0.5,
            cql_weight=0.03,
            soft_weight=0.05,
            q_l2_weight=0.0001,
            device="cpu",
        )
    )

    assert metrics["train_rows"] == 4
    assert metrics["val_rows"] == 3
    assert metrics["eval_hard_jsonl"] == [str(eval_path)]


def test_train_qadv_saves_best_checkpoint_from_external_eval(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_rows = [_qadv_row(f"train-{index}", accepted_top1=index % 2 == 0) for index in range(4)]
    eval_rows = [_qadv_row(f"eval-{index}", accepted_top1=True) for index in range(2)]
    train_path.write_text("\n".join(json.dumps(row) for row in train_rows), encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(row) for row in eval_rows), encoding="utf-8")
    best_path = tmp_path / "best.pt"

    metrics = train(
        argparse.Namespace(
            hard_jsonl=[str(train_path)],
            eval_hard_jsonl=[str(eval_path)],
            model_out=str(tmp_path / "final.pt"),
            best_model_out=str(best_path),
            metrics_out=None,
            monitor_metric="val_accepted_accuracy",
            min_delta=0.0,
            epochs=1,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            dropout=0.0,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            val_ratio=0.5,
            seed=7,
            max_rows=None,
            weighted_sampler=False,
            train_lambda=1.0,
            eval_lambda=0.1,
            margin=0.5,
            accept_weight=1.0,
            pair_weight=0.5,
            cql_weight=0.03,
            soft_weight=0.05,
            q_l2_weight=0.0001,
            device="cpu",
        )
    )

    assert best_path.exists()
    assert metrics["best_epoch"] == 1
    assert metrics["best_model_out"] == str(best_path)


def test_train_qadv_val_ratio_still_works_without_eval_hard_jsonl(tmp_path):
    hard_path = tmp_path / "hard.jsonl"
    rows = [_qadv_row(f"row-{index}", accepted_top1=index % 2 == 0) for index in range(6)]
    hard_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    metrics = train(
        argparse.Namespace(
            hard_jsonl=[str(hard_path)],
            eval_hard_jsonl=[],
            model_out=str(tmp_path / "final.pt"),
            best_model_out=None,
            metrics_out=None,
            monitor_metric="val_accepted_accuracy",
            min_delta=0.0,
            epochs=1,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            dropout=0.0,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            val_ratio=0.5,
            seed=7,
            max_rows=None,
            weighted_sampler=False,
            train_lambda=1.0,
            eval_lambda=0.1,
            margin=0.5,
            accept_weight=1.0,
            pair_weight=0.5,
            cql_weight=0.03,
            soft_weight=0.05,
            q_l2_weight=0.0001,
            device="cpu",
        )
    )

    assert metrics["train_rows"] == 3
    assert metrics["val_rows"] == 3
    assert metrics["eval_hard_jsonl"] == []


def test_train_qadv_rejects_low_fan_hu_rows_before_training(tmp_path):
    row = _qadv_row("bad-hu", accepted_top1=True)
    row["candidate_actions"].append(1)
    row["candidate_norms"].append("HU")
    row["candidate_rule_features"].append([0.0] * 7)
    row["base_logits"].append(10.0)
    row["base_ranks"].append(0.0)
    row["teacher_target_dist"].append(0.0)
    row["allow_hu"] = False
    row["candidate_digest"] = qadv_candidate_digest(row["candidate_actions"], row["candidate_norms"])
    hard_path = tmp_path / "bad.jsonl"
    hard_path.write_text(json.dumps(row), encoding="utf-8")

    try:
        load_qadv_rows([hard_path], strict=True)
    except ValueError as exc:
        assert "low-fan Hu" in str(exc)
    else:
        raise AssertionError("low-fan Hu row should be rejected")


def test_load_qadv_rows_accepts_terminal_return_rows(tmp_path):
    terminal_path = tmp_path / "terminal.jsonl"
    rows = [_terminal_row("terminal-pos", discounted_return=0.4)]
    terminal_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    loaded = load_qadv_rows([terminal_path], strict=True)
    batch = collate_qadv_rows(loaded)

    assert loaded[0]["schema_version"] == "qadv_terminal_v1"
    assert batch["has_return"].tolist() == [True]
    assert batch["chosen_action_index"].tolist() == [0]
    assert batch["return_target"].tolist() == pytest.approx([0.4])


def test_collate_qadv_rows_can_choose_terminal_return_target():
    row = _terminal_row("terminal-pos", discounted_return=0.4)
    row["return"]["terminal_return"] = 0.7
    row["terminal_return"] = 0.7

    batch = collate_qadv_rows([row], value_target_key="terminal_return")

    assert batch["has_return"].tolist() == [True]
    assert batch["return_target"].tolist() == pytest.approx([0.7])


def test_train_qadv_mixes_hard_and_terminal_rows(tmp_path):
    hard_path = tmp_path / "hard.jsonl"
    terminal_path = tmp_path / "terminal.jsonl"
    hard_rows = [_qadv_row(f"hard-{index}", accepted_top1=index % 2 == 0) for index in range(4)]
    terminal_rows = [
        _terminal_row("terminal-pos", discounted_return=0.4),
        _terminal_row("terminal-neg", discounted_return=-0.3),
    ]
    hard_path.write_text("\n".join(json.dumps(row) for row in hard_rows), encoding="utf-8")
    terminal_path.write_text("\n".join(json.dumps(row) for row in terminal_rows), encoding="utf-8")

    metrics = train(
        argparse.Namespace(
            hard_jsonl=[str(hard_path)],
            eval_hard_jsonl=[],
            terminal_jsonl=[str(terminal_path)],
            eval_terminal_jsonl=[str(terminal_path)],
            model_out=str(tmp_path / "final.pt"),
            best_model_out=None,
            metrics_out=None,
            monitor_metric="val_accepted_accuracy",
            min_delta=0.0,
            epochs=1,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            dropout=0.0,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            val_ratio=0.5,
            seed=7,
            max_rows=None,
            weighted_sampler=False,
            train_lambda=1.0,
            eval_lambda=0.1,
            margin=0.5,
            accept_weight=1.0,
            pair_weight=0.5,
            cql_weight=0.03,
            soft_weight=0.05,
            return_loss_weight=0.20,
            q_l2_weight=0.0001,
            device="cpu",
        )
    )

    assert metrics["hard_rows"] == 4
    assert metrics["terminal_rows"] == 2
    assert metrics["eval_terminal_rows"] == 2
    assert "train_return_loss" in metrics["history"][0]


def test_evaluate_qadv_reports_terminal_return_metrics(tmp_path):
    hard_path = tmp_path / "hard.jsonl"
    terminal_path = tmp_path / "terminal.jsonl"
    hard_rows = [_qadv_row(f"hard-{index}", accepted_top1=index % 2 == 0) for index in range(4)]
    terminal_rows = [
        _terminal_row("terminal-pos", discounted_return=0.4),
        _terminal_row("terminal-neg", discounted_return=-0.3),
    ]
    hard_path.write_text("\n".join(json.dumps(row) for row in hard_rows), encoding="utf-8")
    terminal_path.write_text("\n".join(json.dumps(row) for row in terminal_rows), encoding="utf-8")
    model_path = tmp_path / "final.pt"
    train(
        argparse.Namespace(
            hard_jsonl=[str(hard_path)],
            eval_hard_jsonl=[],
            terminal_jsonl=[str(terminal_path)],
            eval_terminal_jsonl=[],
            model_out=str(model_path),
            best_model_out=None,
            metrics_out=None,
            monitor_metric="val_accepted_accuracy",
            min_delta=0.0,
            epochs=1,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            dropout=0.0,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            val_ratio=0.5,
            seed=7,
            max_rows=None,
            weighted_sampler=False,
            train_lambda=1.0,
            eval_lambda=0.1,
            margin=0.5,
            accept_weight=1.0,
            pair_weight=0.5,
            cql_weight=0.03,
            soft_weight=0.05,
            return_loss_weight=0.20,
            q_l2_weight=0.0001,
            device="cpu",
        )
    )

    metrics = evaluate(
        argparse.Namespace(
            hard_jsonl=[str(hard_path)],
            terminal_jsonl=[str(terminal_path)],
            q_checkpoint=str(model_path),
            metrics_out=None,
            batch_size=2,
            max_rows=None,
            lambdas="0.00,0.10",
            device="cpu",
        )
    )

    assert metrics["terminal_rows"] == 2
    assert metrics["terminal_q"]["return_rows"] == 2
    assert metrics["terminal_q"]["positive_return_rows"] == 1
    assert metrics["terminal_q"]["negative_return_rows"] == 1
