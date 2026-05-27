import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_transformer_candidate import (  # noqa: E402
    ReviewTarget,
    ReviewTargetLookup,
    BOTZONE_ACTION_TYPES,
    TILE_IDS,
    TransformerExample,
    TransformerCandidateModel,
    action_response,
    build_transformer_examples_from_record,
    chaga_candidates_to_action_distribution,
    collate_transformer_examples,
    count_model_parameters,
    encode_history_event,
    filter_reviewed_examples,
    hu_gated_candidate_mask,
    is_better_checkpoint_metric,
    load_review_target_lookup,
    policy_loss_with_optional_teacher,
    reviewed_sampling_weights,
    rule_gated_hu_allowed,
    teacher_accepted_set_loss,
    validate_reviewed_training_args,
)


def all_pass(players=range(4)):
    return {str(player): {"response": "PASS", "raw": "PASS", "verdict": "OK"} for player in players}


def base_record(extra_logs):
    hands = {
        "0": "W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 B1",
        "1": "W1 W1 B1 B2 B3 B4 B5 B6 B7 T1 T2 T3 J1",
        "2": "W2 W3 W4 W5 W6 W7 W8 W9 T4 T5 T6 B8 B9",
        "3": "B1 B2 B3 B4 B5 B6 B7 B8 B9 F1 F2 F3 J2",
    }
    logs = [
        {"output": {"content": {str(player): f"0 {player} 0" for player in range(4)}, "display": {"action": "INIT"}}},
        all_pass(),
        {
            "output": {
                "content": {
                    player: f"1 0 0 0 0 {tiles}"
                    for player, tiles in hands.items()
                },
                "display": {"action": "DEAL"},
            }
        },
        all_pass(),
    ]
    logs.extend(extra_logs)
    return {
        "match_id": "unit",
        "game": "Chinese-Standard-Mahjong",
        "logs": logs,
        "scores": {"0": 16, "1": -8, "2": -4, "3": -4},
    }


def action_id_for_response(response: str) -> int:
    normalized = response.upper()
    for action in range(235):
        if action_response(action).upper() == normalized:
            return action
    raise AssertionError(f"response not found: {response}")


def minimal_transformer_example(
    candidate_responses: list[str],
    *,
    target_response: str,
    teacher_candidate_norms: tuple[str, ...],
    teacher_accept_top3: bool,
) -> TransformerExample:
    candidate_actions = [action_id_for_response(response) for response in candidate_responses]
    action_mask = np.zeros(235, dtype=np.int8)
    action_mask[candidate_actions] = 1
    target_action = action_id_for_response(target_response)
    if target_action not in candidate_actions:
        action_mask[target_action] = 1
    return TransformerExample(
        obs=np.zeros((71, 4, 9), dtype=np.float32),
        action_mask=action_mask,
        act=target_action,
        player=0,
        turn=1,
        response=target_response,
        history_tokens=np.zeros((1,), dtype=np.int64),
        value_target=0.0,
        allow_hu=False,
        candidate_rule_features=np.zeros((235, 7), dtype=np.float32),
        teacher_accept_top3=teacher_accept_top3,
        teacher_candidate_norms=teacher_candidate_norms,
    )


def accepted_actions_from_batch(batch: dict[str, torch.Tensor], row: int = 0) -> list[str]:
    accepted: list[str] = []
    for action, ok in zip(batch["candidate_actions"][row].tolist(), batch["teacher_accept_mask"][row].tolist()):
        if ok:
            accepted.append(action_response(int(action)).upper())
    return accepted


def test_build_transformer_examples_keeps_legal_candidates_and_train_player_filter():
    record = base_record(
        [
            {
                "output": {
                    "content": {"0": "2 B2", "1": "2 B4"},
                    "display": {"action": "DRAW", "player": 0, "tile": "B2"},
                }
            },
            {
                "0": {"response": "PLAY B2", "raw": "PLAY B2", "verdict": "OK"},
                "1": {"response": "PLAY B4", "raw": "PLAY B4", "verdict": "OK"},
            },
        ]
    )
    record["train_players"] = ["1"]

    examples, stats = build_transformer_examples_from_record(record, history_len=8)

    assert examples
    assert {example.player for example in examples} == {1}
    assert stats["filtered_train_player_examples"] >= 1
    first = examples[0]
    assert first.obs.shape == (71, 4, 9)
    assert first.action_mask.shape == (235,)
    assert first.candidate_rule_features.shape[0] == 235
    assert first.candidate_rule_features.shape[1] >= 6
    assert first.action_mask[first.act] == 1
    assert first.candidate_rule_features[first.act].sum() > 0.0
    assert first.history_tokens.shape == (8,)
    assert first.value_target < 0


def test_hu_gated_candidate_mask_fails_closed_for_unknown_fan():
    action_mask = np.zeros(5, dtype=np.int8)
    action_mask[[0, 1, 2]] = 1
    responses = {0: "Pass", 1: "Hu", 2: "Play B1"}

    closed = hu_gated_candidate_mask(action_mask, responses.__getitem__, allow_hu=False)
    opened = hu_gated_candidate_mask(action_mask, responses.__getitem__, allow_hu=True)

    assert closed.tolist() == [1, 0, 1, 0, 0]
    assert opened.tolist() == [1, 1, 1, 0, 0]


def test_collate_and_model_score_only_candidate_actions():
    record = base_record(
        [
            {
                "output": {
                    "content": {"0": "2 B2", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                    "display": {"action": "DRAW", "player": 0, "tile": "B2"},
                }
            },
            {
                "0": {"response": "PLAY B2", "raw": "PLAY B2", "verdict": "OK"},
                **all_pass(players=[1, 2, 3]),
            },
        ]
    )
    examples, _ = build_transformer_examples_from_record(record, history_len=6)
    batch = collate_transformer_examples(examples[:1], max_candidates=16)

    assert batch["candidate_actions"].shape == (1, 16)
    assert batch["candidate_mask"].shape == (1, 16)
    assert batch["candidate_rule_features"].shape[:2] == (1, 16)
    assert int(batch["target_index"][0]) >= 0
    assert batch["candidate_actions"][0, batch["target_index"][0]] == examples[0].act

    model = TransformerCandidateModel(
        act_size=235,
        history_vocab_size=512,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
    )
    logits, values = model(batch)

    assert logits.shape == (1, 16)
    assert values.shape == (1,)
    masked_positions = batch["candidate_mask"] == 0
    assert torch.isneginf(logits[masked_positions]).all()


def test_chaga_candidate_distribution_maps_soft_targets_to_legal_actions_by_type():
    action_mask = np.zeros(8, dtype=np.int8)
    action_mask[[0, 1, 2, 3, 4, 7]] = 1
    responses = {
        0: "Pass",
        1: "Play W1",
        2: "Play W2",
        3: "Peng",
        4: "Chi W2",
        7: "Hu",
    }
    candidates = [[10.0, "Play W2"], [8.0, "Peng J1"], [7.0, "Play W9"], [6.0, "Hu"]]

    dist = chaga_candidates_to_action_distribution(
        candidates,
        action_mask,
        responses.__getitem__,
        allow_hu=False,
        temperature=1.0,
    )

    assert dist is not None
    assert dist.shape == (8,)
    assert dist[2] > dist[3] > 0.0
    assert dist[7] == 0.0
    assert dist[1] == 0.0
    assert np.isclose(float(dist.sum()), 1.0)


def test_rule_gated_hu_allowed_uses_legal_mask_not_recorded_response():
    action_mask = np.zeros(8, dtype=np.int8)
    action_mask[[0, 7]] = 1
    responses = {0: "Pass", 7: "Hu"}

    assert rule_gated_hu_allowed(action_mask, responses.__getitem__) is True

    action_mask[7] = 0
    assert rule_gated_hu_allowed(action_mask, responses.__getitem__) is False


def test_review_target_lookup_uses_turn_to_disambiguate_repeated_states():
    lookup = ReviewTargetLookup(
        {
            ("record-a", "0", "2 J3", "PLAY J3", "1"): [ReviewTarget([[1.0, "Play W1"]])],
            ("record-a", "0", "2 J3", "PLAY J3", "2"): [ReviewTarget([[1.0, "Play W2"]])],
        }
    )

    second = lookup(record_id="record-a", player=0, request="2 J3", response="PLAY J3", turn=2)
    first = lookup(record_id="record-a", player=0, request="2 J3", response="PLAY J3", turn=1)

    assert second is not None and second.candidates[0][1] == "Play W2"
    assert first is not None and first.candidates[0][1] == "Play W1"


def test_collate_carries_teacher_distribution_over_candidate_slots():
    record = base_record(
        [
            {
                "output": {
                    "content": {"0": "2 B2", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                    "display": {"action": "DRAW", "player": 0, "tile": "B2"},
                }
            },
            {
                "0": {"response": "PLAY B2", "raw": "PLAY B2", "verdict": "OK"},
                **all_pass(players=[1, 2, 3]),
            },
        ]
    )

    def teacher_lookup(**kwargs):
        return ReviewTarget([[2.0, "Play B2"], [1.0, "Play B1"]], accept_top3=True)

    examples, stats = build_transformer_examples_from_record(record, history_len=6, teacher_lookup=teacher_lookup)
    batch = collate_transformer_examples(examples[:1], max_candidates=16)

    assert stats["teacher_targets"] == 1
    assert batch["has_teacher_target"].tolist() == [True]
    assert batch["teacher_accept_top3"].tolist() == [True]
    assert torch.isclose(batch["teacher_target_dist"][0].sum(), torch.tensor(1.0))
    target_slot = int(batch["target_index"][0])
    assert batch["teacher_target_dist"][0, target_slot] > 0.0


def test_collate_teacher_accept_mask_top1_only_for_non_relaxed_rows():
    example = minimal_transformer_example(
        ["Play W1", "Play W2", "Play W3"],
        target_response="Play W1",
        teacher_candidate_norms=("PLAY W1", "PLAY W2", "PLAY W3"),
        teacher_accept_top3=False,
    )

    batch = collate_transformer_examples([example], max_candidates=235)

    assert accepted_actions_from_batch(batch) == ["PLAY W1"]
    assert batch["has_teacher_accept_set"].tolist() == [True]


def test_collate_teacher_accept_mask_top3_for_first_six_play_rows():
    example = minimal_transformer_example(
        ["Play W1", "Play W2", "Play W3", "Play W4"],
        target_response="Play W1",
        teacher_candidate_norms=("PLAY W1", "PLAY W2", "PLAY W3", "PLAY W4"),
        teacher_accept_top3=True,
    )

    batch = collate_transformer_examples([example], max_candidates=235)

    assert set(accepted_actions_from_batch(batch)) == {"PLAY W1", "PLAY W2", "PLAY W3"}
    assert batch["has_teacher_accept_set"].tolist() == [True]


def test_teacher_accepted_set_loss_accepts_any_relaxed_top3_slot():
    logits = torch.tensor([[0.0, 0.0, 10.0, -10.0]], dtype=torch.float32)
    batch = {
        "has_teacher_accept_set": torch.tensor([True]),
        "teacher_accept_mask": torch.tensor([[True, True, True, False]]),
    }

    loss = teacher_accepted_set_loss(logits, batch)

    assert loss.item() < 0.01


def test_teacher_accepted_set_loss_rejects_second_choice_when_not_relaxed():
    logits = torch.tensor([[0.0, 10.0, -10.0]], dtype=torch.float32)
    batch = {
        "has_teacher_accept_set": torch.tensor([True]),
        "teacher_accept_mask": torch.tensor([[True, False, False]]),
    }

    loss = teacher_accepted_set_loss(logits, batch)

    assert loss.item() > 9.0


def test_row_aware_policy_loss_treats_accept_set_rows_as_reviewed_without_distribution():
    logits = torch.tensor([[0.0, 5.0, -5.0]], dtype=torch.float32)
    batch = {
        "target_index": torch.tensor([2], dtype=torch.long),
        "has_teacher_target": torch.tensor([False]),
        "teacher_target_dist": torch.zeros((1, 3), dtype=torch.float32),
        "has_teacher_accept_set": torch.tensor([True]),
        "teacher_accept_mask": torch.tensor([[False, True, False]]),
    }

    loss, parts = policy_loss_with_optional_teacher(
        logits,
        batch,
        hard_loss_weight=1.0,
        teacher_loss_weight=0.0,
        reviewed_hard_loss_weight=0.0,
        reviewed_teacher_loss_weight=0.0,
        reviewed_accept_set_loss_weight=1.0,
        unreviewed_hard_loss_weight=0.2,
    )

    assert loss.item() < 0.01
    assert parts["reviewed_accept_set_policy_loss"].item() < 0.01
    assert parts["unreviewed_hard_policy_loss"].item() == 0.0


def test_policy_loss_uses_soft_teacher_rows_when_available():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=torch.float32)
    batch = {
        "target_index": torch.tensor([0, 0], dtype=torch.long),
        "has_teacher_target": torch.tensor([True, False]),
        "teacher_target_dist": torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }

    combined, parts = policy_loss_with_optional_teacher(
        logits,
        batch,
        hard_loss_weight=1.0,
        teacher_loss_weight=2.0,
    )

    hard_only = torch.nn.functional.cross_entropy(logits, batch["target_index"])
    assert parts["teacher_policy_loss"] > 2.0
    assert combined > hard_only


def test_policy_loss_can_weight_reviewed_and_unreviewed_rows_separately():
    logits = torch.tensor(
        [
            [0.0, 5.0, -5.0],
            [5.0, 0.0, -5.0],
        ],
        dtype=torch.float32,
    )
    batch = {
        "target_index": torch.tensor([2, 0], dtype=torch.long),
        "has_teacher_target": torch.tensor([True, False]),
        "teacher_target_dist": torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }

    loss, parts = policy_loss_with_optional_teacher(
        logits,
        batch,
        hard_loss_weight=1.0,
        teacher_loss_weight=0.0,
        reviewed_hard_loss_weight=0.0,
        reviewed_teacher_loss_weight=1.0,
        unreviewed_hard_loss_weight=0.2,
    )

    assert loss.item() < 0.01
    assert parts["reviewed_hard_policy_loss"].item() > 5.0
    assert parts["reviewed_teacher_policy_loss"].item() < 0.01
    assert parts["unreviewed_hard_policy_loss"].item() < 0.01


def test_policy_loss_ignores_zero_teacher_mass_on_padded_negative_infinity_logits():
    logits = torch.tensor([[2.0, 0.0, float("-inf")]], dtype=torch.float32)
    batch = {
        "target_index": torch.tensor([0], dtype=torch.long),
        "has_teacher_target": torch.tensor([True]),
        "teacher_target_dist": torch.tensor([[0.75, 0.25, 0.0]], dtype=torch.float32),
    }

    combined, parts = policy_loss_with_optional_teacher(
        logits,
        batch,
        hard_loss_weight=1.0,
        teacher_loss_weight=1.0,
    )

    assert torch.isfinite(combined)
    assert torch.isfinite(parts["teacher_policy_loss"])


def test_load_review_target_lookup_keys_by_record_seat_request_and_normalized_response(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    good_entry = {
        "record_id": "r1",
        "seat": 2,
        "human_action": "Play W2",
        "state_actual_response": "PLAY W2",
        "chaga_top5_candidates": [[3.0, "Play W2"]],
        "play_ordinal": 6,
        "checks": {
            "offered_tile_matches": True,
            "drawn_tile_matches": True,
            "current_actor_matches": True,
            "window_matches": True,
            "hand_size_mod_ok": True,
            "top1_in_legal_mask": True,
        },
        "state_context": {"request": "2 W9"},
    }
    bad_entry = dict(good_entry)
    bad_entry["record_id"] = "r2"
    bad_entry["checks"] = dict(good_entry["checks"], window_matches=False)
    audit_path.write_text(
        json.dumps(good_entry, ensure_ascii=False) + "\n" + json.dumps(bad_entry, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lookup = load_review_target_lookup(audit_path)

    target = lookup(record_id="r1", player=2, request="2 W9", response="PLAY W2")
    assert target == ReviewTarget([[3.0, "Play W2"]], accept_top3=True)
    assert lookup(record_id="r1", player=2, request="2 W9", response="PLAY W2") is None
    assert lookup(record_id="r2", player=2, request="2 W9", response="PLAY W2") is None


def test_review_target_lookup_instances_are_independent(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    entry = {
        "record_id": "r1",
        "seat": 0,
        "state_turn": 7,
        "state_context": {"turn": 7, "request": "2 W1", "state_actual_response": "Play W1"},
        "chaga_top5_candidates": [[10.0, "Play W1"], [9.0, "Play W2"]],
        "checks": {
            "offered_tile_matches": True,
            "drawn_tile_matches": True,
            "current_actor_matches": True,
            "window_matches": True,
            "hand_size_mod_ok": True,
            "top1_in_legal_mask": True,
        },
    }
    audit_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    kwargs = {"record_id": "r1", "player": 0, "turn": 7, "request": "2 W1", "response": "Play W1"}

    shared = load_review_target_lookup(audit_path)
    assert shared(**kwargs) is not None
    assert shared(**kwargs) is None

    lookup_train = load_review_target_lookup(audit_path)
    lookup_val = load_review_target_lookup(audit_path)
    assert lookup_train(**kwargs) is not None
    assert lookup_val(**kwargs) is not None


def test_filter_reviewed_examples_uses_candidate_norms_and_distribution_requirement():
    reviewed = SimpleNamespace(
        teacher_candidate_norms=("PLAY W1",),
        teacher_action_distribution=np.array([1.0], dtype=np.float32),
    )
    unmapped = SimpleNamespace(
        teacher_candidate_norms=("PLAY W2",),
        teacher_action_distribution=None,
    )
    unreviewed = SimpleNamespace(
        teacher_candidate_norms=(),
        teacher_action_distribution=None,
    )

    filtered, summary = filter_reviewed_examples(
        [reviewed, unmapped, unreviewed],
        require_distribution=True,
    )

    assert filtered == [reviewed]
    assert summary["reviewed_examples_before_filter"] == 2
    assert summary["reviewed_without_teacher_distribution"] == 1
    assert summary["reviewed_examples_after_filter"] == 1


def test_reviewed_sampling_weights_target_requested_fraction():
    reviewed_a = SimpleNamespace(teacher_candidate_norms=("PLAY W1",))
    reviewed_b = SimpleNamespace(teacher_candidate_norms=("PLAY W2",))
    unreviewed = SimpleNamespace(teacher_candidate_norms=())

    weights = reviewed_sampling_weights(
        [reviewed_a, reviewed_b, unreviewed],
        reviewed_batch_fraction=0.75,
    )

    assert weights is not None
    reviewed_mass = float(weights[0] + weights[1])
    unreviewed_mass = float(weights[2])
    assert abs(reviewed_mass / (reviewed_mass + unreviewed_mass) - 0.75) < 1e-6


def test_reviewed_only_training_requires_full_candidate_width():
    args = SimpleNamespace(
        review_audit_jsonl=None,
        reviewed_batch_fraction=None,
        reviewed_accept_set_loss_weight=None,
        train_reviewed_only=True,
        val_reviewed_only=False,
        max_candidates=96,
    )

    with pytest.raises(ValueError, match="235"):
        validate_reviewed_training_args(args)


def test_chaga_review_training_requires_full_candidate_width_for_mixed_review_loss():
    args = SimpleNamespace(
        review_audit_jsonl="runs/chaga_review_alignment_audit_all.jsonl",
        reviewed_batch_fraction=0.7,
        reviewed_accept_set_loss_weight=1.0,
        train_reviewed_only=False,
        val_reviewed_only=False,
        max_candidates=96,
    )

    with pytest.raises(ValueError, match="235"):
        validate_reviewed_training_args(args)


def test_history_event_tokens_preserve_full_tile_identity():
    play_tokens = {
        tile: encode_history_event(0, f"2 {tile}", f"Play {tile}")
        for tile in TILE_IDS
    }

    assert len(set(play_tokens.values())) == len(TILE_IDS)
    assert play_tokens["W1"] != play_tokens["T4"]
    assert play_tokens["W1"] != play_tokens["F1"]


def test_default_history_vocab_can_represent_full_history_token_range():
    model = TransformerCandidateModel()

    required_size = 1 + 4 * len(BOTZONE_ACTION_TYPES) * 35
    assert model.history_vocab_size >= required_size


def test_teacher_only_policy_loss_ignores_hard_target():
    batch = {
        "target_index": torch.tensor([2], dtype=torch.long),
        "has_teacher_target": torch.tensor([True]),
        "teacher_target_dist": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    }

    good_logits = torch.tensor([[0.0, 5.0, -5.0]], dtype=torch.float32)
    good_loss, _ = policy_loss_with_optional_teacher(
        good_logits,
        batch,
        hard_loss_weight=0.0,
        teacher_loss_weight=1.0,
    )
    bad_logits = torch.tensor([[0.0, -5.0, 5.0]], dtype=torch.float32)
    bad_loss, _ = policy_loss_with_optional_teacher(
        bad_logits,
        batch,
        hard_loss_weight=0.0,
        teacher_loss_weight=1.0,
    )

    assert good_loss.item() < 0.01
    assert bad_loss.item() > 5.0


def test_checkpoint_selection_prefers_play_exact_then_accuracy_then_loss():
    incumbent = {
        "val_play_exact_accuracy": 0.40,
        "val_accuracy": 0.50,
        "val_loss": 1.0,
    }

    assert is_better_checkpoint_metric(
        {"val_play_exact_accuracy": 0.41, "val_accuracy": 0.45, "val_loss": 2.0},
        incumbent,
    )
    assert is_better_checkpoint_metric(
        {"val_play_exact_accuracy": 0.40, "val_accuracy": 0.51, "val_loss": 2.0},
        incumbent,
    )
    assert is_better_checkpoint_metric(
        {"val_play_exact_accuracy": 0.40, "val_accuracy": 0.50, "val_loss": 0.9},
        incumbent,
    )
    assert not is_better_checkpoint_metric(
        {"val_play_exact_accuracy": 0.39, "val_accuracy": 0.90, "val_loss": 0.1},
        incumbent,
    )


def test_count_model_parameters_reports_trainable_and_state_bytes():
    model = TransformerCandidateModel(
        act_size=235,
        history_vocab_size=512,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
    )

    counts = count_model_parameters(model)

    assert counts["parameters"] > 0
    assert counts["trainable_parameters"] == counts["parameters"]
    assert counts["state_tensor_bytes"] >= counts["parameters"] * 4
