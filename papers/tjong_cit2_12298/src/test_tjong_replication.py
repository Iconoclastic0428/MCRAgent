import argparse
import sys
from pathlib import Path
from collections import Counter
import json
import tempfile
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tjong_replication.audit_policy_replay as audit_policy_replay  # noqa: E402
import tjong_replication.collect_selfplay as collect_selfplay_module  # noqa: E402
import tjong_replication.train_supervised as train_supervised_module  # noqa: E402
import tjong_replication.verify_platform_wrappers as verify_platform_wrappers  # noqa: E402
from tjong_replication.actions import ACTION_NAMES, ACTION_TO_INDEX, CLAIM_SIZE, flatten_claim, unflatten_claim  # noqa: E402
from tjong_replication.audit_tziakcha_botzone_coverage import audit_coverage  # noqa: E402
from tjong_replication.build_ppo_rollouts import reward_to_go, rollout_rewards  # noqa: E402
from tjong_replication.collect_selfplay import (  # noqa: E402
    SelfplaySummaryAccumulator,
    TjongBotzoneJsonReplayPolicy,
    summarize as summarize_selfplay,
)
from tjong_replication.convert_tziakcha_to_botzone import convert_tziakcha_file  # noqa: E402
from tjong_replication.convert_tziakcha_to_initdata import convert_tziakcha_initdata_file  # noqa: E402
from tjong_replication.encoding import VISIBLE_ROW_NAMES  # noqa: E402
from tjong_replication.evaluate_supervised import load_model  # noqa: E402
from tjong_replication.evaluate_checkpoints import evaluate_checkpoints  # noqa: E402
from tjong_replication.evaluate_slide_resnet_supervised import evaluate as evaluate_slide_resnet  # noqa: E402
from tjong_replication.fan_attribution import attribute_display_fans  # noqa: E402
from tjong_replication.fan_backward import FanItem, calculate_score, winning_reward  # noqa: E402
from tjong_replication.merge_selfplay_shards import merge_shards  # noqa: E402
from tjong_replication.merge_botzone_corpora import merge_botzone_corpora  # noqa: E402
from tjong_replication.model import TITPolicyBackbone, TjongConfig, TjongNetwork  # noqa: E402
from tjong_replication.paper_metrics import (  # noqa: E402
    DEFAULT_PAPER_METRIC_TOLERANCE,
    PAPER_REPORTED_SUPERVISED_METRICS,
)
from tjong_replication.pipeline_status import audit_pipeline  # noqa: E402
from tjong_replication.ppo import PPOConfig  # noqa: E402
from tjong_replication.policy_bot import (  # noqa: E402
    TjongCheckpointPredictor,
    build_runtime_context,
    encode_runtime_state,
    respond_json_with_predictor,
    response_discard_label,
    response_to_labels,
    runtime_hidden_schema_rows,
)
from tjong_replication.populate_fan_backward_rewards import infer_loser, populate_file  # noqa: E402
from tjong_replication.plot_supervised_progress import load_epoch_metrics, summarize_epochs  # noqa: E402
from tjong_replication.selfplay_dashboard import analyze_records, format_dashboard_text  # noqa: E402
from tjong_replication.slide_resnet import (  # noqa: E402
    SLIDE_V1_CHANNELS,
    SLIDE_V2_CHANNELS,
    SLIDE_GRID_SUIT_ROWS,
    SlideMahjongResNetDueling,
    SlideResNetConfig,
    build_slide_feature_planes,
    tile_vectors_to_grid,
    transform_claim_labels,
    transform_discard_labels,
    transform_tile_id,
    transform_tile_tensor,
)
from tjong_replication.slide_v2_features import (  # noqa: E402
    SlideV2SearchContext,
    build_batch_search_features_from_tensors,
    build_search_feature_planes,
)
from tjong_replication.tensorize_botzone import (  # noqa: E402
    HIDDEN_TILE_ROW_NAMES,
    SHARD_INDEX_FORMAT,
    TENSOR_ENCODING_VERSION,
    ReplayState,
    action_type_mask,
    chow_claim_index,
    metadata_column,
    tensor_encoding_schema,
    tensorize_file,
    tensorize_record,
)
from tjong_replication.tiles import TILE_NAMES, tile_id  # noqa: E402
from tjong_replication.train_ppo import (  # noqa: E402
    hierarchical_log_prob_and_entropy,
    train as train_ppo,
    validate_paper_ppo_args,
    validate_rollout_reward_source,
)
from tjong_replication.train_slide_resnet_supervised import (  # noqa: E402
    divergence_guard_reasons,
    elastic_net_penalty,
    fast_local_supervised_loss_components,
    slide_checkpoint_payload,
)
from tjong_replication.train_supervised import (  # noqa: E402
    ShardedTensorDataset,
    batch_metric_sums,
    evaluate_model,
    finalize_metric_sums,
    freeze_supervised_value_parameters,
    iter_optimized_sharded_batches,
    load_tensor_dataset,
    nonfinite_supervised_context,
    safe_clip_grad_norm_,
    supervised_loss_components,
    train as train_supervised,
    validate_supervised_dataset_contract,
    validate_supervised_labels,
    validate_paper_supervised_args,
)
from tjong_replication.validate_initdata_corpus import validate_initdata_corpus  # noqa: E402
from tjong_replication.validate_paper_corpus import validate_corpus  # noqa: E402
from tjong_replication.verify_paper_compliance import verify_paper_compliance, verify_paper_compliance_file  # noqa: E402
from tjong_replication.verify_paper_metrics import compare_supervised_metrics, verify_metrics_file  # noqa: E402


def test_claim_flattening_round_trip():
    idx = flatten_claim("PONG", 3)
    claim = unflatten_claim(idx)
    assert claim.family == "PONG"
    assert claim.local_index == 3
    assert CLAIM_SIZE == 199


def test_network_forward_shapes():
    config = TjongConfig(d_model=128, n_heads=4, ffn_dim=256, dropout=0.0)
    model = TjongNetwork(config)
    batch = 2
    visible = torch.zeros(batch, config.memory_len, 22, 34)
    game = torch.zeros(batch, config.memory_len, 24)
    sub_visible = torch.zeros(batch, config.memory_len, 22, 34)
    sub_game = torch.zeros(batch, config.memory_len, 24)
    sub_game[:, -1, -8:] = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0]).float()
    hidden = torch.zeros(batch, 5, 34)
    out = model(
        visible_tiles=visible,
        game_features=game,
        sub_visible_tiles=sub_visible,
        sub_game_features=sub_game,
        hidden_tiles=hidden,
    )
    assert out["action_logits"].shape == (batch, 8)
    assert out["claim_logits"].shape == (batch, 199)
    assert out["discard_logits"].shape == (batch, 34)
    assert out["value"].shape == (batch,)


def test_network_reuse_identical_substate_matches_full_eval_path():
    config = TjongConfig(d_model=32, n_heads=4, ffn_dim=64, dropout=0.0)
    model = TjongNetwork(config)
    model.eval()
    batch = 3
    visible = torch.randn(batch, config.memory_len, 22, 34)
    game = torch.randn(batch, config.memory_len, 24)
    rewards = torch.randn(batch, config.memory_len)
    previous = torch.zeros(batch, config.memory_len, dtype=torch.long)
    sub_visible = visible.clone()
    sub_game = game.clone()
    sub_rewards = rewards.clone()
    sub_previous = previous.clone()
    sub_game[1, -1, 0] += 1.0
    sub_previous[2, -1] = ACTION_TO_INDEX["PONG"]

    with torch.no_grad():
        model.reuse_identical_substate = False
        full = model(
            visible_tiles=visible,
            game_features=game,
            rewards=rewards,
            previous_actions=previous,
            sub_visible_tiles=sub_visible,
            sub_game_features=sub_game,
            sub_rewards=sub_rewards,
            sub_previous_actions=sub_previous,
        )
        model.reuse_identical_substate = True
        reused = model(
            visible_tiles=visible,
            game_features=game,
            rewards=rewards,
            previous_actions=previous,
            sub_visible_tiles=sub_visible,
            sub_game_features=sub_game,
            sub_rewards=sub_rewards,
            sub_previous_actions=sub_previous,
        )

    assert torch.allclose(reused["action_logits"], full["action_logits"])
    assert torch.allclose(reused["claim_logits"], full["claim_logits"])
    assert torch.allclose(reused["discard_logits"], full["discard_logits"])


def test_slide_feature_planes_match_v1_and_v2_channel_counts():
    batch = 2
    visible = torch.zeros(batch, 4, 22, 34)
    game = torch.zeros(batch, 4, 24)
    hidden = torch.zeros(batch, 5, 34)
    visible[0, 0, 0, tile_id("W1")] = 2.0
    game[0, 0, 0] = 3.0
    hidden[0, 0, tile_id("J3")] = 1.0

    tile_grid = tile_vectors_to_grid(visible[0, 0, 0])
    assert tile_grid.shape == (4, 9)
    assert SLIDE_GRID_SUIT_ROWS == ("T", "W", "B")
    assert tile_grid[1, 0] == 2.0

    v1 = build_slide_feature_planes(
        visible_tiles=visible,
        game_features=game,
        hidden_tiles=hidden,
        feature_version="v1",
        use_hidden_tiles=True,
    )
    assert v1.shape == (batch, SLIDE_V1_CHANNELS, 4, 9)
    assert v1[0, 0, 1, 0] == 2.0
    assert v1[0, 88, :, :].unique().tolist() == [3.0]
    assert v1[0, 184, 3, 6] == 1.0
    assert v1[:, -1].eq(1.0).all()

    search = torch.ones(batch, 30, 4, 9)
    v2 = build_slide_feature_planes(
        visible_tiles=visible,
        game_features=game,
        hidden_tiles=hidden,
        search_features=search,
        feature_version="v2",
    )
    assert v2.shape == (batch, SLIDE_V2_CHANNELS, 4, 9)
    assert v2[:, SLIDE_V1_CHANNELS:].eq(1.0).all()


def test_slide_feature_planes_can_require_v2_search_features():
    visible = torch.zeros(1, 4, 22, 34)
    game = torch.zeros(1, 4, 24)
    try:
        build_slide_feature_planes(
            visible_tiles=visible,
            game_features=game,
            feature_version="v2",
            require_search_features=True,
        )
    except ValueError as exc:
        assert "search feature" in str(exc)
    else:
        raise AssertionError("expected v2 feature builder to require search features")


def test_slide_symmetry_transforms_tile_and_claim_labels():
    transform = (("T", "B", "W"), True)
    suit_permutation, mirror = transform
    assert transform_tile_id(tile_id("W1"), suit_permutation, mirror=mirror) == tile_id("T9")
    assert transform_tile_id(tile_id("T2"), suit_permutation, mirror=mirror) == tile_id("B8")
    assert transform_tile_id(tile_id("B3"), suit_permutation, mirror=mirror) == tile_id("W7")
    assert transform_tile_id(tile_id("J2"), suit_permutation, mirror=mirror) == tile_id("J2")

    tile_tensor = torch.zeros(1, 34)
    tile_tensor[0, tile_id("W1")] = 2.0
    tile_tensor[0, tile_id("J2")] = 1.0
    transformed_tiles = transform_tile_tensor(tile_tensor, suit_permutation, mirror=mirror)
    assert transformed_tiles[0, tile_id("T9")] == 2.0
    assert transformed_tiles[0, tile_id("J2")] == 1.0

    discard = torch.tensor([tile_id("W1"), tile_id("J2")])
    transformed_discard = transform_discard_labels(discard, suit_permutation, mirror=mirror)
    assert transformed_discard.tolist() == [tile_id("T9"), tile_id("J2")]

    chow = flatten_claim("CHOW", chow_claim_index("W2", "W1"))
    pong = flatten_claim("PONG", tile_id("B3"))
    transformed_claim = transform_claim_labels(torch.tensor([chow, pong]), suit_permutation, mirror=mirror)
    assert transformed_claim.tolist() == [
        flatten_claim("CHOW", chow_claim_index("T8", "T9")),
        flatten_claim("PONG", tile_id("W7")),
    ]


def test_slide_divergence_guard_reasons():
    args = argparse.Namespace(
        divergence_guard_after_epochs=2,
        divergence_guard_max_discard_loss=5.0,
        divergence_guard_max_optimization_loss=10.0,
        divergence_guard_max_grad_norm=1000.0,
        loss_mode="head_sum",
    )
    assert (
        divergence_guard_reasons(
            args,
            {
                "epoch": 1,
                "discard_loss": 100.0,
                "optimization_loss": 100.0,
                "max_grad_norm_before_clip": 10000.0,
            },
        )
        == []
    )
    reasons = divergence_guard_reasons(
        args,
        {
            "epoch": 2,
            "discard_loss": 10.0,
            "optimization_loss": 2.0,
            "max_grad_norm_before_clip": 2000.0,
        },
    )
    assert any("discard_loss" in reason for reason in reasons)
    assert any("max_grad_norm_before_clip" in reason for reason in reasons)
    assert not any("optimization_loss" in reason for reason in reasons)

    masked_args = argparse.Namespace(
        divergence_guard_after_epochs=2,
        divergence_guard_max_discard_loss=5.0,
        divergence_guard_max_optimization_loss=10.0,
        divergence_guard_max_grad_norm=1000.0,
        loss_mode="masked_hierarchical",
    )
    masked_reasons = divergence_guard_reasons(
        masked_args,
        {
            "epoch": 2,
            "discard_loss": 10.0,
            "optimization_loss": 2.0,
            "max_grad_norm_before_clip": 200.0,
        },
    )
    assert masked_reasons == []
    masked_args.loss_mode = "masked_hierarchical_balanced"
    assert divergence_guard_reasons(masked_args, {"epoch": 2, "discard_loss": 10.0}) == []


def test_slide_elastic_net_penalty_uses_l1_and_l2_lambdas():
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0]]))
        model.bias.copy_(torch.tensor([3.0]))

    l1_penalty, l2_penalty = elastic_net_penalty(model, l1_lambda=0.1, l2_lambda=0.01)

    assert torch.isclose(l1_penalty, torch.tensor(0.6))
    assert torch.isclose(l2_penalty, torch.tensor(0.14))


def test_slide_masked_hierarchical_loss_is_optimized_total():
    n = 2
    action_label = torch.tensor([ACTION_TO_INDEX["DISCARD"], ACTION_TO_INDEX["PONG"]], dtype=torch.long)
    claim_label = torch.tensor([0, flatten_claim("PONG", tile_id("W1"))], dtype=torch.long)
    discard_label = torch.tensor([tile_id("W1"), 0], dtype=torch.long)
    action_logits = torch.zeros(n, len(ACTION_NAMES))
    action_logits[0, ACTION_TO_INDEX["HU"]] = 20.0
    action_logits[1, ACTION_TO_INDEX["CHOW"]] = 20.0
    claim_logits = torch.zeros(n, CLAIM_SIZE)
    claim_logits[1, flatten_claim("CHOW", 0)] = 20.0
    claim_logits[1, flatten_claim("PONG", tile_id("W1"))] = 8.0
    discard_logits = torch.zeros(n, len(TILE_NAMES))
    discard_logits[0, tile_id("T9")] = 20.0
    game = torch.zeros(n, 4, 24)
    sub_visible = torch.zeros(n, 4, 22, 34)

    components = fast_local_supervised_loss_components(
        {
            "action_logits": action_logits,
            "claim_logits": claim_logits,
            "discard_logits": discard_logits,
        },
        (action_label, claim_label, discard_label),
        {
            "game_features": game,
            "sub_visible_tiles": sub_visible,
        },
        loss_mode="masked_hierarchical",
    )

    assert torch.isclose(components["total"], components["masked_hierarchical_nll"])
    assert components["total"] < 0.1
    assert components["legacy_head_sum_ce"] > 30.0
    assert int(components["action_target_mask_repairs"].item()) == 2
    assert int(components["discard_target_mask_repairs"].item()) == 1


def test_slide_masked_hierarchical_balanced_loss_uses_legal_smoothing():
    n = 2
    action_label = torch.tensor([ACTION_TO_INDEX["DISCARD"], ACTION_TO_INDEX["PONG"]], dtype=torch.long)
    claim_label = torch.tensor([0, flatten_claim("PONG", tile_id("W1"))], dtype=torch.long)
    discard_label = torch.tensor([tile_id("W1"), 0], dtype=torch.long)
    action_logits = torch.zeros(n, len(ACTION_NAMES))
    action_logits[0, ACTION_TO_INDEX["DISCARD"]] = 8.0
    action_logits[1, ACTION_TO_INDEX["PONG"]] = 8.0
    claim_logits = torch.zeros(n, CLAIM_SIZE)
    claim_logits[1, flatten_claim("PONG", tile_id("W1"))] = 8.0
    discard_logits = torch.zeros(n, len(TILE_NAMES))
    discard_logits[0, tile_id("W1")] = 8.0
    game = torch.zeros(n, 4, 24)
    game[0, -1, 16 + ACTION_TO_INDEX["DISCARD"]] = 1.0
    game[1, -1, 16 + ACTION_TO_INDEX["PASS"]] = 1.0
    game[1, -1, 16 + ACTION_TO_INDEX["PONG"]] = 1.0
    sub_visible = torch.zeros(n, 4, 22, 34)
    sub_visible[0, -1, 0, tile_id("W1")] = 1.0
    sub_visible[0, -1, 0, tile_id("W2")] = 1.0

    components = fast_local_supervised_loss_components(
        {
            "action_logits": action_logits,
            "claim_logits": claim_logits,
            "discard_logits": discard_logits,
        },
        (action_label, claim_label, discard_label),
        {
            "game_features": game,
            "sub_visible_tiles": sub_visible,
        },
        loss_mode="masked_hierarchical_balanced",
        action_loss_weight=1.0,
        discard_loss_weight=0.5,
        claim_loss_weight=0.2,
        action_label_smoothing=0.02,
        discard_label_smoothing=0.05,
        claim_label_smoothing=0.01,
    )

    assert torch.isclose(components["total"], components["balanced_masked_hierarchical_ce"])
    assert components["total"] > 0
    assert components["masked_hierarchical_nll"] > 0
    assert components["masked_discard_loss_conditional"] > 0
    assert int(components["action_target_mask_repairs"].item()) == 0
    assert int(components["discard_target_mask_repairs"].item()) == 0


def test_slide_checkpoint_payload_records_resume_state():
    config = SlideResNetConfig(base_channels=8, head_hidden=16, dropout=0.0)
    model = SlideMahjongResNetDueling(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    metrics = {
        "checkpoint_encoding_version": TENSOR_ENCODING_VERSION,
        "epochs": [{"epoch": 1, "discard_loss": 1.0}],
        "epochs_completed": 1,
        "global_batch": 12,
    }

    payload = slide_checkpoint_payload(
        raw_model=model,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        metrics=metrics,
        completed_epochs=1,
        data_parallel_enabled=False,
    )

    assert payload["epoch"] == 1
    assert payload["tensor_encoding_version"] == TENSOR_ENCODING_VERSION
    assert payload["parallel_model_state_dict"] is None
    assert payload["metrics"]["global_batch"] == 12
    assert "model_state_dict" in payload
    assert "optimizer" in payload
    assert "scheduler" in payload
    assert payload["config"]["base_channels"] == 8


def test_evaluate_slide_resnet_reports_full_pass_metrics():
    config = SlideResNetConfig(base_channels=8, head_hidden=16, dropout=0.0, use_hidden_tiles=True)
    model = SlideMahjongResNetDueling(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    metrics = {
        "checkpoint_encoding_version": TENSOR_ENCODING_VERSION,
        "epochs": [],
        "epochs_completed": 0,
        "global_batch": 0,
        "elastic_net_l1_lambda": 1e-7,
        "elastic_net_l2_lambda": 1e-5,
    }
    visible = torch.zeros(3, 4, 22, 34)
    game = torch.zeros(3, 4, 24)
    hidden = torch.zeros(3, 5, 34)
    action_label = torch.tensor(
        [
            ACTION_TO_INDEX["DISCARD"],
            ACTION_TO_INDEX["CHOW"],
            ACTION_TO_INDEX["PONG"],
        ],
        dtype=torch.long,
    )
    claim_label = torch.tensor(
        [
            0,
            flatten_claim("CHOW", chow_claim_index("W2", "W1")),
            flatten_claim("PONG", tile_id("W1")),
        ],
        dtype=torch.long,
    )
    discard_label = torch.tensor([tile_id("W1"), tile_id("W2"), tile_id("W3")], dtype=torch.long)

    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp = Path(tmp_dir)
        checkpoint = tmp / "slide.pt"
        eval_pt = tmp / "eval.pt"
        metrics_out = tmp / "metrics.json"
        torch.save(
            slide_checkpoint_payload(
                raw_model=model,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                metrics=metrics,
                completed_epochs=0,
                data_parallel_enabled=False,
            ),
            checkpoint,
        )
        torch.save(
            {
                "visible_tiles": visible,
                "game_features": game,
                "hidden_tiles": hidden,
                "action_label": action_label,
                "claim_label": claim_label,
                "discard_label": discard_label,
                "examples": 3,
                "encoding_schema": tensor_encoding_schema(),
            },
            eval_pt,
        )

        result = evaluate_slide_resnet(
            argparse.Namespace(
                checkpoint=str(checkpoint),
                eval_pt=str(eval_pt),
                metrics_out=str(metrics_out),
                batch_size=2,
                num_workers=0,
                device="cpu",
                amp="off",
                allow_tf32=False,
                require_encoding_version=TENSOR_ENCODING_VERSION,
                seed=0,
                strict_deterministic=False,
                optimized_sharded_loader=False,
                shard_prefetch=1,
                pin_memory=False,
                drop_shard_file_cache=False,
                mmap_shards=False,
                max_batches=0,
                progress_every_batches=0,
                l1_lambda=None,
                l2_lambda=None,
                loss_mode="masked_hierarchical",
                v2_search_levels=1,
                v2_use_official_fan=False,
                v2_require_official_fan=False,
            )
        )
        written_examples = json.loads(metrics_out.read_text(encoding="utf-8"))["evaluated_examples"]

    assert result["full_pass"]
    assert result["evaluated_examples"] == 3
    assert result["action_count"] == 3
    assert result["claim_count"] == 2
    assert result["discard_count"] == 1
    assert result["l1_lambda"] == 1e-7
    assert result["l2_lambda"] == 1e-5
    assert result["loss_mode"] == "masked_hierarchical"
    assert result["regularization_loss"] > 0
    assert result["cross_entropy_loss"] > 0
    assert result["masked_hierarchical_nll"] == pytest.approx(result["cross_entropy_loss"])
    assert result["head_sum"] > 0
    assert result["head_sum_cross_entropy_loss"] > 0
    assert result["hierarchical_per_example_loss"] > 0
    assert result["per_decision_loss"] > 0
    assert result["claim_fraction"] == pytest.approx(2 / 3)
    assert result["discard_fraction"] == pytest.approx(1 / 3)
    assert written_examples == 3


def test_slide_v2_search_features_from_encoded_tensors():
    visible = torch.zeros(1, 4, 22, 34)
    game = torch.zeros(1, 4, 24)
    visible[0, -1, VISIBLE_ROW_NAMES.index("hand_self"), tile_id("W1")] = 2.0
    visible[0, -1, VISIBLE_ROW_NAMES.index("remaining_tiles_p0"), tile_id("T5")] = 4.0
    game[0, -1, 1] = 1.0

    search = build_batch_search_features_from_tensors(visible, game, levels=1)
    assert search.shape == (1, 30, 4, 9)
    assert search[0, 0, 1, 0] == 0.5
    assert search[0, 1, 0, 4] == 1.0

    context_search = build_search_feature_planes(
        SlideV2SearchContext(
            hand=Counter({"W1": 2}),
            live_counts=Counter({"T5": 4}),
            visible_counts=Counter(),
        ),
        levels=1,
    )
    assert context_search.shape == (30, 4, 9)
    assert context_search[0, 1, 0] == 0.5


def test_slide_resnet_dueling_forward_shapes():
    config = SlideResNetConfig(feature_version="v1", base_channels=8, head_hidden=32, dropout=0.0)
    model = SlideMahjongResNetDueling(config)
    assert isinstance(model.policy_norm, torch.nn.LayerNorm)
    assert isinstance(model.tile_norm, torch.nn.LayerNorm)
    batch = 2
    visible = torch.zeros(batch, 4, 22, 34)
    game = torch.zeros(batch, 4, 24)
    sub_visible = torch.zeros(batch, 4, 22, 34)
    sub_game = torch.zeros(batch, 4, 24)
    hidden = torch.zeros(batch, 5, 34)
    out = model(
        visible_tiles=visible,
        game_features=game,
        sub_visible_tiles=sub_visible,
        sub_game_features=sub_game,
        hidden_tiles=hidden,
    )
    assert out["action_logits"].shape == (batch, 8)
    assert out["claim_logits"].shape == (batch, 199)
    assert out["discard_logits"].shape == (batch, 34)


def test_tjong_runtime_context_replays_memory_and_public_discards():
    input_text = "\n".join(
        [
            "REQ 0 0 1",
            "RES PASS",
            "REQ 1 0 0 0 0 W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 F1 F2",
            "RES PASS",
            "REQ 2 W5",
            "RES PLAY F2",
            "REQ 3 0 PLAY F2",
            "RES PASS",
            "REQ 3 1 PLAY W1",
        ]
    )
    hand = Counter("W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 W5".split())
    context = build_runtime_context(
        input_text=input_text,
        hand=hand,
        player_id=0,
        current_action_mask=[1, 0, 0, 0, 1, 0, 0, 0],
        memory_len=4,
    )

    memory = context.memory_for_mask([1, 0, 0, 0, 1, 0, 0, 0], previous_action=context.last_action)

    assert context.last_action == ACTION_TO_INDEX["DISCARD"]
    assert memory.previous_actions.tolist()[-2:] == [ACTION_TO_INDEX["DISCARD"], ACTION_TO_INDEX["DISCARD"]]
    discard_row = VISIBLE_ROW_NAMES.index("discard_p0")
    assert memory.visible_tiles[-1, discard_row, tile_id("F2")] == 1
    assert not torch.equal(memory.visible_tiles[-2], memory.visible_tiles[-1])


class _HierarchicalFakeTjong(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TjongConfig()
        self.previous_action_calls = []

    def forward(
        self,
        *,
        visible_tiles,
        game_features,
        rewards=None,
        previous_actions=None,
        sub_visible_tiles=None,
        sub_game_features=None,
        sub_rewards=None,
        sub_previous_actions=None,
        hidden_tiles=None,
    ):
        self.previous_action_calls.append(
            {
                "previous_actions": None if previous_actions is None else previous_actions.detach().cpu().clone(),
                "sub_previous_actions": (
                    None if sub_previous_actions is None else sub_previous_actions.detach().cpu().clone()
                ),
            }
        )
        batch = visible_tiles.shape[0]
        action_logits = torch.full((batch, len(ACTION_NAMES)), -20.0)
        action_logits[:, ACTION_TO_INDEX["PONG"]] = 20.0
        claim_logits = torch.full((batch, CLAIM_SIZE), -20.0)
        claim_logits[:, flatten_claim("PONG", tile_id("W1"))] = 20.0
        discard_logits = torch.full((batch, len(TILE_NAMES)), -20.0)
        discard_logits[:, tile_id("B2")] = 20.0
        return {
            "action_logits": action_logits,
            "claim_logits": claim_logits,
            "discard_logits": discard_logits,
            "value": torch.zeros(batch),
        }


def test_tjong_predictor_uses_hierarchical_claim_then_forced_discard():
    predictor = TjongCheckpointPredictor.__new__(TjongCheckpointPredictor)
    predictor.device = torch.device("cpu")
    predictor.model = _HierarchicalFakeTjong()
    input_text = "\n".join(
        [
            "REQ 0 0 1",
            "RES PASS",
            "REQ 1 0 0 0 0 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3",
            "RES PASS",
            "REQ 3 1 PLAY W1",
        ]
    )

    response = predictor.predict_legal_response(
        input_text=input_text,
        hand=Counter("W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3".split()),
        player_id=0,
        request="3 1 PLAY W1",
        candidates=["PASS", "PENG B1", "PENG B2"],
    )

    assert response == "PENG B2"
    assert predictor.model.previous_action_calls[0]["previous_actions"] is not None


def test_tjong_respond_json_teacher_forces_previous_responses():
    class RecordingPredictor:
        kind = "legal_action_ranker"

        def __init__(self):
            self.calls = []

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            self.calls.append(
                {
                    "input_text": input_text,
                    "hand": Counter(hand),
                    "player_id": player_id,
                    "request": request,
                    "candidates": list(candidates),
                }
            )
            return "PENG B2"

    predictor = RecordingPredictor()
    payload = {
        "requests": [
            "0 0 1",
            "1 0 0 0 0 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3",
            "2 W5",
            "3 1 PLAY W1",
        ],
        "responses": ["PASS", "PASS", "PLAY F2"],
    }

    response = respond_json_with_predictor(payload, predictor)

    assert response == "PENG B2"
    assert len(predictor.calls) == 1
    call = predictor.calls[0]
    assert "REQ 2 W5\nRES PLAY F2" in call["input_text"]
    assert call["input_text"].endswith("REQ 3 1 PLAY W1")
    assert call["hand"]["F2"] == 0
    assert call["hand"]["W5"] == 1
    assert "PENG B2" in call["candidates"]


def test_tjong_respond_json_fails_when_recorded_replay_is_not_legal():
    class RecordingPredictor:
        kind = "legal_action_ranker"

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            return "PASS"

    payload = {
        "requests": [
            "0 0 1",
            "1 0 0 0 0 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3",
            "2 W5",
            "3 1 PLAY W1",
        ],
        "responses": ["PASS", "PASS", "PLAY W9"],
    }

    try:
        respond_json_with_predictor(payload, RecordingPredictor())
    except ValueError as exc:
        assert "Botzone replay diverged before current decision" in str(exc)
        assert "PLAY W9" in str(exc)
    else:
        raise AssertionError("invalid recorded replay should fail loudly")


def test_tjong_policy_replay_audit_uses_json_wrapper(monkeypatch, tmp_path):
    class FakePredictor:
        kind = "legal_action_ranker"

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            if request == "2 W5":
                return "PLAY W5"
            return "PASS"

    monkeypatch.setattr(audit_policy_replay, "TjongCheckpointPredictor", lambda *args, **kwargs: FakePredictor())
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "match_id": "audit-smoke",
                "logs": [
                    {"output": {"content": {"0": "0 0 0"}}},
                    {"0": {"response": "PASS"}},
                    {
                        "output": {
                            "content": {
                                "0": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
                            }
                        }
                    },
                    {"0": {"response": "PASS"}},
                    {"output": {"content": {"0": "2 W5"}}},
                    {"0": {"response": "PLAY W5"}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = audit_policy_replay.audit_botzone_replay(
        checkpoint=tmp_path / "fake.pt",
        raw_path=raw,
        max_states=1,
        require_encoding_version=None,
    )

    assert summary["states"] == 1
    assert summary["exact_accuracy"] == 1.0
    assert summary["family_accuracy"] == 1.0
    assert summary["error_count"] == 0
    assert summary["examples"][0]["request"] == "2 W5"


def test_tjong_policy_replay_audit_keeps_pass_history_aligned(monkeypatch, tmp_path):
    class FakePredictor:
        pass

    payloads = []

    def fake_respond_json_with_predictor(payload, predictor):
        payloads.append(payload)
        return "PLAY W5"

    monkeypatch.setattr(audit_policy_replay, "TjongCheckpointPredictor", lambda *args, **kwargs: FakePredictor())
    monkeypatch.setattr(audit_policy_replay, "respond_json_with_predictor", fake_respond_json_with_predictor)
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "match_id": "audit-pass-alignment",
                "logs": [
                    {"output": {"content": {"0": "0 0 0"}}},
                    {"0": {"response": "PASS"}},
                    {
                        "output": {
                            "content": {
                                "0": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
                            }
                        }
                    },
                    {"0": {"response": "PASS"}},
                    {"output": {"content": {"0": "2 W4"}}},
                    {"0": {"response": "PASS"}},
                    {"output": {"content": {"0": "2 W5"}}},
                    {"0": {"response": "PLAY W5"}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = audit_policy_replay.audit_botzone_replay(
        checkpoint=tmp_path / "fake.pt",
        raw_path=raw,
        max_states=1,
        nonpass_only=True,
        require_encoding_version=None,
    )

    assert summary["states"] == 1
    assert payloads == [
        {
            "requests": [
                "0 0 0",
                "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
                "2 W4",
                "2 W5",
            ],
            "responses": ["PASS", "PASS", "PASS"],
        }
    ]


def test_tjong_selfplay_policy_uses_botzone_json_replay(monkeypatch):
    calls = []

    def fake_respond_json_with_predictor(payload, predictor, *, fan_checker=None):
        calls.append(
            {
                "payload": payload,
                "predictor": predictor,
                "fan_checker": fan_checker,
            }
        )
        return "PASS" if len(payload["requests"]) == 1 else "PLAY W5"

    monkeypatch.setattr(
        collect_selfplay_module,
        "respond_json_with_predictor",
        fake_respond_json_with_predictor,
    )
    predictor = object()
    fan_checker = object()
    policy = TjongBotzoneJsonReplayPolicy(predictor, fan_checker=fan_checker)

    assert policy.respond("0 0 1") == "PASS"
    assert policy.respond("2 W5") == "PLAY W5"

    assert calls[0]["payload"] == {"requests": ["0 0 1"], "responses": []}
    assert calls[1]["payload"] == {"requests": ["0 0 1", "2 W5"], "responses": ["PASS"]}
    assert calls[1]["predictor"] is predictor
    assert calls[1]["fan_checker"] is fan_checker
    assert policy.diagnostics()["kind"] == "tjong_botzone_json_replay"


def test_collect_selfplay_uses_botzone_json_replay_policy(monkeypatch, tmp_path):
    calls = []

    class FakePredictor:
        pass

    class FakeFanChecker:
        @classmethod
        def default(cls):
            return "default-fan-checker"

    def fake_respond_json_with_predictor(payload, predictor, *, fan_checker=None):
        calls.append(
            {
                "payload": payload,
                "predictor": predictor,
                "fan_checker": fan_checker,
            }
        )
        request = payload["requests"][-1]
        return "PLAY W5" if request == "2 W5" else "PASS"

    def fake_load_initdata(path, limit=None, offset=0):
        assert str(path).endswith("initdata.jsonl")
        assert limit == 1
        assert offset == 0
        return [{"srand": 1234}]

    def fake_run_match(policies, initdata, exe_path=None, max_turns=500):
        responses = [
            policies[0].respond("0 0 1"),
            policies[0].respond("1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"),
            policies[0].respond("2 W5"),
        ]
        assert responses == ["PASS", "PASS", "PLAY W5"]
        return {
            "terminal_reason": "finish",
            "turns": 3,
            "scores": [0, 0, 0, 0],
            "final_output": {"display": {"action": "HUANG"}},
            "log": [
                {"output": {"content": {"0": "0 0 1"}}},
                {"0": {"response": responses[0], "raw": responses[0], "verdict": "OK"}},
                {
                    "output": {
                        "content": {
                            "0": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
                        }
                    }
                },
                {"0": {"response": responses[1], "raw": responses[1], "verdict": "OK"}},
                {"output": {"content": {"0": "2 W5"}}},
                {"0": {"response": responses[2], "raw": responses[2], "verdict": "OK"}},
            ],
        }

    monkeypatch.setattr(collect_selfplay_module, "TjongCheckpointPredictor", lambda *args, **kwargs: FakePredictor())
    monkeypatch.setattr(
        collect_selfplay_module,
        "respond_json_with_predictor",
        fake_respond_json_with_predictor,
    )
    monkeypatch.setitem(
        sys.modules,
        "official_judge_match",
        types.SimpleNamespace(load_initdata=fake_load_initdata, run_match=fake_run_match),
    )
    monkeypatch.setitem(
        sys.modules,
        "official_fan",
        types.SimpleNamespace(OfficialFanChecker=FakeFanChecker),
    )

    raw_out = tmp_path / "selfplay_raw.jsonl"
    fan_out = tmp_path / "fan_items.jsonl"
    summary_out = tmp_path / "summary.json"
    args = argparse.Namespace(
        checkpoint="checkpoint.pt",
        raw=str(tmp_path / "initdata.jsonl"),
        games=1,
        offset=0,
        max_turns=8,
        judge="judge.exe",
        fan_checker=None,
        out_raw=str(raw_out),
        out_fan_items=str(fan_out),
        summary_out=str(summary_out),
        device="cpu",
        require_encoding_version="tjong_cit2_12298_v3_hidden_concealed_kong",
        require_paper_config=True,
    )

    summary = collect_selfplay_module.collect(args)

    assert summary["policy_interface"] == "botzone_json_replay_inprocess"
    assert json.loads(summary_out.read_text(encoding="utf-8"))["policy_interface"] == (
        "botzone_json_replay_inprocess"
    )
    record = json.loads(raw_out.read_text(encoding="utf-8").splitlines()[0])
    assert record["logs"][-1]["0"]["response"] == "PLAY W5"
    assert calls[2]["payload"] == {
        "requests": [
            "0 0 1",
            "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
            "2 W5",
        ],
        "responses": ["PASS", "PASS"],
    }
    assert calls[2]["fan_checker"] == "default-fan-checker"


def test_selfplay_summary_reports_deal_in_and_fans():
    summary = collect_selfplay_module.summarize(
        [
            {
                "scores": {"0": 24, "1": -8, "2": -4, "3": -12},
                "turn_count": 18,
                "final_output": {
                    "display": {
                        "action": "HU",
                        "player": 0,
                        "fanCnt": 16,
                        "fan": [
                            {"name": "清一色", "value": 24, "cnt": 1},
                            {"name": "门前清", "value": 2, "cnt": 1},
                        ],
                    }
                },
            },
            {
                "scores": {"0": 0, "1": 0, "2": 0, "3": 0},
                "turn_count": 24,
                "final_output": {"display": {"action": "HUANG"}},
            },
        ]
    )

    assert summary["games"] == 2
    assert summary["hu_rate"] == 0.5
    assert summary["huang_rate"] == 0.5
    assert summary["hu_turn_average"] == 18.0
    assert summary["hu_fan_average"] == 16.0
    assert summary["per_player"]["0"]["hu_rate"] == 0.5
    assert summary["per_player"]["3"]["deal_in_rate"] == 0.5
    assert summary["fan_breakdown"]["清一色"] == {
        "hu_hands": 1,
        "occurrences": 1.0,
        "score_total": 24.0,
    }


def test_platform_wrapper_verification_reports_cross_platform_and_replay_checks(monkeypatch, tmp_path):
    class FakePredictor:
        requires_botzone_history = True
        kind = "legal_action_ranker"

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            return "PLAY W5"

    def fake_audit_botzone_replay(**kwargs):
        return {
            "states": 8,
            "error_count": 0,
            "exact_accuracy": 0.75,
            "family_accuracy": 1.0,
        }

    monkeypatch.setattr(verify_platform_wrappers, "TjongCheckpointPredictor", lambda *args, **kwargs: FakePredictor())
    monkeypatch.setattr(verify_platform_wrappers, "audit_botzone_replay", fake_audit_botzone_replay)

    summary = verify_platform_wrappers.run_platform_wrapper_verification(
        checkpoint=tmp_path / "checkpoint.pt",
        raw_path=tmp_path / "raw.jsonl",
        min_states=8,
        min_exact_accuracy=0.7,
        min_family_accuracy=0.99,
    )

    assert summary["ok"] is True
    assert summary["checks"] == {
        "tziakcha_matches_botzone_draw": True,
        "illegal_draw_fallback_matches_botzone": True,
        "illegal_reaction_fallback_matches_botzone": True,
        "training_replay_no_errors": True,
        "training_replay_min_states": True,
        "training_replay_min_exact_accuracy": True,
        "training_replay_min_family_accuracy": True,
    }
    assert summary["botzone_tziakcha_equivalence"]["ok"] is True
    assert summary["illegal_draw_fallback"]["ok"] is True
    assert summary["illegal_reaction_fallback"]["ok"] is True


def test_platform_wrapper_verification_honors_optional_accuracy_thresholds(monkeypatch, tmp_path):
    class FakePredictor:
        requires_botzone_history = True
        kind = "legal_action_ranker"

        def predict_legal_response(self, input_text, hand, player_id, request, candidates):
            return "PLAY W5"

    monkeypatch.setattr(verify_platform_wrappers, "TjongCheckpointPredictor", lambda *args, **kwargs: FakePredictor())
    monkeypatch.setattr(
        verify_platform_wrappers,
        "audit_botzone_replay",
        lambda **kwargs: {
            "states": 8,
            "error_count": 0,
            "exact_accuracy": 0.75,
            "family_accuracy": 1.0,
        },
    )

    summary = verify_platform_wrappers.run_platform_wrapper_verification(
        checkpoint=tmp_path / "checkpoint.pt",
        raw_path=tmp_path / "raw.jsonl",
        min_exact_accuracy=0.9,
    )

    assert summary["ok"] is False
    assert summary["checks"]["training_replay_min_exact_accuracy"] is False


def test_platform_wrapper_gate_manifest_is_cpu_only_and_avoids_reserved_nodes():
    manifest = (
        ROOT.parent
        / "k8s"
        / "tjong-platform-wrapper-gate-cpu-20260612a.yaml"
    ).read_text(encoding="utf-8")

    assert "tjong_replication.verify_platform_wrappers" in manifest
    assert "--fail-on-error" in manifest
    assert "nvidia.com/gpu" not in manifest
    assert "nautilus.io/reservation" not in manifest
    assert "csu-tide" not in manifest
    assert "rci-tide-gpu-11.sdsu.edu" in manifest
    assert "TJONG_PLATFORM_CHECKPOINT" in manifest
    assert "TJONG_PLATFORM_RAW" in manifest
    assert "TJONG_BRANCH" in manifest
    assert "downloaded_branch_zip" in manifest
    assert "DATA_ROOT=/data/idl/mcr_agent_transformer_20260527_0029/mcr_transformer_sync_20260527" in manifest
    assert "scripts/tjong_botzone_json_policy_bot.py" in manifest


def test_supervised_metrics_include_epoch_breakdowns():
    action_labels = torch.tensor(
        [
            ACTION_TO_INDEX["HU"],
            ACTION_TO_INDEX["DISCARD"],
            ACTION_TO_INDEX["CHOW"],
            ACTION_TO_INDEX["PONG"],
            ACTION_TO_INDEX["MINGKONG"],
            ACTION_TO_INDEX["BUKONG"],
            ACTION_TO_INDEX["ANKONG"],
            ACTION_TO_INDEX["PASS"],
        ],
        dtype=torch.long,
    )
    claim_labels = torch.tensor(
        [
            0,
            0,
            flatten_claim("CHOW", 0),
            flatten_claim("PONG", 0),
            flatten_claim("MINGKONG", 0),
            flatten_claim("BUKONG", 0),
            flatten_claim("ANKONG", 0),
            0,
        ],
        dtype=torch.long,
    )
    discard_labels = torch.zeros(len(action_labels), dtype=torch.long)
    action_logits = torch.full((len(action_labels), len(ACTION_NAMES)), -10.0)
    claim_logits = torch.full((len(action_labels), CLAIM_SIZE), -10.0)
    discard_logits = torch.full((len(action_labels), 34), -10.0)
    for row, label in enumerate(action_labels.tolist()):
        action_logits[row, label] = 10.0
    for row, label in enumerate(claim_labels.tolist()):
        claim_logits[row, label] = 10.0
    discard_logits[:, 0] = 10.0

    metrics = finalize_metric_sums(
        batch_metric_sums(
            {
                "action_logits": action_logits,
                "claim_logits": claim_logits,
                "discard_logits": discard_logits,
            },
            (action_labels, claim_labels, discard_labels),
        )
    )

    assert metrics["action_breakdown"]["HU"]["count"] == 1
    assert metrics["action_breakdown"]["HU"]["accuracy"] == 1.0
    assert metrics["action_breakdown"]["DISCARD"]["count"] == 1
    assert metrics["claim_breakdown"]["CHOW"]["accuracy"] == 1.0
    assert metrics["claim_breakdown"]["PONG"]["accuracy"] == 1.0
    assert metrics["claim_breakdown"]["MINGKONG"]["accuracy"] == 1.0
    assert metrics["claim_breakdown"]["BUKONG"]["accuracy"] == 1.0
    assert metrics["claim_breakdown"]["ANKONG"]["accuracy"] == 1.0
    assert metrics["discard_tile_breakdown"]["W1"]["accuracy"] == 1.0


def test_supervised_safe_grad_clip_uses_manual_norm():
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    parameter.grad = torch.tensor([3.0, 4.0])

    total_norm = safe_clip_grad_norm_([parameter], 0.5)

    assert abs(total_norm - 5.0) < 1e-6
    assert abs(float(parameter.grad.norm().item()) - 0.5) < 1e-6


def test_supervised_label_validation_rejects_selected_sentinel():
    labels = (
        torch.tensor([ACTION_TO_INDEX["DISCARD"]]),
        torch.tensor([0]),
        torch.tensor([255]),
    )

    try:
        validate_supervised_labels(labels)
    except ValueError as exc:
        assert "discard_selected" in str(exc)
        assert "invalid_count" in str(exc)
    else:
        raise AssertionError("expected selected discard sentinel to be rejected before CE")


def test_supervised_nonfinite_context_pinpoints_head():
    labels = (
        torch.tensor([ACTION_TO_INDEX["DISCARD"]]),
        torch.tensor([0]),
        torch.tensor([tile_id("W1")]),
    )
    outputs = {
        "action_logits": torch.zeros(1, len(ACTION_NAMES)),
        "claim_logits": torch.zeros(1, CLAIM_SIZE),
        "discard_logits": torch.zeros(1, 34),
    }
    outputs["discard_logits"][0, 0] = float("nan")

    components = supervised_loss_components(outputs, labels)
    context = nonfinite_supervised_context(outputs, labels, components)

    assert not torch.isfinite(components["total"]).item()
    assert context["loss_components"]["discard"]["finite"] is False
    assert context["output_tensors"]["discard_logits"]["nan_count"] == 1
    assert context["label_ranges"]["discard_selected"]["invalid_count"] == 0


def test_supervised_training_freezes_value_branch_for_policy_ce():
    model = TjongNetwork(TjongConfig(d_model=32, n_heads=4, ffn_dim=64, dropout=0.0))

    frozen = freeze_supervised_value_parameters(model)

    assert frozen > 0
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("value_inner.") or name.startswith("value_head.")
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not (name.startswith("value_inner.") or name.startswith("value_head."))
    )


def _supervised_contract_payload(
    action_labels: list[int],
    claim_labels: list[int],
    discard_labels: list[int],
) -> dict:
    n = len(action_labels)
    game = torch.zeros(n, 4, 24)
    sub_game = torch.zeros(n, 4, 24)
    sub_visible = torch.zeros(n, 4, 22, 34)
    for row, action in enumerate(action_labels):
        game[row, -1, 16 + action] = 1.0
        sub_game[row, -1, 16 + action] = 1.0
        if action == ACTION_TO_INDEX["DISCARD"]:
            sub_visible[row, -1, 0, discard_labels[row]] = 1.0
    return {
        "visible_tiles": torch.zeros(n, 4, 22, 34),
        "game_features": game,
        "previous_actions": torch.zeros(n, 4, dtype=torch.long),
        "sub_visible_tiles": sub_visible,
        "sub_game_features": sub_game,
        "sub_previous_actions": torch.zeros(n, 4, dtype=torch.long),
        "hidden_tiles": torch.zeros(n, 5, 34),
        "action_label": torch.tensor(action_labels, dtype=torch.long),
        "claim_label": torch.tensor(claim_labels, dtype=torch.long),
        "discard_label": torch.tensor(discard_labels, dtype=torch.long),
        "encoding_schema": tensor_encoding_schema(),
    }


def test_supervised_dataset_contract_accepts_valid_hierarchical_labels():
    payload = _supervised_contract_payload(
        [
            ACTION_TO_INDEX["DISCARD"],
            ACTION_TO_INDEX["CHOW"],
            ACTION_TO_INDEX["PONG"],
            ACTION_TO_INDEX["MINGKONG"],
            ACTION_TO_INDEX["BUKONG"],
            ACTION_TO_INDEX["ANKONG"],
        ],
        [
            0,
            flatten_claim("CHOW", 0),
            flatten_claim("PONG", 0),
            flatten_claim("MINGKONG", 0),
            flatten_claim("BUKONG", 0),
            flatten_claim("ANKONG", 0),
        ],
        [tile_id("W1"), 0, 0, 0, 0, 0],
    )

    summary = validate_supervised_dataset_contract(payload)

    assert summary["passed"] is True
    assert summary["issue_counts"] == {}


def test_supervised_dataset_contract_rejects_claim_family_mismatch():
    payload = _supervised_contract_payload(
        [ACTION_TO_INDEX["PONG"]],
        [flatten_claim("CHOW", 0)],
        [0],
    )

    try:
        validate_supervised_dataset_contract(payload)
    except ValueError as exc:
        assert "claim_family_mismatch:PONG" in str(exc)
    else:
        raise AssertionError("expected PONG row with CHOW claim label to fail preflight")


def test_supervised_dataset_contract_rejects_discard_missing_from_sub_hand():
    payload = _supervised_contract_payload(
        [ACTION_TO_INDEX["DISCARD"]],
        [0],
        [tile_id("W1")],
    )
    payload["sub_visible_tiles"].zero_()

    try:
        validate_supervised_dataset_contract(payload)
    except ValueError as exc:
        assert "active_discard_not_in_sub_hand" in str(exc)
    else:
        raise AssertionError("expected discard target absent from sub-hand to fail preflight")


def test_supervised_paths_skip_value_hidden_tiles(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        tensor_path = tmp_path / "tiny_tensor.pt"
        torch.save(
            {
                "visible_tiles": torch.zeros(2, 4, 22, 34),
                "game_features": torch.zeros(2, 4, 24),
                "hidden_tiles": torch.ones(2, 5, 34),
                "action_label": torch.tensor([ACTION_TO_INDEX["DISCARD"], ACTION_TO_INDEX["PASS"]]),
                "claim_label": torch.zeros(2, dtype=torch.long),
                "discard_label": torch.tensor([tile_id("W1"), 0]),
                "encoding_schema": tensor_encoding_schema(),
            },
            tensor_path,
        )
        original_forward = TjongNetwork.forward
        calls = []

        def wrapped_forward(self, *args, **kwargs):
            assert kwargs.get("hidden_tiles") is None
            calls.append(True)
            return original_forward(self, *args, **kwargs)

        monkeypatch.setattr(TjongNetwork, "forward", wrapped_forward)
        dataset = load_tensor_dataset(tensor_path, expected_encoding_version=TENSOR_ENCODING_VERSION)
        model = TjongNetwork(TjongConfig(d_model=32, n_heads=4, ffn_dim=64, dropout=0.0))

        evaluate_model(model, dataset, device=torch.device("cpu"), batch_size=2, num_workers=0)

        args = argparse.Namespace(
            train_pt=tensor_path,
            checkpoint_out=tmp_path / "supervised.pt",
            metrics_out=tmp_path / "metrics.json",
            epochs=1,
            batch_size=2,
            lr=1e-4,
            d_model=32,
            n_heads=4,
            ffn_dim=64,
            dropout=0.0,
            num_workers=0,
            device="cpu",
            data_parallel=False,
            require_encoding_version=TENSOR_ENCODING_VERSION,
            checkpoint_every_epochs=0,
            checkpoint_dir=None,
            resume_checkpoint=None,
            metrics_jsonl=None,
            grad_clip=0.5,
            force_math_sdp=False,
            fail_on_nonfinite=True,
            cuda_sync_debug=False,
            max_steps=1,
        )
        train_supervised(args)

        assert calls


def test_policy_outer_transformer_uses_causal_memory_mask():
    config = TjongConfig(d_model=32, n_heads=4, ffn_dim=64, dropout=0.0)
    backbone = TITPolicyBackbone(config)
    captured = {}
    original_forward = backbone.outer.forward

    def wrapped_forward(src, mask=None, **kwargs):
        captured["mask"] = mask.detach().cpu().clone()
        return original_forward(src, mask=mask, **kwargs)

    backbone.outer.forward = wrapped_forward
    visible = torch.zeros(1, config.memory_len, 22, 34)
    game = torch.zeros(1, config.memory_len, 24)

    with torch.no_grad():
        backbone(visible, game)

    mask = captured["mask"]
    assert mask.shape == (config.memory_len, config.memory_len)
    assert torch.isneginf(mask[0, 1])
    assert torch.isneginf(mask[0, -1])
    assert mask[1, 0].item() == 0.0
    assert mask.diag().eq(0.0).all()


def test_policy_inner_transformer_consumes_game_features():
    config = TjongConfig(d_model=128, n_heads=4, ffn_dim=256, dropout=0.0)
    model = TjongNetwork(config)
    model.eval()
    visible = torch.zeros(1, config.memory_len, 22, 34)
    game = torch.zeros(1, config.memory_len, 24)
    hidden = torch.zeros(1, 5, 34)

    with torch.no_grad():
        base = model(visible_tiles=visible, game_features=game, hidden_tiles=hidden)["action_logits"]
        game[:, -1, 0] = 1.0
        changed = model(visible_tiles=visible, game_features=game, hidden_tiles=hidden)["action_logits"]

    assert not torch.allclose(base, changed)


def test_value_network_uses_global_hidden_features_only():
    config = TjongConfig(d_model=128, n_heads=4, ffn_dim=256, dropout=0.0)
    model = TjongNetwork(config)
    model.eval()
    visible = torch.zeros(1, config.memory_len, 22, 34)
    game = torch.zeros(1, config.memory_len, 24)
    hidden = torch.zeros(1, 5, 34)

    with torch.no_grad():
        base = model(visible_tiles=visible, game_features=game, hidden_tiles=hidden)["value"]
        visible[:, -1, 0, 0] = 1.0
        game[:, -1, 0] = 1.0
        same_hidden = model(visible_tiles=visible, game_features=game, hidden_tiles=hidden)["value"]
        hidden[:, 0, 0] = 1.0
        changed_hidden = model(visible_tiles=visible, game_features=game, hidden_tiles=hidden)["value"]

    assert torch.allclose(base, same_hidden)
    assert not torch.allclose(base, changed_hidden)


def test_hidden_global_matrix_encodes_concealed_kong_tiles():
    state = ReplayState()
    state.hands[0].update(["W1", "W1", "W1", "W1"])
    state.apply_display({"action": "GANG", "player": 0, "tile": "W1"})

    _, _, hidden = state.encode(1, [1.0] + [0.0] * 7)

    assert tensor_encoding_schema()["version"] == TENSOR_ENCODING_VERSION
    assert HIDDEN_TILE_ROW_NAMES[3] == "concealed_kongs"
    assert hidden[3, tile_id("W1")].item() == 4.0
    assert hidden[4].sum().item() == 0.0


def test_runtime_policy_hidden_matrix_matches_tensor_schema():
    _, _, hidden = encode_runtime_state(
        input_text="\n".join(
            [
                "REQ 0 0 0",
                "REQ 2 W1",
                "REQ 3 0 GANG W1",
            ]
        ),
        hand=Counter({"W2": 1}),
        player_id=0,
        action_mask=[1.0] + [0.0] * 7,
    )

    assert runtime_hidden_schema_rows() == HIDDEN_TILE_ROW_NAMES
    assert hidden[3, tile_id("W1")].item() == 4.0
    assert hidden[4, tile_id("W1")].item() == 0.0
    assert hidden[4, tile_id("W2")].item() == 3.0


def test_fan_backward_score_math():
    hand_private = torch.zeros(34)
    hand_private[0] = 2
    hand_private[1] = 1
    score = calculate_score(hand_private, torch.zeros(34), [FanItem(score=6, tiles=(0, 1))])
    assert score[0].item() == 1.5
    assert score[1].item() == 3.0
    hand_hu = hand_private.clone()
    hand_now = torch.zeros(34)
    hand_now[0] = 1
    assert winning_reward(hand_hu, hand_now, score) == 1.5


def test_structural_fan_attribution_from_display_names():
    hand_hu = torch.zeros(34)
    tiles = "W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 B1 B2".split()
    for tile in tiles:
        hand_hu[tile_id(tile)] += 1

    attributed = attribute_display_fans(
        {"fan": [{"name": "无字", "value": 1, "cnt": 1}, {"name": "自摸", "value": 1, "cnt": 1}]},
        hand_hu=hand_hu,
        winner=0,
        winning_tile=tile_id("B2"),
        prevailing_wind=0,
    )

    assert attributed.unsupported == ()
    assert attributed.fans[0] == FanItem(score=1.0, tiles=tuple(tile_id(tile) for tile in tiles))
    assert attributed.fans[1] == FanItem(score=1.0, tiles=(tile_id("B2"),))


def test_validate_paper_corpus_records_count_and_shape():
    record = {
        "initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)},
        "logs": [{"output": {"display": {"action": "HU"}}}],
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        path = Path(tmp_dir) / "tiny.jsonl"
        summary_path = Path(tmp_dir) / "summary.json"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        summary = validate_corpus(path, expected_records=1, summary_out=summary_path)

        assert summary["valid_for_paper_supervised"]
        assert summary["records"] == 1
        assert summary["records_match_expected"]
        assert not summary["records_match_original_paper_count"]
        assert not summary["records_match_paper"]
        assert summary["expected_records_source"] == "override_same_format_dataset"
        assert summary["terminal_actions"] == {"HU": 1}
        assert json.loads(summary_path.read_text(encoding="utf-8"))["records"] == 1


def test_validate_accepts_converted_botzone_logs_without_initdata():
    record = {
        "match_id": "converted",
        "scores": {"0": 0, "1": 0, "2": 0, "3": 0},
        "logs": [
            {
                "output": {
                    "content": {"0": "0 0 0", "1": "0 1 0", "2": "0 2 0", "3": "0 3 0"},
                    "display": {"action": "INIT"},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "1": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "2": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "3": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                    },
                    "display": {"action": "DEAL"},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
        ],
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        path = Path(tmp_dir) / "converted.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        summary = validate_corpus(path, expected_records=1)

        assert summary["valid_for_paper_supervised"]
        assert summary["error_count"] == 0


def test_merge_botzone_corpora_dedupes_match_ids():
    record_a = {
        "match_id": "same",
        "initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)},
        "logs": [{"output": {"display": {"action": "HU"}}}],
    }
    record_b = {
        "match_id": "other",
        "initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)},
        "logs": [{"output": {"display": {"action": "HUANG"}}}],
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp = Path(tmp_dir)
        first = tmp / "first.jsonl"
        second = tmp / "second.jsonl"
        out = tmp / "merged.jsonl"
        summary_out = tmp / "summary.json"
        first.write_text(json.dumps(record_a) + "\n", encoding="utf-8")
        second.write_text(json.dumps(record_a) + "\n" + json.dumps(record_b) + "\n", encoding="utf-8")

        summary = merge_botzone_corpora(
            inputs=[first, second],
            out_path=out,
            summary_out=summary_out,
            min_written=2,
            fail_on_error=True,
        )

        assert summary["records_seen"] == 3
        assert summary["records_written"] == 2
        assert summary["duplicates_skipped"] == 1
        assert summary["error_count"] == 0
        assert summary["terminal_actions"] == {"HU": 1, "HUANG": 1}
        assert len(out.read_text(encoding="utf-8").splitlines()) == 2
        assert json.loads(summary_out.read_text(encoding="utf-8"))["records_written"] == 2


def test_tziakcha_botzone_coverage_audit_requires_human32_source():
    def write_jsonl(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        root = Path(tmp_dir)
        write_jsonl(root / "data/raw/tziakcha_human_records_32.jsonl", [{"id": "h32"}])
        write_jsonl(root / "data/raw/tziakcha_human_botzone_raw_32.jsonl", [{"match_id": "h32"}])
        write_jsonl(root / "data/raw/tziakcha_human_records_256.jsonl", [{"id": "h256"}])
        write_jsonl(root / "data/raw/tziakcha_human_records_512.jsonl", [{"id": "h512"}])
        write_jsonl(root / "data/raw/tziakcha_human_botzone_raw_517.jsonl", [{"match_id": "h256"}, {"match_id": "h512"}])
        write_jsonl(root / "data/raw/tziakcha_chaga0208_records.jsonl", [{"id": "c"}])
        write_jsonl(root / "data/raw/tziakcha_chaga0208_botzone_raw.jsonl", [{"match_id": "c"}])
        validation = root / "validation.json"
        validation.write_text(
            json.dumps({"valid_for_paper_supervised": True, "error_count": 0, "records": 4}),
            encoding="utf-8",
        )

        missing_merge = root / "missing_merge.json"
        missing_merge.write_text(
            json.dumps(
                {
                    "inputs": [
                        {"path": "data/raw/tziakcha_all_plus_live_elo2300_botzone.jsonl", "records_seen": 1, "records_written": 1},
                        {"path": "data/raw/tziakcha_human_botzone_raw_517.jsonl", "records_seen": 2, "records_written": 2},
                        {"path": "data/raw/tziakcha_chaga0208_botzone_raw.jsonl", "records_seen": 1, "records_written": 1},
                    ],
                    "records_written": 4,
                    "duplicates_skipped": 0,
                    "error_count": 0,
                }
            ),
            encoding="utf-8",
        )
        complete_merge = root / "complete_merge.json"
        complete_merge.write_text(
            json.dumps(
                {
                    "inputs": [
                        {"path": "data/raw/tziakcha_all_plus_live_elo2300_botzone.jsonl", "records_seen": 1, "records_written": 1},
                        {"path": "data/raw/tziakcha_human_botzone_raw_32.jsonl", "records_seen": 1, "records_written": 1},
                        {"path": "data/raw/tziakcha_human_botzone_raw_517.jsonl", "records_seen": 2, "records_written": 2},
                        {"path": "data/raw/tziakcha_chaga0208_botzone_raw.jsonl", "records_seen": 1, "records_written": 1},
                    ],
                    "records_written": 5,
                    "duplicates_skipped": 0,
                    "error_count": 0,
                }
            ),
            encoding="utf-8",
        )

        missing = audit_coverage(repo_root=root, merge_summary=missing_merge, validation_summary=validation)
        complete = audit_coverage(repo_root=root, merge_summary=complete_merge, validation_summary=validation)

        assert not missing["passed"]
        assert not missing["required_real_botzone_inputs"][1]["present_in_merge"]
        assert complete["passed"]


def test_tziakcha_converter_writes_botzone_like_logs():
    record = {
        "id": "tziakcha-synthetic",
        "belongs": "unit",
        "script": "<Decoded>",
        "step": {
            "w": "".join(f"{tile:02x}" for tile in range(144)),
            "d": 0,
            "i": 0,
            "a": [[0, 0, 0]],
            "s": [0, 0, 0, 0],
        },
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw = tmp_path / "tziakcha.jsonl"
        converted = tmp_path / "converted.jsonl"
        summary_path = tmp_path / "summary.json"
        raw.write_text(json.dumps(record) + "\n", encoding="utf-8")

        summary = convert_tziakcha_file(
            in_path=raw,
            out_path=converted,
            summary_out=summary_path,
            min_written=1,
            fail_on_error=True,
        )
        converted_record = json.loads(converted.read_text(encoding="utf-8").splitlines()[0])

        assert summary["records_written"] == 1
        assert summary["min_written_met"]
        assert converted_record["source"] == "tziakcha_record_miner"
        assert converted_record["logs"][0]["output"]["display"]["action"] == "INIT"
        assert converted_record["logs"][2]["output"]["display"]["action"] == "DEAL"

        initdata = tmp_path / "initdata.jsonl"
        initdata_summary = convert_tziakcha_initdata_file(
            in_path=converted,
            out_path=initdata,
            summary_out=tmp_path / "initdata_summary.json",
            min_written=1,
            fail_on_error=True,
        )
        initdata_record = json.loads(initdata.read_text(encoding="utf-8").splitlines()[0])

        assert initdata_summary["records_written"] == 1
        assert initdata_summary["target_format"] == "official_judge_initdata_jsonl"
        assert len(initdata_record["initdata"]["walltiles"].split()) == 144
        assert validate_initdata_corpus(initdata, expected_records=1)["valid_for_selfplay_initdata"]


def test_tziakcha_initdata_converter_random_fallback_for_overlong_replay():
    deal_content = {
        str(player): "1 0 0 0 0 " + " ".join(TILE_NAMES[:13])
        for player in range(4)
    }
    logs = [
        {"output": {"content": {"0": "0 0 0", "1": "0 1 0", "2": "0 2 0", "3": "0 3 0"}, "display": {"action": "INIT"}}},
        {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
        {"output": {"content": deal_content, "display": {"action": "DEAL"}}},
        {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
    ]
    for _ in range(24):
        logs.extend(
            [
                {"output": {"content": {"0": "2 W1"}, "display": {"action": "DRAW", "player": 0, "tile": "W1"}}},
                {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            ]
        )
    record = {"match_id": "overlong", "logs": logs}
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw = tmp_path / "overlong.jsonl"
        out = tmp_path / "initdata.jsonl"
        raw.write_text(json.dumps(record) + "\n", encoding="utf-8")

        summary = convert_tziakcha_initdata_file(
            in_path=raw,
            out_path=out,
            min_written=1,
            fail_on_error=True,
            allow_random_fallback=True,
        )

        assert summary["records_written"] == 1
        assert summary["fallback_random_wall_records"] == 1
        assert validate_initdata_corpus(out, expected_records=1)["valid_for_selfplay_initdata"]


def test_validate_paper_corpus_rejects_wrong_count():
    record = {
        "initdata": {"quan": 0},
        "logs": [{"output": {"display": {"action": "HUANG"}}}],
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        path = Path(tmp_dir) / "tiny.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        try:
            validate_corpus(path, expected_records=2)
        except ValueError as exc:
            assert '"records_match_paper": false' in str(exc)
        else:
            raise AssertionError("expected wrong corpus count to fail")


def test_tensorize_file_embeds_corpus_validation_summary():
    record = {
        "match_id": "validation-embed",
        "scores": {"0": 0, "1": 0, "2": 0, "3": 0},
        "initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)},
        "logs": [
            {
                "output": {
                    "content": {"0": "0 0 0", "1": "0 1 0", "2": "0 2 0", "3": "0 3 0"},
                    "display": {"action": "INIT", "quan": 0, "tileCnt": [21, 21, 21, 21]},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "1": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "2": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "3": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                    },
                    "display": {
                        "action": "DEAL",
                        "hand": [
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                        ],
                        "tileCnt": [21, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
        ],
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_path = tmp_path / "raw.jsonl"
        tensor_path = tmp_path / "tensor.pt"
        validation_path = tmp_path / "validation.json"
        raw_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        validation = validate_corpus(raw_path, expected_records=1, summary_out=validation_path)

        summary = tensorize_file(raw_path, tensor_path, corpus_validation=validation_path)
        tensor = torch.load(tensor_path, map_location="cpu")

        assert summary["corpus_validation"]["sha256"] == validation["sha256"]
        assert summary["encoding_schema"]["version"] == TENSOR_ENCODING_VERSION
        assert tensor["corpus_validation"]["records"] == 1
        assert tensor["encoding_schema"]["hidden_tile_rows"] == list(HIDDEN_TILE_ROW_NAMES)
        assert len(load_tensor_dataset(tensor_path, expected_encoding_version=TENSOR_ENCODING_VERSION)) == 0


def test_streaming_tensorize_compacts_metadata_and_keeps_hu_labels():
    record = {
        "match_id": "streaming-hu",
        "scores": {"0": 24, "1": -8, "2": -8, "3": -8},
        "initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)},
        "logs": [
            {
                "output": {
                    "content": {"0": "0 0 0", "1": "0 1 0", "2": "0 2 0", "3": "0 3 0"},
                    "display": {"action": "INIT", "quan": 0, "tileCnt": [21, 21, 21, 21]},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "1": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "2": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "3": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                    },
                    "display": {
                        "action": "DEAL",
                        "hand": [
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                        ],
                        "tileCnt": [21, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {"0": "2 W1", "1": "3 0 DRAW W1", "2": "3 0 DRAW W1", "3": "3 0 DRAW W1"},
                    "display": {
                        "action": "DRAW",
                        "player": 0,
                        "tile": "W1",
                        "canHu": [8, -4, -4, -4],
                        "baseFanCnt": 8,
                        "fanCnt": 8,
                        "tileCnt": [20, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "HU"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {"0": 24, "1": -8, "2": -8, "3": -8},
                    "display": {"action": "HU", "player": 0, "score": [24, -8, -8, -8], "fanCnt": 8},
                }
            },
        ],
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_path = tmp_path / "raw.jsonl"
        tensor_path = tmp_path / "tensor.pt"
        raw_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        summary = tensorize_file(raw_path, tensor_path, streaming=True, compact_metadata=True)
        tensor = torch.load(tensor_path, map_location="cpu")

        assert summary["streaming"] is True
        assert summary["compact_metadata"] is True
        assert summary["stats"]["action:HU"] == 1
        assert "label_outside_mask:HU" not in summary["stats"]
        assert int(tensor["action_label"][0]) == ACTION_TO_INDEX["HU"]
        assert tensor["metadata"]["format"] == "compact_v1"
        assert metadata_column(tensor["metadata"], "match_id") == ["streaming-hu"]
        returns = reward_to_go(torch.tensor([3.0]), metadata=tensor["metadata"], gamma=1.0)
        assert torch.equal(returns, torch.tensor([3.0]))


def test_tensor_loader_rejects_missing_paper_encoding_version():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tensor_path = Path(tmp_dir) / "legacy_tensor.pt"
        torch.save(
            {
                "visible_tiles": torch.zeros(1, 4, 22, 34),
                "game_features": torch.zeros(1, 4, 24),
                "action_label": torch.zeros(1, dtype=torch.long),
                "claim_label": torch.zeros(1, dtype=torch.long),
                "discard_label": torch.zeros(1, dtype=torch.long),
            },
            tensor_path,
        )

        try:
            load_tensor_dataset(tensor_path, expected_encoding_version=TENSOR_ENCODING_VERSION)
        except ValueError as exc:
            assert "tensor encoding version mismatch" in str(exc)
        else:
            raise AssertionError("expected legacy tensor without encoding schema to fail")


def test_original_supervised_trainer_writes_final_checkpoint_only():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        tensor_path = tmp_path / "tiny_tensor.pt"
        final_checkpoint = tmp_path / "supervised.pt"
        metrics_path = tmp_path / "metrics.json"
        torch.save(
            {
                "visible_tiles": torch.zeros(2, 4, 22, 34),
                "game_features": torch.zeros(2, 4, 24),
                "action_label": torch.tensor([ACTION_TO_INDEX["DISCARD"], ACTION_TO_INDEX["PASS"]]),
                "claim_label": torch.zeros(2, dtype=torch.long),
                "discard_label": torch.tensor([tile_id("W1"), 0]),
                "encoding_schema": tensor_encoding_schema(),
            },
            tensor_path,
        )
        args = argparse.Namespace(
            train_pt=tensor_path,
            checkpoint_out=final_checkpoint,
            metrics_out=metrics_path,
            epochs=1,
            batch_size=2,
            lr=1e-4,
            d_model=32,
            n_heads=4,
            ffn_dim=64,
            dropout=0.0,
            num_workers=0,
            device="cpu",
            data_parallel=False,
            require_encoding_version=TENSOR_ENCODING_VERSION,
        )

        metrics = train_supervised(args)
        final_payload = torch.load(final_checkpoint, map_location="cpu")

        assert len(metrics["epochs"]) == 1
        assert final_payload["metrics"]["epochs"][-1]["epoch"] == 1
        assert "optimizer" not in final_payload
        assert "epoch" not in final_payload
        assert json.loads(metrics_path.read_text(encoding="utf-8"))["epochs"][-1]["epoch"] == 1


def test_supervised_trainer_rejects_nonpaper_config_when_required():
    args = argparse.Namespace(epochs=1, batch_size=1024, lr=1e-4, require_paper_config=True)
    try:
        validate_paper_supervised_args(args, TjongConfig())
    except ValueError as exc:
        assert "paper supervised training config mismatch" in str(exc)
        assert "epochs" in str(exc)
    else:
        raise AssertionError("expected paper supervised config gate to reject non-paper epoch count")

    good_args = argparse.Namespace(epochs=125, batch_size=1024, lr=1e-4, require_paper_config=True)
    validate_paper_supervised_args(good_args, TjongConfig())

    try:
        validate_paper_supervised_args(good_args, TjongConfig(d_model=32, n_heads=4, ffn_dim=64, dropout=0.0))
    except ValueError as exc:
        assert "paper supervised training config mismatch" in str(exc)
        assert "d_model" in str(exc)
    else:
        raise AssertionError("expected paper supervised config gate to reject non-paper model shape")


def test_ppo_trainer_rejects_nonpaper_constants_when_required():
    args = argparse.Namespace(batch_size=512, lr=1e-4, require_paper_config=True)
    try:
        validate_paper_ppo_args(args, PPOConfig(policy_clip=0.2, value_clip=0.3, grad_clip=0.5))
    except ValueError as exc:
        assert "paper PPO training config mismatch" in str(exc)
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("expected paper PPO config gate to reject non-paper batch size")

    good_args = argparse.Namespace(batch_size=1024, lr=1e-4, require_paper_config=True)
    validate_paper_ppo_args(good_args, PPOConfig(policy_clip=0.2, value_clip=0.3, grad_clip=0.5))

    try:
        validate_paper_ppo_args(good_args, PPOConfig(policy_clip=0.1, value_clip=0.3, grad_clip=0.5))
    except ValueError as exc:
        assert "paper PPO training config mismatch" in str(exc)
        assert "policy_clip" in str(exc)
    else:
        raise AssertionError("expected paper PPO config gate to reject non-paper policy clip")


def test_original_supervised_trainer_allows_tiny_nonpaper_smoke():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        tensor_path = tmp_path / "tiny_tensor.pt"
        torch.save(
            {
                "visible_tiles": torch.zeros(1, 4, 22, 34),
                "game_features": torch.zeros(1, 4, 24),
                "action_label": torch.tensor([ACTION_TO_INDEX["PASS"]]),
                "claim_label": torch.zeros(1, dtype=torch.long),
                "discard_label": torch.zeros(1, dtype=torch.long),
                "encoding_schema": tensor_encoding_schema(),
            },
            tensor_path,
        )
        args = argparse.Namespace(
            train_pt=tensor_path,
            checkpoint_out=tmp_path / "supervised.pt",
            metrics_out=tmp_path / "metrics.json",
            epochs=1,
            batch_size=1,
            lr=1e-4,
            d_model=32,
            n_heads=4,
            ffn_dim=64,
            dropout=0.0,
            num_workers=0,
            device="cpu",
            data_parallel=False,
            seed=0,
            require_encoding_version=TENSOR_ENCODING_VERSION,
            require_paper_config=False,
            checkpoint_every_epochs=0,
            checkpoint_dir=None,
            resume_checkpoint=None,
            metrics_jsonl=None,
        )

        metrics = train_supervised(args)

        assert metrics["epochs"][-1]["epoch"] == 1
        assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["epochs"][-1]["epoch"] == 1


def test_supervised_progress_plotter_reads_powershell_utf16_logs():
    log = (
        '+ python -m tjong_replication.train_supervised\n'
        '{"epoch": 1, "action_accuracy": 0.5, "claim_accuracy": 0.6, '
        '"discard_accuracy": 0.25, "decision_loss": 1.5}\n'
        '{"epoch": 2, "action_accuracy": 0.75, "claim_accuracy": 0.8, '
        '"discard_accuracy": 0.5, "decision_loss": 1.0}\n'
    )
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        log_path = Path(tmp_dir) / "progress.log"
        log_path.write_bytes(log.encode("utf-16"))

        epochs = load_epoch_metrics(log_path)
        summary = summarize_epochs(epochs)

        assert [epoch["epoch"] for epoch in epochs] == [1, 2]
        assert summary["latest_epoch"] == 2
        assert summary["latest_discard_accuracy"] == 0.5


def test_supervised_checkpoint_sweep_uses_same_eval_tensor_and_reports_best_metrics():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        tensor_path = tmp_path / "tiny_tensor.pt"
        metrics_path = tmp_path / "sweep.json"
        results_jsonl = tmp_path / "sweep.jsonl"
        torch.save(
            {
                "visible_tiles": torch.zeros(2, 4, 22, 34),
                "game_features": torch.zeros(2, 4, 24),
                "action_label": torch.tensor([ACTION_TO_INDEX["DISCARD"], ACTION_TO_INDEX["PASS"]]),
                "claim_label": torch.zeros(2, dtype=torch.long),
                "discard_label": torch.tensor([tile_id("W1"), 0]),
                "encoding_schema": tensor_encoding_schema(),
            },
            tensor_path,
        )
        checkpoints = []
        for epoch in (1, 2):
            config = TjongConfig(d_model=32, n_heads=4, ffn_dim=64, dropout=0.0)
            model = TjongNetwork(config)
            checkpoint_path = tmp_path / f"epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "epoch": epoch,
                    "encoding_schema": tensor_encoding_schema(),
                    "tensor_encoding_version": TENSOR_ENCODING_VERSION,
                },
                checkpoint_path,
            )
            checkpoints.append(str(checkpoint_path))
        args = argparse.Namespace(
            checkpoint=checkpoints,
            checkpoint_glob=[],
            eval_pt=str(tensor_path),
            metrics_out=str(metrics_path),
            results_jsonl=str(results_jsonl),
            batch_size=2,
            num_workers=0,
            device="cpu",
            require_encoding_version=TENSOR_ENCODING_VERSION,
            require_paper_config=False,
            seed=0,
            strict_deterministic=False,
        )

        summary = evaluate_checkpoints(args)

        assert summary["deterministic_eval"]
        assert summary["checkpoint_count"] == 2
        assert {result["checkpoint_epoch"] for result in summary["results"]} == {1, 2}
        assert summary["best_by_metric"]["action_accuracy"]["checkpoint_epoch"] in {1, 2}
        assert json.loads(metrics_path.read_text(encoding="utf-8"))["checkpoint_count"] == 2
        rows = [json.loads(line) for line in results_jsonl.read_text(encoding="utf-8").splitlines()]
        assert [row["checkpoint_epoch"] for row in rows] == [1, 2]


def test_checkpoint_loader_requires_paper_encoding_version():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "legacy_checkpoint.pt"
        model = TjongNetwork(TjongConfig())
        torch.save({"model": model.state_dict(), "config": model.config.__dict__}, checkpoint_path)

        try:
            load_model(
                checkpoint_path,
                torch.device("cpu"),
                expected_encoding_version=TENSOR_ENCODING_VERSION,
                require_paper_config=True,
            )
        except ValueError as exc:
            assert "checkpoint tensor encoding version mismatch" in str(exc)
        else:
            raise AssertionError("expected legacy checkpoint without encoding schema to fail")


def test_checkpoint_loader_rejects_non_paper_config_when_required():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "tiny_checkpoint.pt"
        config = TjongConfig(d_model=128, n_heads=4, ffn_dim=256, dropout=0.0)
        model = TjongNetwork(config)
        torch.save(
            {
                "model": model.state_dict(),
                "config": config.__dict__,
                "encoding_schema": tensor_encoding_schema(),
                "tensor_encoding_version": TENSOR_ENCODING_VERSION,
            },
            checkpoint_path,
        )

        try:
            load_model(
                checkpoint_path,
                torch.device("cpu"),
                expected_encoding_version=TENSOR_ENCODING_VERSION,
                require_paper_config=True,
            )
        except ValueError as exc:
            assert "checkpoint config mismatch" in str(exc)
            assert "d_model" in str(exc)
        else:
            raise AssertionError("expected non-paper checkpoint config to fail")


def test_checkpoint_loader_accepts_paper_config_with_encoding_metadata():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "paper_checkpoint.pt"
        model = TjongNetwork(TjongConfig())
        torch.save(
            {
                "model": model.state_dict(),
                "config": model.config.__dict__,
                "encoding_schema": tensor_encoding_schema(),
                "tensor_encoding_version": TENSOR_ENCODING_VERSION,
            },
            checkpoint_path,
        )

        loaded = load_model(
            checkpoint_path,
            torch.device("cpu"),
            expected_encoding_version=TENSOR_ENCODING_VERSION,
            require_paper_config=True,
        )

        assert loaded.parameter_count() == model.parameter_count()


def test_supervised_metric_sums_report_three_decision_heads():
    outputs = {
        "action_logits": torch.tensor(
            [
                [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            ]
        ),
        "claim_logits": torch.zeros(2, 199),
        "discard_logits": torch.zeros(2, 34),
    }
    outputs["claim_logits"][1, 3] = 5.0
    outputs["discard_logits"][0, 4] = 5.0
    labels = (
        torch.tensor([2, 4]),
        torch.tensor([0, 3]),
        torch.tensor([4, 0]),
    )

    metrics = finalize_metric_sums(batch_metric_sums(outputs, labels))

    assert metrics["action_accuracy"] == 1.0
    assert metrics["claim_accuracy"] == 1.0
    assert metrics["discard_accuracy"] == 1.0
    assert metrics["action_count"] == 2
    assert metrics["claim_count"] == 1
    assert metrics["discard_count"] == 1


def test_paper_metric_gate_accepts_reported_values():
    metrics = {
        **PAPER_REPORTED_SUPERVISED_METRICS,
        "paper_reported": dict(PAPER_REPORTED_SUPERVISED_METRICS),
    }

    summary = compare_supervised_metrics(metrics, tolerance=DEFAULT_PAPER_METRIC_TOLERANCE)

    assert summary["passed"]
    assert summary["comparisons"]["action_accuracy"]["absolute_delta"] == 0.0


def test_paper_metric_gate_requires_strict_deterministic_when_requested():
    metrics = {
        **PAPER_REPORTED_SUPERVISED_METRICS,
        "paper_reported": dict(PAPER_REPORTED_SUPERVISED_METRICS),
        "deterministic_eval": True,
        "strict_deterministic": False,
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        metrics_path = tmp_path / "metrics.json"
        summary_path = tmp_path / "summary.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

        try:
            verify_metrics_file(
                metrics_path,
                summary_out=summary_path,
                require_strict_deterministic=True,
            )
        except ValueError as exc:
            assert "metrics_determinism" in str(exc)
        else:
            raise AssertionError("expected non-strict deterministic metrics to fail paper gate")

        metrics["strict_deterministic"] = True
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        summary = verify_metrics_file(
            metrics_path,
            summary_out=summary_path,
            require_strict_deterministic=True,
        )

        assert summary["passed"]
        assert summary["metrics_determinism"]["required"] is True
        assert summary["metrics_determinism"]["passed"] is True


def test_paper_metric_gate_rejects_missed_target_and_writes_summary():
    metrics = {
        **PAPER_REPORTED_SUPERVISED_METRICS,
        "action_accuracy": PAPER_REPORTED_SUPERVISED_METRICS["action_accuracy"] - 0.001,
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        metrics_path = tmp_path / "metrics.json"
        summary_path = tmp_path / "summary.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

        try:
            verify_metrics_file(
                metrics_path,
                tolerance=DEFAULT_PAPER_METRIC_TOLERANCE,
                summary_out=summary_path,
            )
        except ValueError as exc:
            assert '"action_accuracy"' in str(exc)
        else:
            raise AssertionError("expected paper metric gate to fail")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert not summary["passed"]
        assert not summary["comparisons"]["action_accuracy"]["passed"]


def test_paper_compliance_gate_covers_ocr_architecture_and_pods():
    summary = verify_paper_compliance(ROOT.parent)

    assert summary["passed"]
    assert summary["model_parameters"] >= 14_000_000
    check_names = {check["name"] for check in summary["checks"]}
    assert "ocr_figure_transcript_present" in check_names
    assert "supervised_l40_manifest_matches_paper_training" in check_names
    assert "ppo_l40_manifest_matches_paper_rl_constants" in check_names


def test_paper_compliance_gate_writes_summary():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        summary_path = Path(tmp_dir) / "paper_compliance.json"

        summary = verify_paper_compliance_file(ROOT.parent, summary_out=summary_path)

        assert summary["passed"]
        assert json.loads(summary_path.read_text(encoding="utf-8"))["passed"]


def test_pipeline_status_reports_first_incomplete_stage_and_next_action():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        root = Path(tmp_dir)
        (root / "papers/tjong_cit2_12298/notes").mkdir(parents=True)
        (root / "papers/tjong_cit2_12298/ocr").mkdir(parents=True)
        (root / "papers/tjong_cit2_12298/notes/extraction.md").write_text("notes", encoding="utf-8")
        (root / "papers/tjong_cit2_12298/ocr/figure_ocr.md").write_text("ocr", encoding="utf-8")
        (root / "data/raw").mkdir(parents=True)
        (root / "runs").mkdir(parents=True)
        (root / "data/raw/tziakcha_all_sources_botzone_20260605b.jsonl").write_text("{}", encoding="utf-8")
        (root / "runs/tjong_validate_tziakcha_all_sources_botzone_20260605b.json").write_text(
            json.dumps({"records": 2, "error_count": 0, "valid_for_paper_supervised": True}),
            encoding="utf-8",
        )

        summary = audit_pipeline(root, expected_records=0)

        assert not summary["passed"]
        assert summary["stages"][0]["passed"]
        assert summary["stages"][1]["passed"]
        assert summary["stages"][0]["next_action"] == ""
        assert summary["stages"][1]["next_action"] == ""
        assert summary["first_incomplete_stage"] == "supervised_tensor"
        assert summary["expected_records"] == 2
        assert "tjong-tziakcha-all-sources-prep-cpu-20260605c.yaml" in summary["next_action"]


def test_hierarchical_ppo_log_prob_includes_tile_head():
    action_logits = torch.full((3, 8), -20.0)
    action_logits[0, 2] = 0.0
    action_logits[1, 4] = 0.0
    action_logits[2, 0] = 0.0
    claim_logits = torch.full((3, 199), -20.0)
    claim_logits[1, 7] = 0.0
    discard_logits = torch.full((3, 34), -20.0)
    discard_logits[0, 9] = 0.0
    outputs = {
        "action_logits": action_logits,
        "claim_logits": claim_logits,
        "discard_logits": discard_logits,
    }

    log_prob, entropy = hierarchical_log_prob_and_entropy(
        outputs,
        action_label=torch.tensor([2, 4, 0]),
        claim_label=torch.tensor([0, 7, 0]),
        discard_label=torch.tensor([9, 0, 0]),
    )

    assert torch.allclose(log_prob, torch.zeros_like(log_prob), atol=1e-5)
    assert torch.all(entropy >= 0)


def test_rollout_reward_to_go_groups_by_match_and_player():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    metadata = {
        "match_id": ["a", "a", "a", "a"],
        "player": [0, 1, 0, 1],
    }

    returns = reward_to_go(rewards, metadata=metadata, gamma=1.0)

    assert torch.equal(returns, torch.tensor([4.0, 6.0, 3.0, 4.0]))


def test_rollout_rewards_prefers_fan_backward_key_over_terminal_fallback():
    source = {
        "fan_backward_reward": torch.tensor([2.0, 3.0]),
        "value_target": torch.tensor([100.0, 100.0]),
    }

    rewards, source_name = rollout_rewards(source, reward_key="fan_backward_reward", terminal_scale=0.01)

    assert torch.equal(rewards, torch.tensor([2.0, 3.0]))
    assert source_name == "fan_backward_reward"


def test_rollout_rewards_labels_terminal_fallback():
    rewards, source_name = rollout_rewards(
        {"value_target": torch.tensor([100.0])},
        reward_key="fan_backward_reward",
        terminal_scale=0.01,
    )

    assert torch.equal(rewards, torch.tensor([1.0]))
    assert source_name == "value_target_terminal_fallback"


def test_rollout_rewards_can_require_fan_backward_key():
    try:
        rollout_rewards(
            {"value_target": torch.tensor([100.0])},
            reward_key="fan_backward_reward",
            terminal_scale=0.01,
            require_reward_key=True,
        )
    except ValueError as exc:
        assert "required reward key" in str(exc)
    else:
        raise AssertionError("expected paper PPO rollout builder to reject terminal fallback")


def test_ppo_trainer_can_require_fan_backward_rollout_source():
    validate_rollout_reward_source(
        {"rollout_summary": {"actual_reward_source": "fan_backward_reward"}},
        required_source="fan_backward_reward",
    )
    try:
        validate_rollout_reward_source(
            {"rollout_summary": {"actual_reward_source": "value_target_terminal_fallback"}},
            required_source="fan_backward_reward",
        )
    except ValueError as exc:
        assert "rollout reward source mismatch" in str(exc)
        assert "value_target_terminal_fallback" in str(exc)
    else:
        raise AssertionError("expected paper PPO trainer to reject non-fan-backward rollouts")


def test_ppo_trainer_writes_periodic_checkpoints_and_resumes():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        checkpoint_path = tmp_path / "init.pt"
        rollout_path = tmp_path / "rollout.pt"
        final_checkpoint = tmp_path / "ppo.pt"
        metrics_path = tmp_path / "ppo_metrics.json"
        metrics_jsonl = tmp_path / "ppo_metrics.jsonl"
        checkpoint_dir = tmp_path / "ppo_checkpoints"
        config = TjongConfig(d_model=32, n_heads=4, ffn_dim=64, dropout=0.0)
        model = TjongNetwork(config)
        torch.save(
            {
                "model": model.state_dict(),
                "config": config.__dict__,
                "encoding_schema": tensor_encoding_schema(),
                "tensor_encoding_version": TENSOR_ENCODING_VERSION,
            },
            checkpoint_path,
        )
        torch.save(
            {
                "visible_tiles": torch.zeros(2, 4, 22, 34),
                "game_features": torch.zeros(2, 4, 24),
                "hidden_tiles": torch.zeros(2, 5, 34),
                "action_label": torch.tensor([ACTION_TO_INDEX["DISCARD"], ACTION_TO_INDEX["PASS"]]),
                "claim_label": torch.zeros(2, dtype=torch.long),
                "discard_label": torch.tensor([tile_id("W1"), 0]),
                "old_log_prob": torch.zeros(2),
                "advantage": torch.ones(2),
                "returns": torch.zeros(2),
                "old_value": torch.zeros(2),
                "encoding_schema": tensor_encoding_schema(),
                "rollout_summary": {"actual_reward_source": "fan_backward_reward"},
            },
            rollout_path,
        )
        args = argparse.Namespace(
            init_checkpoint=checkpoint_path,
            rollout_pt=rollout_path,
            checkpoint_out=final_checkpoint,
            metrics_out=metrics_path,
            epochs=1,
            batch_size=2,
            lr=1e-4,
            policy_clip=0.2,
            value_clip=0.3,
            grad_clip=0.5,
            entropy_coef=0.0,
            value_coef=0.5,
            num_workers=0,
            device="cpu",
            data_parallel=False,
            require_encoding_version=TENSOR_ENCODING_VERSION,
            require_paper_config=False,
            require_rollout_reward_source="fan_backward_reward",
            checkpoint_every_epochs=1,
            checkpoint_dir=checkpoint_dir,
            resume_checkpoint=None,
            metrics_jsonl=metrics_jsonl,
        )

        metrics = train_ppo(args)
        first_latest = torch.load(checkpoint_dir / "latest.pt", map_location="cpu")

        assert len(metrics["epochs"]) == 1
        assert first_latest["epoch"] == 1
        assert first_latest["optimizer"]
        assert (checkpoint_dir / "epoch_0001.pt").exists()

        args.epochs = 2
        args.resume_checkpoint = checkpoint_dir / "latest.pt"
        resumed_metrics = train_ppo(args)
        final_payload = torch.load(final_checkpoint, map_location="cpu")

        assert [epoch["epoch"] for epoch in resumed_metrics["epochs"]] == [1, 2]
        assert final_payload["epoch"] == 2
        assert torch.load(checkpoint_dir / "latest.pt", map_location="cpu")["epoch"] == 2


def test_selfplay_summary_reports_per_seat_hu_huang_turn_and_raw_score():
    records = [
        {
            "turn_count": 18,
            "scores": {"0": 24, "1": -8, "2": -8, "3": -8},
            "final_output": {"display": {"action": "HU", "player": 0, "score": [24, -8, -8, -8]}},
        },
        {
            "turn_count": 26,
            "scores": {"0": -4, "1": -4, "2": 12, "3": -4},
            "final_output": {"display": {"action": "HU", "player": 2, "score": [-4, -4, 12, -4]}},
        },
        {
            "turn_count": 44,
            "scores": {"0": 0, "1": 0, "2": 0, "3": 0},
            "final_output": {"display": {"action": "HUANG", "score": [0, 0, 0, 0]}},
        },
    ]

    summary = summarize_selfplay(records)

    assert summary["hu_count"] == 2
    assert summary["hu_rate"] == 2 / 3
    assert summary["huang_count"] == 1
    assert summary["huang_rate"] == 1 / 3
    assert summary["hu_turn_average"] == 22.0
    assert summary["per_player"]["0"]["score_label"] == "A"
    assert summary["per_player"]["0"]["hu_count"] == 1
    assert summary["per_player"]["0"]["hu_rate"] == 1 / 3
    assert summary["per_player"]["0"]["average_hu_turn"] == 18.0
    assert summary["per_player"]["0"]["average_raw_score"] == (24 - 4 + 0) / 3
    assert summary["per_player"]["2"]["hu_count"] == 1
    assert summary["per_player"]["2"]["average_hu_turn"] == 26.0
    assert summary["per_player"]["1"]["hu_count"] == 0
    assert summary["per_player"]["1"]["average_hu_turn"] is None
    assert summary["score_table"][0]["score_label"] == "A"


def test_selfplay_streaming_accumulator_matches_batch_summary():
    records = [
        {
            "turn_count": 18,
            "scores": {"0": 24, "1": -8, "2": -8, "3": -8},
            "final_output": {"display": {"action": "HU", "player": 0}},
        },
        {
            "turn_count": 44,
            "scores": {"0": 0, "1": 0, "2": 0, "3": 0},
            "final_output": {"display": {"action": "HUANG"}},
        },
    ]
    accumulator = SelfplaySummaryAccumulator()
    for record in records:
        accumulator.add(record)

    assert accumulator.to_summary() == summarize_selfplay(records)


def test_selfplay_dashboard_reports_turn_fans_and_coufan_sections():
    records = [
        {
            "match_id": "dashboard-self-draw",
            "scores": {"0": 48, "1": -16, "2": -16, "3": -16},
            "final_output": {
                "display": {
                    "action": "HU",
                    "player": 0,
                    "fanCnt": 8,
                    "fan": [
                        {"name": "门前清", "value": 2, "cnt": 1},
                        {"name": "平和", "value": 2, "cnt": 1},
                        {"name": "断幺", "value": 2, "cnt": 1},
                        {"name": "自摸", "value": 1, "cnt": 1},
                        {"name": "无字", "value": 1, "cnt": 1},
                    ],
                }
            },
            "logs": [
                {
                    "output": {
                        "display": {"action": "DRAW", "player": 0, "canHu": [8, -4, -4, -4]},
                        "content": {
                            "0": "2 W1",
                            "1": "3 0 DRAW",
                            "2": "3 0 DRAW",
                            "3": "3 0 DRAW",
                        },
                    }
                },
                {
                    "0": {"raw": "HU"},
                    "1": {"raw": "PASS"},
                    "2": {"raw": "PASS"},
                    "3": {"raw": "PASS"},
                },
            ],
        },
        {
            "match_id": "dashboard-discard-hu",
            "scores": {"0": -8, "1": -20, "2": 44, "3": -8},
            "final_output": {
                "display": {
                    "action": "HU",
                    "player": 2,
                    "fanCnt": 12,
                    "fan": [{"name": "全不靠", "value": 12, "cnt": 1}],
                }
            },
            "logs": [
                {
                    "output": {
                        "display": {"action": "DRAW", "player": 1, "canHu": [-4, -4, -4, -4]},
                        "content": {
                            "0": "3 1 DRAW",
                            "1": "2 W2",
                            "2": "3 1 DRAW",
                            "3": "3 1 DRAW",
                        },
                    }
                },
                {
                    "0": {"raw": "PASS"},
                    "1": {"raw": "PLAY W2"},
                    "2": {"raw": "PASS"},
                    "3": {"raw": "PASS"},
                },
                {
                    "output": {
                        "display": {"action": "PLAY", "player": 1, "canHu": [-4, -4, 12, -4]},
                        "content": {
                            "0": "3 1 PLAY W2",
                            "1": "3 1 PLAY W2",
                            "2": "3 1 PLAY W2",
                            "3": "3 1 PLAY W2",
                        },
                    }
                },
                {
                    "0": {"raw": "PASS"},
                    "1": {"raw": "PASS"},
                    "2": {"raw": "HU"},
                    "3": {"raw": "PASS"},
                },
            ],
        },
    ]

    summary = analyze_records(records)
    text = format_dashboard_text(summary)

    assert summary["hu_count"] == 2
    assert summary["turns"]["和牌巡数"] == 1.0
    assert summary["turns"]["点和巡数"] == 1.0
    assert summary["turns"]["自摸巡数"] == 1.0
    assert summary["turns"]["听牌巡数"] == 1.0
    assert summary["fans"]["平均和牌番"] == 10.0
    assert summary["fan_table"]["12番"][0]["name"] == "全不靠"
    assert summary["fan_table"]["12番"][0]["count"] == 1
    assert summary["coufan_hands"] == 1
    assert summary["coufan_table"]["基础"][1]["name"] == "门清平和断幺"
    assert summary["coufan_table"]["基础"][1]["count"] == 1
    assert "巡数相关" in text
    assert "全不靠\t1 (50.000%)\t1.000" in text


def test_merge_selfplay_shards_preserves_order_and_quality_summary():
    records = [
        {
            "match_id": "shard-0-game-0",
            "turn_count": 18,
            "scores": {"0": 24, "1": -8, "2": -8, "3": -8},
            "final_output": {"display": {"action": "HU", "player": 0, "score": [24, -8, -8, -8]}},
        },
        {
            "match_id": "shard-1-game-0",
            "turn_count": 26,
            "scores": {"0": -4, "1": -4, "2": 12, "3": -4},
            "final_output": {"display": {"action": "HU", "player": 2, "score": [-4, -4, 12, -4]}},
        },
    ]
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        root = Path(tmp_dir)
        shard_dir = root / "shards"
        shard_dir.mkdir()
        for index, record in enumerate(records):
            raw = shard_dir / f"tjong_selfplay_raw_shard_{index:04d}.jsonl"
            fan = shard_dir / f"tjong_selfplay_fan_items_shard_{index:04d}.jsonl"
            summary = shard_dir / f"tjong_selfplay_summary_shard_{index:04d}.json"
            raw.write_text(json.dumps(record) + "\n", encoding="utf-8")
            fan.write_text(json.dumps({"match_id": record["match_id"], "fans": []}) + "\n", encoding="utf-8")
            summary.write_text(json.dumps({"games": 1}), encoding="utf-8")

        merged_summary = merge_shards(
            shard_dir=shard_dir,
            shards=2,
            out_raw=root / "merged_raw.jsonl",
            out_fan_items=root / "merged_fans.jsonl",
            summary_out=root / "merged_summary.json",
            expected_games=2,
            min_games=2,
            min_total_hu_rate=0.99,
            max_huang_rate=0.01,
        )

        merged_records = [
            json.loads(line)["match_id"]
            for line in (root / "merged_raw.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert merged_records == ["shard-0-game-0", "shard-1-game-0"]
        assert merged_summary["games"] == 2
        assert merged_summary["hu_rate"] == 1.0
        assert merged_summary["huang_rate"] == 0.0
        assert merged_summary["per_player"]["0"]["hu_rate"] == 0.5
        assert merged_summary["per_player"]["2"]["hu_rate"] == 0.5


def test_infer_loser_only_for_unique_direct_loss():
    assert infer_loser([-8, -38, -8, 54], 3) == 1
    assert infer_loser([-18, 54, -18, -18], 1) is None


def test_tile_order_and_chow_claim_index_match_botzone_layout():
    assert len(TILE_NAMES) == 34
    assert tile_id("W1") == 0
    assert tile_id("T1") == 9
    assert tile_id("B1") == 18
    assert chow_claim_index("W4", "W5") == 8


def test_tjong_botzone_candidate_labels_include_forced_discards():
    peng = response_to_labels("3 0 PLAY W2", "PENG T1")
    chi = response_to_labels("3 0 PLAY W2", "CHI W3 B1")

    assert TjongCheckpointPredictor.kind == "legal_action_ranker"
    assert peng == (ACTION_TO_INDEX["PONG"], flatten_claim("PONG", tile_id("W2")), tile_id("T1"))
    assert chi == (ACTION_TO_INDEX["CHOW"], flatten_claim("CHOW", chow_claim_index("W3", "W2")), tile_id("B1"))
    assert response_discard_label("PENG T1") == tile_id("T1")
    assert response_discard_label("CHI W3 B1") == tile_id("B1")


def test_runtime_post_claim_state_matches_forced_discard_decision_state():
    input_text = "\n".join(
        [
            "REQ 0 1 0",
            "REQ 3 0 PLAY W2",
        ]
    )
    hand = Counter({"W2": 2, "W3": 1, "W4": 1, "T1": 1, "B1": 1})
    discard_mask = [0.0] * 8
    discard_mask[ACTION_TO_INDEX["DISCARD"]] = 1.0

    peng_visible, peng_game, _ = encode_runtime_state(
        input_text=input_text,
        hand=hand,
        player_id=1,
        action_mask=discard_mask,
        post_claim_request="3 0 PLAY W2",
        post_claim_response="PENG T1",
    )

    assert peng_visible[VISIBLE_ROW_NAMES.index("discard_p3"), tile_id("W2")].item() == 0.0
    assert peng_visible[VISIBLE_ROW_NAMES.index("peng_p0"), tile_id("W2")].item() == 3.0
    assert peng_visible[VISIBLE_ROW_NAMES.index("hand_self"), tile_id("W2")].item() == 0.0
    assert peng_visible[VISIBLE_ROW_NAMES.index("hand_self"), tile_id("T1")].item() == 1.0
    assert peng_game[-8 + ACTION_TO_INDEX["DISCARD"]].item() == 1.0

    chi_visible, _, _ = encode_runtime_state(
        input_text=input_text,
        hand=hand,
        player_id=1,
        action_mask=discard_mask,
        post_claim_request="3 0 PLAY W2",
        post_claim_response="CHI W3 B1",
    )

    assert chi_visible[VISIBLE_ROW_NAMES.index("discard_p3"), tile_id("W2")].item() == 0.0
    assert chi_visible[VISIBLE_ROW_NAMES.index("chow_p0"), tile_id("W2")].item() == 1.0
    assert chi_visible[VISIBLE_ROW_NAMES.index("chow_p0"), tile_id("W3")].item() == 1.0
    assert chi_visible[VISIBLE_ROW_NAMES.index("chow_p0"), tile_id("W4")].item() == 1.0
    assert chi_visible[VISIBLE_ROW_NAMES.index("hand_self"), tile_id("W3")].item() == 0.0
    assert chi_visible[VISIBLE_ROW_NAMES.index("hand_self"), tile_id("W4")].item() == 0.0
    assert chi_visible[VISIBLE_ROW_NAMES.index("hand_self"), tile_id("B1")].item() == 1.0


def test_botzone_tensorizer_smoke_with_post_claim_discard():
    record = {
        "match_id": "synthetic",
        "scores": {"0": 10, "1": -2, "2": -4, "3": -4},
        "initdata": {
            "quan": 0,
            "walltiles": " ".join(TILE_NAMES * 4),
        },
        "logs": [
            {
                "output": {
                    "content": {"0": "0 0 0", "1": "0 1 0", "2": "0 2 0", "3": "0 3 0"},
                    "display": {"action": "INIT", "quan": 0, "tileCnt": [21, 21, 21, 21]},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4 T5",
                        "1": "1 0 0 0 0 W2 W2 T1 T2 T3 T4 T5 T6 T7 T8 T9 B1 B2",
                        "2": "1 0 0 0 0 B3 B4 B5 B6 B7 B8 B9 F1 F2 F3 F4 J1 J2",
                        "3": "1 0 0 0 0 J3 W1 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                    },
                    "display": {
                        "action": "DEAL",
                        "hand": [
                            ["W1", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4", "T5"],
                            ["W2", "W2", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "B1", "B2"],
                            ["B3", "B4", "B5", "B6", "B7", "B8", "B9", "F1", "F2", "F3", "F4", "J1", "J2"],
                            ["J3", "W1", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                        ],
                        "tileCnt": [21, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {"0": "2 W2", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                    "display": {
                        "action": "DRAW",
                        "player": 0,
                        "tile": "W2",
                        "canHu": [8, -4, -4, -4],
                        "tileCnt": [20, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PLAY W2"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {"0": "3 0 PLAY W2", "1": "3 0 PLAY W2", "2": "3 0 PLAY W2", "3": "3 0 PLAY W2"},
                    "display": {
                        "action": "PLAY",
                        "player": 0,
                        "tile": "W2",
                        "canHu": [-4, -3, -3, -3],
                        "tileCnt": [20, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PENG T1"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
        ],
    }

    examples, stats = tensorize_record(record)

    assert stats["examples"] == 3
    assert examples[0].visible_tiles.shape == (4, 22, 34)
    assert examples[0].game_features.shape == (4, 24)
    assert examples[0].sub_visible_tiles.shape == (4, 22, 34)
    assert examples[0].sub_game_features.shape == (4, 24)
    assert examples[0].hidden_tiles.shape == (5, 34)
    assert examples[0].action_label == 2
    assert examples[0].sub_previous_actions[-1].item() == examples[0].action_label
    assert examples[1].action_label == 4
    assert examples[1].sub_game_features[-1, -8 + examples[1].action_label].item() == 1.0
    assert examples[2].kind == "post_claim_discard"
    assert examples[2].discard_label == tile_id("T1")


def test_tensorizer_rejects_under_eight_can_hu_values():
    state = ReplayState.from_record({"initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)}})
    state.apply_display(
        {
            "action": "DEAL",
            "hand": [
                ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
            ],
        }
    )

    under_threshold = action_type_mask(0, ["2", "W1"], state, {"canHu": [7, -4, -4, -4]})
    at_threshold = action_type_mask(0, ["2", "W1"], state, {"canHu": [8, -4, -4, -4]})

    assert under_threshold[ACTION_TO_INDEX["HU"]] == 0.0
    assert under_threshold[ACTION_TO_INDEX["DISCARD"]] == 1.0
    assert at_threshold[ACTION_TO_INDEX["HU"]] == 1.0


def test_tensorizer_replays_converted_deal_content_for_chow_legality():
    record = {
        "match_id": "converted-deal-chow",
        "scores": {"0": -8, "1": 24, "2": -8, "3": -8},
        "initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)},
        "logs": [
            {
                "output": {
                    "content": {"0": "0 0 0", "1": "0 1 0", "2": "0 2 0", "3": "0 3 0"},
                    "display": {"action": "INIT"},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W3 W6 W7 W8 W9 T1 T2 T3 B1 B2 B3 F1",
                        "1": "1 0 0 0 0 W2 W4 W5 W7 W8 W9 T1 T2 T3 B1 B2 B3 F2",
                        "2": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 B1",
                        "3": "1 0 0 0 0 B1 B2 B3 B4 B5 B6 B7 B8 B9 T1 T2 T3 W1",
                    },
                    "display": {"action": "DEAL"},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {"0": "2 W6", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                    "display": {"action": "DRAW", "player": 0, "tile": "W6", "canHu": [-4, -4, -4, -4]},
                }
            },
            {"0": {"raw": "PLAY W3"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "3 0 PLAY W3",
                        "1": "3 0 PLAY W3",
                        "2": "3 0 PLAY W3",
                        "3": "3 0 PLAY W3",
                    },
                    "display": {"action": "PLAY", "player": 0, "tile": "W3", "canHu": [-4, -4, -4, -4]},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "CHI W3 W5"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
        ],
    }

    examples, stats = tensorize_record(record)

    assert stats["action:CHOW"] == 1
    assert stats["claim_discard_examples"] == 1
    assert stats.get("action_mask_repaired:CHOW", 0) == 0
    assert stats.get("label_outside_mask:CHOW", 0) == 0
    action_counts = Counter(example.action_label for example in examples)
    assert action_counts[ACTION_TO_INDEX["CHOW"]] == 1
    assert action_counts[ACTION_TO_INDEX["DISCARD"]] == 2


def test_tensorizer_keeps_sampled_single_action_discards():
    record = {
        "match_id": "single-discard-sampling",
        "scores": {"0": 24, "1": -8, "2": -8, "3": -8},
        "initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)},
        "logs": [
            {
                "output": {
                    "content": {"0": "0 0 0", "1": "0 1 0", "2": "0 2 0", "3": "0 3 0"},
                    "display": {"action": "INIT", "quan": 0, "tileCnt": [21, 21, 21, 21]},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "1": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "2": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "3": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                    },
                    "display": {
                        "action": "DEAL",
                        "hand": [
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                        ],
                        "tileCnt": [21, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
        ],
    }
    for tile in ("W1", "W2", "W3"):
        record["logs"].extend(
            [
                {
                    "output": {
                        "content": {"0": f"2 {tile}", "1": f"3 0 DRAW {tile}", "2": f"3 0 DRAW {tile}", "3": f"3 0 DRAW {tile}"},
                        "display": {
                            "action": "DRAW",
                            "player": 0,
                            "tile": tile,
                            "canHu": [-4, -4, -4, -4],
                            "tileCnt": [20, 21, 21, 21],
                        },
                    }
                },
                {"0": {"raw": f"PLAY {tile}"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
                {
                    "output": {
                        "content": {
                            "0": f"3 0 PLAY {tile}",
                            "1": f"3 0 PLAY {tile}",
                            "2": f"3 0 PLAY {tile}",
                            "3": f"3 0 PLAY {tile}",
                        },
                        "display": {
                            "action": "PLAY",
                            "player": 0,
                            "tile": tile,
                            "canHu": [-4, -4, -4, -4],
                            "tileCnt": [20, 21, 21, 21],
                        },
                    }
                },
                {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            ]
        )
    record["logs"].extend(
        [
            {
                "output": {
                    "content": {"0": "2 W4", "1": "3 0 DRAW W4", "2": "3 0 DRAW W4", "3": "3 0 DRAW W4"},
                    "display": {
                        "action": "DRAW",
                        "player": 0,
                        "tile": "W4",
                        "canHu": [8, -4, -4, -4],
                        "tileCnt": [20, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "HU"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
        ]
    )

    examples, stats = tensorize_record(record, single_action_discard_stride=2)

    assert stats["single_action_discard_seen"] == 3
    assert stats["single_action_discard_kept"] == 2
    assert stats["single_action_discard_sampled_out"] == 1
    assert stats["action:DISCARD"] == 2
    assert stats["action:HU"] == 1
    assert stats["action:DISCARD"] > stats["action:HU"]
    action_counts = Counter(example.action_label for example in examples)
    assert action_counts[ACTION_TO_INDEX["DISCARD"]] == 2
    assert action_counts[ACTION_TO_INDEX["HU"]] == 1


def _single_action_discard_hu_record(match_id: str) -> dict:
    record = {
        "match_id": match_id,
        "scores": {"0": 24, "1": -8, "2": -8, "3": -8},
        "initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)},
        "logs": [
            {
                "output": {
                    "content": {"0": "0 0 0", "1": "0 1 0", "2": "0 2 0", "3": "0 3 0"},
                    "display": {"action": "INIT", "quan": 0, "tileCnt": [21, 21, 21, 21]},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "1": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "2": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                        "3": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
                    },
                    "display": {
                        "action": "DEAL",
                        "hand": [
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4"],
                        ],
                        "tileCnt": [21, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
        ],
    }
    for tile in ("W1", "W2", "W3"):
        record["logs"].extend(
            [
                {
                    "output": {
                        "content": {"0": f"2 {tile}", "1": f"3 0 DRAW {tile}", "2": f"3 0 DRAW {tile}", "3": f"3 0 DRAW {tile}"},
                        "display": {
                            "action": "DRAW",
                            "player": 0,
                            "tile": tile,
                            "canHu": [-4, -4, -4, -4],
                            "tileCnt": [20, 21, 21, 21],
                        },
                    }
                },
                {"0": {"raw": f"PLAY {tile}"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
                {
                    "output": {
                        "content": {
                            "0": f"3 0 PLAY {tile}",
                            "1": f"3 0 PLAY {tile}",
                            "2": f"3 0 PLAY {tile}",
                            "3": f"3 0 PLAY {tile}",
                        },
                        "display": {
                            "action": "PLAY",
                            "player": 0,
                            "tile": tile,
                            "canHu": [-4, -4, -4, -4],
                            "tileCnt": [20, 21, 21, 21],
                        },
                    }
                },
                {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            ]
        )
    record["logs"].extend(
        [
            {
                "output": {
                    "content": {"0": "2 W4", "1": "3 0 DRAW W4", "2": "3 0 DRAW W4", "3": "3 0 DRAW W4"},
                    "display": {
                        "action": "DRAW",
                        "player": 0,
                        "tile": "W4",
                        "canHu": [8, -4, -4, -4],
                        "tileCnt": [20, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "HU"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
        ]
    )
    return record


def test_sharded_tensorizer_keeps_all_discards_and_trains_from_index():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_path = tmp_path / "raw.jsonl"
        shard_dir = tmp_path / "shards"
        index_path = shard_dir / "index.json"
        metrics_path = tmp_path / "metrics.json"
        checkpoint_path = tmp_path / "supervised.pt"
        raw_path.write_text(
            "\n".join(
                json.dumps(_single_action_discard_hu_record(f"all-discard-{index}"))
                for index in range(2)
            )
            + "\n",
            encoding="utf-8",
        )

        summary = tensorize_file(
            raw_path,
            None,
            shard_dir=shard_dir,
            shard_index_out=index_path,
            compact_metadata=True,
            compact_storage=True,
            shard_max_examples=7,
            single_action_discard_stride=1,
        )
        loaded_summary = json.loads(index_path.read_text(encoding="utf-8"))
        first_shard = torch.load(shard_dir / summary["shards"][0]["path"], map_location="cpu")
        dataset = load_tensor_dataset(index_path, expected_encoding_version=TENSOR_ENCODING_VERSION)

        assert summary["format"] == SHARD_INDEX_FORMAT
        assert loaded_summary["format"] == SHARD_INDEX_FORMAT
        assert summary["sharded"] is True
        assert summary["compact_storage"] is True
        assert summary["single_action_discard_stride"] == 1
        assert summary["stats"]["single_action_discard_seen"] == 6
        assert summary["stats"]["single_action_discard_kept"] == 6
        assert summary["stats"].get("single_action_discard_sampled_out", 0) == 0
        assert summary["stats"]["action:DISCARD"] == 6
        assert summary["stats"]["action:HU"] == 2
        assert summary["stats"]["action:DISCARD"] > summary["stats"]["action:HU"]
        assert len(summary["shards"]) == 2
        assert all(len(shard["sha256"]) == 64 for shard in summary["shards"])
        assert first_shard["visible_tiles"].dtype == torch.uint8
        assert first_shard["action_label"].dtype == torch.uint8
        assert "rewards" not in first_shard
        assert isinstance(dataset, ShardedTensorDataset)
        assert len(dataset) == summary["examples"] == 14

        args = argparse.Namespace(
            train_pt=index_path,
            checkpoint_out=checkpoint_path,
            metrics_out=metrics_path,
            epochs=1,
            batch_size=2,
            lr=1e-4,
            d_model=32,
            n_heads=4,
            ffn_dim=64,
            dropout=0.0,
            num_workers=0,
            device="cpu",
            data_parallel=False,
            seed=0,
            require_encoding_version=TENSOR_ENCODING_VERSION,
            require_paper_config=False,
            checkpoint_every_epochs=0,
            checkpoint_dir=None,
            resume_checkpoint=None,
            metrics_jsonl=None,
            checkpoint_every_batches=0,
            checkpoint_at_global_batches=None,
            checkpoint_at_epoch_batches=None,
            grad_clip=0.5,
            force_math_sdp=False,
            fail_on_nonfinite=True,
            cuda_sync_debug=False,
            max_steps=1,
        )
        metrics = train_supervised(args)

        assert metrics["train_data_format"] == SHARD_INDEX_FORMAT
        assert metrics["train_examples"] == 14
        assert metrics["train_shard_count"] == 2
        assert metrics["epochs"][-1]["batches"] == 1
        assert metrics["epochs"][-1]["action_count"] == 2


def test_optimized_sharded_loader_preserves_global_batches_across_ranks():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        schema = tensor_encoding_schema()
        shard_specs = [(0, 5), (5, 5)]
        shards = []
        for shard_index, (start, count) in enumerate(shard_specs):
            ids = torch.arange(start, start + count)
            visible_tiles = torch.zeros(count, 4, 22, 34)
            visible_tiles[:, 0, 0, 0] = ids.float()
            payload = {
                "visible_tiles": visible_tiles,
                "game_features": torch.zeros(count, 4, 24),
                "rewards": torch.zeros(count, 4),
                "previous_actions": torch.zeros(count, 4, dtype=torch.long),
                "sub_visible_tiles": visible_tiles.clone(),
                "sub_game_features": torch.zeros(count, 4, 24),
                "sub_rewards": torch.zeros(count, 4),
                "sub_previous_actions": torch.zeros(count, 4, dtype=torch.long),
                "hidden_tiles": torch.zeros(count, 5, 34),
                "action_label": torch.full((count,), ACTION_TO_INDEX["DISCARD"], dtype=torch.long),
                "claim_label": torch.zeros(count, dtype=torch.long),
                "discard_label": ids.remainder(34).long(),
                "encoding_schema": schema,
                "examples": count,
            }
            path = shard_dir / f"shard_{shard_index}.pt"
            torch.save(payload, path)
            shards.append({"path": path.name, "examples": count})
        index_path = shard_dir / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "format": SHARD_INDEX_FORMAT,
                    "encoding_schema": schema,
                    "examples": 10,
                    "shards": shards,
                }
            ),
            encoding="utf-8",
        )
        dataset = load_tensor_dataset(index_path, expected_encoding_version=TENSOR_ENCODING_VERSION)
        assert isinstance(dataset, ShardedTensorDataset)

        rank_batches = [
            list(
                iter_optimized_sharded_batches(
                    dataset,
                    batch_size=4,
                    shuffle=False,
                    rank=rank,
                    world_size=2,
                    prefetch_shards=2,
                )
            )
            for rank in range(2)
        ]

        assert [batch[0].shape[0] for batch in rank_batches[0]] == [2, 2, 1]
        assert [batch[0].shape[0] for batch in rank_batches[1]] == [2, 2, 1]
        recovered = []
        for rank in range(2):
            for batch in rank_batches[rank]:
                recovered.extend(int(value.item()) for value in batch[0][:, 0, 0, 0])
        assert sorted(recovered) == list(range(10))
        assert len(recovered) == len(set(recovered)) == 10


def test_optimized_sharded_loader_can_skip_within_shard_shuffle():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        schema = tensor_encoding_schema()
        shards = []
        for shard_index in range(3):
            start = shard_index * 4
            ids = torch.arange(start, start + 4)
            visible_tiles = torch.zeros(4, 4, 22, 34)
            visible_tiles[:, 0, 0, 0] = ids.float()
            payload = {
                "visible_tiles": visible_tiles,
                "game_features": torch.zeros(4, 4, 24),
                "rewards": torch.zeros(4, 4),
                "previous_actions": torch.zeros(4, 4, dtype=torch.long),
                "sub_visible_tiles": visible_tiles.clone(),
                "sub_game_features": torch.zeros(4, 4, 24),
                "sub_rewards": torch.zeros(4, 4),
                "sub_previous_actions": torch.zeros(4, 4, dtype=torch.long),
                "hidden_tiles": torch.zeros(4, 5, 34),
                "action_label": torch.full((4,), ACTION_TO_INDEX["DISCARD"], dtype=torch.long),
                "claim_label": torch.zeros(4, dtype=torch.long),
                "discard_label": ids.remainder(34).long(),
                "encoding_schema": schema,
                "examples": 4,
            }
            path = shard_dir / f"shard_{shard_index}.pt"
            torch.save(payload, path)
            shards.append({"path": path.name, "examples": 4})
        index_path = shard_dir / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "format": SHARD_INDEX_FORMAT,
                    "encoding_schema": schema,
                    "examples": 12,
                    "shards": shards,
                }
            ),
            encoding="utf-8",
        )
        dataset = load_tensor_dataset(index_path, expected_encoding_version=TENSOR_ENCODING_VERSION)

        batches = list(
            iter_optimized_sharded_batches(
                dataset,
                batch_size=4,
                shuffle=True,
                seed=123,
                prefetch_shards=2,
                shuffle_within_shards=False,
            )
        )

        assert len(batches) == 3
        recovered = [[int(value.item()) for value in batch[0][:, 0, 0, 0]] for batch in batches]
        assert sorted(value for batch in recovered for value in batch) == list(range(12))
        assert all(batch == list(range(batch[0], batch[0] + 4)) for batch in recovered)


def test_sharded_dataset_records_mmap_shard_preference():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        schema = tensor_encoding_schema()
        payload = {
            "visible_tiles": torch.zeros(1, 4, 22, 34),
            "game_features": torch.zeros(1, 4, 24),
            "action_label": torch.full((1,), ACTION_TO_INDEX["DISCARD"], dtype=torch.long),
            "claim_label": torch.zeros(1, dtype=torch.long),
            "discard_label": torch.zeros(1, dtype=torch.long),
            "encoding_schema": schema,
            "examples": 1,
        }
        torch.save(payload, shard_dir / "shard_0.pt")
        index_path = shard_dir / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "format": SHARD_INDEX_FORMAT,
                    "encoding_schema": schema,
                    "examples": 1,
                    "shards": [{"path": "shard_0.pt", "examples": 1}],
                }
            ),
            encoding="utf-8",
        )

        dataset = load_tensor_dataset(
            index_path,
            expected_encoding_version=TENSOR_ENCODING_VERSION,
            mmap_shards=True,
        )

        assert isinstance(dataset, ShardedTensorDataset)
        assert dataset.mmap_shards is True
        assert len(dataset.load_shard_payload(0)["action_label"]) == 1


def test_sharded_dataset_can_drop_file_cache_after_loading_shard(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        schema = tensor_encoding_schema()
        count = 2
        payload = {
            "visible_tiles": torch.zeros(count, 4, 22, 34),
            "game_features": torch.zeros(count, 4, 24),
            "rewards": torch.zeros(count, 4),
            "previous_actions": torch.zeros(count, 4, dtype=torch.long),
            "sub_visible_tiles": torch.zeros(count, 4, 22, 34),
            "sub_game_features": torch.zeros(count, 4, 24),
            "sub_rewards": torch.zeros(count, 4),
            "sub_previous_actions": torch.zeros(count, 4, dtype=torch.long),
            "hidden_tiles": torch.zeros(count, 5, 34),
            "action_label": torch.full((count,), ACTION_TO_INDEX["DISCARD"], dtype=torch.long),
            "claim_label": torch.zeros(count, dtype=torch.long),
            "discard_label": torch.zeros(count, dtype=torch.long),
            "encoding_schema": schema,
            "examples": count,
        }
        shard_path = shard_dir / "shard_0.pt"
        torch.save(payload, shard_path)
        index_path = shard_dir / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "format": SHARD_INDEX_FORMAT,
                    "encoding_schema": schema,
                    "examples": count,
                    "shards": [{"path": shard_path.name, "examples": count}],
                }
            ),
            encoding="utf-8",
        )
        dropped_paths = []

        def fake_drop_file_cache(path):
            dropped_paths.append(Path(path))
            return True

        monkeypatch.setattr(train_supervised_module, "_drop_file_cache_for_path", fake_drop_file_cache)
        dataset = load_tensor_dataset(
            index_path,
            expected_encoding_version=TENSOR_ENCODING_VERSION,
            drop_shard_file_cache=True,
        )

        assert isinstance(dataset, ShardedTensorDataset)
        assert dataset.drop_file_cache is True
        loaded = dataset.load_shard_payload(0)
        assert loaded["examples"] == count
        assert dropped_paths == [shard_path]


def test_parallel_sharded_tensorizer_matches_single_worker_stats():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_path = tmp_path / "raw.jsonl"
        single_dir = tmp_path / "single"
        parallel_dir = tmp_path / "parallel"
        records = [_single_action_discard_hu_record(f"parallel-{index}") for index in range(6)]
        raw_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

        single = tensorize_file(
            raw_path,
            None,
            shard_dir=single_dir,
            shard_index_out=single_dir / "index.json",
            compact_metadata=True,
            compact_storage=True,
            shard_max_examples=9,
            single_action_discard_stride=1,
        )
        parallel = tensorize_file(
            raw_path,
            None,
            shard_dir=parallel_dir,
            shard_index_out=parallel_dir / "index.json",
            compact_metadata=True,
            compact_storage=True,
            shard_max_examples=9,
            single_action_discard_stride=1,
            num_workers=2,
        )
        dataset = load_tensor_dataset(parallel_dir / "index.json", expected_encoding_version=TENSOR_ENCODING_VERSION)

        assert parallel["format"] == SHARD_INDEX_FORMAT
        assert parallel["parallel_workers"] == 2
        assert parallel["examples"] == single["examples"]
        assert parallel["stats"] == single["stats"]
        assert sum(int(shard["examples"]) for shard in parallel["shards"]) == parallel["examples"]
        assert {int(shard["worker"]) for shard in parallel["shards"]} == {0, 1}
        assert len(dataset) == parallel["examples"]


def test_populate_fan_backward_rewards_with_explicit_fan_items():
    record = {
        "match_id": "fan-smoke",
        "scores": {"0": -16, "1": 24, "2": -4, "3": -4},
        "initdata": {"quan": 0, "walltiles": " ".join(TILE_NAMES * 4)},
        "logs": [
            {
                "output": {
                    "content": {"0": "0 0 0", "1": "0 1 0", "2": "0 2 0", "3": "0 3 0"},
                    "display": {"action": "INIT", "quan": 0, "tileCnt": [21, 21, 21, 21]},
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 T1 T2 T3 T4 T5",
                        "1": "1 0 0 0 0 W1 W1 W2 W3 W4 W5 W6 W7 W8 T1 T2 T3 T4",
                        "2": "1 0 0 0 0 B1 B2 B3 B4 B5 B6 B7 B8 B9 F1 F2 F3 F4",
                        "3": "1 0 0 0 0 J1 J2 J3 W9 T5 T6 T7 T8 T9 B1 B2 B3 B4",
                    },
                    "display": {
                        "action": "DEAL",
                        "hand": [
                            ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "T1", "T2", "T3", "T4", "T5"],
                            ["W1", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "T1", "T2", "T3", "T4"],
                            ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "F1", "F2", "F3", "F4"],
                            ["J1", "J2", "J3", "W9", "T5", "T6", "T7", "T8", "T9", "B1", "B2", "B3", "B4"],
                        ],
                        "tileCnt": [21, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {"0": "2 W9", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                    "display": {
                        "action": "DRAW",
                        "player": 0,
                        "tile": "W9",
                        "canHu": [8, -4, -4, -4],
                        "tileCnt": [20, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PLAY W9"}, "1": {"raw": "PASS"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "content": {"0": "3 0 PLAY W9", "1": "3 0 PLAY W9", "2": "3 0 PLAY W9", "3": "3 0 PLAY W9"},
                    "display": {
                        "action": "PLAY",
                        "player": 0,
                        "tile": "W9",
                        "canHu": [-4, 8, -3, -3],
                        "tileCnt": [20, 21, 21, 21],
                    },
                }
            },
            {"0": {"raw": "PASS"}, "1": {"raw": "HU"}, "2": {"raw": "PASS"}, "3": {"raw": "PASS"}},
            {
                "output": {
                    "command": "finish",
                    "content": {"0": -16, "1": 24, "2": -4, "3": -4},
                    "display": {
                        "action": "HU",
                        "player": 1,
                        "score": [-16, 24, -4, -4],
                        "fan": [{"cnt": 1, "name": "无字", "value": 8}],
                        "fanCnt": 8,
                        "tileCnt": [20, 21, 21, 21],
                    },
                }
            },
        ],
    }
    examples, _ = tensorize_record(record, include_single_action=True)
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        tensor_path = tmp_path / "tensor.pt"
        raw_path = tmp_path / "raw.jsonl"
        fan_path = tmp_path / "fans.jsonl"
        out_path = tmp_path / "rewarded.pt"
        torch.save(
            {
                "visible_tiles": torch.stack([example.visible_tiles for example in examples]),
                "game_features": torch.stack([example.game_features for example in examples]),
                "rewards": torch.stack([example.rewards for example in examples]),
                "previous_actions": torch.stack([example.previous_actions for example in examples]),
                "sub_visible_tiles": torch.stack([example.sub_visible_tiles for example in examples]),
                "sub_game_features": torch.stack([example.sub_game_features for example in examples]),
                "sub_rewards": torch.stack([example.sub_rewards for example in examples]),
                "sub_previous_actions": torch.stack([example.sub_previous_actions for example in examples]),
                "hidden_tiles": torch.stack([example.hidden_tiles for example in examples]),
                "action_label": torch.tensor([example.action_label for example in examples]),
                "claim_label": torch.tensor([example.claim_label for example in examples]),
                "discard_label": torch.tensor([example.discard_label for example in examples]),
                "value_target": torch.tensor([example.value_target for example in examples]),
                "metadata": {
                    "match_id": [example.match_id for example in examples],
                    "player": [example.player for example in examples],
                    "turn_index": [example.turn_index for example in examples],
                    "kind": [example.kind for example in examples],
                },
            },
            tensor_path,
        )
        raw_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        fan_path.write_text(
            json.dumps({"match_id": "fan-smoke", "fans": [{"score": 8, "tiles": ["W9"]}]}) + "\n",
            encoding="utf-8",
        )

        summary = populate_file(
            tensor_pt=tensor_path,
            raw_path=raw_path,
            out_path=out_path,
            fan_items_jsonl=fan_path,
            require_hu_reward=True,
        )
        rewarded = torch.load(out_path, map_location="cpu")

        assert summary["fan_tile_sources"] == {"explicit_fan_items": 1}
        assert summary["hu_matches"] == 1
        assert summary["huang_matches"] == 0
        assert summary["hu_matches_with_fans"] == 1
        assert summary["hu_matches_with_reward"] == 1
        assert summary["hu_matches_without_reward"] == []
        assert int((rewarded["fan_backward_reward"] != 0).sum().item()) > 0
        del rewarded

        bad_fan_path = tmp_path / "bad_fans.jsonl"
        bad_fan_path.write_text(
            json.dumps({"match_id": "fan-smoke", "fans": [{"score": 8, "tiles": ["J3"]}]}) + "\n",
            encoding="utf-8",
        )
        try:
            populate_file(
                tensor_pt=tensor_path,
                raw_path=raw_path,
                out_path=tmp_path / "bad_rewarded.pt",
                fan_items_jsonl=bad_fan_path,
                require_hu_reward=True,
            )
        except ValueError as exc:
            assert "produced zero fan-backward reward" in str(exc)
        else:
            raise AssertionError("expected impossible fan attribution to fail paper reward gate")

        structural_out_path = tmp_path / "rewarded_structural.pt"
        structural_summary = populate_file(
            tensor_pt=tensor_path,
            raw_path=raw_path,
            out_path=structural_out_path,
            require_hu_reward=True,
        )
        structural_rewarded = torch.load(structural_out_path, map_location="cpu")

        assert structural_summary["fan_tile_sources"] == {"structural_fan_attribution_from_display": 1}
        assert int((structural_rewarded["fan_backward_reward"] != 0).sum().item()) > 0
        del structural_rewarded


def test_populate_fan_backward_rewards_penalizes_huang_player_terminals():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        tensor_path = tmp_path / "huang_tensor.pt"
        raw_path = tmp_path / "huang_raw.jsonl"
        out_path = tmp_path / "huang_rewarded.pt"
        torch.save(
            {
                "visible_tiles": torch.zeros(5, 4, 22, 34),
                "game_features": torch.zeros(5, 4, 24),
                "hidden_tiles": torch.zeros(5, 5, 34),
                "action_label": torch.zeros(5, dtype=torch.long),
                "claim_label": torch.zeros(5, dtype=torch.long),
                "discard_label": torch.zeros(5, dtype=torch.long),
                "metadata": {
                    "match_id": ["huang-smoke"] * 5,
                    "player": [0, 0, 1, 2, 3],
                    "turn_index": [1, 3, 2, 2, 2],
                },
            },
            tensor_path,
        )
        raw_path.write_text(
            json.dumps(
                {
                    "match_id": "huang-smoke",
                    "logs": [{"output": {"display": {"action": "HUANG", "score": [0, 0, 0, 0]}}}],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        summary = populate_file(
            tensor_pt=tensor_path,
            raw_path=raw_path,
            out_path=out_path,
            huang_penalty=100.0,
        )
        rewarded = torch.load(out_path, map_location="cpu")

        assert summary["huang_matches"] == 1
        assert summary["huang_penalty"] == 100.0
        assert summary["huang_penalized_matches"] == 1
        assert summary["huang_penalized_player_terminals"] == 4
        assert torch.equal(
            rewarded["fan_backward_reward"],
            torch.tensor([0.0, -100.0, -100.0, -100.0, -100.0]),
        )
