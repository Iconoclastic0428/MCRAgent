"""Audit local implementation against paper-stated Tjong requirements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .actions import ACTION_NAMES, CLAIM_GROUP_SIZES, CLAIM_SIZE, DISCARD_SIZE
from .encoding import GAME_FEATURES, HIDDEN_TILE_ROWS, TILE_TYPES, VISIBLE_TILE_ROWS
from .model import TjongConfig, TjongNetwork
from .paper_metrics import PAPER_NAME, PAPER_REPORTED_SUPERVISED_METRICS
from .policy_bot import runtime_hidden_schema_rows
from .tensorize_botzone import HIDDEN_TILE_ROW_NAMES, TENSOR_ENCODING_VERSION


PAPER_SOURCE_LINKS = {
    "doi": "https://doi.org/10.1049/cit2.12298",
    "wiley": "https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12298",
    "njit_metadata": "https://digitalcommons.njit.edu/fac_pubs/267/",
}

EXPECTED_ACTION_NAMES = ("PASS", "HU", "DISCARD", "CHOW", "PONG", "MINGKONG", "BUKONG", "ANKONG")
EXPECTED_CLAIM_GROUP_SIZES = {
    "CHOW": 63,
    "PONG": 34,
    "MINGKONG": 34,
    "BUKONG": 34,
    "ANKONG": 34,
}


def _check(name: str, passed: bool, *, observed: Any = None, expected: Any = None, detail: str | None = None) -> dict:
    item = {
        "name": name,
        "passed": bool(passed),
    }
    if observed is not None:
        item["observed"] = observed
    if expected is not None:
        item["expected"] = expected
    if detail:
        item["detail"] = detail
    return item


def _contains_all(path: Path, needles: list[str]) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, needles
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def _manifest_contains(path: Path, needles: list[str]) -> tuple[bool, list[str]]:
    return _contains_all(path, needles)


def verify_paper_compliance(paper_root: Path | None = None) -> dict:
    root = paper_root or Path(__file__).resolve().parents[2]
    root = root.resolve()
    config = TjongConfig()
    model = TjongNetwork(config)
    model_parameters = model.parameter_count()
    outer_causal_ok, outer_causal_missing = _contains_all(
        root / "src" / "tjong_replication" / "model.py",
        [
            "causal_mask = torch.triu",
            "float(\"-inf\")",
            "diagonal=1",
            "mask=causal_mask",
        ],
    )

    checks: list[dict] = [
        _check("visible_tile_shape", (VISIBLE_TILE_ROWS, TILE_TYPES) == (22, 34), observed=[VISIBLE_TILE_ROWS, TILE_TYPES], expected=[22, 34]),
        _check("game_feature_shape", GAME_FEATURES == 24, observed=GAME_FEATURES, expected=24),
        _check("hidden_tile_shape", (HIDDEN_TILE_ROWS, TILE_TYPES) == (5, 34), observed=[HIDDEN_TILE_ROWS, TILE_TYPES], expected=[5, 34]),
        _check("memory_length", config.memory_len == 4, observed=config.memory_len, expected=4),
        _check("inner_transformer_layers", config.inner_layers == 3, observed=config.inner_layers, expected=3),
        _check("outer_transformer_layers", config.outer_layers == 3, observed=config.outer_layers, expected=3),
        _check(
            "policy_outer_transformer_uses_causal_memory_mask",
            outer_causal_ok,
            observed="src/tjong_replication/model.py",
            expected="outer TIT memory frames are masked from future frames",
            detail=f"missing: {outer_causal_missing}" if outer_causal_missing else None,
        ),
        _check(
            "policy_inner_transformer_receives_game_features",
            getattr(model.policy_backbone.inner, "game_projection", None) is not None
            and not hasattr(model.policy_backbone, "game_projection"),
            observed={
                "inner_game_feature_dim": model.policy_backbone.inner.game_feature_dim,
                "backbone_has_post_inner_game_projection": hasattr(model.policy_backbone, "game_projection"),
            },
            expected="game features are a token inside the policy Transformer blocks",
        ),
        _check(
            "value_network_uses_hidden_global_matrix",
            model.value_inner.rows == HIDDEN_TILE_ROWS and getattr(model.value_inner, "game_projection", None) is None,
            observed={
                "value_inner_rows": model.value_inner.rows,
                "value_inner_has_game_projection": getattr(model.value_inner, "game_projection", None) is not None,
            },
            expected="value input is the 5 x 34 hidden/global tile matrix",
        ),
        _check(
            "hidden_global_rows_include_concealed_kongs",
            tuple(HIDDEN_TILE_ROW_NAMES)
            == (
                "opponent_hand_1",
                "opponent_hand_2",
                "opponent_hand_3",
                "concealed_kongs",
                "remaining_wall",
            ),
            observed=list(HIDDEN_TILE_ROW_NAMES),
            expected="three opponent hands, concealed Kong tile counts, remaining wall",
        ),
        _check(
            "runtime_policy_hidden_schema_matches_tensorizer",
            runtime_hidden_schema_rows() == HIDDEN_TILE_ROW_NAMES,
            observed=list(runtime_hidden_schema_rows()),
            expected=list(HIDDEN_TILE_ROW_NAMES),
        ),
        _check("action_space", tuple(ACTION_NAMES) == EXPECTED_ACTION_NAMES, observed=list(ACTION_NAMES), expected=list(EXPECTED_ACTION_NAMES)),
        _check("claim_group_sizes", dict(CLAIM_GROUP_SIZES) == EXPECTED_CLAIM_GROUP_SIZES, observed=dict(CLAIM_GROUP_SIZES), expected=EXPECTED_CLAIM_GROUP_SIZES),
        _check("claim_head_size", CLAIM_SIZE == 199, observed=CLAIM_SIZE, expected=199),
        _check("discard_head_size", DISCARD_SIZE == 34, observed=DISCARD_SIZE, expected=34),
        _check(
            "model_parameter_count_about_15m",
            14_000_000 <= model_parameters <= 17_000_000,
            observed=model_parameters,
            expected="approximately 15M",
        ),
        _check(
            "paper_reported_supervised_metrics",
            PAPER_REPORTED_SUPERVISED_METRICS
            == {"action_accuracy": 0.9463, "claim_accuracy": 0.9855, "discard_accuracy": 0.8151},
            observed=dict(PAPER_REPORTED_SUPERVISED_METRICS),
            expected={"action_accuracy": 0.9463, "claim_accuracy": 0.9855, "discard_accuracy": 0.8151},
        ),
    ]

    notes_ok, notes_missing = _contains_all(
        root / "notes" / "extraction.md",
        [
            "Memory length: 4",
            "Inner blocks: 3 layers",
            "Outer blocks: 3 layers",
            "Supervised learning data: 519,338 Botzone battle logs",
            "PPO policy clip: `0.2`",
            "PPO value clip: `0.3`",
            "PPO gradient clip: `0.5`",
        ],
    )
    checks.append(
        _check(
            "paper_extraction_notes_present",
            notes_ok,
            observed="notes/extraction.md",
            expected="paper constants recorded",
            detail=f"missing: {notes_missing}" if notes_missing else None,
        )
    )

    ocr_ok, ocr_missing = _contains_all(
        root / "ocr" / "figure_ocr.md",
        [
            "Figure 1 - TIT Network Structure",
            "22 x 34 x 1",
            "Figure 3 - State Representation",
            "Figure 5 - Hierarchical Decision-Making Model",
            "Figure 6 - Structure of Tjong Network",
            "Figure 7 - Fan Backward Example",
        ],
    )
    checks.append(
        _check(
            "ocr_figure_transcript_present",
            ocr_ok,
            observed="ocr/figure_ocr.md",
            expected="OCR-derived architecture figures recorded",
            detail=f"missing: {ocr_missing}" if ocr_missing else None,
        )
    )

    repo_root = root.parent.parent
    tziakcha_backend = repo_root / "scripts" / "tziakcha_records.py"
    checks.append(
        _check(
            "tziakcha_to_botzone_converter_backend_present",
            tziakcha_backend.exists(),
            observed=str(tziakcha_backend),
            expected="scripts/tziakcha_records.py backend for converting user-provided Tziakcha data",
        )
    )
    checks.append(
        _check(
            "tziakcha_to_initdata_converter_present",
            (root / "src" / "tjong_replication" / "convert_tziakcha_to_initdata.py").exists()
            and (root / "src" / "tjong_replication" / "validate_initdata_corpus.py").exists(),
            observed="convert_tziakcha_to_initdata.py + validate_initdata_corpus.py",
            expected="available same-format Tziakcha logs can seed official-judge self-play",
        )
    )

    supervised_ok, supervised_missing = _manifest_contains(
        root / "k8s" / "tjong-supervised-rl-l40-all-sources-20260605b.yaml",
        [
            "NVIDIA-L40",
            "nvidia.com/gpu: 2",
            "EXPECTED_RECORDS=\"${TJONG_EXPECTED_RECORDS:-}\"",
            "python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding=\"utf-8\"))[\"records\"])' \"${VALIDATION_SUMMARY}\"",
            "tziakcha_all_sources_botzone_20260605b.jsonl",
            "tziakcha_all_sources_botzone_20260605b_tensorized.pt",
            "tjong_supervised_all_sources_20260605b.pt",
            "--epochs 125",
            "--batch-size 1024",
            "--lr 0.0001",
            "TRAIN_SEED=\"${TJONG_TRAIN_SEED:-0}\"",
            "--seed \"${TRAIN_SEED}\"",
            "--num-workers 0",
            "CUBLAS_WORKSPACE_CONFIG",
            "--require-encoding-version",
            "--require-paper-config",
            "--data-parallel",
            "--checkpoint-every-epochs 1",
            "CHECKPOINT_EVERY_BATCHES=\"${TJONG_CHECKPOINT_EVERY_BATCHES:-0}\"",
            "--checkpoint-every-batches",
            "--checkpoint-at-global-batches",
            "--checkpoint-at-epoch-batches",
            "CHECKPOINT_DIR=\"${TJONG_SUPERVISED_CHECKPOINT_DIR:-models/tjong_supervised_all_sources_20260605b_checkpoints}\"",
            "--checkpoint-dir \"${CHECKPOINT_DIR}\"",
            "METRICS_JSONL=\"${TJONG_METRICS_JSONL:-runs/tjong_supervised_all_sources_20260605b_metrics.jsonl}\"",
            "--metrics-jsonl \"${METRICS_JSONL}\"",
            "TJONG_RESUME_CHECKPOINT",
            "elif [ -s \"${CHECKPOINT_DIR}/latest.pt\" ]; then",
            "RESUME_ARGS=\"--resume-checkpoint ${CHECKPOINT_DIR}/latest.pt\"",
            "evaluate_supervised",
            "--strict-deterministic",
            "verify_paper_metrics",
            "--require-strict-deterministic",
            f'TENSOR_ENCODING_VERSION="${{TJONG_TENSOR_ENCODING_VERSION:-{TENSOR_ENCODING_VERSION}}}"',
        ],
    )
    checks.append(
        _check(
            "supervised_l40_manifest_matches_paper_training",
            supervised_ok,
            observed="k8s/tjong-supervised-rl-l40-all-sources-20260605b.yaml",
            expected="2-GPU SL pod with Tziakcha-to-Botzone conversion, Adam lr, batch, epochs, deterministic metric gate",
            detail=f"missing: {supervised_missing}" if supervised_missing else None,
        )
    )

    hu_supervised_manifest = root / "k8s" / "tjong-supervised-hu-l40-20260607f.yaml"
    hu_supervised_ok, hu_supervised_missing = _manifest_contains(
        hu_supervised_manifest,
        [
            "DATA_TAG=20260607f",
            "tziakcha_all_sources_botzone_hu_${DATA_TAG}_shards/index.json",
            "expected_actions = {",
            "\"CHOW\": 1829984",
            "\"PONG\": 1274394",
            "\"MINGKONG\": 58468",
            "\"BUKONG\": 71554",
            "\"ANKONG\": 44483",
            "\"HU\": 745368",
            "parallel_workers",
            "validate_paper_supervised_args",
            "CrossEntropyLoss",
            "ACTION_NAMES",
            "CLAIM_GROUP_SIZES",
            "--epochs \"${EPOCHS}\"",
            "EPOCHS=\"${TJONG_SUPERVISED_EPOCHS:-125}\"",
            "--batch-size \"${BATCH_SIZE}\"",
            "BATCH_SIZE=\"${TJONG_BATCH_SIZE:-1024}\"",
            "--lr 0.0001",
            "--data-parallel",
            "nvidia.com/gpu: 2",
            "rci-tide-gpu-17.sdsu.edu",
        ],
    )
    hu_supervised_text = hu_supervised_manifest.read_text(encoding="utf-8", errors="replace") if hu_supervised_manifest.exists() else ""
    checks.append(
        _check(
            "current_hu_supervised_manifest_matches_paper_hierarchy",
            hu_supervised_ok
            and "nautilus.io/reservation=csu-tide" not in hu_supervised_text
            and "--grad-clip" not in hu_supervised_text,
            observed="k8s/tjong-supervised-hu-l40-20260607f.yaml",
            expected="verified HU shard index, 2-GPU hierarchical CE supervised training, paper lr/batch/epochs, no CSU-TIDE toleration",
            detail=f"missing: {hu_supervised_missing}" if hu_supervised_missing else None,
        )
    )

    strict_eval_ok, strict_eval_missing = _manifest_contains(
        root / "k8s" / "tjong-supervised-final-strict-eval-l40-20260605b.yaml",
        [
            "NVIDIA-L40",
            "nvidia.com/gpu: 1",
            "CUBLAS_WORKSPACE_CONFIG",
            "evaluate_supervised",
            "models/tjong_supervised_all_sources_20260605b.pt",
            "tziakcha_all_sources_botzone_20260605b_tensorized.pt",
            "--strict-deterministic",
            "verify_paper_metrics",
            "--require-strict-deterministic",
        ],
    )
    checks.append(
        _check(
            "strict_final_supervised_eval_manifest_rebuilds_gate",
            strict_eval_ok,
            observed="k8s/tjong-supervised-final-strict-eval-l40-20260605b.yaml",
            expected="strict deterministic final supervised eval/gate job for existing checkpoints",
            detail=f"missing: {strict_eval_missing}" if strict_eval_missing else None,
        )
    )

    eval_sweep_ok, eval_sweep_missing = _manifest_contains(
        root / "k8s" / "tjong-supervised-eval-sweep-l40-20260605b.yaml",
        [
            "NVIDIA-L40",
            "nvidia.com/gpu: 1",
            "evaluate_checkpoints",
            "--checkpoint \"${FINAL_CHECKPOINT}\"",
            "--checkpoint-glob \"${CHECKPOINT_DIR}/epoch_*.pt\"",
            "--checkpoint-glob \"${CHECKPOINT_DIR}/batch_*.pt\"",
            "models/tjong_supervised_all_sources_20260605b.pt",
            "models/tjong_supervised_all_sources_20260605b_checkpoints",
            "--eval-pt \"${EVAL_PT}\"",
            "tziakcha_all_sources_botzone_20260605b_tensorized.pt",
            "--metrics-out runs/tjong_supervised_all_sources_checkpoint_sweep_20260605b.json",
            "--results-jsonl \"${RESULTS_JSONL}\"",
            "--require-encoding-version",
            "--require-paper-config",
            "--seed 0",
            "--strict-deterministic",
            "CUBLAS_WORKSPACE_CONFIG",
            "--device cuda",
        ],
    )
    checks.append(
        _check(
            "supervised_checkpoint_sweep_manifest_evaluates_same_full_validation_set",
            eval_sweep_ok and (root / "src" / "tjong_replication" / "evaluate_checkpoints.py").exists(),
            observed="k8s/tjong-supervised-eval-sweep-l40-20260605b.yaml",
            expected="deterministic full-validation checkpoint sweep for final, epoch, and optional batch checkpoints",
            detail=f"missing: {eval_sweep_missing}" if eval_sweep_missing else None,
        )
    )

    status_ok, status_missing = _manifest_contains(
        root / "k8s" / "tjong-pipeline-status-cpu-20260605a.yaml",
        [
            "pipeline_status",
            "--repo-root .",
            "--expected-records \"${EXPECTED_RECORDS}\"",
            "--summary-out runs/tjong_pipeline_status_20260605a.json",
            "TJONG_EXPECTED_RECORDS",
        ],
    )
    checks.append(
        _check(
            "pipeline_status_manifest_audits_end_to_end_artifacts",
            status_ok and (root / "src" / "tjong_replication" / "pipeline_status.py").exists(),
            observed="k8s/tjong-pipeline-status-cpu-20260605a.yaml",
            expected="CPU status job reports which paper-replication artifact stage is next",
            detail=f"missing: {status_missing}" if status_missing else None,
        )
    )

    convert_ok, convert_missing = _manifest_contains(
        root / "k8s" / "tjong-tziakcha-convert-cpu-20260605a.yaml",
        [
            "convert_tziakcha_to_botzone",
            "validate_paper_corpus",
            "EXPECTED_RECORDS=\"${TJONG_EXPECTED_RECORDS:-763358}\"",
            "tziakcha_all_plus_live_elo2300_session_audit_raw.jsonl",
            "tziakcha_all_plus_live_elo2300_botzone.jsonl",
            "--fail-on-error",
        ],
    )
    checks.append(
        _check(
            "tziakcha_conversion_cpu_manifest_prepares_own_dataset",
            convert_ok,
            observed="k8s/tjong-tziakcha-convert-cpu-20260605a.yaml",
            expected="CPU pod converts the available Tziakcha corpus into validated Botzone-style logs",
            detail=f"missing: {convert_missing}" if convert_missing else None,
        )
    )

    merge_all_sources_ok, merge_all_sources_missing = _manifest_contains(
        root / "k8s" / "tjong-tziakcha-all-botzone-merge-cpu-20260605b.yaml",
        [
            "merge_botzone_corpora",
            "validate_paper_corpus",
            "audit_tziakcha_botzone_coverage",
            "tziakcha_all_plus_live_elo2300_botzone.jsonl",
            "tziakcha_human_botzone_raw_32.jsonl",
            "tziakcha_human_botzone_raw_517.jsonl",
            "tziakcha_chaga0208_botzone_raw.jsonl",
            "tziakcha_all_sources_botzone_20260605b.jsonl",
            "--min-written 763471",
            "--fail-on-error",
        ],
    )
    checks.append(
        _check(
            "tziakcha_all_sources_merge_manifest_prepares_full_own_dataset",
            merge_all_sources_ok,
            observed="k8s/tjong-tziakcha-all-botzone-merge-cpu-20260605b.yaml",
            expected="CPU pod merges every available Tziakcha Botzone source into the consolidated same-format corpus",
            detail=f"missing: {merge_all_sources_missing}" if merge_all_sources_missing else None,
        )
    )

    initdata_ok, initdata_missing = _manifest_contains(
        root / "k8s" / "tjong-tziakcha-initdata-cpu-20260605b.yaml",
        [
            "convert_tziakcha_to_initdata",
            "validate_initdata_corpus",
            "EXPECTED_RECORDS=\"${TJONG_RAW_INITDATA_EXPECTED_RECORDS:-763475}\"",
            "tziakcha_all_sources_botzone_20260605b.jsonl",
            "tziakcha_all_sources_botzone_20260605b_initdata.jsonl",
            "--allow-random-fallback",
            "--fail-on-error",
        ],
    )
    checks.append(
        _check(
            "tziakcha_initdata_cpu_manifest_prepares_selfplay_seeds",
            initdata_ok,
            observed="k8s/tjong-tziakcha-initdata-cpu-20260605b.yaml",
            expected="CPU pod converts the available Tziakcha Botzone logs into official-judge initdata",
            detail=f"missing: {initdata_missing}" if initdata_missing else None,
        )
    )

    all_sources_prep_ok, all_sources_prep_missing = _manifest_contains(
        root / "k8s" / "tjong-tziakcha-all-sources-prep-cpu-20260605c.yaml",
        [
            "validate_paper_corpus",
            "tensorize_botzone",
            "convert_tziakcha_to_initdata",
            "validate_initdata_corpus",
            "EXPECTED_RECORDS=\"${TJONG_EXPECTED_RECORDS:-}\"",
            "python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding=\"utf-8\"))[\"records\"])' \"${VALIDATION_SUMMARY}\"",
            "tziakcha_all_sources_botzone_20260605b.jsonl",
            "tziakcha_all_sources_botzone_20260605b_tensorized.pt",
            "tziakcha_all_sources_botzone_20260605b_initdata.jsonl",
            "--allow-random-fallback",
            "--fail-on-error",
        ],
    )
    checks.append(
        _check(
            "tziakcha_all_sources_prep_manifest_tensorizes_and_seeds_selfplay",
            all_sources_prep_ok,
            observed="k8s/tjong-tziakcha-all-sources-prep-cpu-20260605c.yaml",
            expected="CPU pod validates, tensorizes, and builds initdata for the consolidated all-source Tziakcha corpus",
            detail=f"missing: {all_sources_prep_missing}" if all_sources_prep_missing else None,
        )
    )

    selfplay_ok, selfplay_missing = _manifest_contains(
        root / "k8s" / "tjong-selfplay-rollout-l40-20260605b.yaml",
        [
            "NVIDIA-L40",
            "nvidia.com/gpu: 1",
            "apt-get install -y libboost-dev libjsoncpp-dev",
            "external/Chinese-Standard-Mahjong/judge/main.cpp",
            "build/official_judge/jsoncpp",
            "build/official_judge/mcr_judge.exe",
            "build/official_judge/mcr_fan_check.exe",
            "python -m pip install --no-cache-dir PyMahjongGB",
            "MahjongGB import ok",
            "convert_tziakcha_to_initdata",
            "validate_initdata_corpus",
            "EXPECTED_RECORDS=\"${TJONG_RAW_INITDATA_EXPECTED_RECORDS:-763475}\"",
            "GAMES=\"${TJONG_SELFPLAY_GAMES:-763475}\"",
            "SELFPLAY_MIN_GAMES=\"${TJONG_SELFPLAY_MIN_GAMES:-519338}\"",
            "MIN_TOTAL_HU_RATE=\"${TJONG_MIN_TOTAL_HU_RATE:-0.99}\"",
            "MAX_HUANG_RATE=\"${TJONG_MAX_HUANG_RATE:-0.01}\"",
            "tziakcha_all_sources_botzone_20260605b.jsonl",
            "tziakcha_all_sources_botzone_20260605b_initdata.jsonl",
            "--allow-random-fallback",
            "collect_selfplay",
            "models/tjong_supervised_all_sources_20260605b.pt",
            "summary.get(\"games\"",
            "summary[\"score_table\"]",
            "summary[\"huang_rate\"]",
            "tjong_selfplay_20260605b_fan_items.jsonl",
            "tensorize_botzone",
            "--require-encoding-version",
            "--require-paper-config",
            f'TENSOR_ENCODING_VERSION="${{TJONG_TENSOR_ENCODING_VERSION:-{TENSOR_ENCODING_VERSION}}}"',
        ],
    )
    checks.append(
        _check(
            "selfplay_l40_manifest_uses_paper_initdata_and_fan_items",
            selfplay_ok,
            observed="k8s/tjong-selfplay-rollout-l40-20260605b.yaml",
            expected="self-play pod generates/validates Tziakcha-derived initdata and emits fan-display items",
            detail=f"missing: {selfplay_missing}" if selfplay_missing else None,
        )
    )

    sharded_ok, sharded_missing = _manifest_contains(
        root / "k8s" / "tjong-selfplay-rollout-sharded-l40-20260605b.yaml",
        [
            "completionMode: Indexed",
            "completions: 32",
            "parallelism: 8",
            "NVIDIA-L40",
            "TJONG_SELFPLAY_TOTAL_GAMES:-763475",
            "TJONG_SELFPLAY_SHARDS:-32",
            "tziakcha_all_sources_botzone_20260605b_initdata.jsonl",
            "models/tjong_supervised_all_sources_20260605b.pt",
            "SHARD_GAMES",
            "OFFSET",
            "--fan-checker",
            "tjong_selfplay_raw_shard_${SHARD_TAG}.jsonl",
            "summary[\"score_table\"]",
            "MIN_TOTAL_HU_RATE",
            "MAX_HUANG_RATE",
        ],
    )
    merge_ok, merge_missing = _manifest_contains(
        root / "k8s" / "tjong-selfplay-merge-cpu-20260605b.yaml",
        [
            "merge_selfplay_shards",
            "--expected-games \"${EXPECTED_GAMES}\"",
            "--min-total-hu-rate \"${MIN_TOTAL_HU_RATE}\"",
            "--max-huang-rate \"${MAX_HUANG_RATE}\"",
            "tensorize_botzone",
            "tjong_selfplay_20260605b_tensorized.pt",
            "record_error_lines",
            "TENSOR_ENCODING_VERSION",
        ],
    )
    checks.append(
        _check(
            "sharded_selfplay_manifests_cover_paper_scale_rollout",
            sharded_ok
            and merge_ok
            and (root / "src" / "tjong_replication" / "merge_selfplay_shards.py").exists(),
            observed="k8s/tjong-selfplay-rollout-sharded-l40-20260605b.yaml + k8s/tjong-selfplay-merge-cpu-20260605b.yaml",
            expected="Indexed L40 self-play shards plus CPU merge/tensorize path for the full paper-scale rollout",
            detail=(
                f"sharded missing: {sharded_missing}; merge missing: {merge_missing}"
                if sharded_missing or merge_missing
                else None
            ),
        )
    )

    ppo_ok, ppo_missing = _manifest_contains(
        root / "k8s" / "tjong-ppo-l40-20260605b.yaml",
        [
            "NVIDIA-L40",
            "nvidia.com/gpu: 2",
            "models/tjong_supervised_all_sources_20260605b.pt",
            "populate_fan_backward_rewards",
            "--reward-key fan_backward_reward",
            "--require-reward-key",
            "--require-rollout-reward-source fan_backward_reward",
            "--require-hu-reward",
            "HUANG_PENALTY=\"${TJONG_HUANG_PENALTY:-100.0}\"",
            "--huang-penalty \"${HUANG_PENALTY}\"",
            "summary[\"huang_penalty\"] > 0",
            "summary[\"huang_penalized_matches\"] == summary[\"huang_matches\"]",
            "hu_matches_without_reward",
            "--require-encoding-version",
            "--require-paper-config",
            "--policy-clip 0.2",
            "--value-clip 0.3",
            "--grad-clip 0.5",
            "--data-parallel",
            "--checkpoint-every-epochs 1",
            "--checkpoint-dir models/tjong_ppo_20260605b_checkpoints",
            "--metrics-jsonl runs/tjong_ppo_20260605b_metrics.jsonl",
            "TJONG_PPO_RESUME_CHECKPOINT",
            "rci-tide-gpu-17.sdsu.edu",
        ],
    )
    ppo_text = (root / "k8s" / "tjong-ppo-l40-20260605b.yaml").read_text(encoding="utf-8", errors="replace")
    checks.append(
        _check(
            "ppo_l40_manifest_matches_paper_rl_constants",
            ppo_ok and "nautilus.io/reservation=csu-tide" not in ppo_text,
            observed="k8s/tjong-ppo-l40-20260605b.yaml",
            expected="2-GPU PPO pod with fan-backward rewards and paper clipping constants",
            detail=f"missing: {ppo_missing}" if ppo_missing else None,
        )
    )

    summary = {
        "paper": PAPER_NAME,
        "paper_root": str(root),
        "source_links": PAPER_SOURCE_LINKS,
        "passed": all(check["passed"] for check in checks),
        "model_parameters": model_parameters,
        "checks": checks,
    }
    return summary


def verify_paper_compliance_file(paper_root: Path | None = None, summary_out: Path | None = None) -> dict:
    summary = verify_paper_compliance(paper_root)
    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not summary["passed"]:
        raise ValueError(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", default=None)
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()
    try:
        summary = verify_paper_compliance_file(
            Path(args.paper_root) if args.paper_root else None,
            Path(args.summary_out) if args.summary_out else None,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
