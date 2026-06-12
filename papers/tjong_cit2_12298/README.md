# Tjong Replication

This directory is a clean replication workspace for:

Li et al., "Tjong: A transformer-based Mahjong AI via hierarchical decision-making and fan backward", CAAI Transactions on Intelligence Technology, 2024, DOI `10.1049/cit2.12298`.

Source status:

- The article is gold open access and the metadata points to Wiley `pdfdirect`.
- Local shell/browser downloads from Wiley and ResearchGate are blocked by Cloudflare/ResearchGate security checks in this environment.
- The paper text and rendered page screenshots were accessed through the web reader and recorded in `notes/` and `ocr/`.
- The original GitHub URL in the paper is `https://github.com/750251985/Tjong`; it did not resolve as a public repository during this pass.

Implementation status:

- `src/tjong_replication/` contains a standalone PyTorch scaffold for the paper method:
  - TIT backbone with inner and outer Transformer blocks.
  - Memory length 4.
  - Three-layer inner and outer blocks.
  - Hierarchical heads: action, claiming tile, discard tile.
  - Two-pass hierarchical policy forward path: action logits come from the original state, while claim/discard logits can come from an action-conditioned sub-state passed through the same TIT blocks again.
  - Value network using the global `5 x 34` hidden tile matrix.
  - Fan backward reward implementation.
  - Supervised-learning and PPO training skeletons with paper hyperparameters.

The code keeps paper ambiguities explicit in configs and comments instead of silently hard-coding guesses.

## Reproduction Commands

Convert the available Tziakcha records into Botzone-style MCR logs, merge every Tziakcha source into one deduplicated all-source corpus, then tensorize those logs into the paper-shaped tensors:

```powershell
$env:PYTHONPATH="D:\MCR_Agent\papers\tjong_cit2_12298\src"
python -m tjong_replication.convert_tziakcha_to_botzone `
  --in data\raw\tziakcha_all_plus_live_elo2300_session_audit_raw.jsonl `
  --out data\raw\tziakcha_all_plus_live_elo2300_botzone.jsonl `
  --summary-out runs\tjong_tziakcha_to_botzone_summary.json `
  --min-written 763358 `
  --fail-on-error

python -m tjong_replication.merge_botzone_corpora `
  --in data\raw\tziakcha_all_plus_live_elo2300_botzone.jsonl `
  --in data\raw\tziakcha_human_botzone_raw_32.jsonl `
  --in data\raw\tziakcha_human_botzone_raw_517.jsonl `
  --in data\raw\tziakcha_chaga0208_botzone_raw.jsonl `
  --out data\raw\tziakcha_all_sources_botzone_20260605b.jsonl `
  --summary-out runs\tjong_tziakcha_all_sources_botzone_20260605b_summary.json `
  --min-written 763475 `
  --fail-on-error

python -m tjong_replication.audit_tziakcha_botzone_coverage `
  --repo-root . `
  --merge-summary runs\tjong_tziakcha_all_sources_botzone_20260605b_summary.json `
  --validation-summary runs\tjong_validate_tziakcha_all_sources_botzone_20260605b.json `
  --summary-out runs\tjong_tziakcha_all_sources_botzone_coverage_20260605b.json `
  --fail-on-missing

python -m tjong_replication.verify_paper_compliance `
  --paper-root D:\MCR_Agent\papers\tjong_cit2_12298 `
  --summary-out runs\tjong_paper_compliance.json

python -m tjong_replication.validate_paper_corpus `
  --in data\raw\tziakcha_all_sources_botzone_20260605b.jsonl `
  --expected-records <records_written_from_merge_summary> `
  --summary-out runs\tjong_validate_tziakcha_all_sources_botzone_20260605b.json

python -m tjong_replication.tensorize_botzone `
  --in data\raw\tziakcha_all_sources_botzone_20260605b.jsonl `
  --out data\processed\tjong\tziakcha_all_sources_botzone_20260605b_tensorized.pt `
  --summary-out runs\tjong_tensorize_all_sources_20260605b_summary.json `
  --corpus-validation runs\tjong_validate_tziakcha_all_sources_botzone_20260605b.json
```

The earlier `20260605a` all-source merge is superseded because it omitted the real `tziakcha_human_botzone_raw_32.jsonl` shard. Keep it only as historical evidence for the already-running job; use the `20260605b` corpus for the corrected all-real-source run.

Run the supervised-learning phase with the paper hyperparameters:

```powershell
$env:PYTHONPATH="D:\MCR_Agent\papers\tjong_cit2_12298\src"
python -m tjong_replication.train_supervised `
  --train-pt data\processed\tjong\tziakcha_all_sources_botzone_20260605b_tensorized.pt `
  --checkpoint-out models\tjong_supervised_all_sources_20260605b.pt `
  --metrics-out runs\tjong_supervised_all_sources_20260605b_metrics.json `
  --metrics-jsonl runs\tjong_supervised_all_sources_20260605b_metrics.jsonl `
  --epochs 125 `
  --batch-size 1024 `
  --lr 0.0001 `
  --num-workers 0 `
  --require-encoding-version tjong_cit2_12298_v3_hidden_concealed_kong `
  --data-parallel `
  --checkpoint-every-epochs 1 `
  --checkpoint-dir models\tjong_supervised_all_sources_20260605b_checkpoints
```

Evaluate the supervised checkpoint deterministically with the paper-style offline metrics:

```powershell
$env:PYTHONPATH="D:\MCR_Agent\papers\tjong_cit2_12298\src"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
python -m tjong_replication.evaluate_supervised `
  --checkpoint models\tjong_supervised_all_sources_20260605b.pt `
  --eval-pt data\processed\tjong\tziakcha_all_sources_botzone_20260605b_tensorized.pt `
  --metrics-out runs\tjong_supervised_all_sources_20260605b_eval.json `
  --batch-size 1024 `
  --num-workers 0 `
  --require-encoding-version tjong_cit2_12298_v3_hidden_concealed_kong `
  --require-paper-config `
  --strict-deterministic

python -m tjong_replication.verify_paper_metrics `
  --metrics runs\tjong_supervised_all_sources_20260605b_eval.json `
  --summary-out runs\tjong_supervised_all_sources_20260605b_paper_metric_gate.json `
  --require-strict-deterministic
```

Evaluate the final supervised checkpoint and any numbered epoch checkpoints on the same full validation tensor:

```powershell
$env:PYTHONPATH="D:\MCR_Agent\papers\tjong_cit2_12298\src"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
python -m tjong_replication.evaluate_checkpoints `
  --checkpoint models\tjong_supervised_all_sources_20260605b.pt `
  --checkpoint-glob "models\tjong_supervised_all_sources_20260605b_checkpoints\epoch_*.pt" `
  --checkpoint-glob "models\tjong_supervised_all_sources_20260605b_checkpoints\batch_*.pt" `
  --eval-pt data\processed\tjong\tziakcha_all_sources_botzone_20260605b_tensorized.pt `
  --metrics-out runs\tjong_supervised_all_sources_checkpoint_sweep_20260605b.json `
  --results-jsonl runs\tjong_supervised_all_sources_checkpoint_sweep_20260605b.jsonl `
  --batch-size 1024 `
  --num-workers 0 `
  --require-encoding-version tjong_cit2_12298_v3_hidden_concealed_kong `
  --require-paper-config `
  --strict-deterministic `
  --seed 0
```

Verify that the deployable platform wrappers feed the model the same state that training used before using a checkpoint for Botzone, Tziakcha, smoke self-play, or paper self-play:

```powershell
$env:PYTHONPATH="D:\MCR_Agent\papers\tjong_cit2_12298\src"
python -m tjong_replication.verify_platform_wrappers `
  --checkpoint models\tjong_supervised_all_sources_20260605b.pt `
  --raw data\raw\tziakcha_all_sources_botzone_20260605b.jsonl `
  --max-states 1000 `
  --nonpass-only `
  --device cpu `
  --require-encoding-version tjong_cit2_12298_v3_hidden_concealed_kong `
  --require-paper-config `
  --min-states 100 `
  --fail-on-error `
  --out runs\tjong_platform_wrapper_gate_20260605b.json
```

This gate checks three deployment invariants in one report: Botzone JSON and Tziakcha complete-history snapshots produce identical predictor input/candidates on the same state, illegal model outputs fall back exactly like the Botzone runtime instead of using a smoke-test heuristic, and the wrapper can replay held-out Botzone-format training logs with zero state-replay errors while reporting exact and action-family agreement. The Kubernetes manifest `k8s/tjong-platform-wrapper-gate-cpu-20260612a.yaml` runs the same gate on CPU against a PVC checkpoint path, defaults to `models/tjong_supervised_hu_ddp_full_4x4090_20260608d_checkpoints/latest.pt`, requests no GPUs, and excludes `rci-tide-*` nodes. Like the long training manifest, it downloads the configured GitHub branch archive into `/tmp` before running so the verifier code is not accidentally taken from stale PVC files.

Collect Tjong self-play games through the official judge, then tensorize the resulting decisions:

```powershell
$env:PYTHONPATH="D:\MCR_Agent\papers\tjong_cit2_12298\src"
python -m tjong_replication.convert_tziakcha_to_initdata `
  --in data\raw\tziakcha_all_sources_botzone_20260605b.jsonl `
  --out data\processed\tjong\tziakcha_all_sources_botzone_20260605b_initdata.jsonl `
  --summary-out runs\tjong_tziakcha_all_sources_initdata_20260605b_summary.json `
  --min-written <records_written_from_merge_summary> `
  --allow-random-fallback `
  --fail-on-error

python -m tjong_replication.validate_initdata_corpus `
  --in data\processed\tjong\tziakcha_all_sources_botzone_20260605b_initdata.jsonl `
  --expected-records <records_written_from_merge_summary> `
  --summary-out runs\tjong_validate_tziakcha_all_sources_initdata_20260605b.json

python -m tjong_replication.collect_selfplay `
  --checkpoint models\tjong_supervised_all_sources_20260605b.pt `
  --raw data\processed\tjong\tziakcha_all_sources_botzone_20260605b_initdata.jsonl `
  --games 763475 `
  --judge build\official_judge\mcr_judge.exe `
  --out-raw data\processed\tjong\tjong_selfplay_20260605b_raw.jsonl `
  --out-fan-items data\processed\tjong\tjong_selfplay_20260605b_fan_items.jsonl `
  --summary-out runs\tjong_selfplay_20260605b_summary.json `
  --require-encoding-version tjong_cit2_12298_v3_hidden_concealed_kong `
  --require-paper-config `
  --device cuda

python -m tjong_replication.tensorize_botzone `
  --in data\processed\tjong\tjong_selfplay_20260605b_raw.jsonl `
  --out data\processed\tjong\tjong_selfplay_20260605b_tensorized.pt `
  --summary-out runs\tjong_selfplay_tensorize_20260605b_summary.json
```

The self-play initdata converter first reconstructs official-judge walls from the converted Tziakcha replay. If a replay cannot fit the judge's 36-tile per-seat wall segment layout, `--allow-random-fallback` emits a deterministic legal 144-tile random wall for that record, matching the user-approved fallback for insufficient usable games. The self-play collector records fan names and values from the judge display. Botzone display records do not contain per-fan tile sets, so `populate_fan_backward_rewards` first prefers exact fan item JSONL `tiles`, then uses deterministic structural attribution for supported fan names. If any terminal fan name cannot be mapped to a 34-tile support set, the faithful path fails instead of falling back to approximation.

Run the PPO phase on fan-backward self-play rollout tensors:

```powershell
$env:PYTHONPATH="D:\MCR_Agent\papers\tjong_cit2_12298\src"
python -m tjong_replication.train_ppo `
  --init-checkpoint models\tjong_supervised_all_sources_20260605b.pt `
  --rollout-pt data\processed\tjong\tjong_selfplay_20260605b_fan_backward_rollouts.pt `
  --checkpoint-out models\tjong_ppo_20260605b.pt `
  --metrics-out runs\tjong_ppo_20260605b_metrics.json `
  --policy-clip 0.2 `
  --value-clip 0.3 `
  --grad-clip 0.5 `
  --require-encoding-version tjong_cit2_12298_v3_hidden_concealed_kong `
  --require-paper-config `
  --require-rollout-reward-source fan_backward_reward `
  --data-parallel
```

Prepare a PPO rollout tensor from tensorized self-play decisions and fan-backward rewards:

```powershell
$env:PYTHONPATH="D:\MCR_Agent\papers\tjong_cit2_12298\src"
python -m tjong_replication.build_ppo_rollouts `
  --tensor-pt data\processed\tjong\tjong_selfplay_20260605b_fan_backward_tensorized.pt `
  --checkpoint models\tjong_supervised_all_sources_20260605b.pt `
  --out data\processed\tjong\tjong_selfplay_20260605b_fan_backward_rollouts.pt `
  --reward-key fan_backward_reward `
  --require-reward-key `
  --require-encoding-version tjong_cit2_12298_v3_hidden_concealed_kong `
  --require-paper-config `
  --gamma 1.0 `
  --normalize-advantage
```

Populate exact fan-backward rewards before rollout building:

```powershell
$env:PYTHONPATH="D:\MCR_Agent\papers\tjong_cit2_12298\src"
python -m tjong_replication.populate_fan_backward_rewards `
  --tensor-pt data\processed\tjong\tjong_selfplay_20260605b_tensorized.pt `
  --raw data\processed\tjong\tjong_selfplay_20260605b_raw.jsonl `
  --fan-items-jsonl data\processed\tjong\tjong_selfplay_20260605b_fan_items.jsonl `
  --out data\processed\tjong\tjong_selfplay_20260605b_fan_backward_tensorized.pt `
  --summary-out runs\tjong_fan_backward_20260605b_summary.json
```

The Kubernetes manifest `k8s/tjong-tziakcha-convert-cpu-20260605a.yaml` converts the main available Tziakcha source corpus into Botzone-style logs on the shared PVC without occupying L40 GPUs. The corrected follow-up all-source merge job `k8s/tjong-tziakcha-all-botzone-merge-cpu-20260605b.yaml` merges those logs with `tziakcha_human_botzone_raw_32.jsonl`, `tziakcha_human_botzone_raw_517.jsonl`, and `tziakcha_chaga0208_botzone_raw.jsonl`, deduplicates by `match_id`, validates the canonical `data/raw/tziakcha_all_sources_botzone_20260605b.jsonl` corpus, and writes a coverage audit proving all real local Tziakcha Botzone sources are represented while synthetic suit augmentation is excluded.

The Kubernetes manifest `k8s/tjong-pipeline-status-cpu-20260605a.yaml` writes `runs/tjong_pipeline_status_20260605a.json`, an end-to-end artifact readiness report covering OCR extraction, corpus conversion, tensorization, initdata, supervised training/eval, checkpoint sweep, self-play, fan-backward rollouts, and PPO.

The Kubernetes manifest `k8s/tjong-supervised-rl-l40-all-sources-20260605b.yaml` is the corrected all-real-source supervised run. It must start after `k8s/tjong-tziakcha-all-sources-prep-cpu-20260605c.yaml` completes and uses the `20260605b` tensor with the same 2-GPU L40 paper training constants, strict deterministic final evaluation, and `verify_paper_metrics` gate. The earlier `20260605a` supervised pod is historical only because it trained over the superseded tensor.

The Kubernetes manifest `k8s/tjong-supervised-eval-sweep-l40-20260605b.yaml` is the post-SL deterministic checkpoint sweep. It evaluates `models/tjong_supervised_all_sources_20260605b.pt` plus any `models/tjong_supervised_all_sources_20260605b_checkpoints/epoch_*.pt` and optional `batch_*.pt` files on the same full tensorized all-source validation corpus, records each action/claim/discard metric to an incremental JSONL file, and reports the best checkpoint by each accuracy/loss metric in the final JSON summary. Batch snapshots are eval-only and can be requested in future/resumed supervised jobs with `TJONG_CHECKPOINT_EVERY_BATCHES`, `TJONG_CHECKPOINT_AT_GLOBAL_BATCHES`, or `TJONG_CHECKPOINT_AT_EPOCH_BATCHES`.

The local monitor `scripts/monitor_tjong_pipeline_autolaunch.ps1` can be run from the repo root to watch the long paper pipeline and apply the next manifest only after the previous job has succeeded and the expected PVC artifacts exist. In order, it gates on the final supervised checkpoint plus paper metric gate, applies the deterministic checkpoint sweep, then applies sharded self-play, merge/tensorization, and PPO. Run with `-Once` to log current state without launching anything, or `-StopAfterEval` to stop after the checkpoint sweep.

The self-play manifest `k8s/tjong-selfplay-rollout-l40-20260605b.yaml` runs four legal Tjong checkpoint policies through the official judge, writes raw self-play logs plus fan-display records, and tensorizes the decisions. Its paper-mode default is the full available `763,475` all-source initdata seeds, with `TJONG_SELFPLAY_MIN_GAMES` defaulting to the paper's `519,338` supervised-log count as a lower bound; smoke runs must override `TJONG_SELFPLAY_GAMES` explicitly. The self-play summary reports each seat's score label A-D, HU rate over total games, average HU turn, average raw score, and placement counts. The pod gates rollout health with `TJONG_MIN_TOTAL_HU_RATE=0.99` and `TJONG_MAX_HUANG_RATE=0.01` by default, matching the requirement that all four seats collectively HU in about 99% of games and HUANG stays below 1%. Exact fan item JSONL remains the strongest source; otherwise the PPO preparation step can use structural display attribution for covered fan names.

For the paper-scale run, prefer the sharded rollout manifests over the single-pod fallback:

```powershell
kubectl apply -f papers\tjong_cit2_12298\k8s\tjong-selfplay-rollout-sharded-l40-20260605b.yaml
kubectl wait --for=condition=complete job/tjong-selfplay-rollout-sharded-l40-20260605b -n nourish-sdsc --timeout=259200s
kubectl apply -f papers\tjong_cit2_12298\k8s\tjong-selfplay-merge-cpu-20260605b.yaml
```

The sharded L40 job is an Indexed Job with `32` shards and `8` concurrent pods by default. Each shard computes its deterministic `offset`/`games` slice from `JOB_COMPLETION_INDEX`, writes raw/fan/summary JSONL under `data/processed/tjong/selfplay_shards_20260605b`, and uses per-pod temporary official-judge binaries to avoid shared-PVC build races. The CPU merge job concatenates shards in index order, recomputes the A-D HU/HUANG/raw-score summary, enforces the paper-scale rollout gates, and tensorizes the merged raw file into `data/processed/tjong/tjong_selfplay_20260605b_tensorized.pt` for PPO.

The PPO manifest `k8s/tjong-ppo-l40-20260605b.yaml` consumes a supervised checkpoint and self-play rollout tensor file containing fan-backward rewards/advantages. If `TJONG_ROLLOUT_SOURCE_PT` is absent, it first populates fan-backward rewards from `TJONG_SELFPLAY_TENSOR_PT`, `TJONG_SELFPLAY_RAW`, and optional `TJONG_FAN_ITEMS_JSONL`; exact JSON fan tiles are preferred, otherwise deterministic structural fan attribution is used for supported terminal fan names. If `TJONG_ROLLOUT_PT` is absent, it then builds it from `TJONG_ROLLOUT_SOURCE_PT` by freezing old policy log-probabilities and value estimates from the supervised checkpoint. The pod passes `--require-reward-key`, asserts `actual_reward_source=fan_backward_reward`, and passes `--require-rollout-reward-source fan_backward_reward` into the trainer, so terminal-score fallback cannot silently enter a paper PPO run. It intentionally fails fast if real self-play tensors are absent or fan names cannot be attributed, because the paper-scale RL phase depends on actual self-play data rather than supervised-log labels.

For a faithful PPO replication, `runs/tjong_rollout_build_20260605b_summary.json` must report `actual_reward_source` as `fan_backward_reward`. The local smoke path can fall back to scaled terminal scores only to verify tensor plumbing.

## Local Smoke Results

- Focused tests: `48 passed`.
- Cluster data audit: `k8s/tjong-data-audit-cpu-20260605a.yaml` and `k8s/tjong-storage-search-cpu-20260605a.yaml` verified that `llama-model-pvc` is mounted and bound. The exact paper corpus is unavailable, so the faithful run now uses the available same-format Tziakcha corpus after conversion to Botzone-style logs. Evidence is recorded in `artifacts/tjong_cluster_data_audit_20260605a.json`.
- Real-log tensorizer smoke: first 3 local Botzone matches produced 61 trainable examples, including 7 post-claim discard examples.
- Wider tensorizer smoke: first 25 local Botzone matches produced 488 trainable examples and covered Pass, Hu, Discard, Chow, Pong, MingKong, and BuKong.
- Second-pass hierarchy smoke: first 5 local Botzone matches produced 99 examples with `sub_visible_tiles`/`sub_game_features`; a tiny CPU trainer consumed the generated `.pt` file and completed one epoch.
- Deterministic evaluator smoke: a tiny CPU checkpoint produced action, claim, and discard loss/accuracy metrics and embeds the paper's reported offline targets for comparison.
- Rollout-builder smoke: a temporary tensorized dataset plus tiny checkpoint produced old log-probs, old values, returns, and normalized advantages. It reports `actual_reward_source=value_target_terminal_fallback`, so it is explicitly a plumbing smoke rather than a faithful fan-backward rollout.
- Explicit fan-backward reward test: a controlled discard-win game with explicit fan-tile attribution produced nonzero `fan_backward_reward` without approximation.
- PPO smoke: a tiny CPU supervised checkpoint and temporary rollout tensor completed one clipped PPO epoch with `policy_clip=0.2`, `value_clip=0.3`, and `grad_clip=0.5`.
