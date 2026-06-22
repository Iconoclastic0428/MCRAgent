# MCRAgent River-Elastic Feature Agent

This branch contains the final promoted feature-agent model and only the code
needed to reproduce, load, and evaluate it.

## Final Checkpoint

- Model: `models/feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a/16.pkl`
- SHA256: `2b982a838a15d9632f85ffdeee993105e077c5d92d236800140819607d9e2f3d`
- Git LFS is used for the checkpoint.

The checkpoint is the river-elastic epoch-16 model trained from the Botzone
98209-game dataset with 12x suit/number augmentation. Games containing LVYISE
or TUIBUDAO hands are kept unaugmented. The promoted model uses:

- 185 observation planes: the 85-plane vec-fix OBS representation plus
  25 river-property planes for each of four players.
- The original 117-dimensional VEC feature vector.
- Mixed-kernel input.
- Dueling value/advantage output head.
- AdamW with elastic regularization in training:
  `L1 = 0.01 * lr`, `L2 = 0.1 * lr`.

## Source Layout

- `third_party/mahjong-agent-2025-aug12-excludespecial-3pkl/feature-agent/`
  contains the architecture, feature construction, preprocessing, dataset,
  training loop, and validation helpers for this model.
- `scripts/feature_agent_checkpoint_json_bot.py` loads the promoted checkpoint
  as a Botzone-compatible JSON/text policy.
- `scripts/feature_repo_json_runtime.py` replays Botzone requests into the
  feature-agent runtime.
- `scripts/benchmark_json_policies.py` runs seat-rotated local benchmarks
  against persistent JSON/text policy wrappers.

## Quick Smoke Test

```powershell
'{}' | python scripts\feature_agent_checkpoint_json_bot.py
```

Expected response:

```json
{"response": "PASS"}
```

## Text-Mode Policy

```powershell
python scripts\feature_agent_checkpoint_json_bot.py --protocol text
```

The process reads Botzone text requests from stdin and emits Botzone responses.

## Training Entry Point

```powershell
cd third_party\mahjong-agent-2025-aug12-excludespecial-3pkl\feature-agent
python supervised.py `
  --data-folder <botzone98209_vec_dir> `
  --augment-mode all12 `
  --special-matches <lv_yi_se_tui_bu_dao_special_matches.json> `
  --split-ratio 0.9 `
  --test-ratio 0 `
  --batch-size 8192 `
  --lr 0.0005 `
  --mixed-kernel-input `
  --dueling-head
```

The dataset folder is expected to contain preprocessed feature-agent NPZ data
with the 185-plane river-elastic OBS layout.
