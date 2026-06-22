# Feature Agent River Elastic Epoch 16

Promoted final checkpoint for the Botzone 98,209-game feature-agent run with:

- 185 OBS planes: the 85-plane vec-fix feature set plus 25 river planes for each of 4 players.
- Mixed-kernel input support enabled in `SelfVecModel`.
- Dueling value/advantage output head enabled.
- 12x suit/number augmentation with special-hand exclusions preserved.

Checkpoint:

- `16.pkl`
- SHA256: `2B982A838A15D9632F85FFDEEE993105E077C5D92D236800140819607D9E2F3D`

Use with:

```powershell
$env:MCR_FEATURE_AGENT_DIR="D:\MCR_Agent\third_party\mahjong-agent-2025-aug12-excludespecial-3pkl\feature-agent"
$env:MCR_FEATURE_AGENT_CHECKPOINT="D:\MCR_Agent\models\feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a\16.pkl"
$env:MCR_FEATURE_AGENT_RIVER="1"
python D:\MCR_Agent\scripts\feature_agent_checkpoint_json_bot.py --protocol text
```

Training entry point:

```powershell
cd D:\MCR_Agent\third_party\mahjong-agent-2025-aug12-excludespecial-3pkl\feature-agent
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

The training code uses contiguous 90/10 train/validation match splits by
default, validates the target action against the legal mask, and preserves
special `绿一色` / `推不倒` games during the 12x augmentation pass. It applies
elastic regularization as `L1 = 0.01 * lr` and `L2 = 0.1 * lr`, with AdamW
`weight_decay=0.0`.
