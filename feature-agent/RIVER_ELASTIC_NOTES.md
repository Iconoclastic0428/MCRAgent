# River Elastic Aug12 Training

This branch extends the 3.pkl aug12 exclude-special training path.

## Feature Layout

- Observation planes are now `185*4*9`.
- Planes `0:60` keep the original game-state layout.
- Planes `60:85` keep the vec-fix tile attributes: point, tile type, yaojiu, zipai, tuibudao, lvyise.
- Planes `85:185` are river-property planes: `4 players * 25 tile attributes`.
- Existing 85-plane NPZ files are upgraded inside `MahjongGBDataset` by deriving river-property planes from `PLAY` planes `16:20`.
- New preprocessing writes 185-plane NPZ files directly through `FeatureAgent.OBS_SIZE`.

## Augmentation

`all12` still applies all 6 suit permutations and both rank directions. Rank mirrors also swap fan-searcher feature names for:

- `dayu5` <-> `xiaoyu5`
- `quanda` <-> `quanxiao`
- `大于五` <-> `小于五`
- `全大` <-> `全小`

## Model And Loss

The training script supports the architecture changes from the slide:

- `--mixed-kernel-input`: optional parallel `3*3` and `1*1` input stem.
- `--dueling-head`: value stream plus advantage stream, returning `V(s) + A(s,a) - mean(A)`.
- Elastic-net loss is always active in this branch:
  - `lambda_l1 = 0.01 * lr`
  - `lambda_l2 = 0.1 * lr`
  - `loss = cross_entropy + lambda_l1 * L1 + lambda_l2 * L2`

## Launch Shape

Use the existing 98209 Botzone vector folder and exclude special games:

```sh
python supervised.py \
  --data-folder "${VEC_DIR}" \
  --version feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a \
  --log-root "${DATA_ROOT}/models/feature_agent_checkpoints" \
  --runs-root "${DATA_ROOT}/runs/feature_agent_tensorboard" \
  --epochs 30 \
  --batch-size 8192 \
  --lr 0.0005 \
  --split-ratio 0.9 \
  --device cuda \
  --data-parallel \
  --num-workers 0 \
  --no-augment \
  --augment-mode all12 \
  --special-matches "${SPECIAL_JSON}" \
  --exclude-special-matches \
  --dueling-head \
  --mixed-kernel-input
```
