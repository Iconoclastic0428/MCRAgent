# Slide FPN No-Aug Experiment

This branch adds a network shaped after the provided slide:

- Input representation: first 60 observation planes only, shaped `60*4*9`.
- 2D branch: `3*3` convolution to 128 channels, then FPN-style parallel `3*3` convolutions with dilation 1, 2, and 3.
- 1D branch: flatten the `4*9` tile grid to length 36, then FPN-style parallel `3`, `5`, and `7` convolutions.
- The 1D branch is reshaped back to `4*9`, concatenated with the 2D branch, passed through a final residual `3*3` block, flattened, and projected directly to 235 actions.
- The model applies the same 235-dimensional legal action mask.

The first training run should use the normal 98209 Botzone vector data, 90% train and 10% validation, without augmentation:

```sh
python supervised.py \
  --data-folder "${VEC_DIR}" \
  --version feature_agent_botzone98209_slide_fpn_noaug_2xgpu_20260617a \
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
  --augment-mode none \
  --model-kind slide-fpn \
  --hidden 128 \
  --slide-fpn-obs-planes 60 \
  --slide-fpn-blocks 1 \
  --fc-hidden 0
```

If validation accuracy does not approach the expected 89% best result, the next knobs to test are likely:

- Increase `--slide-fpn-blocks`.
- Add one hidden FC layer with `--fc-hidden 256`.
- Add vec features back after the flattened FPN output.

## No-Aug Follow-Up Controls

The first run was a literal slide transcription. The follow-up code keeps
augmentation disabled but adds controls for the likely mismatches:

- `--split-mode contiguous|random` supports checking whether the last-10%
  validation fold is harder than a seeded random match split.
- `audit_data_vec.py` checks the existing `.npz` files without rewriting them:
  target label inside action mask, action-family counts, candidate-count counts,
  samples per match, and tensor shapes.
- `--slide-fpn-residual input` switches the FPN residual to the slide-like
  input skip, instead of the first run's merged-activation skip.
- `--slide-fpn-obs-planes 85` keeps the vec-fix tile-attribute planes.
- `--slide-fpn-use-vec --slide-fpn-vec-hidden 128` concatenates a learned vector
  adapter after the flattened FPN features.

Useful no-augmentation commands:

```sh
python audit_data_vec.py \
  --data-folder "${VEC_DIR}" \
  --split-ratio 0.9 \
  --split-mode contiguous
```

```sh
python supervised.py \
  --data-folder "${VEC_DIR}" \
  --version feature_agent_botzone98209_selfvec_noaug_control_2xgpu_20260617a \
  --log-root "${DATA_ROOT}/models/feature_agent_checkpoints" \
  --runs-root "${DATA_ROOT}/runs/feature_agent_tensorboard" \
  --epochs 30 \
  --batch-size 8192 \
  --lr 0.0005 \
  --split-ratio 0.9 \
  --split-mode contiguous \
  --device cuda \
  --data-parallel \
  --num-workers 0 \
  --no-augment \
  --augment-mode none \
  --model-kind selfvec \
  --hidden 128 \
  --num-blocks 20
```

```sh
python supervised.py \
  --data-folder "${VEC_DIR}" \
  --version feature_agent_botzone98209_slide_fpn85vec_exact_noaug_2xgpu_20260617b \
  --log-root "${DATA_ROOT}/models/feature_agent_checkpoints" \
  --runs-root "${DATA_ROOT}/runs/feature_agent_tensorboard" \
  --epochs 30 \
  --batch-size 8192 \
  --lr 0.0005 \
  --split-ratio 0.9 \
  --split-mode contiguous \
  --device cuda \
  --data-parallel \
  --num-workers 0 \
  --no-augment \
  --augment-mode none \
  --model-kind slide-fpn \
  --hidden 128 \
  --slide-fpn-obs-planes 85 \
  --slide-fpn-blocks 2 \
  --slide-fpn-residual input \
  --slide-fpn-use-vec \
  --slide-fpn-vec-hidden 128 \
  --fc-hidden 256
```
