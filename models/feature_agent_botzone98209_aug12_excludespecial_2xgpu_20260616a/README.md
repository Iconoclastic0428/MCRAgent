# Feature-Agent Aug12 Exclude-Special Epoch 3

This directory contains the checkpoint used as the selected local/self-play model:

- Checkpoint: `3.pkl`
- Training job: `feature-agent-botzone98209-aug12-excludespecial-2xgpu-20260616a`
- Source checkpoint path on PVC: `/data/idl/mcr_agent_transformer_20260527_0029/mcr_transformer_sync_20260527/models/feature_agent_checkpoints/feature_agent_botzone98209_aug12_excludespecial_2xgpu_20260616a/3.pkl`
- SHA256: `3c5d7627540dd61f3cb47f4ad062b423641f21ad00799e9bb3ff4273d162ffc3`

Training setup:

- Base data: Botzone 98209-record dataset only.
- Base vector run: `feature_agent_botzone98209_2xgpu_20260615b`.
- Model run: `feature_agent_botzone98209_aug12_excludespecial_2xgpu_20260616a`.
- Augmentation: `--augment-mode all12`, covering 6 suit permutations and rank mirroring.
- Special-match handling: excludes games containing `推不倒` or `绿一色` from augmentation/training via `detect_special_matches.py`.
- Split: match-level 90% train / 10% validation.
- Training command and Kubernetes resource setup are captured in `k8s/feature-agent-botzone98209-aug12-excludespecial-2xgpu-20260616a.yaml`.
- The exact feature-agent source snapshot used for the code path is vendored under `third_party/mahjong-agent-2025-aug12-excludespecial-3pkl/`.
