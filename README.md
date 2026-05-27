# MCRAgent

Local research workspace for Chinese Standard Mahjong / MCR Botzone agents.

The current best verified local model is:

`models/ensemble_draw_public1000_2026_tziakcha256_winner_025_060_015_reaction1000_prefer_hu.pkl`

It is intentionally not committed because model and dataset artifacts are generated locally and are too large for a normal GitHub source commit. The latest documented gate remains:

- Official sample-bot gate: 21/64 wins, average score 19.84375.
- Terminal results: 21 `HU`, 43 `HUANG`.
- Safety invariant: runtime `HU` is allowed only when the official fan checker proves fan >= 8.

Recent work added:

- Suit-permuted official-judge evaluation generation.
- Full-record suit augmentation for Tziakcha training records.
- Fan-aware draw-ranker learning-rate sweep support.
- Official-judge policy-search screens for the new candidates.

Useful commands:

```powershell
python -m pytest tests -q --basetemp tmp\pytest_suit_rl_full
python scripts\generate_suit_permuted_initdata.py --raw data\raw\botzone_mcr_1000.jsonl --out data\eval\botzone_mcr_first64_suit_permuted_384.jsonl --limit 64 --include-identity
python scripts\train_feature_action_ranker.py --raw data\raw\tziakcha_human_botzone_raw_517.jsonl --model-out models\feature_draw_ranker_tziakcha_human_517_winner_only_fanv1_lr006.pkl --metrics-out runs\feature_draw_ranker_tziakcha_human_517_winner_only_fanv1_lr006_metrics.json --request-kind draw --winner-only --feature-mode numeric_fan_v1 --learning-rate 0.06 --max-iter 240 --min-samples-leaf 12
```

The repo code is designed around local generated artifacts under `data/`, `models/`, and `runs/`. See `tasks/TODO.md` and `tasks/SUMMARIES.md` for current experiment history.
