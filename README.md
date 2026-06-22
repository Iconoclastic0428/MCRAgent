# MCRAgent

Local research workspace for Chinese Standard Mahjong / MCR Botzone agents.

The current implementation is reset to a Lawlorentz-first policy:

- `external/Chinese-Standard-Mahjong-DRL` supplies the Botzone text protocol state updates, 71-channel observation shape, and 235-action legal mask.
- `scripts/lawlorentz_policy.py` applies the MCR effective-tile rule as the base draw-discard scorer.
- `HU` is emitted only through legal masked actions that satisfy the MahjongGB fan gate; incidental fan sources are disabled for future-structure scoring.
- Old generated model, run, and distribution artifacts under `models/`, `runs/`, and `dist/` are not promotion candidates after the reset.

Promoted final feature-agent baseline:

- `models/feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a/16.pkl`
  is the promoted final checkpoint.
- The matching training/architecture code lives in
  `third_party/mahjong-agent-2025-aug12-excludespecial-3pkl/feature-agent`.
- This river-elastic epoch-16 feature-agent uses 185 OBS planes, the mixed
  kernel input option, and the dueling output head.
- `scripts/feature_agent_checkpoint_json_bot.py` is the reusable Botzone
  JSON/text wrapper for local official-judge evaluation of this checkpoint.
- `advisor_service.model_advisor.DEFAULT_TRANSFORMER_MODEL` remains the
  fallback transformer checkpoint at
  `models/transformer_candidate_finetune_medbest_l40_20260529a.pt`.
- `models/qadv_reranker_medbest_v1_terminal_best.pt` remains the transformer
  fallback QADV reranker at validation-selected `qadv_lambda=0.10`.
- The QADV-v1 promotion is based on the saved GPT Pro historical-terminal plan:
  lambda zero reproduces the base model, Hu/mask safety is clean, terminal Q
  separates positive and negative historical returns, and validation/test
  CHAGA relaxed metrics do not regress.
- The Chrome extension remains read-only; it observes tziakcha traffic and
  displays recommendations from the local advisor service.

Useful commands:

```powershell
python -m pytest tests -q --basetemp tmp\pytest
python -m advisor_service.server --port 8765 --local-only
python scripts\feature_agent_checkpoint_json_bot.py --protocol text
python scripts\official_judge_match.py --policy lawlorentz_effective --lawlorentz-levels 1 --opponent sample --games 4 --raw data\raw\botzone_mcr_sample.jsonl --max-turns 500 --out runs\lawlorentz_effective_sample_smoke4.json
python scripts\evaluate_lawlorentz_effective_all_datasets.py --games-per-dataset 4 --lawlorentz-levels 1 --out runs\lawlorentz_effective_all_datasets_report_4each.json
```

Datasets and protocol/interface code remain repo-local. Generated artifacts stay out of git unless deliberately selected for a source-focused handoff.
