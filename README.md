# MCRAgent

Local research workspace for Chinese Standard Mahjong / MCR Botzone agents.

The current implementation is reset to a Lawlorentz-first policy:

- `external/Chinese-Standard-Mahjong-DRL` supplies the Botzone text protocol state updates, 71-channel observation shape, and 235-action legal mask.
- `scripts/lawlorentz_policy.py` applies the MCR effective-tile rule as the base draw-discard scorer.
- `HU` is emitted only through legal masked actions that satisfy the MahjongGB fan gate; incidental fan sources are disabled for future-structure scoring.
- Old generated model, run, and distribution artifacts under `models/`, `runs/`, and `dist/` are not promotion candidates after the reset.

Current Chrome advisor baseline:

- `advisor_service.model_advisor.DEFAULT_TRANSFORMER_MODEL` points to
  `models/transformer_candidate_finetune_medhard_l40_20260528b.pt`.
- This strict finetune checkpoint is the current local Chrome-facing baseline
  for `http://127.0.0.1:8765`.
- The Chrome extension remains read-only; it observes tziakcha traffic and
  displays recommendations from the local advisor service.

Useful commands:

```powershell
python -m pytest tests -q --basetemp tmp\pytest
python -m advisor_service.server --port 8765 --local-only
python scripts\official_judge_match.py --policy lawlorentz_effective --lawlorentz-levels 1 --opponent sample --games 4 --raw data\raw\botzone_mcr_sample.jsonl --max-turns 500 --out runs\lawlorentz_effective_sample_smoke4.json
python scripts\evaluate_lawlorentz_effective_all_datasets.py --games-per-dataset 4 --lawlorentz-levels 1 --out runs\lawlorentz_effective_all_datasets_report_4each.json
```

Datasets and protocol/interface code remain repo-local. Generated artifacts stay out of git unless deliberately selected for a source-focused handoff.
