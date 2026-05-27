# MCR RL Model Goal

Objective: research available Mahjong Competition Rules (MCR / Chinese Standard Mahjong / Guobiao) AI models online, use sufficiently large open datasets for model training, train an MCR reinforcement-learning model, and verify it can beat the current online models found.

## Current Checklist

- [x] Confirm current workspace state.
  - Evidence: `D:\MCR_Agent` exists but was initially empty and not a git repository.
- [x] Build an evidence-backed research inventory.
  - Include online MCR/Chinese-Standard-Mahjong models, competition bots, papers, toolkits, datasets, and evaluation targets.
- [x] Acquire and validate a real open MCR dataset.
  - Primary candidate: Botzone matchpacks for `Chinese-Standard-Mahjong`.
  - Validation must include real file paths, month/game identifier, line counts, JSON parse checks, and sample schema notes.
- [x] Implement a baseline training dataset converter.
  - Convert Botzone matches into supervised behavior-cloning examples first.
  - Keep raw samples and converted shards inspectable.
- [ ] Implement or integrate an MCR simulator/evaluator.
  - It must evaluate Botzone protocol actions and legal move masks.
  - It must support offline matches against rule/sample bots and later external benchmark bots.
  - Current status: lightweight Python self-play proxy exists and the official C++ Botzone judge is built locally.
- [x] Train first supervised MCR behavior-cloning baselines.
  - Save checkpoints, config, metrics, and training logs under repo-local artifacts.
- [ ] Train an MCR RL model.
  - Start from a stronger supervised/legal baseline, then add self-play/RL fine-tuning.
  - Save checkpoints, config, metrics, and training logs under repo-local artifacts.
  - Current status: first reward-weighted self-play baseline exists, but it is weaker than the source composite policy and is not a final RL model.
- [ ] Benchmark against found online baselines.
  - At minimum: Botzone random/sample bot, Aleo-style heuristic/search baseline if buildable, and selected top Botzone public replay opponents when code is unavailable.
  - Completion requires statistically meaningful match counts and score/win-rate confidence notes.
- [ ] Document remaining gaps honestly.
  - Do not claim the model beats all current models until current-state evidence proves it.

## Current Iteration

- [ ] Scale the public Botzone replay corpus from 1,000 to 5,000 games.
  - Use the existing scraper, not new scraping logic.
  - Save raw records to `data/raw/botzone_mcr_5000.jsonl`.
  - Convert to `data/processed/botzone_mcr_bc_5000.jsonl`.
  - Validate JSONL counts, parse errors, action counts, and active draw count.
  - Record whether this corpus is large enough for the next neural supervised pretraining step, while noting it is still much smaller than ALONG's reported 500K-game pretraining scale.
  - 2026-05-25 attempt: `python scripts\scrape_botzone_mcr.py --pages 300 --max-matches 5000 --workers 24 --sleep 0.02 --progress-every 250 --verbose --out data\raw\botzone_mcr_5000.jsonl` timed out from the caller after about 15 minutes.
  - Later validation found `data/raw/botzone_mcr_5000.jsonl` present with 4,975 valid JSONL records and no parse errors. The processed BC conversion is still pending.
- [x] Acquire the official strong-AI dataset linked from the 2026 Botzone page.
  - Source page: `https://botzone.org.cn/static/gamecontest2026a.html`.
  - Dataset link: `https://disk.pku.edu.cn/link/AA8CB7A57AFDCD48CAA7C749E04B5B6FAA` (`data.zip`).
  - Inspect download/API path, save the archive under `data/raw/`, and convert it if it uses official Botzone logs.
- [x] Convert the organizer strong-AI text logs into repo raw JSONL.
  - Add tests first for draw, no-reaction PASS, Chi/Peng response folding, Hu, ignored lower-priority declarations, and score parsing.
  - Create `scripts/strong_ai_text.py` to parse `data.txt` from `data/raw/botzone_2026_strong_ai_data.zip`.
  - Save converted raw-style records to `data/raw/botzone_2026_strong_ai_raw.jsonl`.
  - Validate match count and compatibility with existing legal-candidate loading.
  - 2026-05-25 result: `data/raw/botzone_2026_strong_ai_data.zip` downloaded and converted to `data/raw/botzone_2026_strong_ai_raw.jsonl` with 98,209 records and zero parse errors.
  - Focused converter tests passed with `python -m pytest tests\test_strong_ai_text.py -q --basetemp tmp\pytest`.
  - Compatibility check on `tmp/botzone_2026_strong_ai_raw_1000.jsonl`: 702,556 legal candidates, 440,237 draw candidates, 254,319 reaction candidates.
- [x] Train an updated supervised base from the 2026 organizer dataset.
  - Use `data/raw/botzone_2026_strong_ai_raw_5000.jsonl` as the first tractable slice before refactoring the trainers for streaming full-corpus fitting.
  - Train draw and reaction components, compose the policy, and compare against the previous best official-filtered checkpoint.
  - Best updated-data candidate so far: `models/composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000.pkl`.
  - The updated candidate improves official 128-game fallback/shanten scores versus the old best, but it is weaker against the official sample bot and is not good enough for Botzone group testing yet.
  - Full-corpus stride draw experiment `models/composite_feature_draw_2026_stride10_reaction1000_nonpass50.pkl` was weaker on the sample-bot benchmark and was not promoted.
- [x] Train and evaluate a public-5k Botzone candidate.
  - Use `data/raw/botzone_mcr_5000.jsonl`, currently validated as 4,975 records with no JSON parse errors.
  - Candidate-loader validation: 4,784,979 legal candidates; 2,345,391 draw candidates; 2,401,058 reaction candidates.
  - Train public-5k draw and reaction rankers, compose candidate policies, and compare against both the old public-1k best and the 2026 organizer candidate.
  - Gate on official judge results against fallback, shanten, official sample bot, and direct model-vs-model matches before considering the Botzone group test.
  - Trained `models/feature_draw_ranker_public5000.pkl`: held-out draw group accuracy 0.5850.
  - Trained `models/reaction_ranker_nonpass50_public5000.pkl`: held-out reaction group accuracy 0.9853.
  - Composed `models/composite_feature_draw_reaction_nonpass50_public5000.pkl`.
  - Replay smoke on `data/raw/botzone_mcr_80.jsonl`: zero illegal/fallback predictions, active-draw exact 0.4651.
  - Official sample-bot gate failed to improve: `runs/official_judge_composite_feature_draw_reaction_nonpass50_public5000_vs_official_sample_64.json` averaged 3.4531 with 5 wins, below the old public-1k best and the 2026 organizer candidate.
- [x] Try official-reward training from the stronger 2026 supervised source.
  - Generated `runs/official_trajectories_composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000_vs_shanten_256.json`: source model averaged 5.3945 in 256 official games vs shanten.
  - Trained `models/official_reward_weighted_policy_2026_source_shanten_256_player0.pkl`: held-out trajectory group accuracy 0.9143.
  - Negative result: `runs/official_judge_official_reward_weighted_policy_2026_source_shanten_256_player0_vs_shanten_64.json` scored 0.0 average with 64 all-zero games.
  - Trained `models/official_reward_reaction_policy_2026_source_shanten_256_player0.pkl` and composed `models/composite_feature_draw_2026_official_reward_reaction_shanten_256_player0.pkl`.
  - Negative result: both unthresholded and margin-0.20 official-reward reaction composites scored 0.0 average in 64 games vs shanten.
  - This confirms the current reward-weighted text policy path is not reliable enough; the next model-quality work should change the policy selection architecture or trainer rather than repeat the same reward-weighting loop.
- [x] Design and implement a draw-ranker ensemble or calibrated selector.
  - Candidate design: average or calibrated-combine old public-1k draw, 2026 organizer draw, and possibly public-5k draw scores while keeping the best known reaction component.
  - Must be test-covered before policy code changes.
  - Goal is to preserve 2026 model gains against shanten/fallback while recovering the old public-1k sample-bot strength.
  - Added `draw_ensemble_composite_policy` support in `scripts/policy_bot.py` and `scripts/create_draw_ensemble_policy.py`.
  - Added tests in `tests/test_policy_bot.py` and `tests/test_create_draw_ensemble_policy.py`; focused RED tests failed before implementation and passed after.
  - Best ensemble artifact: `models/ensemble_draw_public1000_2026_050_050_reaction1000.pkl`.
  - Official sample 64-game gate: `runs/official_judge_ensemble_draw_public1000_2026_050_050_reaction1000_vs_official_sample_64.json` averaged 8.0 with 12 wins, beating the old public-1k best and the 2026-only candidate.
  - Official shanten/fallback 128-game gates:
    - `runs/official_judge_ensemble_draw_public1000_2026_050_050_reaction1000_vs_shanten_128.json`: average 9.0391, 27 wins.
    - `runs/official_judge_ensemble_draw_public1000_2026_050_050_reaction1000_vs_fallback_128.json`: average 8.8203, 26 wins.
  - Held-out offset gates remained strong:
    - `runs/official_judge_ensemble_draw_public1000_2026_050_050_reaction1000_vs_shanten_64_offset128.json`: average 8.1719, 13 wins.
    - `runs/official_judge_ensemble_draw_public1000_2026_050_050_reaction1000_vs_fallback_64_offset128.json`: average 8.9219, 14 wins.
    - `runs/official_judge_ensemble_draw_public1000_2026_050_050_reaction1000_vs_official_sample_64_offset64.json`: average 8.5625, 13 wins.
  - Direct old/new model-vs-model gates were positive for the ensemble as seat 0 but still seat-sensitive, so they are useful but not final proof against online Botzone models.
  - Added `scripts/botzone_json_policy_bot.py`, a Botzone JSON-protocol wrapper for the ensemble policy.
  - Local submission/test package: `dist/mcr_ensemble_botzone_package.zip`; package smoke test from its own directory returned valid `{"response": "PLAY F2"}` JSON.
  - Opened `https://botzone.org.cn/group/69d9eca875ce41301557e099` in Chrome for manual/logged-in group testing. No online submission result has been recorded yet.
  - 2026-05-25 Chrome check of `https://wiki.botzone.org.cn/index.php?title=Bot`: Botzone upload runs on Ubuntu 16.04, supports Python zip uploads with a root `__main__.py`, appears capped at 4 MB, and the listed Python libraries do not include scikit-learn. The existing sklearn pickle package is therefore only a local smoke artifact, not a valid online submission package.
- [ ] Build a source-only Botzone Python 3.6 upload artifact.
  - Export the ensemble draw HistGradientBoosting models and reaction TF-IDF/SGD model to pure Python data/scoring code.
  - Preserve the existing JSON protocol wrapper behavior with no scikit-learn/numpy dependency at runtime.
  - Add parity tests against the local sklearn policy on representative requests.
  - Package as a zip with root `__main__.py`, smoke it locally, then upload/test through Chrome.
  - 2026-05-25: added `scripts/export_botzone_pure_python.py` and `scripts/botzone_pure_runtime.py`; generated `dist/mcr_pure_python_botzone_package.zip` at 406,523 bytes.
  - Safety correction: pure source-only runtime now suppresses `HU` because it cannot call the official fan checker. A complete hand is not enough for MCR and caused official `WH` failures.
  - Focused export tests pass and package JSON smoke passes, but the source-only upload artifact is not the preferred strong model until a verified pure-Python fan evaluator exists.
  - 2026-05-25 update: added a conservative source-only high-fan lower-bound gate for unambiguous patterns such as full flush, seven pairs, thirteen orphans, all honors, all terminals/honors, and big/little wind/dragon hands. The gate still suppresses low-fan complete hands.
  - Export now preserves the model `prefer_hu` flag, but the pure runtime takes `HU` only if the conservative gate first proves at least 8 fan.
  - Regenerated `dist/mcr_pure_python_botzone_package.zip` at 407,043 bytes. Smoke checks: high-fan full-flush payload returns `{"response": "HU"}`; low-fan complete-hand payload returns `{"response": "PASS"}`.
  - 2026-05-25 live Botzone version-0 failure root cause: the uploaded bootstrap attempted to write its embedded package to `/tmp`, and Botzone returned `PermissionError: [Errno 13] Permission denied: '/tmp/mcr_pure_python_botzone_package.zip'`.
  - Fixed `scripts/export_botzone_pure_python.py` so direct-code bootstraps write the embedded package under `os.getcwd()` or the script directory instead of `tempfile.gettempdir()`.
  - Regenerated `dist/mcr_pure_python_botzone_bootstrap.py` at 602,741 bytes and `dist/mcr_pure_python_botzone_package.zip` at 408,270 bytes. Local direct-code smokes still return `PASS` for a low-fan complete hand and `HU` for a high-fan official-backed self-draw.
  - Upload of fixed Botzone version 1 is pending: the Chrome extension currently rejects Botzone file attachment with `Not allowed`, and the Chrome virtual clipboard is unavailable for direct editor paste.
  - Verification after the upload-fix notes: `python -m pytest tests -q --basetemp tmp\pytest` passed with 118 tests.
  - Next hardening step: remove all bootstrap filesystem writes by executing the embedded package from memory. Red test command: `python -m pytest tests\test_export_botzone_pure_python.py::test_exported_bootstrap_file_outputs_json_without_sklearn -q --basetemp tmp\pytest_bootstrap_memory_red` failed because the bootstrap still used the extraction path.
  - Completed no-write bootstrap hardening: `dist/mcr_pure_python_botzone_bootstrap.py` now executes `botzone_pure_runtime.py` and `botzone_model_data.py` from `zipfile.ZipFile(io.BytesIO(payload))` and no longer calls `os.getcwd()` or writes an extracted package.
  - Regenerated artifacts after no-write hardening: `dist/mcr_pure_python_botzone_bootstrap.py` is 602,491 bytes and `dist/mcr_pure_python_botzone_package.zip` is 408,270 bytes.
  - No-write smoke: running the bootstrap from a fresh temp directory returned valid JSON and left only the bootstrap file in that directory, proving no extracted zip was written.
  - HU invariant smokes after regeneration: low-fan complete hand returned `{"response": "PASS"}`; high-fan self-draw returned `{"response": "HU"}`.
  - Verification after no-write hardening: export tests passed with 21 tests, focused HU guard tests passed with 3 tests, and full suite passed with `python -m pytest tests -q --basetemp tmp\pytest`: 118 tests.
- [x] Gate the full local model at roughly 27% win rate before Chrome/Tziakcha play.
  - Use the full local sklearn model with the compiled official fan checker, not the source-only Botzone upload export.
  - Added `prefer_hu` support to `SklearnPredictor`/`BotzonePolicy`: when the official fan checker leaves `HU` in `_legal_responses`, take it immediately.
  - 2026-05-25 correction: `BotzonePolicy` now fails closed when the fan checker is unavailable or errors. `HU` is not legal unless the checker explicitly accepts it.
  - Created `models/ensemble_draw_public1000_2026_030_070_reaction1000_prefer_hu.pkl`.
  - Official sample-bot test set: `runs/official_judge_ensemble030070_prefer_hu_vs_official_sample_64.json` scored 18 wins in 64 games, win rate 0.28125, average score 17.3125, terminal actions 46 `HUANG` / 18 `HU` / 0 `WH`.
  - Full test suite passed after this change with `python -m pytest tests -q --basetemp tmp\pytest`: 73 passed.
  - After the fail-closed HU correction, a 16-game official sample-bot sanity rerun wrote `runs/official_judge_ensemble030070_prefer_hu_vs_official_sample_16_after_hu_failclosed.json`: 4 wins in 16 games, average score 9.625, terminal actions 12 `HUANG` / 4 `HU` / 0 `WH` / 0 `WA`. A full 64-game rerun exceeded the 3-minute command timeout before writing an output file, so the earlier 64-game gate remains the main >=27% artifact.
- [x] Build a read-only Tziakcha Chrome advisor around the full local model.
  - Reuse the prior WebSocket observer shape, but keep it repo-local under `advisor_service/` and `extension/`.
  - Use `models/ensemble_draw_public1000_2026_030_070_reaction1000_prefer_hu.pkl` by default.
  - Preserve the hard invariant: never recommend `HU` unless the official fan checker says the hand is at least 8 fan.
  - Keep the service read-only: `/observe`, `/state`, `/recommendation`, `/events`, `/errors`, `/health`, `/reset`; no `/action`.
  - Verify with focused advisor tests, the full pytest suite, local service health, and a Chrome/Tziakcha live-tab check.
  - 2026-05-25: added repo-local `advisor_service/`, `extension/`, and Tziakcha tests; the default service loads the full local model and official fan checker.
  - Verification: focused Tziakcha tests passed with 24 tests; full suite passed with `python -m pytest tests -q --basetemp tmp\pytest`: 97 passed.
  - Local service now runs on `http://127.0.0.1:8765/dashboard`; `/health` returns `{"ok": true, "model_loaded": true, "read_only": true}`.
  - Chrome live check: `https://tziakcha.net/play/` posts observer events into the model service (`event_count` advanced to 8). A table was created/readied, but it had only one player, so no live hand/decision prompt was available yet.
- [ ] Run a real Tziakcha AI match/decision-prompt check once a table with AI opponents is available.
  - Do not automate Mahjong moves. Keep this as read-only observer/advisor unless the user explicitly changes scope.
  - 2026-05-25 Chrome check: the live create-table dialog exposes rule/time/seat/password settings but no AI/opponent-fill option.
  - Bundle/protocol check: visible table management commands cover create/join/leave/ready/watch/login/rename/progress/start; searches found auto-action toggles for a human seat, but no bot/AI-seat creation control.

- [x] Review the Botzone 2020 Mahjong competition page for RL-target papers.
  - Relevant source: `https://www.botzone.org.cn/static/gamecontest2020a.html`.
  - Relevant paper/PPT: ALONG, `https://www.botzone.org.cn/static/IJCAI2020MahjongPPT/04-ALONG-yue.pdf`.
  - Finding: ALONG used large-scale imitation learning plus PPO-style self-play RL and reached second prize / 1314 in the 2020 final-stage table, so it is a stronger target than the current local fallback/shanten baselines.
- [x] Add official sample-bot and external-process benchmark wrappers.
  - `scripts/official_judge_match.py` can run `sample` JSON bots and `aleo` line-history bots.
  - Official sample bot is built at `build/official_sample_bot.exe`.
  - Aleo binaries are built under `build/`, but Aleo currently crashes on valid official payloads and is not yet a reliable benchmark.
- [ ] Stabilize or precisely bound the Aleo external benchmark.
  - Reproduce the current mid-run Aleo crash with a one-game-at-a-time diagnostic.
  - Identify whether the failure is caused by our line-history wrapper, an unsupported request/action shape, or Aleo's own native code.
  - Add regression coverage before changing wrapper behavior.
  - Run a smoke benchmark after the fix or record the exact unsupported boundary if Aleo remains unusable.
- [x] Run the full test suite after benchmark-wrapper documentation updates.
  - `python -m pytest tests -q --basetemp tmp\pytest` passed with 57 tests.

- [x] Add official-judge trajectory extraction for reward training.
  - Convert official match logs into `trajectory` plus official `scores` and normalized `rewards`.
  - Add held-out evaluation support with `--offset` in `scripts/official_judge_match.py`.
- [x] Train official-reward weighted checkpoints.
  - Train a standalone official-reward player-0 model from 128 official source-vs-shanten games.
  - Train a reaction-only official-reward component and compose it with the existing feature draw ranker.
- [x] Evaluate official-reward checkpoints against the supervised source.
  - Use held-out `initdata` offset 128 so the comparison is not only on trajectory-training games.
  - Record negative results if the reward-weighted models regress.

- [x] Add official-aware policy state for Hu/kong legality.
  - Reproduce the official `WH` and `WA` failures from saved official judge outputs.
  - Track wall counts from initial flower counts and public draw/Buhua events.
  - Fix local hand updates for concealed kong and supplemental kong responses.
  - Add a fan-calculator bridge or equivalent official fan check before allowing `HU`.
  - Re-evaluate the best policy under the official judge without blanket no-Hu/no-kong suppression.
- [x] Verify the improved official-aware policy path.
  - Add focused regression tests before each behavior change.
  - Run official 128-game comparisons against fallback and shanten.
  - Run full pytest.

- [x] Build and smoke-test the official C++ Botzone judge locally.
  - Binary: `build/official_judge/mcr_judge.exe`.
  - Smoke tests matched public replay prefixes after compiling with `-D_BOTZONE_ONLINE`.
- [x] Add a tested official-judge match harness.
  - `scripts/official_judge_match.py` reproduces Botzone-style request/response logs.
  - Regression coverage includes `verdict: "OK"` response-log entries, which the judge requires to avoid immediate penalties.
- [x] Diagnose official-judge penalties for the current best proxy model.
  - Main failures were invalid `HU` claims (`WH`) and late-wall kong attempts (`WA`).
- [x] Create and evaluate an official-safe no-Hu/no-kong composite.
  - Create `models/composite_feature_draw_reaction_nonpass50_1000_nohu_nogang.pkl`.
  - Compare against fallback and shanten in the official judge.
  - Run replay imitation diagnostics for the official-safe artifact.
- [x] Record official-judge results and remaining legality/scoring gaps.
  - The likely next gap is fan-aware Hu and wall-aware kong legality, not more proxy reward tuning.
- [x] Run the full test suite after the official-safe evaluation.

- [x] Add and test reaction-threshold support for composite policy artifacts.
  - `scripts/create_composite_policy.py` accepts reaction threshold options.
  - `policy_bot.py` can gate low-margin non-pass reactions to PASS.
- [x] Evaluate thresholded variants of the best feature-draw policy.
  - Margin thresholds degraded proxy reward; current best remains unthresholded `models/composite_feature_draw_reaction_nonpass50_1000.pkl`.
- [x] Add and test a numeric feature-action ranker path.
  - `scripts/feature_ranker.py`
  - `scripts/train_feature_action_ranker.py`
  - `policy_bot.py` support for `feature_action_ranker` payloads.
- [x] Train and evaluate 1000-match feature rankers.
  - Best current proxy artifact: `models/composite_feature_draw_reaction_nonpass50_1000.pkl`.
- [x] Diagnose reward-weighted self-play regression and add controlled-player filtering.
  - `models/reward_weighted_policy_feature_source_player0_shanten_128.pkl` is improved over previous reward-weighted runs but still regresses from the source model.
- [x] Add a tested outcome filter for legal-action ranker training.
  - Support training only from positive-score or winning player decisions.
- [x] Train outcome-filtered 300-match policies.
  - Compare replay imitation and lightweight self-play against fallback, shanten, broad 300-match composite, and reward-weighted policies.
- [x] Add a tested concurrent replay fetch path for larger Botzone data collection.
  - Use `--workers` in `scripts/scrape_botzone_mcr.py`.
- [x] Scrape and convert a 1000-match Botzone corpus.
  - `data/raw/botzone_mcr_1000.jsonl`: 1,000 matches, no scrape failures.
  - `data/processed/botzone_mcr_bc_1000.jsonl`: 418,112 examples.
- [x] Train and evaluate 1000-match supervised and reward-weighted checkpoints.
  - Best proxy model so far: `models/composite_draw_reaction_nonpass50_1000.pkl`.
- [x] Record artifacts, commands, and negative/positive results in the docs.
- [x] Run the full test suite after code changes.

## Verification Log

- 2026-05-25:
  - Scraped 160 public Botzone `Chinese-Standard-Mahjong` replays to `data/raw/botzone_mcr_160.jsonl`.
  - Scraped 300 public Botzone `Chinese-Standard-Mahjong` replays to `data/raw/botzone_mcr_300.jsonl`.
  - Converted 62,904 behavior-cloning examples to `data/processed/botzone_mcr_bc_160.jsonl`.
  - Converted 121,412 behavior-cloning examples to `data/processed/botzone_mcr_bc_300.jsonl`.
  - Trained first baselines:
    - `models/bc_policy_draw_response_160.pkl`
    - `models/bc_policy_draw_action_160.pkl`
    - `models/discard_ranker_160.pkl`
    - `models/discard_ranker_shanten_160.pkl`
    - `models/discard_ranker_shanten_drawn_160.pkl`
    - `models/legal_action_ranker_160.pkl`
    - `models/reaction_ranker_160.pkl`
    - `models/reaction_ranker_nonpass50_160.pkl`
    - `models/composite_draw_reaction_160.pkl`
    - `models/composite_draw_reaction_nonpass50_160.pkl`
    - `models/reward_weighted_selfplay_policy_128.pkl`
    - `models/discard_ranker_shanten_drawn_300.pkl`
    - `models/reaction_ranker_nonpass50_300.pkl`
    - `models/composite_draw_reaction_nonpass50_300.pkl`
    - `models/reward_weighted_selfplay_policy_300src_128.pkl`
  - Replay evaluator artifacts:
    - `runs/policy_replay_eval_fallback_160.json`
    - `runs/policy_replay_eval_draw_response_160.json`
    - `runs/policy_replay_eval_discard_ranker_160.json`
    - `runs/policy_replay_eval_shanten_heuristic_160.json`
    - `runs/policy_replay_eval_discard_ranker_shanten_160.json`
    - `runs/policy_replay_eval_discard_ranker_shanten_drawn_160.json`
    - `runs/policy_replay_eval_legal_action_ranker_160.json`
    - `runs/policy_replay_eval_composite_draw_reaction_160.json`
    - `runs/policy_replay_eval_composite_draw_reaction_nonpass50_160.json`
    - `runs/selfplay_composite_nonpass50_vs_fallback_128_trajectories.json`
    - `runs/selfplay_reward_weighted_policy_128_vs_fallback_128.json`
    - `runs/reward_weighted_selfplay_policy_128_metrics.json`
    - `runs/policy_replay_eval_composite_draw_reaction_nonpass50_300.json`
    - `runs/selfplay_composite_nonpass50_300_vs_fallback_128_seed9700.json`
    - `runs/selfplay_reward_weighted_policy_300src_128_vs_fallback_128_seed9700.json`
    - `runs/reward_weighted_selfplay_policy_300src_128_metrics.json`
    - `runs/policy_replay_eval_reward_weighted_selfplay_300src_128.json`
  - `python -m pytest tests -q --basetemp tmp\pytest` passed with 39 tests after reaction-threshold policy support.
  - PyMahjongGB install failed because the machine lacks Microsoft C++ Build Tools; local source clone is available under `external/PyMahjongGB`.
  - Added pure-Python regular/seven-pairs/thirteen-orphans shanten features because PyMahjongGB could not be built.
  - Added legal CHI/PENG/GANG/HU candidate generation and reaction-specific replay metrics.
  - Weighted reaction model made 229 non-pass reaction predictions, with 50 exact matches: precision 0.2183, recall 0.0576.
  - Added lightweight self-play simulator and reward-weighted policy training. The reward-weighted model averaged 0.5358 reward vs fallback under seed 9400, below shanten 0.9831 and composite 1.0371 under the same proxy evaluator.
  - Scaling to 300 matches improved composite proxy self-play reward on seed 9700 to 1.0358 average vs fallback, but reward-weighted self-play from that source dropped to 0.8581 and did not improve the source policy.
  - Added outcome-filtered training for legal-action rankers. Positive-score and winner-only 300-match models were legal but weaker than the broad 300-match composite in replay and proxy self-play.
  - Added concurrent replay fetching to `scripts/scrape_botzone_mcr.py` and scraped `data/raw/botzone_mcr_1000.jsonl` with 1,000 matches and no failures.
  - Converted `data/processed/botzone_mcr_bc_1000.jsonl` with 418,112 examples.
  - Trained `models/discard_ranker_shanten_drawn_1000.pkl`, `models/reaction_ranker_nonpass50_1000.pkl`, `models/composite_draw_reaction_nonpass50_1000.pkl`, and `models/reward_weighted_selfplay_policy_1000src_128.pkl`.
  - The 1000-match composite averaged 1.1569 reward vs fallback and 0.0664 vs shanten in the lightweight proxy under seed 9700, but replay imitation still trailed fallback and this is not official MCR scoring.
  - The 1000-source reward-weighted policy regressed to 0.8451 vs fallback and -0.4798 vs shanten under seed 9700.
  - Added numeric feature ranker training. `models/feature_draw_ranker_1000.pkl` reached 0.5806 held-out draw group accuracy, above the TF-IDF 1000-match discard ranker's 0.3228.
  - Best current proxy artifact: `models/composite_feature_draw_reaction_nonpass50_1000.pkl`.
    - Replay: 0.4838 active-draw exact, 0 illegal prediction rate.
    - Lightweight self-play across seeds 9700/9800/9900: mean 1.6024 reward and 153/384 wins vs fallback; mean 0.9403 reward and 151/384 wins vs shanten.
  - Numeric feature-reaction branches over-claimed and were negative in lightweight self-play despite higher replay active-draw exactness.
  - Added `--player-filter` to reward-weighted self-play training. Player-0-only filtered RL-style model beat fallback and shanten in proxy, but still underperformed its supervised feature-draw source.
  - Added reaction-threshold support for composite artifacts and evaluated margin thresholds 0.02, 0.05, 0.10, and 0.20. Thresholding degraded the best feature-draw policy on seed 9700:
    - Unthresholded vs shanten: 0.9167 average reward, 49 wins.
    - Best threshold tested (`min_margin=0.02`) vs shanten: 0.6484 average reward, 33 wins.
    - `min_margin=0.02` vs fallback: 1.4167 average reward, below unthresholded 1.5833 on the same seed.
  - Fresh Botzone top-30 rank snapshot saved to `data/metadata/botzone_ranklist_top30_2026-05-25.json`.
  - The trained baselines are not competitive with current online models; the best replay-imitation baseline remains the conservative drawn-tile fallback.
  - Built and smoke-tested the official C++ Botzone judge at `build/official_judge/mcr_judge.exe`.
  - Added `scripts/official_judge_match.py` with tests for Botzone-style response logs and required `verdict: "OK"` fields.
  - Official 32-game judge results for the proxy-best model were negative because of invalid Hu and kong actions:
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_vs_fallback_32.json`: player 0 total -213.0, average -6.65625, terminal actions HU 3 / HUANG 18 / WA 3 / WH 8.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_vs_shanten_32.json`: player 0 total -273.0, average -8.53125, terminal actions HU 3 / HUANG 16 / WA 3 / WH 10.
  - Created `models/composite_feature_draw_reaction_nonpass50_1000_nohu_nogang.pkl`.
  - Official-safe no-Hu/no-kong 32-game judge results were penalty-free but non-winning:
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_nogang_vs_fallback_32.json`: player 0 total 0.0, average 0.0, 32 HUANG.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_nogang_vs_shanten_32.json`: player 0 total 0.0, average 0.0, 32 HUANG.
  - Replay diagnostic for `models/composite_feature_draw_reaction_nonpass50_1000_nohu_nogang.pkl` saved to `runs/policy_replay_eval_composite_feature_draw_reaction_nonpass50_1000_nohu_nogang.json`: active-draw exact 0.4841, illegal prediction rate 0.0, reaction non-pass predictions 0.
  - `python -m pytest tests -q --basetemp tmp\pytest` passed with 44 tests after official-judge harness and suppression-policy changes.
  - Added official-aware policy legality state:
    - `BotzonePolicy` tracks official wall counts from the 136-tile judge wall, initial flower counts, public draw events, and Buhua events.
    - Concealed `GANG` now removes all four standing tiles from local hand state.
    - Draw/reaction `GANG`, `BUGANG`, `PENG`, and `CHI` candidates are filtered when the official judge would reject them due empty replacement/next wall.
    - `scripts/mcr_fan_check.cpp` and `scripts/official_fan.py` bridge to the official MahjongGB `calculate_fan` implementation before allowing `HU`.
  - Focused tests for policy wall tracking, kong hand updates, wall-filtered kong rejection, Hu fan filtering, and the fan-check helper passed with `python -m pytest tests\test_official_fan.py tests\test_policy_bot.py -q --basetemp tmp\pytest`.
  - Wall-aware filtering fixed the no-Hu model's official `WA` penalties:
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_wallfix2_vs_fallback_32.json`: player 0 total 0.0, 32 HUANG.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_wallfix2_vs_shanten_32.json`: player 0 total 0.0, 32 HUANG.
  - Official-aware filters turned the proxy-best checkpoint positive against local baselines:
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_officialfilter_vs_fallback_128.json`: player 0 total 639.0, average 4.9921875, 15 wins, terminal actions HU 15 / HUANG 113.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_officialfilter_vs_shanten_128.json`: player 0 total 676.0, average 5.28125, 16 wins, terminal actions HU 16 / HUANG 112.
  - `python -m pytest tests -q --basetemp tmp\pytest` passed with 50 tests after official fan filtering and wall-count changes.
  - Official evaluation of the existing player-0 reward-weighted RL-style checkpoint was positive but still weaker than its supervised source:
    - `runs/official_judge_reward_weighted_policy_feature_source_player0_shanten_128_officialfilter_vs_fallback_64.json`: player 0 total 203.0, average 3.171875, 5 wins, terminal actions HU 5 / HUANG 59.
    - `runs/official_judge_reward_weighted_policy_feature_source_player0_shanten_128_officialfilter_vs_shanten_64.json`: player 0 total 207.0, average 3.234375, 5 wins, terminal actions HU 5 / HUANG 59.
    - Same first-64 source checkpoint comparison from the 128-game official runs: average 4.390625 vs fallback and 4.890625 vs shanten, so current reward-weighted RL remains a regression.
  - Added `scripts/official_trajectories.py` and `tests/test_official_trajectories.py` to convert official judge logs into reward-weighted training trajectories.
  - Added `--offset` to `scripts/official_judge_match.py` so official evaluation can use held-out `initdata` ranges.
  - Generated `runs/official_trajectories_composite_feature_draw_reaction_nonpass50_1000_vs_shanten_128.json`, matching the prior 128-game official source-vs-shanten totals.
  - Trained official-reward checkpoints:
    - `models/official_reward_weighted_policy_feature_source_shanten_128.pkl` from all player-0 decisions; metrics in `runs/official_reward_weighted_policy_feature_source_shanten_128_metrics.json`.
    - `models/official_reward_reaction_policy_feature_source_shanten_128.pkl` from player-0 reaction decisions; metrics in `runs/official_reward_reaction_policy_feature_source_shanten_128_metrics.json`.
    - `models/composite_feature_draw_official_reward_reaction_shanten_128.pkl` combining the feature draw ranker with the official-reward reaction component.
  - Held-out offset-128 official evaluation showed both official-reward attempts regressed to all draws:
    - `runs/official_judge_official_reward_weighted_policy_feature_source_shanten_128_vs_fallback_64_offset128.json`: player 0 total 0.0, 64 HUANG.
    - `runs/official_judge_official_reward_weighted_policy_feature_source_shanten_128_vs_shanten_64_offset128.json`: player 0 total 0.0, 64 HUANG.
    - `runs/official_judge_composite_feature_draw_official_reward_reaction_shanten_128_vs_fallback_64_offset128.json`: player 0 total 0.0, 64 HUANG.
    - `runs/official_judge_composite_feature_draw_official_reward_reaction_shanten_128_vs_shanten_64_offset128.json`: player 0 total 0.0, 64 HUANG.
  - The supervised source on the same held-out offset-128 games remained positive:
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_officialfilter_vs_fallback_64_offset128.json`: player 0 total 382.0, average 5.96875, 10 wins, HU 10 / HUANG 54.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_officialfilter_vs_shanten_64_offset128.json`: player 0 total 382.0, average 5.96875, 10 wins, HU 10 / HUANG 54.
  - `python -m pytest tests -q --basetemp tmp\pytest` passed with 54 tests after official trajectory extraction, offset evaluation, and request-kind training changes.
  - Reviewed the Botzone 2020 competition page and ALONG paper/PPT. ALONG is a directly relevant IL+RL target: about 500K pretraining games, PPO-style self-play RL, about 32M steps to convergence, and 1314 in the 2020 final-stage table.
  - Built and wrapped the official sample bot at `build/official_sample_bot.exe`.
  - Official sample benchmark:
    - `runs/official_judge_official_sample_vs_fallback_64.json`: sample player 0 scored 0.0 average with 64 `HUANG`.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_vs_official_sample_64.json`: current supervised source scored 313.0 total, 4.8906 average, and 8 wins in 64 games against three official sample bots.
  - Aleo benchmark status: `build/aleo_bot.exe` and `build/aleo_botzone.exe` exist, but the bot crashes on valid official payloads saved at `tmp/aleo_crash_payload.txt` and `tmp/aleo_botzone_crash_payload.txt`, so Aleo is not yet a reliable benchmark.
  - `python -m pytest tests -q --basetemp tmp\pytest` passed with 57 tests after benchmark-wrapper additions and documentation updates.
  - Downloaded and validated the 2026 organizer strong-AI archive:
    - `data/raw/botzone_2026_strong_ai_data.zip`, 26,386,568 bytes, SHA-256 `476abfcca64b2504973a877d6e5834ca623551914071b3e02da0e55b555ac2e0`.
    - Archive README says `data.txt` contains 98,209 Chinese Standard Mahjong game records.
  - Added `scripts/strong_ai_text.py` and `tests/test_strong_ai_text.py` for organizer text-log conversion, including draw/play folding, synthesized PASS reactions, Chi/Peng discard folding, Hu, ignored claimers, self-draw Hu, and Hu after BuGang.
  - `python -m pytest tests\test_strong_ai_text.py -q --basetemp tmp\pytest` passed with 6 tests.
  - Converted the full organizer dataset to `data/raw/botzone_2026_strong_ai_raw.jsonl`: 98,209 records, zero parse errors.
  - Created and validated training slices:
    - `data/raw/botzone_2026_strong_ai_raw_5000.jsonl`: 5,000 records.
    - `data/raw/botzone_2026_strong_ai_raw_stride10.jsonl`: 9,821 records sampled across the full organizer file.
    - `data/raw/botzone_mcr_1000_plus_2026_strong_ai_5000.jsonl`: 6,000 records, public 1,000 plus organizer 5,000.
  - Trained updated draw/reaction components:
    - `models/feature_draw_ranker_2026_strong_ai_5000.pkl`: held-out draw group accuracy 0.6135.
    - `models/reaction_ranker_nonpass50_2026_strong_ai_5000.pkl`: held-out reaction group accuracy 0.9686.
    - `models/feature_draw_ranker_public1000_plus_2026_strong_ai_5000.pkl`: held-out draw group accuracy 0.5952.
    - `models/feature_draw_ranker_2026_strong_ai_stride10.pkl`: held-out draw group accuracy 0.6117.
  - Composed and evaluated updated candidates. Best updated-data candidate so far is `models/composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000.pkl`:
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000_vs_shanten_128.json`: average 5.9844, 17 wins.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000_vs_fallback_128.json`: average 5.9453, 17 wins.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000_vs_shanten_64_offset128.json`: average 6.2656, 10 wins.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000_vs_fallback_64_offset128.json`: average 7.0313, 11 wins.
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000_vs_official_sample_64.json`: average 4.2969, 6 wins, which is weaker than the old best sample-bot result.
  - Direct old/new model checks are mixed, not decisive enough for Botzone submission:
    - `runs/official_judge_composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000_vs_oldbest_64.json`: updated seat 0 total 166, three old-best opponents combined -166.
    - `runs/official_judge_oldbest_vs_composite_feature_draw_reaction_nonpass50_2026_strong_ai_5000_64.json`: old-best seat 0 total 5, three updated opponents combined -5.
  - The current model is not good enough for the Botzone group test yet. It improves local fallback/shanten scores but does not prove competitiveness with current online models, and the official sample-bot regression needs more work.
  - `python -m pytest tests -q --basetemp tmp\pytest` passed with 63 tests after organizer converter additions.

## 2026-05-25 HU fan-gate hardening

- [x] Record the user correction that `HU` must never be emitted unless fan is enough.
- [x] Add a regression test for a self-drawn all-chows hand that the official checker scores at 8+ fan.
- [x] Update the pure Python Botzone fan lower bound without allowing unproven sub-8 `HU`.
- [x] Re-export the source-only Botzone package and smoke-test low-fan suppression plus high-fan acceptance.
- [x] Rerun focused and full tests.
- [x] Re-evaluate the deployable source package and update progress artifacts.

Review:
- Added source-runtime regressions for official-backed self-draw all-chows, mixed shifted chows, last-tile claim, one-void dragon self-draw, closed-wait self-draw, draw-score parity, and a 7-fan false positive that must stay suppressed.
- Regenerated `dist/mcr_pure_python_botzone_package.zip` at 408,270 bytes.
- Zip smokes:
  - low-fan discard claim -> `PASS`
  - official 7-fan false positive -> `PASS`
  - all-chows self-draw official 10 fan -> `HU`
  - mixed shifted chows self-draw official 13 fan -> `HU`
- Source package official sample gate: `runs/official_judge_source_inprocess_drawparity_commonfan3_vs_official_sample_64.json` scored 19/64 wins, win rate 0.296875, average 17.25, terminal actions 45 `HUANG` / 19 `HU`, no player-0 `WH`/`WA`, and minimum player-0 Hu fan 8.
- Verification: `python -m pytest tests -q --basetemp tmp\pytest` passed with 115 tests.
- Chrome/Botzone handoff: the Botzone new-bot form is filled for `MCRAgent2026A` as a Python 3.6 JSON-interaction Chinese-Standard-Mahjong bot, but file upload of `dist/mcr_pure_python_botzone_package.zip` is blocked by the Codex Chrome Extension with `Not allowed`. User needs to enable file access for the extension before the zip can be attached.

## 2026-05-25 Botzone direct-code upload fallback

- [x] Add a single-file source export path so Botzone direct-code input can be used when file upload is blocked.
- [x] Verify the single-file artifact runs with the same JSON protocol and fan-gated behavior.
  - Also added `dist/mcr_pure_python_botzone_bootstrap.py`, a self-extracting direct-code artifact embedding the verified zip; size 602,229 bytes.
- [x] Paste/submit the single-file bot through the live Botzone Chrome form if the artifact is under the size limit.
  - 2026-05-25: submitted direct-code bot `MCRAgent2026A`; the page returned `创建成功。`.
  - Current caveat: after submission the bot entry was found only in hidden DOM/template content, so a fresh My Bots reload/status check is still required before group testing.
- [x] Confirm Botzone shows the submitted bot/version after reload.
  - Fresh My Bots reload showed `MCRAgent2026A`, latest version 0, rating 1000.00, with bot version `source.py36` created at `2026-5-25 17:04:57`.
- [ ] If Botzone accepts it, run the available group/manual test and capture result evidence.
  - 2026-05-25 blocker: joining `Simulation-6 / 模拟赛-6` from `https://botzone.org.cn/group/69d9eca875ce41301557e099` showed `你不在该小组中。如果你确信你已经加入小组，可以尝试重新登录。`
  - The group join page `https://botzone.org.cn/generalform/IJCAI2026MCRAICOM` is a one-time team registration form requiring team name, team leader, and member count. Do not submit without user-provided registration details.
  - General public game-table creation outside the group is also blocked by a one-digit CAPTCHA; do not bypass it autonomously.
- [x] Update this file and `tasks/SUMMARIES.md` with the live upload/test outcome or the next blocker.

Review:
- Fresh HU-focused verification after the latest correction: `python -m pytest tests\test_policy_bot.py::test_policy_prefer_hu_takes_only_official_fan_checked_hu tests\test_policy_bot.py::test_policy_suppresses_hu_when_fan_checker_unavailable tests\test_export_botzone_pure_python.py::test_pure_runtime_prefer_hu_still_requires_conservative_fan_gate tests\test_export_botzone_pure_python.py::test_pure_runtime_suppresses_official_backed_7_fan_shifted_chow_false_positive tests\test_tziakcha_advisor.py::test_model_advisor_prefer_hu_requires_official_fan_acceptance -q --basetemp tmp\pytest_focused_hu` passed with 5 tests.
- Fresh full-suite verification: `python -m pytest tests -q --basetemp tmp\pytest` passed with 117 tests.
- Direct-code bootstrap smoke using the actual Botzone JSON object protocol:
  - Low-fan complete-hand claim returned `{"response": "PASS"}`.
  - Official-backed high-fan self-draw returned `{"response": "HU"}`.
- Chrome/Botzone status:
  - `MCRAgent2026A` exists in My Bots and is available in the Simulation join dialog.
  - Online group testing is blocked until the account joins the IJCAI 2026 group through its one-time team registration form.

## 2026-05-26 Aleo baseline and Tziakcha human-game path

- [x] Precisely bound the Aleo external benchmark instead of discarding it.
  - Reproduced Aleo's native crash on valid official payloads with `rc=3221225477`.
  - Updated `AleoProcessPolicy` to fail closed to `PASS` and emit diagnostics, so crashes are visible and do not invalidate the whole official-judge run.
  - Added regression coverage for fail-closed Aleo responses and policy diagnostics in official match outputs.
  - Focused verification: `python -m pytest tests\test_official_judge_match.py -q --basetemp tmp\pytest_official_judge_match_aleo` passed with 11 tests.
- [x] Run the 32-game full-model-vs-Aleo comparison requested by the user.
  - Artifact: `runs/official_judge_ensemble030070_prefer_hu_vs_aleo_failclosed_32.json`.
  - Seat 0 current model: score total -243.0, average -7.59375, top-score credit 6/32 = 18.75%, sole top-score 1/32 = 3.125%.
  - Aleo seats: score totals -15.0, 147.0, 111.0; average scores -0.46875, 4.59375, 3.46875; top-score credits 10/32, 14/32, 12/32.
  - Game-level Aleo result: at least one Aleo seat was top-score in 31/32 games = 96.875%; Aleo was sole top-score in 26/32 games = 81.25%; model tied with Aleo for top in 5 games.
  - Aleo crash diagnostics make this conservative for Aleo: 7 total native failures across 4 games were replaced with `PASS`.
- [x] Add Aleo as a usable trajectory source for training experiments.
  - `scripts/official_trajectories.py` now supports external policy kinds including `aleo`, `sample`, and `json`, plus `--offset`, `--aleo-exe`, and `--sample-exe`.
  - Focused verification: `python -m pytest tests\test_official_trajectories.py tests\test_official_judge_match.py -q --basetemp tmp\pytest_external_policy_support` passed with 14 tests.
  - Generated `runs/official_trajectories_aleo_selfplay_failclosed_16.json`.
  - Trained `models/reward_weighted_policy_aleo_selfplay_16.pkl`; metrics show good imitation-style held-out accuracy, but official play was worse than the current full model and all-zero versus the official sample bot.
  - Negative result: the tiny Aleo-teacher run is not a candidate to promote. The useful path is larger/better Aleo-guided data or a stronger architecture, not repeating the same small reward-weighted text policy.
- [x] Clone and inspect the Tziakcha human-game miner.
  - Source: `https://github.com/tziakcha-stats/tziakcha_record_miner`.
  - Local path: `external/tziakcha_record_miner`, commit `3b02ef4` (`chore: apply clang-format`).
  - Submodules are initialized.
  - The README describes `fetcher_cli` for downloading Tziakcha records, `analyzer_cli` for replay/stat extraction, and `calc_cli` for fan/shanten calculation.
  - `docs/fetcher/fetcher_cli.md` says history fetching requires `TZI_HISTORY_COOKIE`; the current shell does not have that environment variable set.
  - Build attempt with CMake failed because this PowerShell environment has no C++ compiler or NMake/Ninja on PATH.
- [ ] Fetch and convert Tziakcha human games for training.
  - Needs either `TZI_HISTORY_COOKIE` supplied by the user from a logged-in `https://tziakcha.net/history/` session or a user-approved authenticated fetch workflow.
  - Needs a working C++17 build environment for `fetcher_cli`/`analyzer_cli`, or a Python converter built directly from the documented Tziakcha record JSON schema.
  - Preserve the invariant: any Tziakcha-derived policy still must never emit `HU` unless fan >= 8 is proven.

### Tziakcha Python conversion subtask

- [x] Add test coverage for converting decoded `tziakcha_record_miner` record JSON into existing Botzone-like raw JSONL.
- [x] Implement a Python converter that:
  - decodes Tziakcha tile IDs into the repo's Botzone tile symbols;
  - decodes fetched Tziakcha `script` payloads with base64+zlib when `step` is absent;
  - reconstructs initial hands from wall/dice using the miner's documented dealing logic;
  - folds draw/discard/chi/peng/gang/hu actions into request/response log pairs usable by existing trainers;
  - maps invalid/abandoned Hu declarations to `PASS` and emits `HU` only for recorded fan counts >= 8.
- [x] Validate converted records with `convert_botzone_bc` and legal-candidate loading.
  - The miner's bundled `example_record.json` lacks `step.w`, so it cannot reconstruct real initial hands; the converter now rejects that case with a clear `step.w` error.
  - A wall-backed decoded synthetic record covers `PLAY`, `PENG`, `CHI`, `GANG`, and fan-legal `HU`, and all appear as legal candidate positives.
- [x] Run focused tests plus the full test suite.
- [x] Fetch and convert a first live no-obvious-CHAGA Tziakcha sample.
  - `POST https://tziakcha.net/_qry/history/` worked without `TZI_HISTORY_COOKIE`; the earlier 404 was caused by using GET.
  - Fetched sessions without player names starting `CHAGA` from the current history page.
  - Raw fetched records: `data/raw/tziakcha_human_records_32.jsonl` with 32 valid JSONL records.
  - Converted training records: `data/raw/tziakcha_human_botzone_raw_32.jsonl` with 32 valid Botzone-like records.
  - Fetch/convert summary: `runs/tziakcha_human_fetch_convert_32_summary.json`; one record fetch timed out, then the batch continued until 32 records were written; conversion errors were zero.
  - Behavior-cloning conversion wrote `data/processed/tziakcha_human_bc_32.jsonl` with 13,164 examples and no parse errors.
  - Legal-candidate loading produced 31,475 candidates across 32 matches. Positive actions included 1,535 `PLAY`, 82 `CHI`, 55 `PENG`, 10 `HU`, and 7 `GANG`.
  - Caveat: the raw converted logs contain 30 fan-legal Hu responses, but the current simplified legal-candidate loader recognizes only 10 as legal positives because it does not fully track exposed packs for Hu completion. This should be fixed before relying on Tziakcha Hu examples for final training.
- [x] Train a small Tziakcha-only ranker as a pipeline smoke.
  - Model: `models/legal_action_ranker_tziakcha_human_32.pkl`.
  - Metrics: `runs/legal_action_ranker_tziakcha_human_32_metrics.json`.
  - Held-out group accuracy was 0.9107908063887806; this is only a pipeline smoke and is not a promoted playing model.

### Pack-aware Tziakcha Hu candidate subtask

- [x] Add regression coverage for exposed-meld Hu candidates in the legal training loader.
- [x] Implement pack/meld tracking for training candidate generation.
  - Do not relax live policy Hu behavior; runtime `HU` remains gated by official fan checking.
  - For training, include recorded fan-legal Tziakcha `HU` actions as legal candidates when exposed meld count plus standing tiles form a complete regular hand.
- [x] Recompute legal-candidate coverage on `data/raw/tziakcha_human_botzone_raw_32.jsonl` and confirm the fan-legal Hu positives are no longer dropped.
  - Before this fix: 30 raw `HU` responses, but only 10 positive `HU` legal candidates.
  - After pack-aware completion plus recorded-Hu fallback: 30 raw `HU` responses, 30 positive `HU` legal candidates, 0 missing.
  - One recorded `HU` is still forced into the candidate set because it appears to use a special hand pattern beyond the simplified checker; it remains training-only and does not affect runtime policy `HU` gating.
- [x] Retrain the Tziakcha-32 smoke ranker after candidate coverage improves.
  - Model: `models/legal_action_ranker_tziakcha_human_32.pkl`.
  - Metrics: `runs/legal_action_ranker_tziakcha_human_32_metrics.json`.
  - Candidate count increased to 31,578; positive actions now include all 30 `HU` examples.
  - Held-out group accuracy after retraining: 0.9097627382341501.
- [x] Run focused tests and the full suite.

Review:
- Added `scripts/tziakcha_records.py`, a Python converter from decoded Tziakcha record JSON to the repo's Botzone-like raw JSONL.
- Added `tests/test_tziakcha_records.py` for tile mapping, conversion shape, legal-candidate compatibility, sub-8-fan Hu suppression, and missing-wall diagnostics.
- Focused verification: `python -m pytest tests\test_tziakcha_records.py tests\test_convert_botzone_bc.py tests\test_train_legal_action_ranker.py -q --basetemp tmp\pytest_tziakcha_focused2` passed with 10 tests.
- Added pack-aware regular-hand completion in `scripts/legal_actions.py` via optional `meld_count`; existing callers default to zero melds.
- Updated `scripts/train_legal_action_ranker.py` to track `CHI`/`PENG`/`GANG` meld counts from actual log responses and to keep actual recorded `HU` as a training candidate when the simplified checker misses a valid special shape.
- Focused verification: `python -m pytest tests\test_legal_actions.py tests\test_train_legal_action_ranker.py -q --basetemp tmp\pytest_pack_hu_focused2` passed with 12 tests.
- Fresh full-suite verification after pack-aware Hu candidate updates: `python -m pytest tests -q --basetemp tmp\pytest` passed with 130 tests.

### Tziakcha fetch scaling subtask

- [x] Add a reusable Python fetcher for Tziakcha public history/session/record endpoints.
  - Use POST `/_qry/history/`, `/_qry/game/?id=...`, and `/_qry/record/` with the same no-cookie public path proven by the 32-record sample.
  - Filter sessions with obvious bot/player names such as `CHAGA*` unless explicitly requested.
  - Deduplicate record IDs, continue past fetch/conversion errors, and write auditable raw/converted JSONL plus a summary JSON.
- [x] Add mocked tests for pagination/body behavior, CHAGA filtering, duplicate handling, and nonfatal fetch/convert errors.
  - Focused verification: `python -m pytest tests\test_fetch_tziakcha_records.py -q --basetemp tmp\pytest_fetch_tziakcha_green` passed with 4 tests.
- [x] Fetch and convert a larger no-obvious-CHAGA Tziakcha sample, starting with 256 records from the current public history page if enough records are available.
  - The first public history page yielded 180 converted no-CHAGA records, so Chrome was used to paginate Tziakcha history and extract additional session IDs.
  - Explicit-session fetch wrote 256 raw records to `data/raw/tziakcha_human_records_256.jsonl` and 256 converted records to `data/raw/tziakcha_human_botzone_raw_256.jsonl`.
  - Summary artifact: `runs/tziakcha_human_fetch_convert_256_summary.json` reports 19 sessions seen, 257 records attempted, 256 raw records written, 256 converted records written, 1 fetch error, and 0 conversion errors.
  - Caveat: the CLI finished writing artifacts but hit a Windows console `UnicodeEncodeError` while printing the non-ASCII summary; rerun output should be switched to ASCII-safe console printing.
- [x] Fix the Tziakcha fetcher CLI summary printing so non-ASCII/emoji session titles do not crash Windows console output.
  - Added `console_summary_json` with ASCII-escaped stdout rendering while keeping the summary artifact UTF-8.
  - Red check: `python -m pytest tests\test_fetch_tziakcha_records.py -q --basetemp tmp\pytest_fetch_console_red` failed on missing `console_summary_json`.
  - Green check: `python -m pytest tests\test_fetch_tziakcha_records.py -q --basetemp tmp\pytest_fetch_console_green` passed with 6 tests.
- [x] Validate the larger sample with behavior-cloning conversion and legal-candidate Hu coverage checks.
  - `python scripts\convert_botzone_bc.py --in data\raw\tziakcha_human_botzone_raw_256.jsonl --out data\processed\tziakcha_human_bc_256.jsonl` wrote 107,716 examples across 256 matches with 0 parse errors.
  - Candidate coverage artifact: `runs/tziakcha_human_256_candidate_coverage.json`.
  - Legal-candidate loading produced 260,407 candidates; all 239 raw `HU` responses are present as positive `HU` candidates, with 0 missing and 9 training-only forced special-shape Hu candidates.
- [x] Train a larger Tziakcha-derived or Tziakcha-combined ranker only after the data checks pass.
  - Trained all-decision model `models/legal_action_ranker_tziakcha_human_256.pkl`; metrics in `runs/legal_action_ranker_tziakcha_human_256_metrics.json` show test group accuracy 0.9178116042305031.
  - Trained reaction-only nonpass-weighted model `models/reaction_ranker_nonpass50_tziakcha_human_256.pkl`; metrics in `runs/reaction_ranker_nonpass50_tziakcha_human_256_metrics.json` show test group accuracy 0.9869772244851127.
  - Added `--prefer-hu` support to `scripts/create_draw_ensemble_policy.py` with red/green coverage in `tests/test_create_draw_ensemble_policy.py`.
  - Composed `models/ensemble_draw_public1000_2026_030_070_reaction_tziakcha256_prefer_hu.pkl` from the existing 30/70 draw ensemble and the Tziakcha-256 reaction ranker.
- [ ] Evaluate the new Tziakcha-derived candidates against official sample bots and Aleo before any promotion.
  - `runs/official_judge_ensemble030070_reaction_tziakcha256_prefer_hu_vs_official_sample_64.json`: 18/64 wins, average 17.3125, so the broad Tziakcha reaction model did not beat the source package's 19/64 sample-bot gate.
  - Winner-only and positive-score Tziakcha reaction models were trained and composed, but both matched the current first-32 sample-bot slice at 9/32.
  - Tziakcha draw feature rankers were trained for all data, winner-only data, and positive-score data.
  - The winner-filtered draw ensemble `models/ensemble_draw_public1000_2026_tziakcha256_winner_025_060_015_reaction1000_prefer_hu.pkl` improved the first-32 sample-bot screen to 10/32 and the Aleo 32-game target to 7/32, average -6.0625.
  - This is a small improvement over the old 6/32 Aleo top-score result but is far below the requested 80% target, so no full test was run.

### Tziakcha adversarial / GAN-style subtask

- [x] Add a tested adversarial outcome-weighted trainer.
  - Added `scripts/train_adversarial_action_ranker.py`, a GAIL-like offline trainer that learns a positive-outcome discriminator on actual human decisions, then uses discriminator scores as generator sample weights over legal candidates.
  - Red check: `python -m pytest tests\test_train_adversarial_action_ranker.py -q --basetemp tmp\pytest_adversarial_red` failed on missing trainer.
  - Green check: `python -m pytest tests\test_train_adversarial_action_ranker.py -q --basetemp tmp\pytest_adversarial_green` passed with 1 test.
- [x] Train and screen adversarial Tziakcha candidates.
  - `models/adversarial_legal_action_ranker_tziakcha_human_256.pkl`: discriminator AUC 0.6305, held-out generator group accuracy 0.9186, but official sample-bot screen collapsed to 0/32.
  - `models/adversarial_reaction_ranker_tziakcha_human_256.pkl`: discriminator AUC 0.6214, held-out reaction group accuracy 0.9870; composed with the current draw ensemble it matched the old first-32 sample-bot result at 9/32.
  - Result: the adversarial/GAN-style branch is currently a negative result for playing strength, not a promoted candidate.

### Official-judge policy-search RL subtask

- [x] Add a tested cross-entropy / policy-search driver over draw-ensemble weights.
  - Use existing trained draw rankers and the safest reaction ranker as fixed components.
  - Treat official sample-bot win credit as the primary reward and average score as the secondary reward.
  - Preserve the hard runtime invariant: `HU` is legal only when the official fan checker proves fan >= 8.
  - Added `scripts/policy_search_draw_ensemble.py`.
  - Red check: `python -m pytest tests\test_policy_search_draw_ensemble.py -q --basetemp tmp\pytest_policy_search_red` failed on the missing module.
  - Green check: `python -m pytest tests\test_policy_search_draw_ensemble.py -q --basetemp tmp\pytest_policy_search_green` passed with 4 tests.
- [x] Run small screened official sample-bot candidates before any long gate.
  - Start from known candidates around public-1k, 2026 organizer, and Tziakcha winner-filtered draw weights.
  - Only run a 32-game Botzone-style gate if a screen suggests a plausible improvement.
  - Three-draw search report: `runs/policy_search_draw_ensemble_sample8/sample8_pub2026tziwinner_report.json`.
    - Best 8-game screen: 3/8 wins, average 15.25, weights `[0.15, 0.55, 0.30]`.
  - Four-draw search report: `runs/policy_search_draw_ensemble_sample8b/sample8_pub2026p5ktziwinner_report.json`.
    - Best 8-game screen: 3/8 wins, average 15.25; adding public-5k draw weight did not improve win credit.
  - Reaction-threshold checks under `runs/policy_search_draw_ensemble_sample8_thresholds/` all matched the same 3/8 result, so margin/score reaction conservatism did not improve this slice.
  - Fresh 32-game gate for the best searched candidate: `runs/policy_search_draw_ensemble_sample32/sample32_pub2026tziwinner_150_550_300_vs_sample_32.json` scored 8/32 wins, average 10.8125.
  - Result: no searched candidate reached >50%; the prior Tziakcha winner-filtered screen at 10/32 remains better than this RL-search candidate, and the source package's 19/64 remains the best deployable sample-bot artifact.
- [ ] Report a >50% Botzone-style result only if a fresh 32-game gate exceeds 16 player-0 wins.

### Fan-aware reward / GAN follow-up

- [ ] Implement fan-aware reward training to target >30% official sample-bot win rate.
  - Plan: `docs/superpowers/plans/2026-05-26-fan-aware-rl.md`.
  - Preserve compatibility with existing numeric feature pickles by adding a separate `numeric_fan_v1` feature mode instead of changing `numeric_v1`.
  - Preserve the hard runtime invariant: `HU` remains legal only when the official fan checker proves fan >= 8.
- [x] Add official finish metadata to trajectory artifacts.
  - Added `finish` metadata with terminal action, winner, fan count, and score to `scripts/official_trajectories.py`.
- [x] Add fan-potential numeric features for draw/reaction ranking.
  - Added `scripts/fan_feature_ranker.py` with separate `numeric_fan_v1` features and policy dispatch, preserving old `numeric_v1` model compatibility.
- [x] Train a fan-aware reward-weighted candidate and screen it against official sample bots.
  - Added `scripts/train_fan_rewarded_feature_ranker.py`.
  - Red check: `python -m pytest tests\test_official_trajectories.py tests\test_fan_feature_ranker.py tests\test_train_fan_rewarded_feature_ranker.py tests\test_policy_bot.py::test_sklearn_predictor_dispatches_numeric_fan_feature_payload -q --basetemp tmp\pytest_fan_red` failed on missing fan feature/trainer modules.
  - Green check: `python -m pytest tests\test_official_trajectories.py tests\test_fan_feature_ranker.py tests\test_train_fan_rewarded_feature_ranker.py tests\test_policy_bot.py::test_sklearn_predictor_dispatches_numeric_fan_feature_payload -q --basetemp tmp\pytest_fan_green2` passed with 10 tests.
  - Generated held-out official trajectories from the previous Tziakcha winner-filtered ensemble: `runs/official_trajectories_tziwinner_vs_sample_offset64_64_fanmeta.json`, 19/64 player-0 wins, average 13.078125, minimum player-0 Hu fan 8.
  - Trained fan-rewarded draw ranker: `models/fan_rewarded_draw_tziwinner_vs_sample_offset64_64.pkl`; metrics in `runs/fan_rewarded_draw_tziwinner_vs_sample_offset64_64_metrics.json` show 12,887 candidates and test group accuracy 0.8577.
  - Negative screen: fan-rewarded model as an ensemble component topped out at 2/8, and standalone fan-rewarded draw model scored 0/8.
  - Added `numeric_fan_v1` to the broader supervised feature trainer and trained `models/feature_draw_ranker_tziakcha_human_256_winner_only_fanv1.pkl`; metrics in `runs/feature_draw_ranker_tziakcha_human_256_winner_only_fanv1_metrics.json` show test group accuracy 0.6192.
  - Best fan-feature mixed candidate `models/policy_search_tzi_old_fanv1_sample8/tzi_old_fanv1_pub2026_iter00_cand00_250_550_100_100.pkl` scored 9/32, win rate 28.125%, average 15.1875, minimum player-0 Hu fan 8.
- [x] Only promote the candidate if a fresh gate beats the current best boundary; target is >30% win rate.
  - The fan-aware candidates were not promoted because they did not beat the old Tziakcha winner-filtered ensemble.
  - Fresh 64-game gate for the old Tziakcha winner-filtered ensemble now clears the >30% target: `runs/official_judge_ensemble_draw_tziakcha256_winner_025_060_015_vs_official_sample_64.json` scored 21/64 wins, win rate 32.8125%, average 19.84375, terminal actions 21 `HU` / 43 `HUANG`, minimum player-0 Hu fan 8.
  - Full verification: `python -m pytest tests -q --basetemp tmp\pytest_fan_aware_full` passed with 150 tests and 1 convergence warning from the adversarial smoke test.

### Outcome-conditioned distillation target 40%

- [x] Design a higher-precision training pass aimed at 40% official sample-bot win rate.
  - Use the current 32.8125% model as a teacher.
  - Generate official trajectories from the teacher on first64 and offset64 seeds.
  - Train a fan-aware student only from positive/high-reward player-0 decisions, instead of weighting every draw/loss decision.
  - Compose the student into small draw ensembles and gate only if screens beat the current first-slice behavior.
  - Preserve the runtime HU invariant: `HU` remains legal only when the official fan checker proves fan >= 8.
- [x] Add tested support for filtering fan-rewarded trajectory candidates by minimum player reward.
  - Red check: `python -m pytest tests\test_train_fan_rewarded_feature_ranker.py::test_iter_fan_rewarded_candidates_can_filter_low_reward_games -q --basetemp tmp\pytest_min_reward_red` failed because `min_player_reward` was unsupported.
  - Green check: `python -m pytest tests\test_train_fan_rewarded_feature_ranker.py::test_iter_fan_rewarded_candidates_can_filter_low_reward_games -q --basetemp tmp\pytest_min_reward_green` passed.
- [x] Generate first64 teacher trajectories and combine them with existing offset64 trajectories.
  - First64 artifact: `runs/official_trajectories_tziwinner_vs_sample_first64_fanmeta.json`, matching the current best first64 gate at 21/64 wins, average 19.84375.
  - Combined artifact: `runs/official_trajectories_tziwinner_vs_sample_first64_offset64_128_fanmeta.json`.
  - Added a UTF-8 BOM regression test for trajectory loading after PowerShell wrote the combined artifact with a BOM.
- [x] Train outcome-filtered fan-aware draw and reaction student candidates.
  - Draw students: `models/fan_rewarded_draw_tziwinner_sample128_minreward050.pkl` and `models/fan_rewarded_draw_tziwinner_sample128_minreward120.pkl`.
  - Reaction students: `models/fan_rewarded_reaction_tziwinner_sample128_minreward050.pkl` and `models/fan_rewarded_reaction_tziwinner_sample128_minreward120.pkl`.
- [x] Screen/gate candidates against official sample bots; promote only if a fresh gate approaches or exceeds 40%.
  - Best distilled draw ensemble from `runs/policy_search_distilled_sample8/distilled_tziwinner128_report.json` was `models/policy_search_distilled_sample8/distilled_tziwinner128_iter00_cand04_200_550_150_000_100.pkl`.
  - Its 32-game gate `runs/official_judge_distilled_tziwinner128_200_550_150_000_100_vs_sample_32.json` scored 10/32 wins, 31.25%, average 15.375, below the current best 21/64 = 32.8125% and below the 40% target.
  - Fan-rewarded reaction swaps screened at 3/8, matching the current first8 behavior but not improving it.
- [x] Train and screen latest Tziakcha-data draw variants for the same 40% target.
  - A public Tziakcha fetch targeting 512 records produced only 261 converted records because later history pages returned 403 and one record timed out; summary: `runs/tziakcha_human_fetch_convert_512_summary.json`.
  - Latest 261 raw artifact: `data/raw/tziakcha_human_botzone_raw_512.jsonl`; it contains 261 parsed records, 255,391 legal candidates, 127,043 draw candidates, and 245 positive Hu examples.
  - The old 256-record and latest 261-record batches had zero `match_id` overlap, so they were merged into `data/raw/tziakcha_human_botzone_raw_517.jsonl`.
  - Trained latest/combined winner-only draw rankers:
    - `models/feature_draw_ranker_tziakcha_human_261_latest_winner_only.pkl`
    - `models/feature_draw_ranker_tziakcha_human_261_latest_winner_only_fanv1.pkl`
    - `models/feature_draw_ranker_tziakcha_human_517_winner_only.pkl`
    - `models/feature_draw_ranker_tziakcha_human_517_winner_only_fanv1.pkl`
  - Short official screens in `runs/policy_search_latest261_sample8/latest261_pub2026tzi_report.json` and `runs/policy_search_tzi517_sample8/tzi517_pub2026_report.json` did not improve: the old 25/60/15 public/2026/Tziakcha-256 ensemble stayed best at 3/8, while latest-data weights scored only 1/8 to 2/8.
  - No retrained candidate was promoted; the verified best remains `models/ensemble_draw_public1000_2026_tziakcha256_winner_025_060_015_reaction1000_prefer_hu.pkl`.

### Suit-permuted RL / learning-rate sweep follow-up

- [x] Add suit-permuted initdata generation for official-judge evaluation.
  - Permute only Botzone numbered suits `W`, `T`, and `B`; leave winds, dragons, and flowers unchanged.
  - Output raw-style JSONL records with `initdata` so existing `scripts/official_judge_match.py` can run them unchanged.
  - Preserve source metadata and permutation tags for auditability.
  - Do not reuse original replay logs as labels after permutation.
- [x] Generate small and full suit-permuted evaluation sets from the first official sample-bot gate slice.
  - Small set for fast policy-search screens.
  - Larger set for confirming whether candidate gains survive suit swaps.
  - Generated `data/eval/botzone_mcr_first4_suit_permuted_24.jsonl` and `data/eval/botzone_mcr_first64_suit_permuted_384.jsonl`.
- [x] Run a learning-rate sweep for the strongest fan-aware Tziakcha draw trainer.
  - Use lower learning rates with higher `max_iter`: 0.03/360, 0.04/300, 0.06/240, and 0.08/180.
  - Train winner-only `numeric_fan_v1` draw rankers from `data/raw/tziakcha_human_botzone_raw_517.jsonl`.
  - Best held-out group accuracy in this sweep was `models/feature_draw_ranker_tziakcha_human_517_winner_only_fanv1_lr006.pkl` at 0.640657 with `learning_rate=0.06`, `max_iter=240`.
  - Lower learning rates did not beat it on held-out group accuracy; `learning_rate=0.08` overfit slightly more and had lower group accuracy.
- [x] Run RL-style policy search over draw-ensemble weights.
  - Use official sample-bot win credit and average score as reward.
  - Include only candidates that keep the runtime official fan gate.
  - Promote only if a fresh original or suit-permuted gate improves on 21/64 = 32.8125%.
  - Suit-permuted 12-game reward search: `runs/policy_search_suitperm_lr_sample12/suitperm_lr_report.json`; the best candidate added 15% weight to the LR-0.06 model and improved average score on that slice but not win count.
  - Suit-permuted 24-game confirmation: `runs/official_judge_suitperm_lr_best_vs_sample_suitperm24.json` scored 6/24, average 8.7917, versus the old-weight candidate's 6/24, average 8.375.
  - Original 32-game gate for that suit-permuted winner regressed to 8/32, average 12.46875, so it was not promoted.
  - Original-slice search with the LR-0.06 model: `runs/policy_search_original_lr_sample8/original_lr_report.json`; the old 25/60/15/0 weights remained best at 3/8.
  - Added full-record suit-augmentation and trained `models/feature_draw_ranker_tziakcha_human_517_suitaug6_winner_only_fanv1_lr006.pkl` from `data/raw/tziakcha_human_botzone_raw_517_suitaug6.jsonl`.
  - The suit-augmented small-weight candidate `runs/official_judge_suitaug_original_250_550_100_100_vs_sample_32.json` tied the current best exactly on the first 32 games: 10/32 wins, average 15.1875, minimum Hu fan 8.
  - No new candidate was promoted; the verified best remains the old 25/60/15 public/2026/Tziakcha-256 winner ensemble with 21/64 wins.

### GitHub handoff and GPT Pro consultation

- [ ] Commit and push a source-focused snapshot to `https://github.com/Iconoclastic0428/MCRAgent`.
  - Include source code, tests, docs, task notes, small evaluation JSONL files, and selected official-screen reports.
  - Exclude local conda, build outputs, temp files, external clones, raw/processed corpora, large model binaries, and large trajectory dumps.
- [ ] Use the committed GitHub link in Chrome to consult GPT Pro.
  - Ask for concrete performance-improvement strategy toward matching or beating the original 64-game result.
  - Ask GPT Pro to consider recent papers/strategies when useful.
  - Wait for the full response before summarizing.

### Fan8 effective-tile overlay and Lawlorentz DRL review

- [x] Review `lawlorentz/Chinese-Standard-Mahjong-DRL` as a primary-source architecture reference.
  - Check whether it contains a runnable checkpoint or benchmark result that can be compared with the current `21/64` official sample-bot gate.
  - Extract reusable ideas for RL, value learning, rule/search, fan gating, and suit augmentation.
  - 2026-05-26 result: cloned commit `52e680174e48f900d299341e5c04e5ae3f5cc623` under `external/Chinese-Standard-Mahjong-DRL`.
  - The repo uses PyMahjongGB, 71-ish observation features, 235 masked actions, supervised pretraining, PPO/GAE actor-learner training, value learning, a model pool, and W/T/B suit augmentation.
  - No comparable model checkpoint or benchmark result is present in GitHub: `eval.log` is empty, `rl1-bot.zip` contains source only, and `__main__.py` expects `/data/model_78789.pt`.
  - Conclusion: use its RL/value/masked-action structure as a reference, but do not replace the current ensemble unless a trained checkpoint is obtained or retrained and beats the official gate.
- [x] Implement the user-specified effective-tile hierarchy as a tested local rule module.
  - Primary objective: reach a stable `>=8` fan structure with the most waits.
  - Exclude incidental fan sources when evaluating future fan potential: `is_last=False`, `is_about_kong=False`, and no rob-kong/after-kong credit.
  - Track immediate fan-valid waits, first-class shanten-reducing effective tiles, second-class tiles that increase first-class options without worsening shanten, and third-class tiles that increase second-class options.
  - Added `scripts/effective_tiles.py` and `tests/test_effective_tiles.py`.
  - Runtime fast mode `levels=0` scores immediate fan8 waits only; offline modes `levels=1..3` add first/second/third-class effective-tile hierarchy.
- [x] Wire the effective-tile rule into policy construction as an optional overlay.
  - Keep the existing best ensemble unchanged by default.
  - Preserve the hard HU invariant: no `HU` unless the official fan checker proves `>=8` fan.
  - Added optional `effective_tile_overlay` payload support in `scripts/policy_bot.py`.
  - Added `scripts/create_effective_overlay_policy.py` to wrap an existing policy pickle without mutating the base model.
  - Created local overlay artifacts:
    - `models/ensemble_draw_public1000_2026_tziakcha256_winner_025_060_015_reaction1000_prefer_hu_effective_l1.pkl`
    - `models/ensemble_draw_public1000_2026_tziakcha256_winner_025_060_015_reaction1000_prefer_hu_effective_fast_l0.pkl`
- [x] Verify with focused tests before any official sample-bot screen.
  - Focused overlay tests: `python -m pytest tests\test_policy_bot.py tests\test_create_effective_overlay_policy.py tests\test_effective_tiles.py -q --basetemp tmp\pytest_effective_policy` passed with 26 tests.
  - Full suite: `python -m pytest tests -q --basetemp tmp\pytest_effective_full` passed with 162 tests and 1 existing sklearn convergence warning.
  - Official smoke: fast `levels=0` overlay scored 1/4 wins, average 9.25 on `data/raw/botzone_mcr_1000.jsonl`; incumbent current best scored 2/4, average 17.0 on the same first four games.
  - Conclusion: the overlay is implemented and safe to search, but the immediate-wait runtime wrapper is not promoted.
- [x] Execute the first GPT Pro plan phase: instrument and search the reachable fan8 overlay path under the official judge.
  - Added policy diagnostics for draw turns, model predictions, overlay choices, overlay-changed draws, fan-check calls/accepts/rejects/errors, legal Hu sightings, and taken Hu actions.
  - Added official-match terminal summaries: terminal action counts plus player-0 Hu fan list/min/max.
  - Added `scripts/policy_search_effective_overlay.py` for official-judge searches over overlay configs, ranked by player-0 win rate then average score.
  - Aggressive immediate-wait overlay search: `runs/policy_search_effective_overlay_sample4/effective_l0_sample4_report.json` tied at 1/4 wins, average 9.25, terminal actions 1 `HU` / 3 `HUANG`, player-0 Hu fan `[13]`, with 35/74 player-0 draw choices changed. This is too disruptive.
  - Tuned fan8-positive overlay search: `runs/policy_search_effective_overlay_sample4_tuned/effective_l0_tuned_sample4_report.json` tied at 1/4 wins, average 8.0; it changed only 1 player-0 draw but still lost one incumbent Hu, with terminal actions 1 `HU` / 3 `HUANG`.
  - Refreshed incumbent diagnostic smoke: `runs/official_judge_current_best_vs_sample_smoke4_diagnostics.json` scored 2/4 wins, average 17.0, terminal actions 2 `HU` / 2 `HUANG`, player-0 Hu fans `[8, 12]`.
  - Verification: `python -m pytest tests -q --basetemp tmp\pytest_gpt_plan_full_final` passed with 170 tests and 1 existing sklearn convergence warning.
  - Conclusion: the immediate fan8-wait overlay is not the right promotion path. The next GPT Pro-plan step should be a state-value/beam-search trainer that evaluates future fan-valid waits over multiple draws, not a one-ply runtime override.
- [ ] Build the next GPT Pro-plan architecture only after the one-ply overlay negative result.
  - Extract reusable state/action tensors with masked legal actions, exposed-pack state, visible tiles, shanten, fan-potential, and wait-count features.
  - Train a value or Q/reranker target from official trajectories and Tziakcha positive-outcome games.
  - Use the user guideline as a reward/feature term: maximize >=8-fan structure with many waits, then first/second/third-class effective tiles.
  - Gate every candidate on the original official sample-bot benchmark before promotion; keep `HU` fail-closed unless official fan checking proves fan >= 8.

### Hard reset: Lawlorentz-first effective-tile base policy

- [x] Record the failure lesson from the prior >30-hour path.
  - The old model-search path optimized proxy/offline labels before enforcing the effective-tile rule as the base policy.
  - Old learned artifacts are no longer promotion candidates.
- [x] Delete old learned artifacts while preserving datasets and interfaces.
  - Preserve `data/`, official judge/fan interfaces, dataset converters, and `external/Chinese-Standard-Mahjong-DRL`.
  - Remove old `models/`, `runs/`, `dist/`, temp outputs, packaged bots, and committed historical search reports.
- 2026-05-26 reset result: deleted local generated `models/`, `runs/`, `dist/`, `tmp/`, `.pytest_cache/`, root `mcr_*.zip` files, old pure sklearn Botzone export code, and stale export tests that depended on removed pickle artifacts.
- [x] Reimplement the main policy from a Lawlorentz-style masked action structure with the effective-tile guideline as the base scorer.
  - Use Lawlorentz action IDs and response mapping as the action interface.
  - PyMahjongGB is a required dependency for fan/shanten evaluation; dependency failures must be fixed rather than skipped.
  - Use the effective-tile hierarchy as the sole draw-discard ranking rule before any learned score.
  - Reactions stay conservative: take legal `HU`; otherwise only take melds when the resulting discard improves the effective-tile score.
- 2026-05-26 dependency result: installed Visual Studio Build Tools 2022 C++ workload and installed `PyMahjongGB`; Lawlorentz imports verified with `FeatureAgent.ACT_SIZE == 235` and `OBS_SIZE == 71`.
- 2026-05-26 implementation result: added `scripts/lawlorentz_policy.py`, made `scripts/botzone_json_policy_bot.py` Lawlorentz-first, added official-judge `lawlorentz_effective` policy support, and moved the read-only Tziakcha advisor default away from deleted sklearn model artifacts.
- [x] Test the new policy on all preserved dataset families.
  - Official sample-bot gates over public Botzone, 2026 organizer, Tziakcha, and suit-permuted initdata files.
  - Offline action/replay diagnostics for datasets whose records include response logs.
  - Summarize results in `runs/lawlorentz_effective_all_datasets_report.json`.
  - 2026-05-26 all-dataset screen: `runs/lawlorentz_effective_all_datasets_report_4each.json` covered 18 usable raw/eval datasets at 4 games/examples each. Official initdata screens ranged from 1/4 to 4/4 player-0 wins depending on slice; replay-action agreement was about 0.74 on 2026 strong-AI logs and about 0.90 on Tziakcha logs. This is a screen, not a statistically meaningful full win-rate claim.
  - 2026-05-26 current 32-game gate: `runs/lawlorentz_effective_vs_botzone1000_sample_32.json` scored 10/32 player-0 wins, 31.25% win rate, 13.6875 average score, terminal actions 10 `HU` / 22 `HUANG`, minimum player-0 Hu fan 8.

### Tziakcha plugin fan-context Hu gate

- [x] Add RED tests for Tziakcha state fan context.
  - Cover after-kong draw, flower replacement after kong, rob-kong prompt, visible-count last-tile tracking, stale-discard prompt classification, and below-8 Hu suppression.
  - Red check: `python -m pytest tests\test_tziakcha_state.py::test_state_tracks_kong_draw_and_robbing_kong_win_context tests\test_tziakcha_state.py::test_state_tracks_visible_tiles_for_last_tile_hu_context tests\test_tziakcha_advisor.py::test_model_advisor_passes_last_tile_and_self_draw_flags_to_fan_checker tests\test_tziakcha_advisor.py::test_model_advisor_passes_after_kong_and_robbing_kong_flags_to_fan_checker tests\test_tziakcha_advisor.py::test_model_advisor_keeps_hu_suppressed_when_fan_checker_rejects_special_context -q --basetemp tmp\pytest_tziakcha_fan_red` failed because the state snapshot did not expose `last_win_event`/`visible_counts` and the advisor still passed false special flags.
- [x] Track official-fan context in the read-only Tziakcha state.
  - Preserve observed discard/chi/pung/kong semantics closely enough to calculate last tile, after-kong draw, rob-kong, self-draw, and no-point-hand contexts.
  - Keep the extension read-only; do not send actions to the site.
- [x] Pass full context into the official fan checker before recommending Hu.
  - `HU` remains legal only when the fan checker explicitly returns `can_hu` with fan >= 8.
  - Unknown fan or checker errors must suppress `HU`.
- [x] Verify focused advisor/state tests and the relevant broader suite.
  - Focused stale-discard regression: `python -m pytest tests\test_tziakcha_advisor.py::test_model_advisor_keeps_hu_suppressed_when_fan_checker_rejects_special_context -q --basetemp tmp\pytest_tziakcha_stale_discard_red` failed before the prompt-classification fix and passed after with `tmp\pytest_tziakcha_stale_discard_green`.
  - Focused fan-context check: `python -m pytest tests\test_tziakcha_state.py::test_state_tracks_kong_draw_and_robbing_kong_win_context tests\test_tziakcha_state.py::test_state_tracks_visible_tiles_for_last_tile_hu_context tests\test_tziakcha_state.py::test_state_preserves_after_kong_flag_through_flower_replacement tests\test_tziakcha_advisor.py::test_model_advisor_passes_last_tile_and_self_draw_flags_to_fan_checker tests\test_tziakcha_advisor.py::test_model_advisor_passes_after_kong_and_robbing_kong_flags_to_fan_checker tests\test_tziakcha_advisor.py::test_model_advisor_keeps_hu_suppressed_when_fan_checker_rejects_special_context -q --basetemp tmp\pytest_tziakcha_fan_green4` passed with 6 tests.
  - Tziakcha suite: `python -m pytest tests\test_tziakcha_state.py tests\test_tziakcha_advisor.py tests\test_tziakcha_server.py -q --basetemp tmp\pytest_tziakcha_fan_targeted3` passed with 24 tests.
  - Full suite: `python -m pytest tests -q --basetemp tmp\pytest_tziakcha_fan_full3` passed with 157 tests and 1 existing sklearn convergence warning.

### Tziakcha live result recording

- [x] Add RED tests for result/stat tracking.
  - Cover a ron/deal-in result with Tziakcha settlement bit flags, a self-draw win, a drawn hand, `/results` exposure, and persistent JSONL logging.
  - Red check: `python -m pytest tests\test_tziakcha_state.py::test_state_records_ron_result_and_deal_in_rate_from_settlement_flags tests\test_tziakcha_state.py::test_state_records_self_draw_win_and_drawn_game_rates tests\test_tziakcha_server.py::test_results_route_exposes_stats_and_persists_jsonl -q --basetemp tmp\pytest_tziakcha_results_red` failed because result fields and `/results` did not exist.
- [x] Track completed game results in the read-only observer state.
  - Record winner, discarder, self-draw/ron/draw, scores, fan when visible, local-seat win/deal-in booleans, win rate, and deal-in rate.
  - Do not send actions or mutate the live site.
- [x] Expose results through the local service and plugin UI.
  - Add `/results` and show the running stats in the dashboard/overlay.
  - Append new completed results to a repo-local JSONL ledger for later audit.
- [x] Verify focused state/server tests and the full suite.
  - Focused green check: `python -m pytest tests\test_tziakcha_state.py::test_state_records_ron_result_and_deal_in_rate_from_settlement_flags tests\test_tziakcha_state.py::test_state_records_self_draw_win_and_drawn_game_rates tests\test_tziakcha_server.py::test_results_route_exposes_stats_and_persists_jsonl -q --basetemp tmp\pytest_tziakcha_results_green` passed with 3 tests.
  - Tziakcha suite: `python -m pytest tests\test_tziakcha_state.py tests\test_tziakcha_advisor.py tests\test_tziakcha_server.py -q --basetemp tmp\pytest_tziakcha_results_targeted` passed with 27 tests.
  - Full suite: `python -m pytest tests -q --basetemp tmp\pytest_tziakcha_results_full` passed with 160 tests and 1 existing sklearn convergence warning.
  - Live service restart: `http://127.0.0.1:8765/health` returned `ok=true`, `model_loaded=true`, `read_only=true`; `/results` returned zero-game stats until the next observed completed game.

### Lawlorentz-format retraining dataset

- [x] Build a GitHub-version-format dataset from Tziakcha and preserved local raw logs.
  - Use `lawlorentz/Chinese-Standard-Mahjong-DRL` `FeatureAgent` observations, action masks, and 235-action labels.
  - Include Tziakcha converted raw logs plus local Botzone/public/organizer logs.
  - Preserve the hard rule: do not invent `HU` labels; only train `HU` where the source replay already accepted it, and runtime fan checks must still fail closed.
- 2026-05-26 result: added `scripts/build_lawlorentz_dataset.py`; built `data/processed/lawlorentz_tziakcha_local_balanced_v1` from 517 Tziakcha records, 1,000 public Botzone records, and 1,000 organizer-local records.
- Dataset artifact: 9 GitHub-style `.npz` shards under `data/processed/lawlorentz_tziakcha_local_balanced_v1/cooked_data_without0`, `count.json`, and `manifest.json`; 161,196 examples; zero invalid shanten one-hot observations after filtering.
- [x] Train a new Lawlorentz CNN supervised checkpoint from that dataset.
  - Use the upstream `CNNModel` architecture, but make the local trainer CPU/GPU-safe and artifact-oriented.
  - Save manifest, metrics, checkpoint, and a quick official-judge screen before considering any promotion.
- 2026-05-26 CPU training result: added `scripts/train_lawlorentz_supervised.py` and `LawlorentzModelPolicy` support in `scripts/lawlorentz_policy.py` / `scripts/official_judge_match.py`.
- Dependency fix: installed missing PyTorch optimizer dependencies (`mpmath<1.4`, `filelock`, `fsspec`, `networkx`; plus `regex` and `psutil` for reported package requirements).
- Full 30k-sample CNN training did not finish one epoch within 30 minutes on CPU, so it was stopped and replaced by CPU-bounded checkpoints:
  - `models/lawlorentz_supervised_tziakcha_local_balanced_v1_8k.pt`: 8,192 train samples, 2,048 validation samples, validation action accuracy 37.84%; official 8-game sample-bot screen `runs/official_judge_lawlorentz_supervised_8k_vs_sample_8.json` scored 1/8 wins, average score 4.25, minimum player-0 Hu fan 10.
  - `models/lawlorentz_supervised_tziakcha_local_balanced_v1_8k_e3.pt`: continued to validation action accuracy 42.58%, but official 8-game screen `runs/official_judge_lawlorentz_supervised_8k_e3_vs_sample_8.json` regressed to 0/8 wins.
- Conclusion: the retraining pipeline works and produces legal fan-gated checkpoints, but the first CPU-bounded neural checkpoints are worse than the current `lawlorentz_effective` 32-game baseline and must not be promoted.

### All-dataset and Tziakcha-focused evaluation

- [x] Extend the all-dataset evaluator so explicit dataset patterns do not accidentally include default patterns, and so it can evaluate either the current Lawlorentz-effective policy or a trained Lawlorentz checkpoint.
  - Updated `scripts/evaluate_lawlorentz_effective_all_datasets.py` with explicit `--dataset-pattern` semantics and `--policy lawlorentz_model --model ...` support.
- [x] Run a full replay-agreement evaluation on the Tziakcha converted test files, especially `data/raw/tziakcha_human_botzone_raw_517.jsonl`.
  - Full `levels=0` Tziakcha 517 artifact: `runs/lawlorentz_effective_l0_tziakcha517_full_replay.json`.
  - Tziakcha 517 result: 517 matches, 213,556 decisions, exact agreement 0.89355, action-type agreement 0.97716, active-draw exact 0.32178, active-draw action-type 0.98208, reaction action-type 0.97598.
  - Full Tziakcha family artifact: `runs/lawlorentz_effective_l0_tziakcha_all_full_replay.json`; suit-augmented 517x6 result: 3,102 matches, 1,281,336 decisions, exact agreement 0.89368, action-type agreement 0.97715.
  - `levels=1` full 517 replay did not finish within 15 minutes on CPU and was stopped. A 32-record `levels=1` Tziakcha run wrote `runs/lawlorentz_effective_l1_tziakcha32_full_replay.json` and slightly improved exact/action-type agreement versus `levels=0` on that 32-record file.
- [x] Run an all preserved dataset coverage pass across `data/raw/*.jsonl` and `data/eval/*.jsonl`.
  - Artifact: `runs/lawlorentz_effective_l0_all_datasets_16each.json`.
  - Covered 18 supported raw/eval JSONL files with 16 records/games each where available; no dataset errors.
  - Official sample slices ranged from 2/8 to 7/16 player-0 wins depending on initdata file; all recorded player-0 Hu fan minima were >=8.
  - Replay slices: Tziakcha action-type agreement stayed around 0.977-0.984; organizer strong-AI replay action-type agreement was about 0.764 and draw action-type only about 0.087, showing that source is not aligned with this rule policy.
- [x] Record the exact artifacts, results, and promotion decision.
  - Built Tziakcha-only cooked eval set `data/processed/lawlorentz_tziakcha517_eval_v1` with 34,424 examples.
  - Full CNN checkpoint evaluation over all 34,424 Tziakcha examples timed out after 15 minutes on CPU and was stopped; prefix-2,048 artifact `runs/lawlorentz_checkpoint_tziakcha517_prefix2048_action_accuracy.json` showed 36.13% accuracy for the 8k checkpoint and 41.99% for the continued checkpoint.
  - Promotion decision remains unchanged: do not promote either trained checkpoint; keep the current Lawlorentz-effective policy as the working model.

### Hu-rate and average-Hu-turn retest

- [x] Add explicit Hu-rate and average-Hu-turn metrics to replay and official-judge reports.
  - Replay metrics now include actual/predicted Hu response counts, match-level Hu rates, and average first-Hu turn.
  - Official judge terminal summaries now include terminal Hu count/rate and average Hu turn, plus player-0 Hu count/rate/average turn.
- [x] Re-run the focused Tziakcha 517 replay test with the new Hu metrics.
  - Artifact: `runs/lawlorentz_effective_l0_tziakcha517_full_replay_hu_metrics.json`.
  - Result: 517 matches, actual Hu match rate 0.93617, average actual first-Hu turn 98.38843, predicted Hu match rate 0.93617, average predicted first-Hu turn 97.85537.
  - Actual Hu responses: 484; predicted Hu responses: 525; action-type agreement remained 0.97716.
- [x] Re-run an official sample-bot gate with the new terminal Hu metrics.
  - Artifact: `runs/official_judge_lawlorentz_effective_l0_vs_sample_32_hu_metrics.json`.
  - Result: 32 games, player-0 wins 8/32, average score 10.4375, terminal Hu rate 0.25, average Hu turn 118.125, minimum player-0 Hu fan 8.
- [x] Re-run all supported dataset coverage with the new Hu metrics.
  - Artifact: `runs/lawlorentz_effective_l0_all_datasets_16each_hu_metrics.json`.
  - Covered 18 supported raw/eval JSONL files with 16 records/games each where available.
  - Tziakcha slices had actual Hu rates from 0.75 to 0.9375 and predicted Hu rates from 0.9375 to 1.0 on the 16-record slices; official initdata slices had terminal Hu rates from 0.1875 to 0.4375.

### CHAGA replay agreement

- [x] Find public Tziakcha records whose visible player names match `CHAGA02` through `CHAGA08`.
  - Public history page 0 exposed six CHAGA-containing sessions; pages 1+ returned HTTP 403 from the shell fetch path.
  - Found selected names in fetched records: `CHAGA02`, `CHAGA03`, `CHAGA04`, `CHAGA07`, `CHAGA08`; no `CHAGA05` or `CHAGA06` in the accessible page-0 sample.
- [x] Fetch those bot-including records and preserve the raw player names needed for seat filtering.
  - Artifacts: `data/raw/tziakcha_chaga0208_records.jsonl`, `data/raw/tziakcha_chaga0208_botzone_raw.jsonl`, `runs/tziakcha_chaga0208_fetch_convert_summary.json`.
  - Result: six sessions, 76 records attempted/written/converted, zero fetch errors and zero conversion errors.
- [x] Add a replay evaluator that counts only CHAGA02-CHAGA08 decisions while still feeding full-table context to the policy.
  - Added `scripts/evaluate_chaga_replay.py` and `tests/test_evaluate_chaga_replay.py`.
- [x] Report CHAGA-filtered Hu exact-match rate, play exact-match rate, play action-type rate, and source artifacts.
  - Artifact: `runs/lawlorentz_effective_l0_tziakcha_chaga0208_replay.json`.
  - Aggregate result: 17,973 selected CHAGA02-08 decisions, exact agreement 90.124%, action-type agreement 98.359%, actual-Hu exact match 51/53 = 96.226%, PLAY exact match 715/2,098 = 34.080%, PLAY action-type match 2,098/2,098 = 100%.

### CHAGA play-exact training pass

- [x] Try to expand CHAGA02-CHAGA08 data beyond page-0 public history.
  - Full Tziakcha history requires logged-in `TZI_HISTORY_COOKIE`; current shell environment does not have it, and page 1+ public POSTs return HTTP 403.
  - Do not print or persist any browser cookie value if one becomes available.
- [x] Build a reproducible CHAGA-only training/eval split.
  - Preserve full-table replay context, but mark only `CHAGA02`-`CHAGA08` seats as trainable.
  - Hold out complete records so the PLAY exact rate is not measured on training records.
- [x] Make existing candidate trainers honor `train_players` so non-CHAGA seats do not become labels.
- [x] Train CHAGA teacher models with PLAY exact rate as the primary target.
  - Start with fast legal feature rankers before slower Lawlorentz CNN/RL work.
  - Keep the official fan gate unchanged; CHAGA training must not relax `HU`.
- [x] Evaluate held-out CHAGA PLAY exact rate against the current Lawlorentz-effective baseline and record artifacts.
  - Split artifacts: `data/processed/chaga0208/tziakcha_chaga0208_train.jsonl`, `data/processed/chaga0208/tziakcha_chaga0208_eval.jsonl`, `data/processed/chaga0208/tziakcha_chaga0208_summary.json`.
  - Baseline held-out artifact: `runs/lawlorentz_effective_l0_tziakcha_chaga0208_eval_replay.json`, PLAY exact `165/535 = 30.841%`.
  - Best CHAGA feature artifact: `models/composite_chaga0208_drawnumeric_lr004_i320_leaf3_reactionfan.pkl`; held-out artifact `runs/composite_chaga0208_drawnumeric_lr004_i320_leaf3_reactionfan_eval_replay.json`, PLAY exact `255/535 = 47.664%`.
  - Official sanity gate: `runs/official_judge_composite_chaga0208_drawnumeric_lr004_i320_leaf3_reactionfan_vs_sample_16.json`, 2/16 wins, average score 5.4375, min player-0 Hu fan 9; this is a CHAGA imitation base, not a promoted play-strength model.
  - Lawlorentz CHAGA base dataset/checkpoint: `data/processed/lawlorentz_chaga0208_train_v1`, `models/lawlorentz_supervised_chaga0208_train_v1.pt`; full eval timed out on CPU, prefix-3 artifact `runs/lawlorentz_supervised_chaga0208_train_v1_eval_prefix3_replay.json` had PLAY exact `42/96 = 43.75%`.

### NRP L40 Transformer training pass

- [ ] Request and verify NRP L40/L40S compute for this repo.
  - Use the signed-in NRP portal to inspect resource availability and the local `nautilus` Kubernetes context.
    - Prefer 1 to 2 L40/L40S GPUs for the first Transformer run; do not create idle sleep pods.
    - Correction: do not use the NRP/Airtable reservation form for this workflow. Use only Kubernetes scheduling and, if no L40/L40S is free, wait and check every 15 minutes.
    - Record the submitted pod/job name, scheduling status, node/GPU type, and training log path.
    - 2026-05-27 update: Kubernetes-only job `mcr-transformer-l40-fit-20260527` completed on `rci-tide-gpu-04.sdsu.edu` (`NVIDIA-L40`).
    - First L40 Transformer result: 4,593,154 parameters, best epoch 23, held-out CHAGA PLAY exact `0.491589`, overall exact `0.524444`; final epoch overfit to PLAY exact `0.364486`.
    - Added best-checkpoint selection by held-out CHAGA PLAY exact rate so future jobs save the best epoch rather than the overfit final epoch.
    - Compact follow-up job `mcr-transformer-l40-sweep-20260527a` is running on `rci-tide-gpu-06.sdsu.edu` (`NVIDIA-L40`); current observed best from logs is compact `d_model=128`, 3 layers, `lr=1e-4`, PLAY exact `0.495327`.
    - Deep-research strategy file `C:\Users\Shengqi Li\Downloads\deep-research-report.md` is now treated as the governing training plan: rule-augmented candidate-action Transformer, exact legal masks, no `HU` below 8 fan, value supervision, high-ELO/tziakcha-first data, CHAGA as optional teacher, and model-size ablations.
    - Size-scaling job `mcr-transformer-l40-size-20260527a` was submitted through Kubernetes only and is pending an additional L40/L40S. It tests 4.6M, 13.5M, 30.0M, and 73.0M parameter Transformers before deciding whether to push size further.
    - 2026-05-27 size update: `mcr-transformer-l40-size-20260527a` started on `rci-tide-gpu-02.sdsu.edu` (`NVIDIA-L40`). Observed completed size metrics so far: 4.6M params PLAY exact `0.491589`; 13.5M params PLAY exact `0.508411`; 30.0M params PLAY exact `0.495327`; 73.0M params reached PLAY exact `0.521495` by epoch 18 while still running.
    - Because the 73.0M model improved the best metric, submitted Kubernetes-only size-push job `mcr-transformer-l40-sizepush-20260527a` for 178.8M and 356.8M parameter tests. It is pending an additional L40/L40S.
    - 2026-05-27 size-push result: 178.8M params reached PLAY exact `0.525234`; 356.8M params reached PLAY exact `0.527103`. Since larger size was still improving, submitted Kubernetes-only `mcr-transformer-l40-sizepush2-20260527a` for a 625.8M parameter test.
    - 2026-05-27 largest-size result: `mcr-transformer-l40-sizepush2-20260527a` completed on `rci-tide-gpu-06.sdsu.edu` (`NVIDIA-L40`). The 625.8M parameter model peaked at PLAY exact `0.512150`, below the 356.8M run, so current scaling no longer improves this flat held-out imitation setup.
    - Downloaded public GitHub ELO source to `data/raw/tziakcha_current_elo.csv` after verifying `tziakcha-stats/tziakcha_records/rank/current_elo.csv` through the GitHub connector. The file has 2,422 player rows and 448 players with ELO greater than 2300.
    - Added ELO/min-score filtering to the training-data prep path and generated score-filtered splits under `data/processed/high_elo2300/`: CHAGA page-0 records select 76/76 records; local human256 records select 160/256 records. This is the current available high-ELO dataset, but it does not yet satisfy the larger "all tziakcha history" requirement.
    - Submitted Kubernetes-only high-ELO training job `mcr-transformer-l40-highelo-20260527a`: train on score-filtered CHAGA page-0 train plus human256 train; evaluate on score-filtered CHAGA held-out eval.
    - 2026-05-27 high-ELO mixed result: `mcr-transformer-l40-highelo-20260527a` completed on `rci-tide-gpu-02.sdsu.edu` (`NVIDIA-L40`). The 178.8M parameter high-ELO-mixed run peaked at PLAY exact `0.516640`, below the CHAGA-only 356.8M run.
    - 2026-05-27 CHAGA review alignment audit: `runs/chaga_review_alignment_audit_1000_summary.json` sampled 1,000 random aligned reviewed states from 2,080 aligned review states. Recorded action vs CHAGA review candidates was top-1 `0.961`, top-3 `0.961`, and first-six-discard-top-3-else-top-1 `0.961`; offered tile, drawn tile, current actor, and hand-size checks were all `1.0` in the sample, with one claim-window mismatch and actual-in-legal-mask `0.962`.
- [ ] Add tests for a rule-augmented Transformer candidate scorer.
  - Cover grouped candidate examples, candidate masks, policy/value forward shapes, and fail-closed `HU` masking.
  - Keep `HU` legal only when the candidate source/rule gate already proves fan >= 8; unknown fan must be masked.
- [ ] Implement a compact Transformer training script that follows the current plan.
  - Use structured Lawlorentz observation tensors, action-history tokens, scalar context, candidate-action features, and a value head.
  - Train only over exact legal candidates; never use a flat unmasked action distribution.
  - Support CHAGA/tziakcha train-player filters and held-out PLAY exact evaluation.
- [ ] Run local CPU smoke tests before using the NRP job.
  - Required checks: focused pytest, tiny overfit/smoke train, and no-low-fan-Hu invariant.
- [ ] Sync the verified repo snapshot to the NRP PVC and launch the L40/L40S training job.
  - Use a finite Kubernetes Job with real training work, mounted persistent storage, and artifact output under `models/` and `runs/`.
  - Periodically check job/pod status until allocated/running, then collect final metrics.
- [ ] Evaluate and record whether the Transformer is promotable.
  - Primary replay metric: held-out CHAGA PLAY exact rate.
  - Strength gate: official sample-bot screen with minimum player-0 Hu fan >= 8.
  - Do not promote a checkpoint that beats replay imitation but regresses official-judge strength.

### CHAGA soft-distillation Transformer pass

- [ ] Add test-first support for CHAGA review candidate distributions in the Transformer trainer.
  - Map CHAGA candidate strings to legal Lawlorentz action IDs by tile type/family, not physical tile ID.
  - Collate optional teacher distributions over the legal candidate list.
  - Train with hard recorded-action loss plus soft CHAGA candidate cross-entropy/KL.
  - Report teacher top-1/top-3 metrics separately from recorded-action PLAY exact.
  - 2026-05-27 implementation update: added tested CHAGA soft-target mapping, optional teacher distributions in batches, finite-loss teacher CE, and model-vs-review top-1/top-3/relaxed metrics. Regression tests cover padded `-inf` logits so teacher loss cannot become `NaN`.
- [ ] Build an all-aligned CHAGA review target file from the current accessible CHAGA02-08 records.
  - Use the CHAGA review API cache and the existing 1,000-state audit logic, but write every aligned state instead of a sample.
  - Keep only states whose offered tile, drawn tile, actor, hand-size, and window checks pass.
  - 2026-05-27 artifact: `runs/chaga_review_alignment_audit_all.jsonl` contains 2,080 aligned review states from 76 accessible CHAGA-containing records.
- [ ] Train the next model on L40 through Kubernetes only after local tests pass.
  - Use the public GitHub ELO file to keep `>2300` filtering.
  - Include every currently available filtered dataset shard; do not silently drop a source.
  - Stop scaling size blindly unless the soft-distillation design improves the validation metrics.
  - 2026-05-27 corrected baseline metric: `runs/transformer_chaga0208_l40_eval_model_vs_chaga_review_metrics_script.json` evaluates the local 4.6M L40 checkpoint against held-out reviewed CHAGA states. Model-vs-review top-1 is `0.371648`, top-3 is `0.557471`, first-six-discard relaxed is `0.471264`; PLAY-only relaxed is `0.484211`.
  - 2026-05-27 Kubernetes-only job `mcr-transformer-l40-softdistill-20260527a` started on `rci-tide-gpu-06.sdsu.edu` (`NVIDIA-L40`) and is training the high-ELO soft-distillation model against this review-candidate metric.

### Turn-level CHAGA mismatch visualization and cleanup

- [x] Generate a turn-level visualization of where model decisions differ from CHAGA review candidates.
  - Use model-predicted action vs CHAGA candidates, not recorded player action vs CHAGA candidates.
  - Apply the corrected relaxed metric: top-3 is accepted only for a player's first six `PLAY`/discard decisions; otherwise require top-1.
  - Artifact: `runs/transformer_chaga0208_l40_eval_mismatch_by_turn.svg`.
  - Companion artifacts: `runs/transformer_chaga0208_l40_eval_mismatch_by_turn.png`, `runs/transformer_chaga0208_l40_eval_mismatch_by_turn.csv`, and `runs/transformer_chaga0208_l40_eval_mismatch_by_turn_summary.json`.
  - Current local 4.6M checkpoint result on 522 held-out reviewed states: 276 relaxed mismatches, relaxed mismatch rate `0.528736`.
- [x] Clear completed Kubernetes pods/jobs without touching the running L40 job.
  - Deleted only completed `app=mcr-transformer` jobs: `mcr-transformer-l40-fit-20260527`, `mcr-transformer-l40-highelo-20260527a`, `mcr-transformer-l40-size-20260527a`, `mcr-transformer-l40-sizepush-20260527a`, `mcr-transformer-l40-sizepush2-20260527a`, and `mcr-transformer-l40-sweep-20260527a`.
  - Verified only `mcr-transformer-l40-softdistill-20260527a` remains active on `rci-tide-gpu-06.sdsu.edu` (`NVIDIA L40`).
- [ ] Push the current repo snapshot to GitHub before requesting the GPT Pro review.
- [ ] Ask GPT Pro for a repo-grounded precision-improvement review and use the advice to choose the next training change.
  - 2026-05-27 GPT Pro review saved to `docs/reviews/gpt-pro-chaga-precision-review-2026-05-27.md`.
  - Key advice: do not scale the flat Transformer further until correctness gates pass; fix rule-gated Hu target attachment, turn-aware review lookup, original-candidate evaluation, action normalization, reviewed-state oversampling, `max_candidates=235`, and a reviewed-only overfit proof.
  - Applied first correctness patch: CHAGA Transformer example building now derives `allow_hu` from the legal/rule action mask instead of the recorded human response; audit entries now preserve `state_turn`/`state_action`; review target lookup can disambiguate repeated same request/response rows by turn with fallback for old artifacts; CHAGA review `Play` tile id `0` is accepted as `W1`.
  - Verification after patch: focused new regressions passed (`3 passed`), broader CHAGA/Transformer suite passed (`31 passed`), and current local checkpoint re-evaluation stayed at top-1 `0.371648`, top-3 `0.557471`, relaxed `0.471264`, PLAY relaxed `0.484211` because the existing checkpoint was trained before the fix.

### GPT Pro pass-1 correctness and overfit gates

- [x] Add rule-gated Hu attachment to the Transformer review target path.
  - Use legal/rule mask Hu availability rather than whether the recorded human chose `HU`.
  - Keep runtime invariant unchanged: `HU` must stay impossible unless the source rule layer made it legal.
- [x] Fix CHAGA review action normalization for physical tile id `0`.
  - `Play` with value `0` is a valid W1 tile id, not a missing action.
- [x] Make review lookup keys turn-aware.
  - New audit entries write `state_turn` and `state_context.turn`.
  - Lookup first uses `(record_id, seat, request, normalized_response, turn)`, then falls back to old no-turn keys for existing artifacts.
- [ ] Make the evaluator compare the predicted normalized action against the original CHAGA candidate strings, not only the reconstructed teacher distribution.
- [ ] Add a reviewed-only train-on-eval overfit gate; require `>95%` relaxed match before trusting more L40 training.
- [ ] Add reviewed-only session split and reviewed-oversampled high-ELO experiments before any new size scaling.
