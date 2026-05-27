# Fan-Aware RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and benchmark a fan-aware reward-training path that can improve official sample-bot win rate beyond the current roughly 30% boundary without weakening the HU fan gate.

**Architecture:** Add official finish metadata to trajectory artifacts, add a separate numeric feature mode for fan-potential features so existing pickled numeric models do not break, then train a fan-aware reward-weighted feature ranker from official trajectories. Compose the new draw ranker with the safest existing reaction model and gate against the official sample bot.

**Tech Stack:** Python, pytest, scikit-learn `HistGradientBoostingClassifier`, existing official judge wrapper, existing Botzone policy payloads.

---

### Task 1: Official Finish Metadata

**Files:**
- Modify: `scripts/official_trajectories.py`
- Test: `tests/test_official_trajectories.py`

- [ ] Add a failing test that `game_from_official_result` preserves final display metadata, including terminal action, winner, fan count, and score vector.
- [ ] Run `python -m pytest tests\test_official_trajectories.py -q --basetemp tmp\pytest_fan_metadata_red` and confirm it fails because `finish` is absent.
- [ ] Implement `finish_metadata_from_output` and include its result in the trajectory game object.
- [ ] Run `python -m pytest tests\test_official_trajectories.py -q --basetemp tmp\pytest_fan_metadata_green` and confirm it passes.

### Task 2: Fan-Potential Feature Mode

**Files:**
- Create: `scripts/fan_feature_ranker.py`
- Modify: `scripts/policy_bot.py`
- Test: `tests/test_fan_feature_ranker.py`, `tests/test_policy_bot.py`

- [ ] Add failing tests for suit concentration, honor/terminal density, pair/triple counts, seven-pairs/orphans proximity, and policy dispatch for `feature_mode = numeric_fan_v1`.
- [ ] Run the focused tests and confirm they fail because the new module/dispatch does not exist.
- [ ] Implement `fan_feature_names`, `featurize_fan_response`, `featurize_fan_candidates`, and `featurize_fan_legal_responses`.
- [ ] Update `SklearnPredictor` to dispatch `numeric_fan_v1` payloads to the new feature functions while leaving `numeric_v1` on the existing feature path.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Fan-Aware Reward Trainer

**Files:**
- Create: `scripts/train_fan_rewarded_feature_ranker.py`
- Test: `tests/test_train_fan_rewarded_feature_ranker.py`

- [ ] Add failing tests for player reward calculation and training payload shape.
- [ ] Run `python -m pytest tests\test_train_fan_rewarded_feature_ranker.py -q --basetemp tmp\pytest_fan_reward_red` and confirm it fails because the trainer does not exist.
- [ ] Implement official-trajectory candidate generation, fan-aware sample weights, numeric fan feature training, metrics, and payload writing.
- [ ] Run `python -m pytest tests\test_train_fan_rewarded_feature_ranker.py -q --basetemp tmp\pytest_fan_reward_green` and confirm it passes.

### Task 4: Training And Gate

**Files:**
- Generated: `runs/official_trajectories_*fan*.json`
- Generated: `models/fan_rewarded_*.pkl`
- Generated: `runs/official_judge_*fan*.json`

- [ ] Generate official trajectories from the strongest current local model versus sample bots.
- [ ] Train a draw-only fan-aware feature ranker from those trajectories.
- [ ] Compose it into a draw ensemble with the existing public-1k, 2026 organizer, and Tziakcha winner-filtered draw rankers.
- [ ] Run an 8- or 16-game screen. If it clears 30%, run a 32-game gate, then a 64-game gate if still promising.
- [ ] Record exact artifact paths and whether the candidate exceeded 30%.

### Task 5: Verification And Notes

**Files:**
- Modify: `tasks/TODO.md`
- Modify: `tasks/SUMMARIES.md`

- [ ] Run focused tests.
- [ ] Run `python -m pytest tests -q --basetemp tmp\pytest_fan_aware_full`.
- [ ] Update task notes with results and remaining gaps.

---

## Execution Result

- Implemented Tasks 1-3 with red/green coverage.
- Generated held-out fan-metadata trajectories at `runs/official_trajectories_tziwinner_vs_sample_offset64_64_fanmeta.json`.
- Trained `models/fan_rewarded_draw_tziwinner_vs_sample_offset64_64.pkl`, but it was not promoted because official sample-bot screens regressed.
- Trained `models/feature_draw_ranker_tziakcha_human_256_winner_only_fanv1.pkl`, but the best mixed fan-feature candidate reached only 9/32 on the first official sample-bot gate.
- Fresh 64-game gate for the older Tziakcha winner-filtered ensemble cleared the >30% target at 21/64 wins.
- Full verification passed: `python -m pytest tests -q --basetemp tmp\pytest_fan_aware_full` reported 150 passed, 1 warning.
