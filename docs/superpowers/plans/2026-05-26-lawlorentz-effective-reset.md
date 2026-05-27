# Lawlorentz Effective Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the previous learned-model path with a Lawlorentz-style masked-action policy whose base draw rule is the MCR effective-tile hierarchy.

**Architecture:** Preserve datasets and official/Botzone interfaces, delete old learned artifacts, then add a deterministic effective-tile policy that uses Lawlorentz action IDs and the repo's official fan checker. Learned components may be added later only as tie-breakers that cannot override the effective-tile rule or the >=8 fan `HU` gate.

**Tech Stack:** Python 3.12, existing official C++ judge/fan checker, Lawlorentz action-map structure, existing Botzone text protocol, pytest.

---

### Task 1: Reset Artifacts

**Files:**
- Modify: `D:\MCR_Agent\.gitignore`
- Modify: `D:\MCR_Agent\tasks\TODO.md`
- Modify: `D:\MCR_Agent\tasks\LESSONS.md`
- Delete generated artifacts under `D:\MCR_Agent\models`, `D:\MCR_Agent\runs`, `D:\MCR_Agent\dist`, `D:\MCR_Agent\tmp`

- [ ] Verify the workspace root resolves to `D:\MCR_Agent`.
- [ ] Delete only generated learned artifacts and temp/package outputs inside the workspace.
- [ ] Remove `.gitignore` exceptions for obsolete run reports so old search artifacts are no longer preserved.
- [ ] Keep `data/`, `external/Chinese-Standard-Mahjong-DRL`, official judge/fan code, and dataset converters.

### Task 2: Lawlorentz Action Adapter

**Files:**
- Create: `D:\MCR_Agent\scripts\lawlorentz_actions.py`
- Test: `D:\MCR_Agent\tests\test_lawlorentz_actions.py`

- [ ] Encode the Lawlorentz 235-action offsets and tile order.
- [ ] Add conversion between action IDs and Botzone responses.
- [ ] Add conversion from Botzone responses back to action IDs for legal-mask diagnostics.
- [ ] Verify pass, hu, play, chi, peng, gang, concealed gang, and bugang mappings.

### Task 3: Effective-Tile Base Policy

**Files:**
- Modify: `D:\MCR_Agent\scripts\policy_bot.py`
- Modify: `D:\MCR_Agent\scripts\official_judge_match.py`
- Test: `D:\MCR_Agent\tests\test_policy_bot.py`
- Test: `D:\MCR_Agent\tests\test_official_judge_match.py`

- [ ] Add an `EffectiveTilePredictor` that ranks draw discards only by the effective-tile hierarchy.
- [ ] Add reaction scoring that takes legal `HU` and otherwise accepts melds only when their forced discard improves effective-tile score over passing.
- [ ] Add `--policy effective` to the official judge harness.
- [ ] Keep `HU` fail-closed unless the official fan checker proves fan >= 8.

### Task 4: All-Dataset Evaluation Harness

**Files:**
- Create: `D:\MCR_Agent\scripts\evaluate_lawlorentz_effective_all_datasets.py`
- Test: `D:\MCR_Agent\tests\test_evaluate_lawlorentz_effective_all_datasets.py`

- [ ] Discover preserved raw/eval dataset files with `initdata`.
- [ ] Run official sample-bot gates for each dataset with a configurable game cap.
- [ ] Report PyMahjongGB availability and Lawlorentz checkpoint availability.
- [ ] Write `runs/lawlorentz_effective_all_datasets_report.json`.

### Task 5: Verification

**Files:**
- Modify: `D:\MCR_Agent\tasks\TODO.md`
- Modify: `D:\MCR_Agent\tasks\SUMMARIES.md`

- [ ] Run focused tests for the Lawlorentz action adapter, effective policy, and evaluation harness.
- [ ] Run the full suite with a repo-local basetemp.
- [ ] Run the all-dataset evaluation harness.
- [ ] Record the actual performance and whether the new policy is strong enough for future RL.
