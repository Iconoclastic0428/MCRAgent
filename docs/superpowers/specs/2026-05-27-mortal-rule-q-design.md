# Mortal Rule-Q Training Design

## Goal

Build a side training path that uses the deterministic mahjong-algorithm/effective-tile scorer as a tactical prior, then trains a stronger learned candidate-action model on the collected tziakcha `>2300` corpus with CHAGA validation/test as the gate.

The target is not to replace the current Transformer job immediately. The target is to create a parallel experiment that can beat the current supervised Transformer on CHAGA relaxed candidate match, especially `PLAY`, while preserving the hard MCR invariants: legal mask only, no low-fan `HU`, and training on all eligible record-local `>2300` players.

## Current Evidence

The deterministic PyMahjongGB/mahjong-algorithm-style baseline is strong on claim and winning windows but weak on discard direction:

| Slice | Relaxed match vs CHAGA |
|---|---:|
| Overall | 54.65% |
| `PLAY` | 50.53% |
| `CHI` | 90.81% |
| `PENG` | 95.17% |
| `HU` | 100.00% |
| `GANG` | 9.68% |

The graph in `docs/figures/mahjong_algorithm_chaga_valtest_play_relaxed_miss_by_turn.svg` shows the worst `PLAY` misses happen early, where a rule-only scorer has weak hand-direction judgment. This makes the baseline useful as a feature source and reranker, not as the primary policy.

The active `20260527d` Transformer jobs should continue running. The new work must create a separate model/training job so we can compare model families without losing the current L40 results.

## Approaches Considered

### Option A: Transformer With Stronger Rule Features

Keep the current candidate-action Transformer and turn rule features back on. Add explicit mahjong-algorithm prior logits, shanten/effective-tile values, fan-valid wait counts, and action family indicators. Continue CE/KL/accepted-set training.

This is the lowest-risk path because it reuses the current trainer, but it is still a policy-imitation model. It does not test the Mortal-style value-learning idea.

### Option B: Mortal-Style Dueling Q Candidate Model

Train a candidate-action model that scores legal actions as Q-values. The encoder can initially reuse the existing Transformer state/history/candidate stack. The output changes to a dueling Q decomposition:

`Q(s,a) = V(s) + A(s,a) - mean_legal(A(s,*))`

Training combines expert imitation, CHAGA soft/accepted-set supervision, offline CQL-style regularization, and value targets from match score/placement. The effective-tile scorer becomes both a feature source and an optional prior term.

This is the recommended first side experiment. It changes the learning objective enough to test Mortal-style training while keeping the existing data and legal-action infrastructure.

### Option C: Full LuckyJ-Style Regret/Search Agent

Build a regret-minimizing self-play/search system using online imperfect-information search outputs as policy features. This is too large for the current phase. The practical LuckyJ idea to import now is "search/rule return as a model feature," which Options A and B already cover.

## Selected Design

Implement Option B first as `mortal_rule_q`.

The first model will be a Transformer-dueling-Q candidate scorer, not a CNN. It uses the current Lawlorentz observation tensor, action-history tokens, player/value context, candidate action embeddings, and candidate rule features. The action head produces per-candidate Q-values using a dueling structure. This allows the same legal-action mask, CHAGA evaluation, and Kubernetes streaming dataset to be reused.

After the Transformer-Q experiment is verified, add a 1-D ResNet/CNN ablation over tile planes if the Q objective helps but the Transformer remains too slow or overfits.

## Model Inputs

Each training example uses:

- `obs`: existing Lawlorentz feature tensor.
- `history_tokens`: existing public-action history window.
- `player`, `allow_hu`, `value_target`: existing scalar inputs.
- `candidate_actions`: legal candidate action IDs after hard `HU` gating.
- `candidate_rule_features`: expanded from 7 to a larger stable feature vector.

Expanded candidate rule features should include:

- action family flags: `HU`, `PLAY`, claim, `PASS`, `GANG`.
- mahjong-algorithm/effective-tile values for discard candidates: fan-valid wait tiles/types, first effective tiles/types, second/third effective tiles/types, min shanten, max structural fan.
- algorithm prior rank and normalized prior score among legal candidates.
- claim-after-discard profile for `CHI`/`PENG` windows when available.

If a feature cannot be computed, it must be zero and accompanied by a counter in the load summary. Unknown fan must never enable `HU`.

## Model Outputs

The model returns candidate-aligned Q-values:

- `q_values`: one scalar per candidate.
- `state_value`: dueling value stream output.
- `advantages`: dueling advantage stream output.
- optional auxiliary heads later: placement bucket, fan bucket, and deal-in risk.

Action selection uses `argmax(q_values)` over legal candidates only.

## Losses

Use a multi-term offline objective:

1. Hard expert CE on the record-local `>2300` human action.
2. CHAGA soft KL on reviewed rows with mapped teacher distributions.
3. CHAGA accepted-set loss for the corrected relaxed metric.
4. Conservative Q-learning regularizer:

   `CQL = logsumexp(Q_legal) - Q(chosen)`

   Apply this to all examples so the model is penalized for inflating unseen legal actions.

5. Q regression target:

   `q_target = value_target + algorithm_prior_bonus`

   Keep the prior bonus small and bounded so it improves credit assignment without overriding labels.

The first run should use conservative weights:

| Loss | Initial weight |
|---|---:|
| hard CE | 1.0 |
| CHAGA soft KL | 0.25 |
| accepted-set | 2.0 |
| CQL | 0.1 |
| Q regression | 0.25 |

## Data Split

Training remains:

- all eligible record-local `>2300` tziakcha player rows from the combined archive+live corpus.
- CHAGA-reviewed training rows may be repeated at the start of the stream to ensure early teacher signal.
- CHAGA validation/test sessions remain held out from the all-player stream.

Validation remains:

- CHAGA validation split for checkpoint selection.
- CHAGA test split for final held-out reporting.
- Main metric: original CHAGA relaxed match.
- Secondary metric: `PLAY` relaxed match.

## Kubernetes Plan

Create a new single-L40 job pair:

- medium `mortal_rule_q`, roughly comparable to current 768x12 Transformer.
- optional smaller debug job with fewer layers for smoke tests.

Do not request multiple GPUs in one pod because the trainer is not distributed. If more L40s are available, run more single-GPU jobs with different model families or loss weights.

The active `20260527d` jobs must not be deleted by this side experiment.

## Evaluation Gates

A checkpoint is promotable only if:

- candidate truncation count is zero at 235 candidates.
- `HU` below 8 fan count is zero.
- original relaxed match beats the deterministic baseline `54.65%`.
- `PLAY` relaxed match beats the deterministic baseline `50.53%`.
- it beats the current best active Transformer snapshot before Chrome plugin deployment.
- full held-out CHAGA test evaluation is run before any plugin integration.

## Files To Add Or Modify

Expected implementation files:

- Add `scripts/train_mortal_rule_q_candidate.py`.
- Add `tests/test_train_mortal_rule_q_candidate.py`.
- Modify `scripts/train_transformer_candidate.py` only if shared helpers need extraction.
- Add `scripts/mortal_rule_features.py` if expanded rule features become too large for the trainer.
- Add `k8s/mcr-mortal-rule-q-l40-20260527a.yaml`.
- Update `tasks/TODO.md` and `tasks/SUMMARIES.md`.

Keep the first implementation narrow. Do not add self-play, regret minimization, browser plugin integration, or a CNN ablation until the Q trainer is verified on a smoke run and one L40 run.

## Risks

The biggest risk is label-objective conflict: CQL/value losses can reduce CHAGA imitation if overweighted. The first run should therefore be teacher/accepted-set dominated, with CQL and Q regression as small regularizers.

The second risk is feature computation speed. The current all-player streaming path is already CPU-sensitive. If expanded rule features slow training too much, precompute only algorithm prior rank/score for reviewed rows first, then expand feature width later.

The third risk is another proxy-metric trap. The promotion gate must remain held-out original CHAGA relaxed match and `PLAY` relaxed match, followed by actual game-strength tests later.

## Self-Review

- No placeholder sections remain.
- The design preserves the user's hard split: train on all record-local `>2300` players and evaluate on CHAGA.
- The design preserves hard MCR legality and no-low-fan-`HU` rules.
- The design imports Mortal through dueling Q and CQL-style offline value learning.
- The design imports LuckyJ only as rule/search returns used as model features, avoiding premature full regret-search implementation.
