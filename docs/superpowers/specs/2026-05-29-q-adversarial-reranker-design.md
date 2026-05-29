# Q-Adversarial Reranker Design

## Goal

Move beyond plain CHAGA-supervised finetuning while preserving the new strict
finetuned Transformer as the safety anchor. The objective is to improve fixed
league game utility and feasibility metrics without losing the strong CHAGA
candidate-match behavior or violating MCR legality.

The frozen base policy is:

`models/transformer_candidate_finetune_medhard_l40_20260528b.pt`

This checkpoint has 92,688,386 parameters and SHA256
`94DAEDACC932C03A902F5099544151E8A03425BE6C86B8FA577DDE4D05DAA22A`.

## Baseline Gate

The checkpoint is locked as the new baseline because it passed the strict
finetune gate:

| Split | Metric | Base | Finetuned | Delta |
|---|---:|---:|---:|---:|
| validation | relaxed | 84.5594% | 86.6389% | +2.0795 pp |
| validation | PLAY relaxed | 84.8054% | 85.9256% | +1.1202 pp |
| test | relaxed | 84.9025% | 86.6957% | +1.7933 pp |
| test | PLAY relaxed | 85.3616% | 85.9147% | +0.5532 pp |

Candidate truncation is zero, reviewed rows without teacher distribution are
zero, and the live advisor Hu gate remains `base_fan >= 8` with flowers
excluded.

## Inference Design

Keep the Transformer frozen. Add a small candidate-aligned Q reranker that
scores only legal candidates already produced by the existing rule layer.

```text
final_score(s, a) = transformer_logit(s, a) + lambda * Q_adv(s, a)
action = argmax(final_score over legal candidates)
```

Initial lambda sweep:

`0.00, 0.05, 0.10, 0.20, 0.35, 0.50`

Choose the smallest lambda that improves fixed-league utility while preserving
CHAGA and Hu-safety gates.

## Training Data

Use three automatic sources.

1. CHAGA reviewed states

Positive accepted set:

- first four `PLAY` decisions: CHAGA top-3
- later `PLAY` decisions: CHAGA top-1
- non-`PLAY` decisions: CHAGA top-1

Hard negatives:

- frozen baseline top-1 when not accepted
- frozen baseline top-3 actions not accepted
- CHAGA rank 2 or 3 when strict top-1 is required
- deterministic effective-tile baseline action when not accepted

2. Self-play rollouts

Use a fixed league:

- frozen strict-finetuned baseline
- previous supervised checkpoint
- older checkpoints
- deterministic mahjong-algorithm/effective-tile baseline
- pushy variant
- defensive variant

Exploration must stay inside legal candidates. Hu must never be sampled unless
the base-fan gate proves `base_fan >= 8`.

3. Hard-example replay

Mine and oversample:

- same-family `PLAY` not in CHAGA top-5
- CHAGA top-3 but rejected under strict top-1 region
- wrong action family
- late-turn `PLAY` mismatch
- high-margin CHAGA disagreement
- claim/pass/Hu mismatch

## Reward

Use a bounded hand-level feasibility return until reliable point delta is
available:

```text
R =
    + 1.00 * win
    + 0.35 * end_wait
    + 0.10 * wait_when_deal_in
    - 0.15 * deal_in
    - 0.25 * normalized_hu_turn_if_win
    + 0.10 * normalized_point_delta
```

Deal-in penalty stays small but nonzero. `wait_when_deal_in` is diagnostic
signal, not permission to push recklessly.

## Loss

```text
L =
    1.00 * accepted_set_loss
  + 0.50 * hard_negative_pair_loss
  + 0.20 * selfplay_return_loss
  + 0.03 * conservative_q_loss
  + 0.05 * soft_chaga_loss
```

Definitions:

- accepted-set loss: `-logsumexp(score[accepted]) + logsumexp(score[legal])`
- hard-negative pair loss: `max(0, margin - Q(accepted) + Q(hard_negative))`
- self-play return loss: Huber regression on normalized Monte Carlo return
- conservative Q loss: `logsumexp(Q_legal) - mean(Q_accepted)`
- soft CHAGA loss: small teacher-weighted ranking anchor

Start with Monte Carlo returns. Do not add TD/IQL until rollout termination and
metrics are stable across seeds.

## Promotion Gate

Hard requirements:

- illegal Hu count = 0
- action outside legal mask = 0
- Hu only when base fan is at least 8 with flowers excluded
- candidate truncation count = 0

Regression limits:

- CHAGA relaxed drop no worse than 0.3 to 0.8 pp from the frozen baseline
- PLAY relaxed drop no worse than 0.3 to 0.8 pp from the frozen baseline
- top-1 must not collapse

Utility requirements:

- average point delta improves against the fixed league, if available
- otherwise improve at least three of: Hu rate up, end-wait rate up, average
  Hu turn down, deal-in rate not materially worse
- deal-in rate must not rise by more than 1 to 2 pp

## Implementation Sequence

1. Add a baseline registry/metadata artifact for the frozen checkpoint.
2. Add a fixed-league feasibility evaluator for the frozen baseline. Done in
   `scripts/evaluate_fixed_league_feasibility.py`; smoke output is in
   `runs/strict_finetune_20260528b_feasibility_smoke4.json`.
3. Mine hard examples from the frozen baseline using strict first-four top-3
   semantics.
4. Generate small legal self-play rollouts and verify zero illegal actions and
   zero low-fan Hu.
5. Add a small Q reranker trainer with the Transformer frozen.
6. Sweep lambda and evaluate CHAGA plus fixed-league feasibility.
7. Promote only if utility improves and safety/regression gates pass.

## Deliberate Non-Goals

- Do not run full PPO/self-play policy finetuning as the next step.
- Do not let Q select actions outside the current legal candidate generator.
- Do not optimize Hu rate alone.
- Do not deploy a reranked model that breaks CHAGA or Hu-safety gates.
