# GPT Pro QADV Historical Terminal Plan - 2026-05-29

Source conversation: https://chatgpt.com/c/6a167dc8-942c-83e8-810c-cf16c97c2c92

Prompt context: The full CHAGA-reviewed NRP CPU32 evaluation completed for `models/transformer_candidate_finetune_medhard_l40_20260528b.pt`. Validation relaxed accuracy was `0.866389`, validation top-1 was `0.818950`, validation PLAY relaxed was `0.859256`, and Hu was clean. Test relaxed accuracy was `0.866957`, test top-1 was `0.816067`, test PLAY relaxed was `0.859147`, and Hu was clean. The current weakness is PLAY/discard ordering, not Hu. Earlier official sample/shanten fixed-league probes were terminal-flat, so the question was how to do actual Q-adversarial/self-play training without reward hacking.

GPT Pro recommendation:

- Build QADV-v1 from exact historical tziakcha terminal returns plus the existing CHAGA hard-negative anchor.
- Do not start full self-play return training yet.
- Use self-play only after an official rollout logger proves nonzero safe terminal signal.
- Keep the Transformer frozen; train a small legal-candidate Q reranker.
- Keep CHAGA agreement as a regression anchor and full utility/safety as the real promotion gate.

Implementation order:

1. Build exact terminal-outcome Q data from historical tziakcha trajectories.
2. Extend QADV training to mix CHAGA hard examples and terminal-return rows.
3. Add an official-league rollout logger for future self-play data.
4. Train QADV-v1 on CHAGA hard negatives plus historical terminal returns.
5. Sweep lambda on CHAGA validation and test.
6. Run fixed-league/self-play feasibility only if the signal gate is nonzero.

Required first script: `scripts/build_qadv_terminal_trajectories.py`

Purpose: convert historical prepared tziakcha records into per-decision Q training rows with terminal utility labels. It must reuse the Transformer trainer/evaluator state-building path and must not use `scripts/selfplay_sim.py`.

Example command:

```bash
python scripts/build_qadv_terminal_trajectories.py \
  --raw data/processed/chaga0208_all_plus_live_elo2400_cached_reviews/session_split_seed20260527/train.prepared.jsonl \
  --review-audit-jsonl data/processed/chaga0208_all_plus_live_elo2400_cached_reviews/session_split_seed20260527/train_val.audit.jsonl \
  --checkpoint models/transformer_candidate_finetune_medhard_l40_20260528b.pt \
  --out-jsonl data/processed/qadv/qadv_terminal_train_v1.jsonl.gz \
  --summary-out data/processed/qadv/qadv_terminal_train_v1_summary.json \
  --max-candidates 235 \
  --relaxed-play-top3-count 4 \
  --device cpu
```

Terminal-row schema should include:

- `schema_version: qadv_terminal_v1`
- `state_key`: record/session/player/turn/request/normalized action/play ordinal
- `candidate_digest`
- legal candidate IDs, normalized names, families, Hu gate, candidate count, truncation count
- frozen-base predicted action, logits, and ranks
- logged/chosen action
- optional CHAGA fields: candidate norms, accepted norms/action IDs, first-four top-3 flag, top-1 norm, top1-top2 margin
- terminal result: Hu/Huang, winner, discarder, self-draw, final turn, player win/deal-in, score delta, placement delta, end-wait, wait-when-deal-in, fan/base fan, low-fan Hu flag
- return fields: normalized point delta, terminal return, discounted return, component terms
- safety fields: action-outside-mask, accepted-Hu-outside-gate, low-fan Hu

Recommended v1 return:

```python
point_delta_norm = clip(score_delta / 64.0, -1.0, 1.0)
hu_turn_norm = hu_turn / final_turn if player_won and final_turn > 0 else 0.0

terminal_return = (
    1.00 * point_delta_norm
  + 0.20 * player_won
  + 0.10 * end_wait
  + 0.05 * wait_when_deal_in
  - 0.15 * player_dealt_in
  - 0.10 * hu_turn_norm * player_won
)

discounted_return = terminal_return * (0.995 ** max(0, final_turn - turn))
```

Terminal trajectory gates:

- preferred rows `>= 100000`; minimum `>= 25000`
- preferred games `>= 2000`; minimum `>= 500`
- `nonzero_score_games > 0`
- `return_std >= 0.03`
- `candidate_truncation_count = 0`
- `action_outside_mask = 0`
- `low_fan_hu = 0`
- `accepted_hu_outside_gate = 0`
- `empty_legal_candidate_rows = 0`
- `missing_terminal_result = 0`

Required trainer extension: `scripts/train_qadv_reranker.py`

Add mixed batches from:

- CHAGA hard examples: `qadv_hard_v1`
- historical terminal rows: `qadv_terminal_v1`

Add arguments:

```text
--terminal-jsonl
--eval-terminal-jsonl
--return-loss-weight 0.20
--terminal-sample-weight 1.0
--hard-sample-weight 1.0
--gamma 0.995
--value-target-key discounted_return
```

Loss:

```python
final_logits = base_logits + train_lambda * q_scores

L_accept = -logsumexp(final_logits[accepted_actions]) + logsumexp(final_logits[legal_actions])
L_pair = mean(max(0, margin - final_logits[pos] + final_logits[hard_neg]))
L_soft = KL(CHAGA_soft_distribution || softmax(final_logits))
L_return = Huber(q_scores[chosen_action], discounted_return)
L_cql = logsumexp(q_scores[legal_actions]) - mean(q_scores[support_actions])
L_q_l2 = mean(q_scores[legal_actions] ** 2)

L = (
    1.00 * L_accept
  + 0.50 * L_pair
  + 0.05 * L_soft
  + 0.20 * L_return
  + 0.03 * L_cql
  + 0.0001 * L_q_l2
)
```

The Transformer remains frozen. Do not use PPO for this step.

Required rollout logger: `scripts/generate_qadv_league_rollouts.py`

Purpose: wrap the official judge and write per-decision rollout rows for future use. Do not train from rollout returns unless signal gates pass.

Suggested opponent pool:

- baseline Transformer
- QADV v0 lambda 0.05
- QADV v0 lambda 0.10
- lawlorentz/effective deterministic baseline if practical
- baseline exploratory top-k variant

Rollout signal gates before using rollout returns:

- preferred games `>= 2000`; minimum smoke `>= 200`
- illegal Hu `= 0`
- action-outside-mask `= 0`
- low-fan Hu `= 0`
- nonzero-score games `>= 5%`
- Hu games `>= 5%` or clearly nonzero point-delta standard deviation
- return standard deviation `>= 0.03`
- seats 0,1,2,3 covered
- policy pool includes baseline and non-baseline opponents

Evaluation requirements:

- Add `--terminal-jsonl` support to `scripts/evaluate_qadv_reranker.py`.
- Sweep lambdas `0.00,0.05,0.10,0.20,0.35,0.50`.
- Lambda zero must reproduce the frozen baseline.
- Candidate lambda gate:
  - validation relaxed no worse than baseline by more than `0.003`
  - validation PLAY relaxed no worse by more than `0.003`
  - changed-to-accepted rate beats changed-from-accepted-to-wrong rate
  - terminal Q separates positive-return and negative-return chosen actions
  - low-fan Hu count `0`
  - action-outside-mask count `0`

Promotion gate:

- test relaxed no worse than `0.866957 - 0.003`
- test PLAY relaxed no worse than `0.859147 - 0.003`
- Hu slice stays clean
- fixed-league safety is perfect
- average point delta improves if measurable
- Hu/end-wait stable or better
- average Hu turn decreases if Hu signal exists
- deal-in rate does not increase by more than 1 to 2 percentage points

Kubernetes plan:

- CPU32 job builds terminal trajectories.
- L40 job trains QADV-v1.
- CPU32 or L40 job runs lambda sweep/evaluation.

Tests to add first:

- terminal row schema round-trip
- nonzero score creates nonzero return
- low-fan Hu fails the gate
- action outside mask fails the gate
- all-zero Huang data is rejected by return-std gate
- terminal return loss separates positive and negative rows
- mixed hard/terminal batches train without missing fields
- lambda zero reproduces base logits
- Q cannot select masked actions
- rollout rows include policy pool and terminal result
- rollout gate rejects all-zero terminal signal
- rollout never samples outside legal mask or low-fan Hu

Do not do yet:

- PPO
- updating the 92.7M Transformer during Q training
- training from sample-bot all-Huang returns
- using `scripts/selfplay_sim.py` as exact MCR return data
- optimizing Hu rate alone
- allowing Q to add or legalize actions
- repeatedly tuning on the fixed test split

Success indicators:

- lambda zero exactly reproduces baseline
- lambda greater than zero changes a small but nonzero fraction of decisions
- more changed decisions become CHAGA-accepted than become wrong
- Q scores are higher for positive-return logged actions than negative-return logged actions
- CHAGA relaxed stays within 0.3 percentage points of baseline or improves
- fixed-league safety remains perfect
- point/feasibility metrics improve when terminal signal is available
