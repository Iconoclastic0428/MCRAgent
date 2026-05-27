# GPT Pro CHAGA Precision Expansion Review - 2026-05-27

Source conversation: https://chatgpt.com/c/6a167dc8-942c-83e8-810c-cf16c97c2c92

Repo commit under review: https://github.com/Iconoclastic0428/MCRAgent/commit/efcf71f

My read: the hardening in efcf71f is sufficient for data expansion, with one training-loop metric gap to fix before the next L40 training run. The next change that can realistically move held-out relaxed precision is still expanding CHAGA-reviewed data, not another large model or a family head.

1. Remaining correctness issue before the next L40 job

The standalone evaluator now measures the correct original-candidate metric, but the training loop still selects checkpoints using distribution-derived validation metrics from evaluate_model(), not the original-candidate metric. evaluate_model() uses teacher_target_dist, computes top1_index = argmax(row_dist), and derives relaxed match from the soft distribution’s top-k slots. That is still a proxy for the metric you actually care about. The checkpoint selection then ranks epochs using monitor_metric from this validation dictionary.

train_transformer_candidate



train_transformer_candidate

This matters because the correct metric is already implemented in evaluate_transformer_chaga_review.py: it normalizes the model prediction and compares it to preserved original CHAGA candidate strings, with top-3 allowed only under teacher_accept_top3.

evaluate_transformer_chaga_revi…

 The training loop should expose the same metric during validation, otherwise the saved “best epoch” may be suboptimal for original_relaxed_accuracy.

Implement this first

File: scripts/train_transformer_candidate.py

Extend collate_transformer_examples() to return original-candidate masks:

Python
Run
teacher_top1_mask = torch.zeros((len(examples), max_candidates), dtype=torch.bool)
teacher_top3_mask = torch.zeros((len(examples), max_candidates), dtype=torch.bool)
has_teacher_original = torch.zeros((len(examples),), dtype=torch.bool)
teacher_top1_is_play = torch.zeros((len(examples),), dtype=torch.bool)

For each example:

Python
Run
top1_norms = set(example.teacher_candidate_norms[:1])
top3_norms = set(example.teacher_candidate_norms[:3])

for slot, action in enumerate(candidates):
    norm = normalize_teacher_action(action_response(int(action)))
    if norm in top1_norms:
        teacher_top1_mask[row, slot] = True
    if norm in top3_norms:
        teacher_top3_mask[row, slot] = True

if top1_norms:
    has_teacher_original[row] = True
    teacher_top1_is_play[row] = next(iter(top1_norms)).startswith("PLAY ")

You already build teacher_accept_mask and has_teacher_accept_set for accepted-set loss, so this is a small extension of existing collator logic.

train_transformer_candidate

Then add validation counts in evaluate_model():

Python
Run
has_original = batch["has_teacher_original"]
pred_slot = pred_index[row]

original_top1 += batch["teacher_top1_mask"][row, pred_slot]
original_top3 += batch["teacher_top3_mask"][row, pred_slot]
original_relaxed += batch["teacher_accept_mask"][row, pred_slot]

Return:

original_top1_accuracy
original_top3_inclusion
original_relaxed_accuracy
original_play_relaxed_accuracy
original_samples
original_play_samples

Then launch future training with:

--monitor-metric val_original_relaxed_accuracy

Keep the standalone evaluator as the external audit, but make the training loop save checkpoints by the same metric.

Tests for this change

Add tests in tests/test_train_transformer_candidate.py:

Python
Run
def test_collate_original_candidate_masks_top1_top3_and_accepted_set():
    # teacher_candidate_norms = ("PLAY W1", "PLAY W2", "PLAY W3")
    # teacher_accept_top3 = True
    # Assert:
    # top1 mask contains only PLAY W1
    # top3 mask contains W1/W2/W3
    # accepted mask contains W1/W2/W3
Python
Run
def test_original_validation_metric_differs_from_soft_distribution_metric():
    # Construct an example where accepted top3 includes slot 2
    # and the soft distribution argmax is slot 0.
    # Force model prediction to slot 2.
    # Assert original_relaxed is true.
Python
Run
def test_training_monitor_can_use_val_original_relaxed_accuracy():
    # Feed two metric dicts into is_better_checkpoint_metric
    # and assert the one with higher val_original_relaxed_accuracy wins.

Post-change gate:

reviewed-only train-on-eval:
  in-loop val_original_relaxed_accuracy >= 0.99
  standalone original_relaxed_accuracy >= 0.99
  absolute difference between in-loop and standalone original_relaxed <= 1e-6

If this fails, do not expand training yet; the in-loop metric is not equivalent to the evaluator.

2. Current hardening looks sufficient for expansion

The key hardening points are in place:

--review-audit-jsonl now forces full candidate width when max_candidates < FeatureAgent.ACT_SIZE, so mixed reviewed training can no longer silently use the old 96-candidate truncation path.

train_transformer_candidate

encode_history_event() now uses full tile identity plus a no-tile sentinel, with token range compatible with the new DEFAULT_HISTORY_VOCAB_SIZE = 2048. The maximum token is around 1400, so 2048 is safe.

train_transformer_candidate



train_transformer_candidate

The audit script now supports --use-train-players, --player-regex, and --no-sample, and the sampling helper correctly keeps all entries when sample_size is zero.

audit_chaga_review_alignment



audit_chaga_review_alignment

The splitter creates session-disjoint raw/audit splits, verifies assigned sessions, and rejects repeated sessions across splits. That is the right leakage boundary.

split_chaga_review_corpus



split_chaga_review_corpus

The only small metric sanity check I would add before expansion is this: count teacher_accept_top3 rows whose CHAGA top-1 candidate is not PLAY. The current relaxation flag is derived from the reviewed row’s recorded action and play_ordinal, not from CHAGA top-1. That is probably fine for discard windows, but verify it. If non-PLAY top-1 rows are being relaxed, inspect them before using the expanded corpus. The acceptance flag is currently computed from the human/state action and play ordinal.

train_transformer_candidate

3. Next highest-leverage step: data fetching and audit expansion

Given only 2,080 aligned reviewed rows across 6 sessions, session-disjoint reviewed-only training will mainly test leakage control; it will not push precision toward 85%. The current split of 1,140 train, 340 val, and 600 test is useful as a smoke test but too small for performance claims.

The next implementation should be a reviewed-corpus expansion pipeline. The audit script can already review train_players, but the repo still needs a robust way to mark high-ELO seats in raw records.

Add a high-ELO marker script

Create:

scripts/mark_high_elo_train_players.py

Inputs:

--raw <raw_tziakcha_records.jsonl>
--elo-csv <current_elo.csv>
--min-elo 2300
--out <highelo2300_marked.raw.jsonl>
--summary-out <highelo2300_marked.summary.json>

Behavior:

Load ELO table keyed by player name.

For each raw record, read record_step(raw_record)["p"].

Mark seats whose player name has ELO >= threshold.

Write the raw record with:

JSON
"train_players": ["0", "2"]

Drop records with no marked seats unless --keep-unmarked is passed.

Summary should include:

raw_records_seen
raw_records_written
sessions_written
marked_player_seats
unique_marked_players
missing_elo_names
records_without_marked_players

Then run the existing audit on this marked file:

Bash
python scripts/audit_chaga_review_alignment.py \
  --raw data/processed/highelo2300_marked.raw.jsonl \
  --cache-dir data/raw/chaga_reviews_highelo2300 \
  --use-train-players \
  --no-sample \
  --out-jsonl data/processed/chaga_expanded/all.audit.jsonl \
  --summary-out data/processed/chaga_expanded/all.audit.summary.json

The audit already fetches CHAGA reviews by session and API seat, caches them, and filters rows by round index plus candidate availability.

audit_chaga_review_alignment



audit_chaga_review_alignment

Then split:

Bash
python scripts/split_chaga_review_corpus.py \
  --raw data/processed/highelo2300_marked.raw.jsonl \
  --audit data/processed/chaga_expanded/all.audit.jsonl \
  --out-dir data/processed/chaga_expanded/split_seed20260527 \
  --seed 20260527 \
  --train-ratio 0.8 \
  --val-ratio 0.1

Because train_transformer_candidate.py currently accepts a single --review-audit-jsonl for both train and eval, create:

Bash
cat split_seed20260527/train.audit.jsonl split_seed20260527/val.audit.jsonl \
  > split_seed20260527/train_val.audit.jsonl

Do not include test.audit.jsonl in the training audit file. Use test only after model selection.

4. Minimum data gates

Do not launch a performance L40 job until the expanded corpus reaches at least:

train reviewed states >= 25,000
val reviewed states   >= 5,000
test reviewed states  >= 5,000
train sessions        >= 50
val sessions          >= 10
test sessions         >= 10

For a serious 85% claim, aim higher:

train reviewed states >= 100,000
val reviewed states   >= 10,000
test reviewed states  >= 10,000

The model has already shown it can memorize 2,077 rows. More L40 time on the current 2,080-row corpus will mostly measure overfitting and split variance.

5. Pre-L40 validation gates

Before launching the next L40 job, require all of this:

Corpus gate
audit summary:
  sample_size_requested = 0
  sample_size_written = all_aligned_review_states
  review_fetch_errors = 0
  state_build_errors = 0 or explicitly inspected
  check_pass:top1_in_legal_mask / aligned_review_states >= 0.995
  check_pass:window_matches / aligned_review_states >= 0.995
  check_pass:hand_size_mod_ok / aligned_review_states >= 0.995

split summary:
  no session_id overlap across train/val/test
  train reviewed states >= 25,000
  val reviewed states >= 5,000
  test reviewed states >= 5,000

The splitter already verifies session disjointness and audit/raw count consistency; rely on that, but enforce minimum split sizes before training.

split_chaga_review_corpus

Load/target gate

Run a dry-load checker, either as a new script or by adding --dry-run to train_transformer_candidate.py.

Required checks:

train_load_summary.totals.teacher_original_targets == expected train reviewed states
train_load_summary.totals.teacher_targets == train_load_summary.totals.teacher_original_targets
train_load_summary.totals.teacher_target_unmapped == 0
val equivalent checks pass
reviewed_without_teacher_distribution = 0
candidate_truncation_count = 0
max_candidates = 235
Relaxation sanity gate

Add a small audit summary:

accept_top3_rows
accept_top3_top1_not_play
accept_top3_non_draw_window
accept_top3_play_ordinal_out_of_range

Require:

accept_top3_top1_not_play = 0, or manually inspect every row
accept_top3_non_draw_window = 0
accept_top3_play_ordinal_out_of_range = 0
Hu safety gate

Because rule_gated_hu_allowed() trusts the legal mask, the key gate is: no accepted Hu candidate may survive unless it is in the legal mask. The helper only checks whether Hu appears in the already-gated mask, so the upstream rule mask must remain the source of truth.

train_transformer_candidate

Add a dry-run assertion:

For every reviewed example:
  if "HU" in accepted set:
    some legal candidate normalizes to "HU"
  after hu_gated_candidate_mask:
    no HU exists when allow_hu is false

If you can cheaply recompute base fan through the rule engine, add:

accepted HU base_fan >= 8
6. First L40 job after expansion

Run reviewed-only, session-disjoint accepted-set training first. Do not mix high-ELO human labels yet.

Use a smaller model than 212M for the first expanded-data run. The previous 212M model overfit a tiny reviewed set; the next question is whether more reviewed CHAGA data generalizes, not whether a larger model memorizes faster.

Suggested first L40 run:

Bash
python scripts/train_transformer_candidate.py \
  --raw data/processed/chaga_expanded/split_seed20260527/train.raw.jsonl \
  --eval-raw data/processed/chaga_expanded/split_seed20260527/val.raw.jsonl \
  --review-audit-jsonl data/processed/chaga_expanded/split_seed20260527/train_val.audit.jsonl \
  --train-reviewed-only \
  --val-reviewed-only \
  --require-teacher-distribution \
  --max-candidates 235 \
  --hard-loss-weight 0.0 \
  --teacher-loss-weight 0.0 \
  --reviewed-hard-loss-weight 0.0 \
  --reviewed-teacher-loss-weight 0.0 \
  --reviewed-accept-set-loss-weight 1.0 \
  --unreviewed-hard-loss-weight 0.0 \
  --value-loss-weight 0.0 \
  --history-vocab-size 2048 \
  --monitor-metric val_original_relaxed_accuracy \
  --epochs 30 \
  --batch-size 128 \
  --lr 5e-5 \
  --weight-decay 0.03 \
  --dropout 0.2 \
  --d-model 512 \
  --nhead 8 \
  --num-layers 8 \
  --dim-feedforward 2048

Post-job gates:

standalone original evaluator:
  evaluated_candidate_width = 235
  candidate_truncation_count = 0
  reviewed_without_teacher_distribution = 0
  val original_relaxed_accuracy improves over 0.616
  val original_play_relaxed_accuracy improves over 0.589

With only 25k train reviewed states, a reasonable first gate is:

val original_relaxed_accuracy >= 0.65

With 100k train reviewed states, raise the gate:

val original_relaxed_accuracy >= 0.75

Do not evaluate the test split until the training recipe is selected.

7. What not to prioritize yet
Session-disjoint reviewed-only training on the current 2,080-row corpus

Do it only as a smoke test. With 1,140 train rows and 340 val rows, the result will be too noisy to guide architecture.

Discard-specific auxiliary features

Useful, but second order until you have more CHAGA labels. The current weak slice is PLAY relaxed accuracy, so discard features are plausible later. I would add them after the expanded reviewed-only run if PLAY relaxed stays below 0.70.

One cheap feature fix to keep on the backlog: build_candidate_rule_features() still passes prevalent_wind=0 into LawlorentzEffectiveScorer, so its fan/effective-tile profile ignores actual round wind.

train_transformer_candidate

 Fixing that is reasonable, but it is not likely to move 0.616 to 0.85 without more CHAGA-reviewed data.

Larger transformer

Do not scale again until the expanded reviewed-only run proves validation gains. The current failure pattern is overfitting, not under-capacity.

Bottom line

Add in-loop original-candidate validation metrics first, so L40 checkpoint selection uses the same metric as the standalone evaluator. Then expand the CHAGA-reviewed corpus using --use-train-players --no-sample, split by session, and run a reviewed-only accepted-set L40 job on at least 25k/5k/5k reviewed states. The current hardening is otherwise sufficient; the next real performance gain needs more reviewed CHAGA targets.
