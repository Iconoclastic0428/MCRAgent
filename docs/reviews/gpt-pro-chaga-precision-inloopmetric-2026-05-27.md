# GPT Pro CHAGA Precision In-Loop Metric Review - 2026-05-27

Source conversation: https://chatgpt.com/c/6a167dc8-942c-83e8-810c-cf16c97c2c92

Repo commit under review: https://github.com/Iconoclastic0428/MCRAgent/commit/ec4f8a1

Yes, assuming the implementation matches your summary, ec4f8a1 closes the metric/checkpoint-selection gap.

The decisive requirement was that train_transformer_candidate.py::evaluate_model() no longer selects checkpoints using teacher_target_dist or distribution-derived top-k. If it now computes validation metrics from teacher_top1_mask, teacher_top3_mask, and teacher_accept_mask, and future jobs use:

Bash
--monitor-metric val_original_relaxed_accuracy

then checkpoint selection is aligned with the standalone original-candidate evaluator.

The next single step before another L40 job should be a dry-run corpus gate, not another model change. Data expansion is the next performance step, but the dry-run gate should be implemented first so the expanded corpus cannot silently contain leakage, missing teacher rows, truncated candidates, or mismatched in-loop versus standalone metrics.

Next step: add a dry-run corpus gate

Create or extend a script:

scripts/check_chaga_training_corpus.py

Its job is to load the exact raw/audit files that a Kubernetes job will use, run the same example-building and collator path as training, and fail fast before any L40 time is used.

Inputs
Bash
python scripts/check_chaga_training_corpus.py \
  --train-raw data/processed/chaga_expanded/train.raw.jsonl \
  --val-raw data/processed/chaga_expanded/val.raw.jsonl \
  --test-raw data/processed/chaga_expanded/test.raw.jsonl \
  --train-audit data/processed/chaga_expanded/train.audit.jsonl \
  --val-audit data/processed/chaga_expanded/val.audit.jsonl \
  --test-audit data/processed/chaga_expanded/test.audit.jsonl \
  --max-candidates 235 \
  --summary-out data/processed/chaga_expanded/corpus_gate.summary.json

If the training script still expects a combined train-val audit file, the gate should also produce:

train_val.audit.jsonl

but it must explicitly assert that test.audit.jsonl is excluded.

File-level implementation
1. Reuse training loaders

File: scripts/check_chaga_training_corpus.py

Import from scripts/train_transformer_candidate.py:

Python
Run
load_examples
load_review_target_lookup
collate_transformer_examples
FeatureAgent
normalize_teacher_action
action_response

The checker should not implement its own parser. It should call the same load_examples() path that training uses. That is the point of the gate.

For each split:

Python
Run
teacher_lookup = load_review_target_lookup(split_audit_path)

examples, load_summary = load_examples(
    raw_paths=[split_raw_path],
    teacher_lookup=teacher_lookup,
    max_records=None,
    ...
)

Then filter reviewed examples using the same reviewed-only predicate used in training.

2. Validate split disjointness

The splitter already checks session-disjoint splits, but the training gate should independently re-check it.

Required checks:

Python
Run
train_sessions.isdisjoint(val_sessions)
train_sessions.isdisjoint(test_sessions)
val_sessions.isdisjoint(test_sessions)

Fail if any overlap exists.

Also check raw/audit consistency:

Every audit session appears in exactly one raw split.
Every reviewed raw session has audit rows, unless explicitly dropped.
No test session appears in train_val.audit.jsonl.
3. Validate target attachment

For each split, report and assert:

raw_records
sessions
examples
reviewed_examples
teacher_original_targets
teacher_targets
teacher_target_unmapped
reviewed_without_teacher_distribution
has_teacher_original
has_teacher_accept_set

Minimum hard gates:

reviewed_examples > 0
teacher_original_targets == reviewed_examples
teacher_targets == reviewed_examples
teacher_target_unmapped == 0
reviewed_without_teacher_distribution == 0
has_teacher_accept_set == reviewed_examples

If teacher_original_targets and teacher_targets differ, the original-candidate evaluator can see rows that the training loss cannot learn. That should block the job.

4. Validate candidate coverage

Run one collator pass over all reviewed examples, or over all examples if the corpus is not too large.

Use:

Python
Run
batch = collate_transformer_examples(reviewed_examples, max_candidates=235)

Assert:

evaluated_candidate_width == 235
candidate_truncation_count == 0
has_teacher_original all true
has_teacher_accept_set all true
each reviewed row has at least one accepted candidate slot

For each row:

Python
Run
assert batch["teacher_accept_mask"][i].any()
assert batch["teacher_top1_mask"][i].any()

Also add this sanity check:

Python
Run
accept_top3 = example.teacher_accept_top3
top1 = example.teacher_candidate_norms[0]

if accept_top3:
    assert top1.startswith("PLAY ")

If this fails, the first-six relaxation is being applied to a non-PLAY row.

5. Validate Hu safety

This gate should be explicit because Hu is a hard invariant.

For every reviewed example:

Python
Run
accepted_actions = candidate_actions[teacher_accept_mask]
accepted_norms = {normalize_teacher_action(action_response(a)) for a in accepted_actions}

Assert:

If "HU" is accepted, the legal mask contains Hu.
If allow_hu is false, no Hu candidate is accepted.

If you have a callable base-fan checker available in the same path, add:

accepted Hu base_fan >= 8

If not, at minimum rely on the legal mask and fail if any Hu accepted set is outside the legal candidate list.

6. Compare in-loop and standalone metrics on a known checkpoint

This is the final proof that ec4f8a1 closed the checkpoint-selection gap.

Add a gate mode:

Bash
python scripts/check_chaga_training_corpus.py \
  ... \
  --checkpoint runs/reviewed_overfit/checkpoint.pt \
  --compare-standalone-evaluator

It should run:

train_transformer_candidate.py::evaluate_model() on the validation split.

evaluate_transformer_chaga_review.py logic on the same checkpoint and split.

Assert:

abs(in_loop_original_relaxed - standalone_original_relaxed) <= 1e-6
abs(in_loop_original_top1 - standalone_original_top1) <= 1e-6
abs(in_loop_original_top3 - standalone_original_top3) <= 1e-6
abs(in_loop_original_play_relaxed - standalone_original_play_relaxed) <= 1e-6

This is the most important validation for the new commit.

Tests to add

File:

tests/test_check_chaga_training_corpus.py
Test 1: session overlap fails
Python
Run
def test_corpus_gate_rejects_session_overlap():
    # train and val contain same session_id
    # assert gate raises ValueError
Test 2: test audit is excluded from train-val audit
Python
Run
def test_corpus_gate_rejects_test_session_in_train_val_audit():
    # test session appears in train_val.audit.jsonl
    # assert gate raises ValueError
Test 3: original target without mapped teacher fails
Python
Run
def test_corpus_gate_rejects_original_target_without_teacher_distribution():
    # reviewed example has teacher_candidate_norms but no teacher_action_distribution
    # assert failure
Test 4: accepted-set mask must be nonempty
Python
Run
def test_corpus_gate_rejects_empty_accept_mask():
    # CHAGA candidate strings exist but none map to legal candidate actions
    # assert failure
Test 5: top-3 relaxation only for PLAY
Python
Run
def test_corpus_gate_rejects_top3_relaxation_for_non_play_top1():
    # teacher_accept_top3=True, teacher_candidate_norms[0]="PENG"
    # assert failure
Test 6: in-loop and standalone metrics agree
Python
Run
def test_in_loop_and_standalone_original_metrics_match_on_synthetic_batch():
    # Construct fixed logits/predictions and teacher masks.
    # Assert both metric paths produce identical top1/top3/relaxed.
Validation gates before the next L40 job

Require all of these.

Corpus size gate

For the current 2,080-row corpus, do not launch another performance job. It is only useful for smoke tests.

Minimum for the next meaningful L40 job:

train reviewed states >= 25,000
val reviewed states   >= 5,000
test reviewed states  >= 5,000
train sessions        >= 50
val sessions          >= 10
test sessions         >= 10

For a serious 85% claim:

train reviewed states >= 100,000
val reviewed states   >= 10,000
test reviewed states  >= 10,000
Split gate
session overlap across train/val/test = 0
test sessions in train_val.audit.jsonl = 0
dropped raw sessions are reported with reasons
Target gate
teacher_original_targets == reviewed_examples
teacher_targets == reviewed_examples
teacher_target_unmapped == 0
reviewed_without_teacher_distribution == 0
has_teacher_accept_set == reviewed_examples
Candidate gate
max_candidates = 235
candidate_truncation_count = 0
every reviewed row has nonempty teacher_top1_mask
every reviewed row has nonempty teacher_accept_mask
Metric gate

On a known checkpoint:

in-loop val_original_relaxed_accuracy == standalone original_relaxed_accuracy
in-loop val_original_top1_accuracy == standalone original_top1_accuracy
in-loop val_original_top3_inclusion == standalone original_top3_inclusion

Use a tolerance of 1e-6.

Hu gate
accepted Hu outside legal mask = 0
Hu accepted when allow_hu=false = 0
below-8 base-fan Hu = 0, if base-fan recomputation is available
After the dry-run gate passes

Then do data fetch expansion, not session-disjoint reviewed-only training on the current tiny corpus.

Use the new expansion path:

Bash
python scripts/audit_chaga_review_alignment.py \
  --raw data/processed/highelo2300_marked.raw.jsonl \
  --cache-dir data/raw/chaga_reviews_highelo2300 \
  --use-train-players \
  --no-sample \
  --out-jsonl data/processed/chaga_expanded/all.audit.jsonl \
  --summary-out data/processed/chaga_expanded/all.audit.summary.json

Then split:

Bash
python scripts/split_chaga_review_corpus.py \
  --raw data/processed/highelo2300_marked.raw.jsonl \
  --audit data/processed/chaga_expanded/all.audit.jsonl \
  --out-dir data/processed/chaga_expanded/split_seed20260527 \
  --seed 20260527 \
  --train-ratio 0.8 \
  --val-ratio 0.1

Then run the corpus gate. Only after that should you launch L40.

First L40 job after expansion

Use reviewed-only accepted-set training first. Do not mix high-ELO human labels until reviewed-only improves.

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
  --epochs 30

Initial success gate:

val original_relaxed_accuracy > 0.616
val original_play_relaxed_accuracy > 0.589
candidate_truncation_count = 0
reviewed_without_teacher_distribution = 0

With 25k train reviewed states, require at least:

val original_relaxed_accuracy >= 0.65

With 100k train reviewed states, require at least:

val original_relaxed_accuracy >= 0.75

Do not evaluate the fixed test split until you select the training recipe.

Bottom line

ec4f8a1 likely closes the metric/checkpoint-selection gap. The next single step is a dry-run corpus gate that proves expanded raw/audit files attach all original CHAGA targets, preserve 235-candidate coverage, keep train/val/test session-disjoint, and make in-loop validation metrics exactly match the standalone evaluator. After that, expand CHAGA-reviewed data; the current 2,080-row corpus is too small for another meaningful L40 performance run.
