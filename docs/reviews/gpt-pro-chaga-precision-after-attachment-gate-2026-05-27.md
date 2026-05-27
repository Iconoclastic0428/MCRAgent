# GPT Pro CHAGA Precision After Attachment Gate Review - 2026-05-27

Source conversation: https://chatgpt.com/c/6a167dc8-942c-83e8-810c-cf16c97c2c92

Repo commit under review: https://github.com/Iconoclastic0428/MCRAgent/commit/9986488

Yes. Given 9986488, choose B: rebuild the full CHAGA02-08 >2300 corpus from the local tziakcha_records archive. Do not spend the next step on richer mismatch diagnostics yet. The current 560-state graph is too small to justify new architecture or feature work, and your latest corpus gates now catch the main raw/prepared/audit attachment failure mode.

The two fixes I previously considered blocking are now present: build_chaga_github_corpus.py annotates raw_out with train_players, train_player_names, and source_record_id, and it writes raw/prepared rows only after conversion and preparation succeed. 

build_chaga_github_corpus

 

build_chaga_github_corpus

 The corpus checker now counts audit_target_entries, computes unattached_audit_targets, and fails if unattached targets or teacher target mismatches exist. 

check_chaga_training_corpus

 

check_chaga_training_corpus

Next single step

Rebuild the full archive-backed CHAGA02-08 corpus and run the existing audit, split, and gate. Treat the stale tiny-split failure as confirmation that the new checker works, not as a dataset worth repairing.

1. Build corpus from local archive

Use the local archive path, not GitHub raw. The script has a local archive fetcher that reads from:

<archive_root>/records/<period>/<record_id>.json

build_chaga_github_corpus

Run:

Bash
python scripts/build_chaga_github_corpus.py \
  --archive-root /path/to/tziakcha_records \
  --elo-csv data/raw/tziakcha_current_elo.csv \
  --raw-out data/raw/chaga0208_elo2300_audit_raw.jsonl \
  --prepared-out data/processed/chaga0208_elo2300/chaga0208_elo2300_all.prepared.jsonl \
  --summary-out data/processed/chaga0208_elo2300/chaga0208_elo2300_build_summary.json \
  --min-elo 2300 \
  --player-pattern '^CHAGA0[2-8]$' \
  --workers 16

Build summary hard gates:

fetch_errors == []
convert_errors == []
prepare_drops == 0
records_fetched == records_converted == records_prepared
audit_raw_records_written == records_prepared
sessions_selected >= current GitHub pass baseline, unless archive is incomplete
record_locations >= current GitHub pass baseline, unless archive is incomplete
every raw_out row has train_players
every prepared_out row has train_players
raw_out and prepared_out line counts match

The >2300 condition is implemented as excluding <= min_elo, so --min-elo 2300 means strictly greater than 2300. That matches your stated filter. 

build_chaga_github_corpus

2. Audit with explicit train players
Bash
python scripts/audit_chaga_review_alignment.py \
  --raw data/raw/chaga0208_elo2300_audit_raw.jsonl \
  --cache-dir data/raw/chaga_reviews_chaga0208_elo2300 \
  --use-train-players \
  --no-sample \
  --out-jsonl data/processed/chaga0208_elo2300/chaga0208_elo2300_all.audit.jsonl \
  --summary-out data/processed/chaga0208_elo2300/chaga0208_elo2300_audit_summary.json

Audit gates:

no sampling was applied
review_fetch_errors == 0, or every failure is cached/retried before training
top1_in_legal_mask pass rate >= 0.995
actual_in_legal_mask pass rate >= 0.995
alignment_missing_state / candidate review rows <= 0.005
state_build_errors == 0, or every failed record is excluded with a reason
3. Split by session

Use the prepared converted file as the training raw file:

Bash
python scripts/split_chaga_review_corpus.py \
  --raw data/processed/chaga0208_elo2300/chaga0208_elo2300_all.prepared.jsonl \
  --audit data/processed/chaga0208_elo2300/chaga0208_elo2300_all.audit.jsonl \
  --out-dir data/processed/chaga0208_elo2300/split_seed20260527 \
  --seed 20260527 \
  --train-ratio 0.8 \
  --val-ratio 0.1

Then build train-val audit:

Bash
cat \
  data/processed/chaga0208_elo2300/split_seed20260527/train.audit.jsonl \
  data/processed/chaga0208_elo2300/split_seed20260527/val.audit.jsonl \
  > data/processed/chaga0208_elo2300/split_seed20260527/train_val.audit.jsonl
4. Run the corpus gate
Bash
python scripts/check_chaga_training_corpus.py \
  --train-raw data/processed/chaga0208_elo2300/split_seed20260527/train.raw.jsonl \
  --val-raw data/processed/chaga0208_elo2300/split_seed20260527/val.raw.jsonl \
  --test-raw data/processed/chaga0208_elo2300/split_seed20260527/test.raw.jsonl \
  --train-audit data/processed/chaga0208_elo2300/split_seed20260527/train.audit.jsonl \
  --val-audit data/processed/chaga0208_elo2300/split_seed20260527/val.audit.jsonl \
  --test-audit data/processed/chaga0208_elo2300/split_seed20260527/test.audit.jsonl \
  --train-val-audit data/processed/chaga0208_elo2300/split_seed20260527/train_val.audit.jsonl \
  --summary-out data/processed/chaga0208_elo2300/split_seed20260527/corpus_gate_summary.json \
  --max-candidates 235 \
  --min-train-reviewed 25000 \
  --min-val-reviewed 5000 \
  --min-test-reviewed 5000 \
  --min-train-sessions 50 \
  --min-val-sessions 10 \
  --min-test-sessions 10

The gate already enforces candidate width, nonempty reviewed examples, top-3 relaxation sanity, candidate truncation, mapped teacher distributions, nonempty top-1/accepted masks, and accepted-HU legality. 

check_chaga_training_corpus

 The default minimum corpus sizes are now 25k/5k/5k reviewed states and 50/10/10 sessions. 

check_chaga_training_corpus

No L40 until these pass

Block L40 if any of these are nonzero:

unattached_audit_targets
teacher_original_targets != teacher_targets
without_mapped_teacher_distribution
candidate_truncation_count
top3_relaxation_non_play
accepted_hu_outside_legal_mask
accepted_hu_when_allow_hu_false
empty_top1_mask
empty_accept_mask
missing_has_teacher_original
missing_has_teacher_accept_set

Also block if:

train reviewed states < 25,000
val reviewed states < 5,000
test reviewed states < 5,000
train sessions < 50
val sessions < 10
test sessions < 10
Do not add richer mismatch diagnostics before the rebuild

Richer diagnostics are useful, but the correct timing is after you have a larger validation set. On the current 560-state eval, the highest-mismatch turn buckets are too noisy. You will get better signal by first expanding the reviewed corpus, then running the current evaluator/graph on the larger validation split.

After the corpus gate passes, add diagnostics only if the first expanded reviewed-only run still stalls. The first diagnostic fields to add then are:

teacher_top1_family
predicted_family
predicted_rank_in_chaga_top5
is_top3_but_not_relaxed
top1_top2_margin
candidate_count
legal_action_count
record_id
session_id

Target files later:

scripts/train_transformer_candidate.py
scripts/evaluate_transformer_chaga_review.py
scripts/visualize_transformer_chaga_review_turns.py

But that is not the next highest-value step now.

Next L40 launch, after corpus gate passes

First run should be reviewed-only accepted-set training, no high-ELO human hard labels:

Bash
python scripts/train_transformer_candidate.py \
  --raw data/processed/chaga0208_elo2300/split_seed20260527/train.raw.jsonl \
  --eval-raw data/processed/chaga0208_elo2300/split_seed20260527/val.raw.jsonl \
  --review-audit-jsonl data/processed/chaga0208_elo2300/split_seed20260527/train_val.audit.jsonl \
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

val original_relaxed_accuracy > 0.616071
val original_play_relaxed_accuracy > 0.588694
candidate_truncation_count == 0
reviewed_without_teacher_distribution == 0

For a 25k train reviewed corpus, require:

val original_relaxed_accuracy >= 0.65

For a 100k train reviewed corpus, require:

val original_relaxed_accuracy >= 0.75

Hold the test split until the training recipe is selected.

Bottom line: choose B now. The latest commit appears to have closed the raw/prepared/audit attachment bug and the unattached-audit gate. The next precision-relevant step is to rebuild the full archive-backed corpus and prove it passes the corpus gate.
