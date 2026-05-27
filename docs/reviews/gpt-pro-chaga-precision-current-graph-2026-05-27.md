# GPT Pro CHAGA Precision Current Graph Review - 2026-05-27

Source conversation: https://chatgpt.com/c/6a167dc8-942c-83e8-810c-cf16c97c2c92

Repo commit under review: https://github.com/Iconoclastic0428/MCRAgent/commit/3562d05

I inspected the relevant files in 3562d05. The plan is basically right, but I see one higher-priority corpus-consistency issue before you spend time rebuilding the full archive.

Main finding

build_chaga_github_corpus.py currently writes unannotated raw records to raw_out, while train_players is only added to the converted/prepared output. Specifically, prepare_record() adds train_players and train_player_names to the prepared converted record, but the build loop writes the original raw_record to raw_out before writing the prepared record. 

build_chaga_github_corpus

 

build_chaga_github_corpus

That matters because audit_chaga_review_alignment.py --use-train-players only uses train_players if the audited raw record contains that field; otherwise it falls back to regex selection from the raw record. 

audit_chaga_review_alignment

 Since raw_out does not currently contain train_players, --use-train-players is likely a no-op for the corpus you plan to audit.

This can create selection drift:

The prepared training file contains only the selected high-ELO CHAGA seats.

The audit file may include every raw seat matching the default CHAGA regex.

The split summary may overstate “reviewed states” because some audit rows will not attach to prepared training examples.

check_chaga_training_corpus.py validates loaded reviewed examples, but it does not yet fail on unmatched audit targets.

There is also a minor pattern mismatch: the new builder default is exact ^CHAGA0[2-8]$, while the older CHAGA selector allows ^CHAGA0[2-8](?:$|\D) case-insensitively. 

build_chaga_github_corpus

 

evaluate_chaga_replay

 That is fine if all names are exact, but it strengthens the case for carrying explicit train_players into the audit raw file instead of relying on two separate regex paths.

Highest-priority next code change

Patch scripts/build_chaga_github_corpus.py so raw_out is an audit raw file annotated with the same selected seats as prepared_out.

Change the build loop order from:

Python
Run
raw_file.write(json.dumps(raw_record) + "\n")
converted = converter(raw_record)
prepared = prepare_record(...)
prepared_file.write(json.dumps(prepared) + "\n")

to roughly:

Python
Run
converted = converter(raw_record)
selected_players = selected_by_session.get(location.session_id, {})
prepared = prepare_record(raw_record, converted, selected_players)
if prepared is None:
    summary["prepare_drops"] += 1
    continue

audit_raw = dict(raw_record)
audit_raw["train_players"] = sorted(selected_players)
audit_raw["train_player_names"] = {
    player: selected_players[player]
    for player in sorted(selected_players)
}
audit_raw["source_record_id"] = str(raw_record.get("id") or raw_record.get("match_id") or "")

raw_file.write(json.dumps(audit_raw, ensure_ascii=False, separators=(",", ":")) + "\n")
prepared_file.write(json.dumps(prepared, ensure_ascii=False, separators=(",", ":")) + "\n")

I would also only write raw_out after conversion and preparation succeed, so raw_out and prepared_out are record-aligned. Keep separate counters for records_fetched, records_converted, records_prepared, and audit_raw_records_written.

Tests to add

In tests/test_build_chaga_github_corpus.py:

Python
Run
def test_build_corpus_raw_out_carries_train_players_matching_prepared_out():
    # Build a synthetic session with CHAGA02 selected.
    # Assert raw_out row has train_players and train_player_names.
    # Assert prepared_out row has identical train_players and train_player_names.
Python
Run
def test_build_corpus_writes_raw_only_after_prepare_success():
    # Converter or prepare_record fails.
    # Assert neither raw_out nor prepared_out receives that record.

In tests/test_audit_chaga_review_alignment.py:

Python
Run
def test_use_train_players_overrides_regex_selection():
    # Raw record contains CHAGA02 and CHAGA03.
    # train_players=["1"].
    # use_train_players=True should select only seat 1.

This makes the full archive rebuild much more reliable.

Second change before L40: make the corpus gate catch unattached audit rows

check_chaga_training_corpus.py already checks session disjointness, nonempty reviewed examples, missing original candidates, missing mapped teacher distributions, candidate truncation, empty accept masks, and accepted HU outside the legal mask. 

check_chaga_training_corpus

 It does not yet fail when the audit file has target rows that never attach to loaded training examples.

Add a count before the lookup is consumed:

Python
Run
def review_lookup_target_count(lookup: ReviewTargetLookup) -> int:
    return sum(len(queue) for queue in lookup.entries_by_key.values())

In load_reviewed_split():

Python
Run
lookup = load_review_target_lookup(audit_path)
audit_target_entries = review_lookup_target_count(lookup)

examples, load_summary = load_examples(...)
teacher_original_targets = int(load_summary["totals"].get("teacher_original_targets", 0))
teacher_targets = int(load_summary["totals"].get("teacher_targets", 0))

unattached_audit_targets = audit_target_entries - teacher_original_targets

Add to summary:

JSON
"audit_target_entries": ...,
"teacher_original_targets": ...,
"teacher_targets": ...,
"unattached_audit_targets": ...

Fail by default if:

unattached_audit_targets != 0
teacher_original_targets != teacher_targets

This is the gate that will catch raw/prepared selection drift.

Test
Python
Run
def test_corpus_gate_rejects_unattached_audit_targets():
    # Audit contains two valid review rows.
    # Prepared raw only builds one matching reviewed example.
    # Gate must raise ValueError with unattached_audit_targets = 1.
Is data expansion now the right next step?

Yes, after the two fixes above. The current graph is not strong evidence for a new architecture. It is based on only 560 reviewed states, and the per-turn buckets are small. For example, turn 4 has only 13 reviewed states, and turn 49 has only 6 reviewed states. 

chaga_mismatch_by_turn_current_…

 

chaga_mismatch_by_turn_current_…

The current eval set is also mostly PLAY decisions: 631 Play examples in the loaded eval file and 560 teacher targets overall. 

chaga_mismatch_by_turn_current_…

 That suggests the 61.6% relaxed score is dominated by discard ranking, and a family head is unlikely to be the immediate fix unless diagnostics show many PLAY versus non-PLAY confusions.

So the next precision step should be:

Patch raw audit metadata.

Patch unattached-audit gate.

Rebuild the CHAGA02-08 high-ELO corpus from local archive.

Audit with exact train_players.

Split by session.

Run the corpus gate.

Launch reviewed-only accepted-set L40 training only after the gate passes.

Exact data pipeline after the two patches
1. Build archive corpus
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

Build summary gates:

fetch_errors = []
convert_errors = []
prepare_drops = 0
audit_raw_records_written == records_prepared
raw_out and prepared_out have the same source record ids
every raw_out row has train_players
every prepared_out row has train_players

The local archive path is the right solution for GitHub raw rate limits because local_archive_record_fetcher() reads records from archive_root / records / period / record_id.json. 

build_chaga_github_corpus

2. Audit CHAGA rows
Bash
python scripts/audit_chaga_review_alignment.py \
  --raw data/raw/chaga0208_elo2300_audit_raw.jsonl \
  --cache-dir data/raw/chaga_reviews_chaga0208_elo2300 \
  --use-train-players \
  --no-sample \
  --out-jsonl data/processed/chaga0208_elo2300/chaga0208_elo2300_all.audit.jsonl \
  --summary-out data/processed/chaga0208_elo2300/chaga0208_elo2300_audit_summary.json

Audit gates:

review_fetch_errors = 0
state_build_errors = 0, or every failed record is inspected
alignment_missing_state / review_rows_with_candidates <= 0.005
top1_in_legal_mask pass rate >= 0.995
actual_in_legal_mask pass rate >= 0.995
sample_size = 0 or no-sample confirmed

The audit already reconstructs model states, fetches CHAGA review rows by session and API seat, filters by rr, and stores state_turn, state_action, legal mask, normalized CHAGA top-5, and alignment checks. 

audit_chaga_review_alignment

 

audit_chaga_review_alignment

3. Split session-disjoint

Use the prepared converted file as the raw/training input to the splitter:

Bash
python scripts/split_chaga_review_corpus.py \
  --raw data/processed/chaga0208_elo2300/chaga0208_elo2300_all.prepared.jsonl \
  --audit data/processed/chaga0208_elo2300/chaga0208_elo2300_all.audit.jsonl \
  --out-dir data/processed/chaga0208_elo2300/split_seed20260527 \
  --seed 20260527 \
  --train-ratio 0.8 \
  --val-ratio 0.1

The splitter groups raw and audit rows by session_id, assigns sessions to splits, and verifies session disjointness and raw/audit count consistency. 

split_chaga_review_corpus

 

split_chaga_review_corpus

4. Corpus gate
Bash
cat \
  data/processed/chaga0208_elo2300/split_seed20260527/train.audit.jsonl \
  data/processed/chaga0208_elo2300/split_seed20260527/val.audit.jsonl \
  > data/processed/chaga0208_elo2300/split_seed20260527/train_val.audit.jsonl

python scripts/check_chaga_training_corpus.py \
  --train-raw data/processed/chaga0208_elo2300/split_seed20260527/train.raw.jsonl \
  --val-raw data/processed/chaga0208_elo2300/split_seed20260527/val.raw.jsonl \
  --test-raw data/processed/chaga0208_elo2300/split_seed20260527/test.raw.jsonl \
  --train-audit data/processed/chaga0208_elo2300/split_seed20260527/train.audit.jsonl \
  --val-audit data/processed/chaga0208_elo2300/split_seed20260527/val.audit.jsonl \
  --test-audit data/processed/chaga0208_elo2300/split_seed20260527/test.audit.jsonl \
  --train-val-audit data/processed/chaga0208_elo2300/split_seed20260527/train_val.audit.jsonl \
  --summary-out data/processed/chaga0208_elo2300/split_seed20260527/corpus_gate_summary.json \
  --min-train-reviewed 25000 \
  --min-val-reviewed 5000 \
  --min-test-reviewed 5000 \
  --min-train-sessions 50 \
  --min-val-sessions 10 \
  --min-test-sessions 10

Block L40 if any of these fail:

session overlap across train, val, test = 0
test sessions in train_val.audit = 0
teacher_original_targets == reviewed_examples
teacher_targets == reviewed_examples
unattached_audit_targets = 0
reviewed_without_teacher_distribution = 0
candidate_truncation_count = 0
empty_top1_mask = 0
empty_accept_mask = 0
accepted_hu_outside_legal_mask = 0
accepted_hu_when_allow_hu_false = 0

The current checker already validates accepted HU against the legal mask and allow_hu; keep that, and add explicit reporting for accepted-HU rows. 

check_chaga_training_corpus

Diagnostics to add before the next L40 run

The current mismatch CSV is too thin. collect_original_prediction_rows() records turn, player, response, predicted action, normalized prediction, CHAGA top-1/top-3 strings, teacher_accept_top3, and match booleans. 

evaluate_transformer_chaga_revi…

 That is enough for a turn graph, but not enough to distinguish data scarcity, feature weakness, and alignment.

Add these fields.

File: scripts/train_transformer_candidate.py

Extend TransformerExample:

Python
Run
record_id: str = ""
session_id: str = ""
teacher_candidate_scores: tuple[float, ...] = ()

Populate in build_transformer_examples_from_record():

Python
Run
session_id = str(record.get("belongs") or record.get("session_id") or "")

teacher_candidate_scores = tuple(
    float(item[0])
    for item in teacher_candidates[:5]
    if isinstance(item, (list, tuple)) and len(item) >= 2
)

record_id is already computed in that function. 

train_transformer_candidate

File: scripts/evaluate_transformer_chaga_review.py

Extend collect_original_prediction_rows() with:

Python
Run
def action_family(norm: str) -> str:
    return norm.split()[0] if norm else ""

def predicted_rank(pred_norm: str, norms: tuple[str, ...]) -> int | None:
    for index, norm in enumerate(norms[:5], start=1):
        if normalize_teacher_action(norm) == pred_norm:
            return index
    return None

Add row fields:

record_id
session_id
teacher_top1_family
predicted_family
human_family
is_relaxed_region
predicted_rank_in_chaga_top5
top1_score
top2_score
top1_top2_margin
top1_top3_margin
candidate_count
legal_action_count
is_top3_but_not_relaxed

You can compute candidate_count from the collated candidate_mask and legal_action_count from hu_gated_candidate_mask(example.action_mask, allow_hu=example.allow_hu).sum().

File: scripts/visualize_transformer_chaga_review_turns.py

Add non-turn summaries:

by_teacher_top1_family
by_predicted_family
by_family_confusion
by_relaxed_region
by_predicted_rank_in_chaga_top5
by_margin_bucket
by_turn_bucket: 0-10, 11-30, 31-60, 61+
by_candidate_count_bucket

Suggested mismatch categories:

top3_but_not_relaxed
wrong_family
same_family_not_top5
same_family_top5_rank_2_5
low_margin_top1_top2
claim_or_hu_error
Verification tests
Python
Run
def test_prediction_row_reports_predicted_rank_in_chaga_top5():
    assert predicted_rank("PLAY W3", ("PLAY W1", "PLAY W2", "PLAY W3")) == 3
    assert predicted_rank("PLAY B9", ("PLAY W1", "PLAY W2", "PLAY W3")) is None
Python
Run
def test_prediction_row_marks_top3_but_not_relaxed():
    # accept_top3=False, prediction equals CHAGA rank 2.
    # top3_match=True, relaxed_match=False, is_top3_but_not_relaxed=True.
Python
Run
def test_family_confusion_summary_counts_wrong_family():
    # predicted=PENG, teacher_top1=PLAY W1.
    # wrong_family bucket increments.
Diagnostic gates

Before L40, the eval report should answer:

What fraction of relaxed mismatches are top3_but_not_relaxed?
What fraction are wrong_family?
What fraction are same_family_not_top5?
What fraction are low-margin CHAGA rank disagreements?
What fraction are claim/HU rows?

Interpretation:

If most errors are same_family_not_top5:
  data scarcity or discard-state features are the likely bottleneck.

If many errors are top3_but_not_relaxed:
  top-1 ordering outside the first-six relaxation zone is the bottleneck; consider small soft-teacher loss after reviewed-only baseline.

If many errors are wrong_family:
  then a family head becomes justified.

If many errors concentrate in low CHAGA margins:
  85% may require much more teacher data and/or a less brittle target.

If alignment failures rise in the expanded corpus:
  fix audit/parser alignment before training.
What I would not do yet

Do not implement a family head before the expanded diagnostic report. The current result is dominated by PLAY decisions, and the graph shows many PLAY-vs-PLAY discard disagreements rather than a proven family-selection failure. The eval summary shows most examples are Play decisions, while Chi, Pass, Peng, Hu, and BuGang are much smaller slices. 

chaga_mismatch_by_turn_current_…

Do not run another large L40 model on the current 2,080-row corpus. The 560-row held-out graph is useful as a smoke test, but several high-mismatch turns have single-digit sample sizes, so it cannot support architecture conclusions. 

chaga_mismatch_by_turn_current_…

 

chaga_mismatch_by_turn_current_…

Do not mix high-ELO human hard labels in the first expanded run. First run reviewed-only accepted-set training so the validation result isolates CHAGA distillation quality.

Next L40 launch criteria

After the corpus and diagnostic patches:

train reviewed states >= 25,000
val reviewed states >= 5,000
test reviewed states >= 5,000
unattached_audit_targets = 0
candidate_truncation_count = 0
reviewed_without_teacher_distribution = 0
empty_accept_mask = 0
accepted_hu_outside_legal_mask = 0
session overlap = 0
test sessions in train_val.audit = 0
diagnostic summary generated

Then launch reviewed-only accepted-set training:

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

Expected first gate after expansion:

val original_relaxed_accuracy > 0.616
val original_play_relaxed_accuracy > 0.589

For a 25k train reviewed corpus, require at least:

val original_relaxed_accuracy >= 0.65

For a 100k train reviewed corpus, require at least:

val original_relaxed_accuracy >= 0.75

Hold the test split until the training recipe is selected.

Bottom line: the next precision step is still data expansion, but fix the audit raw metadata and unattached-audit gate first. Otherwise the full archive build may produce a large-looking reviewed corpus whose audit rows do not exactly match the prepared training examples.
