The accepted-set patch is correct. The overfit result is the decisive evidence: the current model path can learn the CHAGA candidate target when train and eval states are identical. The remaining gap is now mostly **data scarcity plus weak discard-context features**, not another large correctness bug.

The next highest-leverage step is **CHAGA-reviewed data expansion with session-disjoint splits**, preceded by two small code hardening changes: force full candidate width for any reviewed-target training, and remove the history-token tile aliasing before running the next expensive job.

## Why data expansion is now the main blocker

The mixed accepted-set run trained on only **1,517 reviewed CHAGA examples** plus **3,278 unreviewed human examples**, while the model has **212M parameters**. The best checkpoint is epoch 5, and validation relaxed accuracy is only `0.616071`; after that, train loss keeps falling while validation stagnates or declines. That is the shape of a data-limited distillation run, not a capacity-limited run.

The held-out eval has only **560 reviewed states**, with `original_top1 = 0.505357`, `original_top3 = 0.675`, and `original_relaxed = 0.616071`. Candidate width is already 235, there is no truncation, and every reviewed row has a teacher distribution, so the previous evaluator/candidate-coverage issues are no longer the bottleneck.

Also, the problem is mostly discard policy. Eval has **513 PLAY teacher samples out of 560 reviewed samples**, and PLAY relaxed accuracy is only `0.588694`, below the overall relaxed score. A family head will not fix most of that; the model is usually choosing a legal action family but the wrong discard among many plausible discards.

The turn graph is useful, but the per-turn buckets are too small to drive architecture choices. For example, turn 65 has only 7 reviewed states, turn 68 has only 5, and several high-mismatch turns have similarly small counts.   Treat that graph as a symptom report, not a training target.

## Immediate implementation changes before data expansion

### 1. Force `max_candidates=235` for any reviewed-target training

Right now `DEFAULT_MAX_CANDIDATES` is still 96, while the accepted-set run correctly passes `--max-candidates 235`. The collator truncates candidates if `len(candidates) > max_candidates`, and it only guarantees reinserting the recorded human target, not CHAGA鈥檚 accepted action.

Add this to `validate_reviewed_training_args` in `scripts/train_transformer_candidate.py`:

```python
if args.review_audit_jsonl and args.max_candidates < FeatureAgent.ACT_SIZE:
    raise ValueError(
        f"CHAGA-reviewed training/evaluation requires --max-candidates {FeatureAgent.ACT_SIZE}"
    )
```

The current guard only enforces this for `--train-reviewed-only` or `--val-reviewed-only`; mixed reviewed training should also be protected.

Test:

```python
def test_chaga_review_training_requires_full_candidate_width():
    args = SimpleNamespace(
        review_audit_jsonl="audit.jsonl",
        train_reviewed_only=False,
        val_reviewed_only=False,
        max_candidates=96,
        reviewed_batch_fraction=0.7,
        reviewed_accept_set_loss_weight=1.0,
    )
    with pytest.raises(ValueError):
        validate_reviewed_training_args(args)
```

Success gate: any command using `--review-audit-jsonl` fails unless `--max-candidates 235`.

### 2. Fix history-token tile aliasing before the next expensive run

`encode_history_event()` currently maps tile identity with `TILE_IDS.get(tile, 0) % 12`, which aliases suits and honors. This throws away exactly the discard-history information that should help later-turn PLAY choices.

Change it to full tile identity:

```python
def encode_history_event(player: int, request: str, response: str) -> int:
    response_tokens = response.strip().split()
    request_tokens = request.strip().split()
    head = response_tokens[0].upper() if response_tokens else "PASS"
    if head == "PASS" and len(request_tokens) >= 3:
        head = request_tokens[2].upper()
    action_type = BOTZONE_ACTION_TYPES.get(head, 0)
    tile = _first_tile(response_tokens[1:]) or _first_tile(request_tokens[1:]) or ""
    tile_id = TILE_IDS.get(tile, 34)  # 34 = no/unknown tile
    return 1 + int(player) * (len(BOTZONE_ACTION_TYPES) * 35) + action_type * 35 + tile_id
```

Then set `--history-vocab-size 2048` by default or compute the minimum as:

```python
1 + 4 * len(BOTZONE_ACTION_TYPES) * 35
```

Tests:

```python
def test_history_event_tokens_do_not_alias_suits():
    assert encode_history_event(0, "2 W1", "Play W1") != encode_history_event(0, "2 T1", "Play T1")
    assert encode_history_event(0, "2 W1", "Play W1") != encode_history_event(0, "2 B1", "Play B1")

def test_history_event_tokens_do_not_alias_honor_and_suit():
    assert encode_history_event(0, "2 W1", "Play W1") != encode_history_event(0, "2 F1", "Play F1")
```

Verification gate: rerun the reviewed-only overfit gate after this change. It should still reach at least `0.99` relaxed. If it does not, the encoding change broke checkpoint config or embedding sizing.

I would also fix `prevalent_wind=0` in `build_candidate_rule_features` later, because it discards round-wind context, but this is secondary. The tile aliasing is the more obvious feature bug.

## Main next step: expand CHAGA-reviewed targets

Do not run another 200M model on 1.5k reviewed train states. Build a larger reviewed corpus first.

### Required repo changes

#### A. Extend `audit_chaga_review_alignment.py` so it can audit high-ELO selected seats, not only CHAGA02鈥?8

The script currently defaults to `DEFAULT_PLAYER_RE`, which matches `CHAGA0[2-8]`, and `main()` does not expose a player-regex argument.

Add:

```python
parser.add_argument("--player-regex", default=DEFAULT_PLAYER_RE.pattern)
parser.add_argument("--use-train-players", action="store_true")
parser.add_argument("--no-sample", action="store_true")
```

Modify `audit_records()` to accept:

```python
use_train_players: bool = False
```

Selection logic:

```python
if use_train_players and raw_record.get("train_players"):
    step = record_step(raw_record)
    selected = {}
    for player in raw_record["train_players"]:
        player = str(player)
        name = str((step.get("p") or [])[int(player)].get("n", ""))
        selected[player] = name
else:
    selected = selected_players_from_raw_record(raw_record, player_pattern)
```

This lets you audit high-ELO human seats after your preprocessing marks them as `train_players`, instead of only auditing CHAGA-named seats.

#### B. Prevent accidental audit downsampling

`audit_records()` randomly samples entries if `len(entries) > sample_size`, and `main()` defaults `--sample-size` to 1000.   For expansion, this is dangerous.

Change the sampling block to:

```python
if sample_size and sample_size > 0 and len(entries) > sample_size:
    entries = random.Random(seed).sample(entries, sample_size)
```

Then use `--no-sample` or `--sample-size 0` for full-corpus audits.

Test:

```python
def test_audit_no_sample_keeps_all_entries():
    entries = [{"id": i} for i in range(2000)]
    sampled = maybe_sample_entries(entries, sample_size=0, seed=1)
    assert len(sampled) == 2000
```

#### C. Add session-disjoint split builder

Create `scripts/split_chaga_review_corpus.py`.

Inputs:

```text
--raw expanded_high_elo_records.jsonl
--audit expanded_chaga_review_audit.jsonl
--out-dir data/processed/chaga_review_expanded
--seed 20260527
--train-ratio 0.8
--val-ratio 0.1
--test-ratio 0.1
```

Behavior:

1. Read all audit entries.
2. Group audit entries by `session_id`.
3. Read raw records and map each raw record to `session_id = record["belongs"]`.
4. Shuffle sessions deterministically.
5. Split sessions, not rows.
6. Write:

```text
train.raw.jsonl
val.raw.jsonl
test.raw.jsonl
train.audit.jsonl
val.audit.jsonl
test.audit.jsonl
split_summary.json
```

Hard assertions:

```python
assert train_sessions.isdisjoint(val_sessions)
assert train_sessions.isdisjoint(test_sessions)
assert val_sessions.isdisjoint(test_sessions)
assert every audit row's session_id appears in exactly one split
assert every raw record's belongs appears in exactly one split or is explicitly dropped
```

Tests:

```python
def test_split_chaga_review_corpus_is_session_disjoint():
    # two records from same session must not be split across train/val/test

def test_split_chaga_review_corpus_keeps_audit_and_raw_consistent():
    # audit rows and raw rows for one session go to same split

def test_split_chaga_review_corpus_is_deterministic():
    # same seed gives identical split files
```

### Data target sizes

Minimum useful target:

```text
train reviewed states: >= 25,000
val reviewed states:   >= 5,000
test reviewed states:  >= 5,000
```

Better target for an 85% claim:

```text
train reviewed states: >= 100,000
val reviewed states:   >= 10,000
test reviewed states:  >= 10,000
```

The current training run has only **1,517 reviewed train states** and **560 reviewed eval states**, so a 25k reviewed-train set is already a 16x increase.

### Data quality gates

For each split, enforce:

```text
candidate_truncation_count = 0
reviewed_without_teacher_distribution = 0
teacher_original_targets == teacher_targets
top1_in_legal_mask rate = 1.0, or every failure is written to a reject file
Hu below 8 fan = 0
no session_id overlap across train/val/test
```

The audit already records `record_id`, `session_id`, `state_turn`, `legal_action_mask`, `chaga_top5_candidates`, `play_ordinal`, and alignment checks, so this split builder can be implemented without changing the review schema.

The review-fetch path already exists: `fetch_review()` calls `https://tc-api.pesiu.org/review/?id={session_id}&seat={seat}` and caches by `session_id` and seat.  Use the cache aggressively and rate-limit fetches.

## Training plan after expanded data

Run the next experiments in this order.

### Experiment 1: reviewed-only CHAGA training

Purpose: measure pure CHAGA distillation without high-ELO human label interference.

Command pattern:

```bash
python scripts/train_transformer_candidate.py \
  --raw data/processed/chaga_review_expanded/train.raw.jsonl \
  --eval-raw data/processed/chaga_review_expanded/val.raw.jsonl \
  --review-audit-jsonl data/processed/chaga_review_expanded/train_val.audit.jsonl \
  --max-candidates 235 \
  --hard-loss-weight 0.0 \
  --teacher-loss-weight 0.0 \
  --reviewed-batch-fraction 0.9 \
  --reviewed-hard-loss-weight 0.0 \
  --reviewed-teacher-loss-weight 0.0 \
  --reviewed-accept-set-loss-weight 1.0 \
  --unreviewed-hard-loss-weight 0.0 \
  --value-loss-weight 0.0 \
  --monitor-metric val_teacher_relaxed_accuracy \
  --history-vocab-size 2048 \
  --epochs 30
```

For the first expanded run, use a smaller model than 212M, for example 30M鈥?0M. The current 212M model memorizes too quickly for the available data, and scaling is not the active bottleneck.

Success gate for 25k train states:

```text
val original_relaxed_accuracy > 0.65
val original_play_relaxed_accuracy > 0.65
train/val gap not exploding before epoch 5
```

Success gate for 100k train states:

```text
val original_relaxed_accuracy > 0.75
val original_play_relaxed_accuracy > 0.75
test is not evaluated until model family is selected
```

Do not use the fixed test split for iteration.

### Experiment 2: reviewed CHAGA plus small high-ELO human auxiliary loss

Only run this after Experiment 1.

Use:

```text
--unreviewed-hard-loss-weight 0.02 or 0.05
--reviewed-batch-fraction 0.8
```

The current accepted-set run used `unreviewed_hard_loss_weight = 0.1`. With such a small reviewed set, unreviewed human labels can easily regularize in the wrong direction because the metric is CHAGA agreement, not human imitation.

Success condition: mixed human data must beat reviewed-only on validation. If it does not, remove human auxiliary loss for the CHAGA-precision target.

### Experiment 3: add soft teacher ordering as a secondary objective

After the accepted-set model works on larger data, try:

```text
--reviewed-accept-set-loss-weight 1.0
--reviewed-teacher-loss-weight 0.05 or 0.1
```

This can help top-1 without destroying the relaxed top-3 target.

Success condition:

```text
original_relaxed does not drop
original_top1 improves
PLAY relaxed does not drop
```

## Add better mismatch diagnostics before interpreting another turn graph

`collect_original_prediction_rows()` currently records turn, player, response, predicted action, CHAGA top candidates, top-1/top-3/relaxed booleans, and teacher-distribution presence.  Add these fields:

```text
teacher_top1_family
predicted_family
human_family
is_relaxed_region
predicted_rank_in_chaga_top5
candidate_count
legal_action_count
record_id
session_id
state_turn
```

To get `record_id` and `session_id`, extend `TransformerExample` with metadata when building examples. You already compute `record_id` in `build_transformer_examples_from_record`; add `session_id = record.get("belongs")` or a normalized equivalent.

Add an aggregate report:

```text
by teacher_top1_family
by predicted_family
by teacher_top1_family 脳 predicted_family
by relaxed region
by predicted_rank_in_chaga_top5
by turn bucket: 0鈥?0, 11鈥?0, 31鈥?0, 61+
by legal_action_count bucket
```

Tests:

```python
def test_predicted_rank_in_chaga_top5():
    row = build_prediction_row(pred="PLAY W3", teacher_norms=("PLAY W1", "PLAY W2", "PLAY W3"))
    assert row["predicted_rank_in_chaga_top5"] == 3

def test_prediction_row_marks_missing_rank():
    row = build_prediction_row(pred="PLAY B9", teacher_norms=("PLAY W1", "PLAY W2", "PLAY W3"))
    assert row["predicted_rank_in_chaga_top5"] is None
```

Gate: before another model run, generate a mismatch summary that says what fraction of relaxed mismatches are:

```text
wrong discard not in top5
top2/top3 outside relaxed region
wrong action family
claim/pass/Hu error
late-turn-only error
```

If most errors are 鈥渨rong discard not in top5,鈥?more CHAGA-reviewed discard data is the answer. If most errors are family errors, then a family head becomes more attractive.

## Lower-priority architecture changes

### Family head

Do not implement this next. The eval set is overwhelmingly PLAY, and PLAY relaxed accuracy is the weak slice. A family head helps when the model confuses PLAY/PASS/CHI/PENG/HU, but the present metric suggests discard ranking is the main problem.

### Bigger transformer

Do not scale again until the reviewed corpus is much larger. The current 212M model already overfits a tiny reviewed set; the next improvement will come from more CHAGA-labeled states and less lossy history features, not another size increase.

## Practical priority order

1. Add the mixed-reviewed `max_candidates=235` validation.
2. Fix `encode_history_event()` to preserve full tile identity.
3. Extend `audit_chaga_review_alignment.py` with `--player-regex`, `--use-train-players`, and no-sampling mode.
4. Add `split_chaga_review_corpus.py` with session-disjoint train/val/test splits.
5. Build at least 25k train, 5k val, 5k test reviewed states.
6. Train reviewed-only accepted-set model on the expanded split.
7. Add high-ELO human auxiliary loss only if it improves validation over reviewed-only.
8. Add family head only if new diagnostics show family-level mistakes dominate.

My read is that there is no remaining obvious code bug likely to move `0.616` to `0.85` by itself. The next real jump requires a larger, session-disjoint CHAGA-reviewed corpus.
