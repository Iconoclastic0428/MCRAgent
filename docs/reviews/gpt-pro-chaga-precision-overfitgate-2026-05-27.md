The latest patch direction is correct. The evaluator now selects reviewed examples from preserved original CHAGA candidate strings, not from mapped soft distributions, and `score_original_chaga_match` compares the model’s normalized predicted action against original top-1/top-3 candidate strings.  The prediction loop also takes the model’s selected 235-action candidate, converts it back to an action string, normalizes it, and scores it against `teacher_candidate_norms`.  The new aggregate metrics are the right ones for the user’s target: original top-1, original top-3 inclusion, original relaxed, PLAY-only original metrics, and missing-distribution count. 

I would do the **reviewed-only train-on-eval overfit gate next**, but first make one small correctness/harness fix: **do not reuse the same destructive `ReviewTargetLookup` instance for train and eval loads.**

## The one correctness issue to fix before the overfit gate

`ReviewTargetLookup.__call__` consumes entries with `queue.popleft()`.  In `train()`, the same `teacher_lookup` object is created once and then passed into `load_examples()` for training and validation when `--eval-raw` is supplied. 

That is mostly harmless when train and eval raw files are disjoint. It is a problem for the reviewed-only train-on-eval gate, because the intended setup will likely pass the same raw/review data for train and eval. The train load can consume the review queues, leaving the eval load without attached teacher rows or with mismatched fallback behavior.

### Patch first

File: `scripts/train_transformer_candidate.py`

Add independent lookup construction:

```python
def make_review_lookup(path: str | None) -> ReviewTargetLookup | None:
    return load_review_target_lookup(Path(path)) if path else None
```

Then in `train()`:

```python
train_teacher_lookup = make_review_lookup(args.review_audit_jsonl)

train_examples, train_load_summary = load_examples(
    ...,
    teacher_lookup=train_teacher_lookup,
    ...
)

if args.eval_raw:
    val_teacher_lookup = make_review_lookup(args.review_audit_jsonl)
    val_examples, val_load_summary = load_examples(
        ...,
        teacher_lookup=val_teacher_lookup,
        ...
    )
else:
    train_examples, val_examples = split_examples(...)
```

Do not share a consumed lookup object across train and eval loads.

A slightly cleaner version is to add `ReviewTargetLookup.clone()` or `fork()`, but reloading from JSONL is simpler and safer right now.

## Add reviewed-only filtering in the same patch

The overfit gate needs a deterministic way to train only on reviewed CHAGA rows. Add filtering after example loading.

File: `scripts/train_transformer_candidate.py`

```python
def is_reviewed_example(example: TransformerExample) -> bool:
    return bool(getattr(example, "teacher_candidate_norms", ()))

def has_mapped_teacher_distribution(example: TransformerExample) -> bool:
    return example.teacher_action_distribution is not None

def filter_reviewed_examples(
    examples: list[TransformerExample],
    *,
    require_distribution: bool,
) -> tuple[list[TransformerExample], dict]:
    reviewed = [example for example in examples if is_reviewed_example(example)]
    without_distribution = [
        example for example in reviewed
        if not has_mapped_teacher_distribution(example)
    ]
    if require_distribution:
        reviewed = [
            example for example in reviewed
            if has_mapped_teacher_distribution(example)
        ]
    return reviewed, {
        "reviewed_examples_before_filter": len(reviewed) + (len(without_distribution) if require_distribution else 0),
        "reviewed_without_teacher_distribution": len(without_distribution),
        "reviewed_examples_after_filter": len(reviewed),
    }
```

Add CLI flags:

```python
parser.add_argument("--train-reviewed-only", action="store_true")
parser.add_argument("--val-reviewed-only", action="store_true")
parser.add_argument("--require-teacher-distribution", action="store_true")
```

Apply them separately:

```python
if args.train_reviewed_only:
    train_examples, train_review_filter_summary = filter_reviewed_examples(
        train_examples,
        require_distribution=args.require_teacher_distribution,
    )

if args.val_reviewed_only:
    val_examples, val_review_filter_summary = filter_reviewed_examples(
        val_examples,
        require_distribution=args.require_teacher_distribution,
    )
```

Store both summaries in `metrics`.

For the overfit gate, use `--require-teacher-distribution`. The evaluator can score rows using original candidate strings even when the soft distribution failed, but the current training loss needs `teacher_action_distribution` because `policy_loss_with_optional_teacher` trains on `teacher_target_dist`.  Your latest run reports `reviewed_without_teacher_distribution: 0`, so requiring it should not drop rows.

## Add one safety check around candidate width

Training still defaults to `DEFAULT_MAX_CANDIDATES = 96`, while evaluation now defaults to `FeatureAgent.ACT_SIZE`, i.e. the full action width.   For reviewed-only training, force `--max-candidates 235`; otherwise CHAGA top candidates can be dropped from the candidate list even though evaluation has full width.

Add a guard:

```python
if (args.train_reviewed_only or args.val_reviewed_only) and args.max_candidates < FeatureAgent.ACT_SIZE:
    raise ValueError("reviewed-only CHAGA training requires --max-candidates 235")
```

This should be a hard error. The collator truncates candidate lists and only guarantees reinsertion of the recorded human target, not CHAGA’s target. 

## Tests to add now

### 1. Independent lookup for train and eval

Add a unit test proving two independent lookup instances can return the same target, while a single instance is destructive.

File: `tests/test_train_transformer_candidate.py`

```python
def test_review_target_lookup_instances_are_independent(tmp_path):
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({
        "record_id": "r1",
        "seat": 0,
        "state_turn": 7,
        "state_context": {
            "turn": 7,
            "request": "2 W1",
            "state_actual_response": "Play W1",
        },
        "chaga_top5_candidates": [[10.0, "Play W1"], [9.0, "Play W2"]],
        "checks": {
            "offered_tile_matches": True,
            "drawn_tile_matches": True,
            "current_actor_matches": True,
            "window_matches": True,
            "hand_size_mod_ok": True,
            "top1_in_legal_mask": True,
        },
    }) + "\n", encoding="utf-8")

    lookup_train = load_review_target_lookup(audit)
    lookup_val = load_review_target_lookup(audit)

    kwargs = {
        "record_id": "r1",
        "player": 0,
        "turn": 7,
        "request": "2 W1",
        "response": "Play W1",
    }

    assert lookup_train(**kwargs) is not None
    assert lookup_val(**kwargs) is not None
```

### 2. Reviewed-only filter keeps candidate-string rows

```python
def test_filter_reviewed_examples_uses_candidate_norms():
    reviewed = SimpleNamespace(
        teacher_candidate_norms=("PLAY W1",),
        teacher_action_distribution=np.array([1.0]),
    )
    unreviewed = SimpleNamespace(
        teacher_candidate_norms=(),
        teacher_action_distribution=None,
    )

    filtered, summary = filter_reviewed_examples(
        [reviewed, unreviewed],
        require_distribution=True,
    )

    assert filtered == [reviewed]
    assert summary["reviewed_examples_after_filter"] == 1
```

### 3. `require_distribution` drops unmapped reviewed rows

```python
def test_filter_reviewed_examples_can_require_mapped_distribution():
    unmapped = SimpleNamespace(
        teacher_candidate_norms=("PLAY W1",),
        teacher_action_distribution=None,
    )

    filtered, summary = filter_reviewed_examples(
        [unmapped],
        require_distribution=True,
    )

    assert filtered == []
    assert summary["reviewed_without_teacher_distribution"] == 1
```

### 4. Reviewed-only training requires full candidate width

Test the argument validation helper, or factor it out:

```python
def validate_reviewed_training_args(args):
    if (args.train_reviewed_only or args.val_reviewed_only) and args.max_candidates < FeatureAgent.ACT_SIZE:
        raise ValueError(...)
```

Test:

```python
def test_reviewed_only_requires_full_candidate_width():
    args = SimpleNamespace(
        train_reviewed_only=True,
        val_reviewed_only=False,
        max_candidates=96,
    )
    with pytest.raises(ValueError):
        validate_reviewed_training_args(args)
```

### 5. Teacher-only loss ignores the human target

This test is important because the reviewed-only gate is meant to prove the model can learn CHAGA, not human action labels.

```python
def test_teacher_only_policy_loss_ignores_hard_target():
    logits = torch.tensor([[0.0, 5.0, -5.0]], requires_grad=True)
    batch = {
        "target_index": torch.tensor([2]),  # human target is slot 2
        "has_teacher_target": torch.tensor([True]),
        "teacher_target_dist": torch.tensor([[0.0, 1.0, 0.0]]),  # CHAGA target is slot 1
    }

    loss, parts = policy_loss_with_optional_teacher(
        logits,
        batch,
        hard_loss_weight=0.0,
        teacher_loss_weight=1.0,
    )

    assert loss.item() < 0.01
```

Then invert the logits and assert loss is high:

```python
bad_logits = torch.tensor([[0.0, -5.0, 5.0]], requires_grad=True)
...
assert bad_loss.item() > 5.0
```

## Reviewed-only train-on-eval command

After the small lookup/filter patch, run the overfit gate. Use the same raw and eval raw to intentionally train and evaluate on the same reviewed examples.

```bash
python scripts/train_transformer_candidate.py \
  --raw data/processed/chaga_review_targets.jsonl \
  --eval-raw data/processed/chaga_review_targets.jsonl \
  --review-audit-jsonl data/processed/chaga_review_audit.jsonl \
  --train-reviewed-only \
  --val-reviewed-only \
  --require-teacher-distribution \
  --max-candidates 235 \
  --hard-loss-weight 0.0 \
  --teacher-loss-weight 1.0 \
  --value-loss-weight 0.0 \
  --teacher-temperature 0.5 \
  --monitor-metric val_teacher_relaxed_accuracy \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-3 \
  --out runs/reviewed_overfit/checkpoint.pt \
  --metrics-out runs/reviewed_overfit/train_metrics.json
```

Then evaluate using the corrected original-candidate evaluator:

```bash
python scripts/evaluate_transformer_chaga_review.py \
  --checkpoint runs/reviewed_overfit/checkpoint.pt \
  --raw data/processed/chaga_review_targets.jsonl \
  --review-audit-jsonl data/processed/chaga_review_audit.jsonl \
  --max-candidates 235 \
  --metrics-out runs/reviewed_overfit/original_eval.json
```

Use your actual raw/audit paths, but keep `--raw` and `--eval-raw` identical for the first gate.

## Success gate

The reviewed-only train-on-eval run should satisfy all of this:

```text
train_reviewed_examples_after_filter > 0
val_reviewed_examples_after_filter == train_reviewed_examples_after_filter
reviewed_without_teacher_distribution == 0
candidate_truncation_count == 0
evaluated_candidate_width == 235
original_relaxed_accuracy >= 0.95
original_top1_accuracy >= 0.85
original_play_relaxed_accuracy >= 0.95
Hu below 8 fan: 0
```

I would not require `original_top1_accuracy >= 0.95`, because the official relaxed metric explicitly allows top-3 for the first six PLAY decisions. But if top-1 remains very low while relaxed is high, inspect whether the teacher distribution is too soft or the candidate ordering is being flattened.

## Failure interpretation

### Case A: `reviewed_examples` drops below expected count

Fix lookup and audit attachment. The likely causes are consumed lookup queues, key mismatch, turn mismatch, or fallback old-artifact behavior. Do not train further.

### Case B: `reviewed_without_teacher_distribution > 0`

Fix action normalization or legal mask mapping. The evaluator can see original candidates, but the trainer cannot learn rows that fail distribution mapping.

Likely rows to inspect:

```text
PASS vs ABANDON
GANG vs BUGANG
GANG with tile identity
CHI middle-tile normalization
Hu rows where fan gate removes Hu
```

### Case C: distribution metrics are high but original metrics are low

The model is learning the mapped soft target, but original string normalization is still inconsistent. Inspect the CSV from the turn visualizer, especially rows where `teacher_top1_accuracy` is high but `original_top1_accuracy` is false.

### Case D: train-on-eval original relaxed stays below 95%

Do not oversample yet. This means the training/evaluation plumbing is still wrong, or the model cannot memorize the tiny target set because the loss is not pointed at the same target as the metric.

Immediate checks:

```text
Is hard_loss_weight really 0?
Is value_loss_weight really 0?
Are all reviewed examples mapped to teacher_target_dist?
Is max_candidates 235 in both train and eval?
Are candidate action IDs stable between train and eval?
Does the saved checkpoint correspond to the best or final epoch?
```

### Case E: train-on-eval passes, held-out stays near 47%

Then correctness is probably good enough, and the bottleneck becomes data/training. The next implementation should be reviewed oversampling plus row-aware teacher loss, not a family head yet.

## What to do after the overfit gate passes

The next code change after a passing overfit gate should be **reviewed oversampling and row-aware loss**, not the family head.

Current `policy_loss_with_optional_teacher` applies global hard and teacher weights to the whole batch.  For the real mixed run, reviewed rows should mostly follow CHAGA, while unreviewed high-ELO rows should have a smaller hard-label loss.

Add:

```python
--reviewed-batch-fraction 0.7
--reviewed-hard-loss-weight 0.0
--reviewed-teacher-loss-weight 1.0
--unreviewed-hard-loss-weight 0.2
```

But only after the reviewed-only train-on-eval gate proves the teacher target is learnable.

## Bottom line

Do the reviewed-only overfit gate next, but first patch the destructive lookup reuse and add reviewed-only filtering. The current evaluator patch is the right direction; the remaining blocker is making a valid train-on-eval CHAGA gate possible without consumed review targets or mixed human-label dilution.
