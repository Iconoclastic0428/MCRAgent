The patch direction is correct. It fixes the main semantic bug I flagged: `allow_hu` is now derived from the legal action mask via `rule_gated_hu_allowed`, and example construction now uses that gate instead of the recorded response. The new audit parser also correctly treats CHAGA `Play` tile id `0` as valid rather than dropping it.   

One caveat: this is only fully correct if the upstream `obs["action_mask"]` is already rule and fan gated. The new helper checks whether a Hu action exists in the mask; it does not independently recompute `base_fan >= 8`. That is acceptable if the mask is produced by the trusted rule engine, but you should keep a regression that proves every Hu action surviving the mask passes the official base-fan gate.

## Next single highest-impact change

Implement the **original-candidate evaluator** first.

Do not implement the family head next. Do not launch another training variant yet. The current evaluator still measures a derived target, not the requested target. It converts CHAGA candidates into `teacher_target_dist`, then evaluates by comparing the predicted candidate slot to the argmax/top-k slots of that mapped distribution.  The standalone CHAGA evaluator simply loads examples and delegates to `evaluate_model`, so it inherits this distribution-based metric. 

That is still not the target you described. The required metric is:

```text
model predicted normalized action vs original CHAGA review candidates

top-3 counts only for that player's first six PLAY/discard decisions
all other states require top-1
```

Until that evaluator exists, the reviewed-only overfit gate and all training experiments are ambiguous.

## Why this beats the other options right now

**Reviewed-only overfit gate** is the next experiment, but it depends on a faithful evaluator. Otherwise you can overfit the mapped distribution and still fail the actual model-vs-CHAGA candidate metric.

**Reviewed oversampling and `max_candidates=235`** are important, but they are training changes. You need the corrected evaluator first to know whether they work. Also, `max_candidates=235` should be included in the evaluator change because the current default is 96 and the collator truncates legal candidates, reinserting only the human action, not the CHAGA top action.  

**Richer lookup/audit regeneration** is valuable, but the current patch already added `state_turn`, `state_action`, and `state_context.turn`, and `ReviewTargetLookup` now keys by record, seat, request, response, and turn before falling back.   Regenerate the audit after the evaluator patch, but do not make richer lookup the next code change unless the corrected evaluator exposes key collisions.

**Family head** should wait. Architecture work before metric correction risks optimizing the wrong objective.

## Concrete implementation plan

### 1. Extend `ReviewTarget` and `TransformerExample`

File: `scripts/train_transformer_candidate.py`

Current dataclasses:

```python
@dataclass
class TransformerExample:
    ...
    teacher_action_distribution: np.ndarray | None = None
    teacher_accept_top3: bool = False

@dataclass
class ReviewTarget:
    candidates: list
    accept_top3: bool = False
```

Add original normalized candidate metadata:

```python
@dataclass
class TransformerExample:
    ...
    teacher_action_distribution: np.ndarray | None = None
    teacher_accept_top3: bool = False
    teacher_top1_norm: str | None = None
    teacher_top3_norms: tuple[str, ...] = ()
    teacher_top5_norms: tuple[str, ...] = ()
    teacher_lookup_key: tuple[str, ...] | None = None

@dataclass
class ReviewTarget:
    candidates: list
    accept_top3: bool = False
    lookup_key: tuple[str, ...] | None = None
```

Add helper:

```python
def normalized_candidate_list(candidates: list, *, limit: int) -> tuple[str, ...]:
    out: list[str] = []
    for item in candidates[:limit]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            norm = normalize_teacher_action(str(item[1]))
            if norm:
                out.append(norm)
    return tuple(out)
```

In `build_transformer_examples_from_record`, when `teacher_candidates` exists, compute:

```python
teacher_top5_norms = normalized_candidate_list(teacher_candidates, limit=5)
teacher_top3_norms = teacher_top5_norms[:3]
teacher_top1_norm = teacher_top5_norms[0] if teacher_top5_norms else None
```

Store these on the `TransformerExample` regardless of whether `teacher_distribution` mapped successfully. This matters because unmapped teacher rows should become visible as evaluator diagnostics, not disappear.

### 2. Change reviewed-example selection

File: `scripts/evaluate_transformer_chaga_review.py`

Current selection:

```python
def select_reviewed_examples(examples):
    return [example for example in examples if example.teacher_action_distribution is not None]
```

Change to:

```python
def select_reviewed_examples(examples):
    return [example for example in examples if example.teacher_top1_norm]
```

Then report both:

```text
reviewed_examples
reviewed_with_teacher_distribution
reviewed_without_teacher_distribution
```

If `reviewed_without_teacher_distribution` is nonzero, that is not automatically fatal, but it indicates action-normalization or mask-mapping loss.

### 3. Add a string-based scoring helper

File: `scripts/evaluate_transformer_chaga_review.py`

Add pure function:

```python
def score_original_chaga_match(
    pred_norm: str,
    *,
    teacher_top1_norm: str | None,
    teacher_top3_norms: tuple[str, ...],
    accept_top3: bool,
) -> dict[str, bool]:
    top1 = bool(teacher_top1_norm) and pred_norm == teacher_top1_norm
    top3 = pred_norm in set(teacher_top3_norms[:3])
    relaxed = top1 or (accept_top3 and top3)
    return {"top1": top1, "top3": top3, "relaxed": relaxed}
```

Do not use `teacher_target_dist` to compute this metric.

### 4. Implement a custom evaluator loop

Still in `scripts/evaluate_transformer_chaga_review.py`.

Do not call `evaluate_model` for the CHAGA report. Keep `evaluate_model` for generic validation, but add a dedicated function:

```python
def evaluate_model_against_original_chaga(
    model: TransformerCandidateModel,
    examples: list[TransformerExample],
    device: torch.device,
    *,
    batch_size: int,
    max_candidates: int,
) -> dict:
    ...
```

Use a custom collate wrapper:

```python
def collate_with_meta(items):
    batch = collate_transformer_examples(items, max_candidates=max_candidates)
    meta = [
        {
            "teacher_top1_norm": ex.teacher_top1_norm,
            "teacher_top3_norms": ex.teacher_top3_norms,
            "teacher_top5_norms": ex.teacher_top5_norms,
            "accept_top3": ex.teacher_accept_top3,
            "actual_response": ex.response,
            "turn": ex.turn,
            "has_teacher_distribution": ex.teacher_action_distribution is not None,
        }
        for ex in items
    ]
    return batch, meta
```

Then:

```python
logits, _ = model(batch)
pred_index = torch.argmax(logits, dim=1)
pred_actions = batch["candidate_actions"].gather(1, pred_index[:, None]).squeeze(1)

for pred_action, row_meta in zip(pred_actions.cpu().tolist(), meta):
    pred_norm = normalize_teacher_action(action_response(int(pred_action)))
    scores = score_original_chaga_match(
        pred_norm,
        teacher_top1_norm=row_meta["teacher_top1_norm"],
        teacher_top3_norms=row_meta["teacher_top3_norms"],
        accept_top3=row_meta["accept_top3"],
    )
```

Report at least:

```text
original_top1_accuracy
original_top3_inclusion
original_relaxed_accuracy
original_play_first6_relaxed_accuracy
original_play_first6_samples
original_non_relaxed_region_top1_accuracy
reviewed_without_teacher_distribution
pred_in_chaga_top5
by_teacher_top1_family
by_actual_family
by_turn_bucket
```

For the current task, `original_relaxed_accuracy` is the main metric.

### 5. Force full candidate coverage during CHAGA evaluation

In `evaluate_transformer_chaga_review.py`, change this:

```python
max_candidates = int(config.get("max_candidates", args.max_candidates))
```

to this:

```python
max_candidates = int(args.max_candidates)
```

and set CLI default to `235`.

Reason: `max_candidates` is a data-collation choice, not a learned model shape. The model scores an arbitrary number of candidate actions because candidate actions are embedded by action id. The checkpoint config should not force the evaluator to reuse a training-time truncation setting.

Add an explicit metric:

```text
candidate_truncation_count
```

or simply assert no truncation when `max_candidates >= 235`.

## Tests to add

### Test 1: top-3 only counts when the accept flag is true

File: `tests/test_evaluate_transformer_chaga_review.py`

```python
def test_score_original_chaga_match_relaxes_only_with_accept_flag():
    assert score_original_chaga_match(
        "PLAY W3",
        teacher_top1_norm="PLAY W1",
        teacher_top3_norms=("PLAY W1", "PLAY W2", "PLAY W3"),
        accept_top3=True,
    )["relaxed"]

    assert not score_original_chaga_match(
        "PLAY W3",
        teacher_top1_norm="PLAY W1",
        teacher_top3_norms=("PLAY W1", "PLAY W2", "PLAY W3"),
        accept_top3=False,
    )["relaxed"]
```

### Test 2: non-PLAY regions require top-1

```python
def test_score_original_chaga_match_requires_top1_outside_relaxed_region():
    result = score_original_chaga_match(
        "PENG",
        teacher_top1_norm="PASS",
        teacher_top3_norms=("PASS", "PENG", "CHI W3"),
        accept_top3=False,
    )
    assert result["top3"]
    assert not result["relaxed"]
```

### Test 3: reviewed selection includes original candidates even if distribution mapping failed

```python
def test_select_reviewed_examples_uses_original_candidate_metadata_not_distribution():
    ex = make_example(
        teacher_action_distribution=None,
        teacher_top1_norm="PLAY W1",
        teacher_top3_norms=("PLAY W1", "PLAY W2", "PLAY W3"),
    )
    assert select_reviewed_examples([ex]) == [ex]
```

### Test 4: evaluator uses `max_candidates=235`

Construct an example where the CHAGA top action is legal but would be outside the first 96 candidate ids. The test should fail under old truncation and pass with `235`.

Expected assertion:

```python
assert metrics["candidate_truncation_count"] == 0
```

or, if you avoid tracking it:

```python
assert metrics["evaluated_candidate_width"] == 235
```

### Test 5: distribution metric and original-candidate metric can disagree

Create a synthetic example where `teacher_target_dist` maps only one candidate, but original CHAGA top-3 contains a different accepted PLAY action. The original-candidate helper should score against the original candidate strings. This prevents future regressions back to distribution-derived evaluation.

## Verification gate after this change

Before touching oversampling, family heads, or bigger training:

```text
1. Regenerate CHAGA audit with the new patch.
2. Run corrected original-candidate evaluator on the old checkpoint.
3. Confirm:
   reviewed_examples == expected aligned reviewed rows, or explain every drop
   reviewed_without_teacher_distribution is reported
   max candidate width is 235
   original_top1 / original_top3 / original_relaxed are reported separately
4. Compare old distribution-based metric vs new original-candidate metric.
5. Inspect all rows where the two disagree.
```

The old checkpoint score may change after this evaluator patch. That is expected and useful. The goal is not to make the old checkpoint look better; the goal is to stop using a proxy metric.

## Then run the reviewed-only overfit gate

Immediately after the evaluator patch, the next experiment should be:

```text
Train only on reviewed CHAGA rows.
Use max_candidates=235.
Use hard_loss_weight=0 or near 0 for reviewed rows.
Use teacher_loss_weight > 0.
Evaluate on the same reviewed rows with the original-candidate evaluator.
```

Expected gate:

```text
train-set original relaxed accuracy > 95%
train-set original top1 materially higher than current
Hu below 8 fan: 0
candidate_truncation_count: 0
reviewed_without_teacher_distribution: either 0 or explicitly diagnosed
```

If reviewed-only train-set relaxed accuracy cannot exceed 95%, do not proceed to oversampling or architecture. That failure would still indicate a target plumbing, normalization, candidate coverage, or evaluator issue.

So the order should be:

```text
1. Original-candidate evaluator with full 235 candidate coverage.
2. Reviewed-only overfit gate.
3. Reviewed oversampling / row-aware teacher-vs-human loss.
4. Audit regeneration and lookup strictness improvements if evaluator exposes drops/collisions.
5. Family head.
```

The next single code change is therefore the original-candidate evaluator, with `max_candidates=235` baked into that evaluation path.
