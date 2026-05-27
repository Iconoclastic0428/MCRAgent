I would not spend more L40 time yet. The repo has enough evidence that the current bottleneck is target construction and evaluation fidelity, not capacity. The current transformer path builds a candidate scorer over Lawlorentz’s 235-action mask, but several code paths still make the CHAGA target indirect, lossy, or too sparse.

My prioritized plan is below.

## 0. Immediate diagnosis

The current 47% relaxed score is not surprising. The repo notes say the corrected held-out review evaluation has only **522 reviewed states**, with top-1 `0.371648`, top-3 `0.557471`, and relaxed `0.471264`; the all-aligned target artifact has only **2,080 aligned CHAGA review states**.  With that target size, a 4.6M or 625M model can still miss badly if the teacher rows are underweighted, partially unmapped, or keyed incorrectly.

The most important repo-level issues are:

1. `allow_hu` is currently derived from the **human/state response**, not from a rule-engine fan gate.
2. The review lookup key is too coarse and can misassign repeated identical requests.
3. The evaluator compares against a reconstructed teacher distribution, not the original CHAGA candidate strings.
4. Teacher examples are too rare relative to non-review examples in the training mix.
5. The target set is too small for a meaningful >85% held-out CHAGA-candidate metric.

## 1. Fix Hu gating in `train_transformer_candidate.py`

This is the highest-priority correctness fix.

In `build_transformer_examples_from_record`, `allow_hu` is currently set as:

```python
allow_hu = response.split()[0].upper() == "HU"
```

That appears in the example-building path before CHAGA targets are converted.  This means Hu is allowed only when the recorded player actually chose Hu. That conflicts with the target objective. If CHAGA says Hu is best and the player passed, your teacher target cannot map Hu because `hu_gated_candidate_mask` removes Hu unless `allow_hu=True`. The hard invariant is “Hu only when fan-valid,” not “Hu only when the logged player chose Hu.”

Patch direction:

```python
def rule_gated_hu_allowed(agent, obs, request, response, record_context) -> bool:
    # Preferred: call the same official/PyMahjongGB fan gate used by runtime.
    # Minimum acceptable fallback: allow Hu if Lawlorentz action_mask contains a Hu action,
    # then separately assert that every Hu label passing this gate has base fan >= 8.
    for action in np.flatnonzero(obs["action_mask"] > 0):
        if agent.action2response(int(action)).split()[0] == "Hu":
            return True
    return False
```

Then use:

```python
allow_hu = rule_gated_hu_allowed(runtimes[player].agent, obs, request, response, record)
```

Do the same for `build_candidate_rule_features`, because feature slot 0 is also conditioned on `allow_hu`. 

Verification gate:

```text
For every reviewed state where CHAGA top-1 is HU:
  Hu must remain in candidate_actions iff official fan checker proves base fan >= 8.
For every reviewed state where official fan < 8:
  Hu must be absent from candidate_actions.
```

Do not continue training until this passes. This bug alone can block CHAGA Hu/pass imitation.

## 2. Make CHAGA evaluation compare predicted normalized action to original candidate strings

Current evaluation uses `teacher_target_dist` and checks whether the predicted candidate slot equals the argmax slot of the mapped teacher distribution.  That is not exactly the requested metric. The requested metric is model-predicted action versus CHAGA review candidate strings, with top-3 accepted only for a player’s first six PLAY decisions.

Current path:

```text
CHAGA candidate strings
  -> normalized
  -> mapped to legal action IDs
  -> converted to teacher distribution
  -> top1 = argmax(distribution)
  -> compare pred_slot to top1 slot
```

That can undercount or overcount when multiple legal actions normalize to one CHAGA candidate, especially `GANG`, `BUGANG`, `PENG`, and claim-family actions. The soft-target converter also maps CHAGA candidates into action IDs before evaluation. 

Patch direction:

Extend `TransformerExample` to store:

```python
teacher_top1_norm: str | None
teacher_top3_norm: tuple[str, ...]
teacher_relaxed_accept_top3: bool
```

Then in `evaluate_transformer_chaga_review.py`, after model prediction:

```python
pred_norm = normalize_teacher_action(action_response(pred_action))

top1_match = pred_norm == example.teacher_top1_norm
top3_match = pred_norm in example.teacher_top3_norm[:3]

relaxed_match = top1_match or (
    example.teacher_relaxed_accept_top3 and pred_norm.startswith("PLAY ") and top3_match
)
```

Do not infer the evaluation target from `teacher_target_dist`. Use the original audit rows. The audit already stores `chaga_top5_candidates`, normalized top-5 candidates, and first-six PLAY ordinal. 

Verification gate:

```text
For a fixed checkpoint, compare:
  old distribution-based metric
  new string-based metric
on the same 522 states.

Investigate every disagreement class:
  GANG tile collapse
  BUGANG/GANG
  PENG with discard suffix
  CHI middle-tile normalization
  PASS vs ABANDON
  Hu gated/unmapped
```

This may raise or lower the current 47%, but it will make the metric faithful.

## 3. Strengthen the review-target lookup key

`ReviewTargetLookup` currently keys teacher entries by:

```python
(record_id, player, request, normalized state_actual_response)
```

and uses a `deque.popleft()`.  This is fragile. Repeated claim windows and repeated `PLAY` requests can share the same `(record_id, player, request, response)` key. If the model builder and audit rows traverse the record differently, the deque can silently attach the wrong CHAGA candidates to a state.

Patch direction:

Add these fields to audit entries:

```python
"state_turn": state.turn
"state_action": state.action
"state_legal_action_count": len(state.legal_actions)
```

Then key by:

```python
(
  record_id,
  str(seat),
  int(state_turn),
  str(state_context["request"]),
  normalize_teacher_action(state_actual_response),
)
```

Use `ri` as a secondary check, not the only key, because `ri` comes from the Tziakcha action stream while training examples come from reconstructed Botzone-style turns.

Verification gate:

```text
review_audit_aligned_entries = 2080
lookup_entries_loaded = 2080
teacher_targets_attached = 2080
duplicate_lookup_keys = 0
teacher_target_unmapped = 0, or explicitly explained by action type
```

Right now the code has alignment checks for offered tile, drawn tile, actor, window, hand-size, and top1-in-legal-mask.  Keep those, but add a strict no-duplicate-key gate before training.

## 4. Fix action normalization edge cases

Current `normalize_teacher_action` collapses `PENG`, `GANG`, and `BUGANG` to action family only.  That is acceptable for some claim windows, but wrong for draw windows where multiple gang choices can be legal. Also audit handling treats kind-1 as `Abandon`, while the model action space likely represents decline as `Pass`. 

Patch direction:

Use context-aware normalization:

```python
normalize_teacher_action(action, window, request):
    if action is Abandon and window == claim:
        return "PASS"

    if action is PENG in claim window:
        return "PENG"

    if action is CHI in claim window:
        return f"CHI {middle_tile}"

    if action is GANG in draw window:
        return f"GANG {tile}"

    if action is GANG in claim window:
        return "GANG"

    if action is BUGANG:
        return f"BUGANG {tile}" when tile is known
```

Also fix this likely audit bug:

```python
if kind == 2:
    if value <= 0:
        return None
```

For Tziakcha tile IDs, physical tile `0` is a valid tile copy. `tile_id_to_botzone_symbol` accepts `0` and maps by tile kind.  Change `value <= 0` to `value < 0` unless you have confirmed CHAGA review rows use `0` as “no tile” for PLAY, which would be unusual.

Verification gate:

```text
No reviewed PLAY rows are dropped solely because v == 0.
PASS and ABANDON mapping is explicitly counted.
GANG/BUGANG mapping coverage is reported by window.
```

## 5. Stop letting human hard labels fight CHAGA teacher labels

The training loss still always includes hard cross-entropy to the recorded action. Teacher loss is optional and additive.  That is not optimal for a model-vs-CHAGA metric. If the objective is CHAGA candidate precision, the reviewed rows should train primarily against CHAGA, not against the human action.

Patch direction:

In `policy_loss_with_optional_teacher`, use row-level weighting:

```python
reviewed = batch["has_teacher_target"]

loss_reviewed = KL_or_CE_to_CHAGA(logits[reviewed], teacher_dist[reviewed])
loss_unreviewed = CE_to_human(logits[~reviewed], target_index[~reviewed])

loss = teacher_weight * loss_reviewed + human_weight * loss_unreviewed
```

Recommended final distillation stage:

```text
reviewed rows:
  hard_loss_weight = 0.0
  teacher_loss_weight = 1.0 to 5.0

unreviewed high-ELO rows:
  hard_loss_weight = 0.1 to 0.3
  teacher_loss_weight = 0.0
```

Current command-level knobs `--hard-loss-weight` and `--teacher-loss-weight` are global.  They are not enough. Add row-aware loss.

Verification gate:

```text
Train-only reviewed relaxed accuracy should reach >95%.
Held-out reviewed relaxed accuracy should improve without train-only collapse.
If train-only reviewed relaxed accuracy cannot exceed 90%, target mapping is still wrong.
```

That train-only gate is essential. A model should be able to overfit 2,080 reviewed states. If it cannot, there is still a mapping, masking, or metric bug.

## 6. Oversample reviewed states aggressively

The L40 run uses high-ELO filtered data plus CHAGA candidates, but the review targets are only 2,080 aligned states. If the loader mixes them into a much larger raw corpus, the teacher objective is diluted. The repo notes show the project already has much larger non-review corpora, while CHAGA review data is tiny. 

Patch direction:

Build two datasets:

```python
reviewed_examples = [ex for ex in all_examples if ex.teacher_action_distribution is not None]
unreviewed_examples = [ex for ex in all_examples if ex.teacher_action_distribution is None]
```

Use a mixed batch schedule:

```text
Phase A: supervised pretrain on all legal data.
Phase B: 70% reviewed CHAGA rows, 30% high-ELO rows.
Phase C: 100% reviewed CHAGA rows, early stop on held-out reviewed relaxed metric.
```

For the current 2,080 target artifact, use repeated sampling rather than more epochs over a huge mixed dataset.

Verification gate before L40:

```text
Each training epoch must contain at least:
  reviewed_rows_seen >= 20 × number_of_unique_reviewed_train_rows
  unreviewed_rows_seen <= 2 × reviewed_rows_seen during distillation phase
```

## 7. Set `max_candidates=235` for review training and evaluation

The collator truncates legal candidates when the legal action list exceeds `max_candidates`, then forces only the human target action back into the list.  If a CHAGA top candidate is not the human action and gets truncated, the teacher target becomes distorted.

Maybe this rarely happens, but the fix is cheap. For CHAGA distillation, set:

```text
--max-candidates 235
```

and store this in the checkpoint config. The action space is only 235, so there is little reason to use 96 during the review-candidate experiment.

Verification gate:

```text
candidate_truncation_count = 0
teacher_top1_dropped_by_truncation = 0
teacher_top3_dropped_by_truncation = 0
```

## 8. Improve history encoding before scaling model size

The current history token compresses tile identity with:

```python
tile_mod = TILE_IDS.get(tile, 0) % 12
return 1 + player * 96 + action_type * 12 + tile_mod
```



This creates avoidable collisions across suits and honors. CHAGA-style discard decisions are sensitive to exact suit/rank/honor identity. Replace it with full tile identity:

```python
tile_id = TILE_IDS.get(tile, 34)  # 34 = no tile
return 1 + player * (len(ACTION_TYPES) * 35) + action_type * 35 + tile_id
```

Then set:

```text
history_vocab_size >= 4 * 10 * 35 + 1 = 1401
```

Use `2048` or `4096`.

Also add request-window tokens separately. A pass after a discard and a pass after a rob-kong window should not collide.

Verification gate:

```text
History token collision audit:
  old distinct semantic events per token
  new distinct semantic events per token
Expected new max collision: 1 except unknown/no-tile cases.
```

## 9. Add stronger candidate features

The current candidate features are seven numbers, mostly action family plus Lawlorentz effective-tile features for PLAY.  That is too thin for CHAGA imitation.

Add features that are cheap and directly relevant:

```text
Candidate tile features:
  tile id one-hot or embedding index
  suit
  rank
  terminal/honor/dragon/wind flags
  count in hand before action
  visible count
  remaining count
  is drawn tile
  is offered tile
  is last copy visible risk

State features:
  wall count bucket
  round wind
  seat wind
  dealer flag
  self score rank / point delta
  open meld count
  flower count
  shanten regular / seven pairs / thirteen orphans
  current max base fan bucket
```

One concrete bug here: `build_candidate_rule_features` sets `prevalent_wind=0` when constructing `LawlorentzEffectiveScorer`.  That loses round-wind fan context. Pass the runtime’s actual prevalent wind into the example and feature builder.

## 10. Expand CHAGA review targets; 2,080 is too small for the stated target

The current public accessible CHAGA02-08 data is 76 records and 2,080 aligned review states.  That is useful for a smoke test, not for >85% held-out candidate precision.

Necessary data expansion route:

```text
1. Use authenticated Tziakcha history access with TZI_HISTORY_COOKIE.
2. Fetch high-ELO or CHAGA-containing sessions.
3. Convert records with existing tziakcha_records.py.
4. For each selected record and seat, call CHAGA review API and cache result.
5. Run audit_chaga_review_alignment.py.
6. Keep only rows passing strict alignment checks.
7. Build split by session_id, not by row.
```

Target scale:

```text
Minimum useful: 25k aligned reviewed states
Better:         100k aligned reviewed states
Strong:         250k+ aligned reviewed states
```

The repo already notes that public shell fetches hit 403 and that full history requires `TZI_HISTORY_COOKIE`.  That is the real data bottleneck.

## 11. Add mandatory pre-L40 verification gates

Before another L40 job, require these gates to pass locally on CPU/GPU with a small model:

```text
Gate A: audit coverage
  aligned_review_states >= expected
  duplicate_lookup_keys = 0
  teacher_targets_attached / aligned_review_states >= 0.995
  teacher_target_unmapped = 0, or explained by action type

Gate B: Hu safety
  below-8 Hu candidate count = 0
  CHAGA Hu rows map iff official fan gate passes
  runtime Hu still gated by official fan checker

Gate C: metric fidelity
  string-based evaluator implemented
  old distribution-based and new string-based disagreement report generated

Gate D: overfit sanity
  Train on 80% of 2,080 reviewed rows only.
  Evaluate on same train rows.
  Relaxed train accuracy >95%.
  If not, fix labels/masks/model plumbing.

Gate E: held-out sanity
  Split by session_id.
  Held-out relaxed > current 47%.
  Report by decision type:
    PLAY first six
    PLAY after first six
    PASS/HU
    CHI/PENG/GANG
    turn bucket
    shanten bucket
    CHAGA margin bucket
```

The current audit already computes many alignment rates, including top1-in-legal-mask and context checks.  Extend it rather than creating a separate ad hoc validator.

## 12. Architecture changes that are reachable in this codebase

Do not jump to 625M. Make these smaller changes first.

### 12.1 Replace flat candidate scorer with shared state plus action-family heads

Current model encodes the flattened observation, averaged history transformer state, scalar features, then scores each candidate independently.  That is workable but weak for action-family calibration.

Reachable change:

```text
state encoder
  -> family head: PASS / HU / PLAY / CHI / PENG / GANG / BUGANG
  -> candidate tile/action scorer
  -> final logit = candidate_score + family_logit[action_family]
```

This helps with the current mismatch pattern because CHAGA agreement is often first about whether to pass, claim, hu, or discard.

### 12.2 Add per-family loss

Use separate CE/KL terms:

```text
L = L_family + L_candidate_within_family + L_teacher_KL + small value loss
```

If CHAGA top1 is `PLAY W5`, family target is `PLAY`; if top3 contains three discard candidates, candidate-within-family learns the tile choice.

### 12.3 Add a calibration/rerank layer for CHAGA imitation

For the reviewed metric, a simple reranker may improve faster than the transformer:

```text
final_logit =
  transformer_logit
  + a * lawlorentz_effective_score
  + b * chaga_distilled_prior_score
  + c_family[action_family]
```

Fit `a`, `b`, and family biases on reviewed training rows with logistic regression or LBFGS. This gives a quick calibration baseline and tells you whether the neural model is undercalibrated rather than underpowered.

## 13. How I would order the actual work

### Pass 1: correctness

Modify these files first:

```text
scripts/audit_chaga_review_alignment.py
  Add state_turn/state_action to audit entries.
  Fix PLAY value 0 handling.
  Normalize PASS/ABANDON intentionally.
  Report duplicate keys and mapping coverage by action type.

scripts/train_transformer_candidate.py
  Replace human-response Hu gating with rule-gated Hu.
  Use richer lookup key.
  Preserve original teacher top1/top3 normalized candidates in examples.
  Add row-aware teacher-vs-human loss.
  Set max_candidates=235 for CHAGA runs.

scripts/evaluate_transformer_chaga_review.py
  Compute metric from model predicted normalized action vs original CHAGA candidates.
  Report all aligned rows, mapped rows, dropped rows, and action-slice metrics.
```

### Pass 2: overfit proof

Run three tiny experiments:

```text
Experiment 1: reviewed-only, train=evaluation rows
Expected: >95% relaxed

Experiment 2: reviewed-only, session split
Expected: materially above 47%, maybe 55–70% with only 2k rows

Experiment 3: reviewed + high-ELO, reviewed oversampled 20x
Expected: should beat Experiment 2 only if high-ELO data adds useful priors
```

If Experiment 1 fails, the bug is still in mapping, masking, normalization, or evaluator.

### Pass 3: data expansion

Use authenticated Tziakcha fetch plus CHAGA review cache. Do not aim for >85% before at least 25k aligned review states.

### Pass 4: model improvements

Only after Passes 1–3:

```text
1. Add full tile identity history tokens.
2. Add family head.
3. Add candidate tile features.
4. Add family-balanced sampler.
5. Train 30M–80M, not 625M first.
```

## 14. Expected realistic outcomes

With only 2,080 aligned CHAGA states:

```text
Train-on-train relaxed:        should be >95% after fixes
Held-out 522 relaxed:          55–70% is a reasonable first target
>85% held-out relaxed:         unlikely without more review data or leakage
```

With 50k–100k aligned CHAGA states and the fixes above:

```text
Top-1:                         65–80%
Relaxed:                       80–88% may become plausible
PLAY first-six relaxed:        likely the easiest slice to push above 85%
Claim/Hu/PASS top-1:           depends on mapping and fan-gate correctness
```

The main second-opinion conclusion: **do not scale the flat model again until reviewed-target attachment is exact, Hu is rule-gated rather than human-action-gated, and the evaluator compares normalized predictions to the original CHAGA candidate strings.** Those are the highest-yield repo-level fixes.
