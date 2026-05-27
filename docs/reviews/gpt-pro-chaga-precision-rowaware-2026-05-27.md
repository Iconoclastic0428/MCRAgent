# GPT Pro CHAGA Precision Row-Aware Review - 2026-05-27

Reviewed pushed commit: https://github.com/Iconoclastic0428/MCRAgent/commit/28342c24ff95f8c4c76cf0807684171f275c932f

Prompt focus: next step after passing reviewed-only overfit gate and getting 60.5357% corrected relaxed held-out match from row-aware mixed L40 training.

---

The current patch stack looks directionally correct. I do not see a remaining blocker as fundamental as the earlier Hu-gating or original-candidate-evaluator issues. The overfit gate passing at 0.999 relaxed on 2,077 reviewed rows is strong evidence that the model, collator, action mapping, and evaluator can learn the CHAGA target when the train and eval states are identical.

The next highest-leverage change is not another architecture increase. It is to replace the current reviewed-row teacher loss with a metric-aligned CHAGA accepted-set loss.

Why this is the next change

The current mixed run is training a soft CHAGA distribution, but the target metric is a discrete accepted-set rule:

first six PLAY decisions: prediction âˆ?CHAGA top-3
all other reviewed decisions: prediction == CHAGA top-1

Right now, chaga_candidates_to_action_distribution() maps all CHAGA candidates into a soft distribution using scores and temperature. It does not construct the actual metric acceptance set.

train_transformer_candidate

 The row-aware loss then trains reviewed rows by cross-entropy against that soft distribution.

train_transformer_candidate

That objective is close to distillation, but it is not the stated metric. For non-first-six decisions, CHAGA top-2/top-3 choices still receive positive probability, even though they are scored as wrong. For first-six PLAY decisions, the loss still prefers CHAGAâ€™s rank ordering, even though any top-3 action is accepted by the metric. This mismatch matters because your eval shows a gap between top-3 inclusion and relaxed accuracy: top-3 is 0.6696, but relaxed is only 0.6054, and PLAY relaxed is 0.5926.

transformer_candidate_highelo23â€?
The training set is also tiny relative to the 212M model: the L40 run has only 1,517 reviewed teacher targets in train and 560 in eval.

transformer_candidate_highelo23â€?
 The model has 212,373,506 parameters.

transformer_candidate_highelo23â€?
 It peaks around epoch 9 and then overfits, which is exactly what I would expect from a large model trained on ~1.5k reviewed targets plus human auxiliary data.

transformer_candidate_highelo23â€?
So the next change should make every reviewed training example optimize exactly what the evaluator scores.

Next implementation: metric-aligned accepted-set loss
Desired behavior

For each reviewed example, construct a boolean mask over candidate slots:

accepted_mask[j] = True if candidate j is accepted by the official CHAGA metric

The accepted set should be:

if example.teacher_accept_top3:
    accepted normalized actions = teacher_candidate_norms[:3]
else:
    accepted normalized actions = teacher_candidate_norms[:1]

Then use a multi-positive loss:

Python
Run
loss = -logsumexp(log_probs[accepted_slots])

This directly rewards putting probability mass on any metric-accepted action. For non-relaxed states, the accepted set has size 1, so this becomes ordinary top-1 CE. For first-six PLAY states, it rewards any top-3 prediction.

Do not remove the existing soft teacher distribution immediately. Keep it for diagnostics and optional auxiliary distillation. But the primary reviewed-row loss should become the metric-aligned accepted-set loss.

Concrete file changes
1. Add accepted-set masks to the collator

File: scripts/train_transformer_candidate.py

The TransformerExample already stores teacher_candidate_norms and teacher_accept_top3.

train_transformer_candidate

 The collator already builds candidate_actions, candidate_mask, teacher_target_dist, and has_teacher_target.

train_transformer_candidate

 Add two tensors:

Python
Run
teacher_accept_mask = torch.zeros((len(examples), max_candidates), dtype=torch.bool)
has_teacher_accept_set = torch.zeros((len(examples),), dtype=torch.bool)

Inside the per-example loop:

Python
Run
accepted_norms = tuple(example.teacher_candidate_norms[:3] if example.teacher_accept_top3 else example.teacher_candidate_norms[:1])
accepted_norms = {normalize_teacher_action(action) for action in accepted_norms if action}

if accepted_norms:
    for slot, action in enumerate(candidates):
        action_norm = normalize_teacher_action(action_response(int(action)))
        if action_norm in accepted_norms:
            teacher_accept_mask[row, slot] = True

    if bool(teacher_accept_mask[row, :len(candidates)].any().item()):
        has_teacher_accept_set[row] = True

Return them:

Python
Run
"teacher_accept_mask": teacher_accept_mask,
"has_teacher_accept_set": has_teacher_accept_set,

Important: this should use normalized original CHAGA candidate strings, not the soft mapped distribution.

2. Add metric-set loss

File: scripts/train_transformer_candidate.py

Add:

Python
Run
def teacher_accepted_set_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    has_set = batch.get("has_teacher_accept_set")
    if has_set is None:
        return logits.new_tensor(0.0)
    has_set = has_set.bool()
    if not bool(has_set.any().item()):
        return logits.new_tensor(0.0)

    log_probs = F.log_softmax(logits[has_set], dim=1)
    accept_mask = batch["teacher_accept_mask"][has_set].bool()

    # logsumexp over accepted candidate slots.
    accepted_log_probs = log_probs.masked_fill(~accept_mask, float("-inf"))
    return -torch.logsumexp(accepted_log_probs, dim=1).mean()

This is the key loss. It optimizes the actual relaxed metric.

3. Extend policy_loss_with_optional_teacher

Current reviewed teacher loss is:

Python
Run
reviewed_teacher_loss = -(teacher_dist * safe_log_probs).sum(dim=1).mean()

train_transformer_candidate

Add arguments:

Python
Run
reviewed_accept_set_loss_weight: float | None = None

Compute:

Python
Run
reviewed_accept_set_loss = teacher_accepted_set_loss(logits, batch)

Then in the row-aware branch:

Python
Run
loss = (
    reviewed_hard_weight * reviewed_hard_loss
    + reviewed_teacher_weight * reviewed_teacher_loss
    + reviewed_accept_set_weight * reviewed_accept_set_loss
    + unreviewed_hard_weight * unreviewed_hard_loss
)

Recommended defaults for the next experiment:

reviewed_hard_loss_weight = 0.0
reviewed_teacher_loss_weight = 0.0 or 0.2
reviewed_accept_set_loss_weight = 1.0
unreviewed_hard_loss_weight = 0.1 to 0.2
value_loss_weight = 0.0 initially

I would start with reviewed_teacher_loss_weight=0.0 for a clean test. Add 0.1â€?.2 only if the model becomes poorly calibrated or unstable.

4. Add CLI flags

File: scripts/train_transformer_candidate.py

Add:

Python
Run
parser.add_argument("--reviewed-accept-set-loss-weight", type=float, default=None)

Record it in metrics next to the other row-aware weights. The current metrics already record reviewed_hard_loss_weight, reviewed_teacher_loss_weight, and unreviewed_hard_loss_weight.

train_transformer_candidate

5. Add train/eval metrics for accepted-set loss

In policy_loss_with_optional_teacher, return:

Python
Run
"reviewed_accept_set_policy_loss": reviewed_accept_set_loss.detach()

The epoch loop currently logs total train loss, train policy loss, train value loss, and train accuracy, but not per-loss components.

train_transformer_candidate

 Add averaged component logging for:

train_reviewed_teacher_policy_loss
train_reviewed_accept_set_policy_loss
train_unreviewed_hard_policy_loss

Without these, you will not know whether the metric loss is actually decreasing before overfit.

Tests to add
Test 1. Collator builds top-1 accepted set for non-relaxed rows

File: tests/test_train_transformer_candidate.py

Python
Run
def test_collate_teacher_accept_mask_top1_only_for_non_relaxed():
    ex = make_transformer_example(
        action_mask_with_play_w1_w2_w3=True,
        teacher_candidate_norms=("PLAY W1", "PLAY W2", "PLAY W3"),
        teacher_accept_top3=False,
    )

    batch = collate_transformer_examples([ex], max_candidates=235)

    accepted_actions = [
        action_response(int(action))
        for action, ok in zip(batch["candidate_actions"][0].tolist(), batch["teacher_accept_mask"][0].tolist())
        if ok
    ]

    assert [normalize_teacher_action(action) for action in accepted_actions] == ["PLAY W1"]
    assert batch["has_teacher_accept_set"][0].item()
Test 2. Collator builds top-3 accepted set for relaxed first-six PLAY rows
Python
Run
def test_collate_teacher_accept_mask_top3_for_first_six_play():
    ex = make_transformer_example(
        action_mask_with_play_w1_w2_w3=True,
        teacher_candidate_norms=("PLAY W1", "PLAY W2", "PLAY W3", "PLAY W4"),
        teacher_accept_top3=True,
    )

    batch = collate_transformer_examples([ex], max_candidates=235)

    accepted = accepted_normalized_actions_from_batch(batch)
    assert set(accepted) == {"PLAY W1", "PLAY W2", "PLAY W3"}
Test 3. Accepted-set loss accepts any top-3 action on relaxed rows
Python
Run
def test_teacher_accepted_set_loss_accepts_any_relaxed_top3_slot():
    logits = torch.tensor([[0.0, 0.0, 10.0, -10.0]], requires_grad=True)
    batch = {
        "has_teacher_accept_set": torch.tensor([True]),
        "teacher_accept_mask": torch.tensor([[True, True, True, False]]),
    }

    loss = teacher_accepted_set_loss(logits, batch)
    assert loss.item() < 0.01
Test 4. Accepted-set loss rejects top-2 on non-relaxed rows
Python
Run
def test_teacher_accepted_set_loss_rejects_second_choice_when_not_relaxed():
    logits = torch.tensor([[0.0, 10.0, -10.0]], requires_grad=True)
    batch = {
        "has_teacher_accept_set": torch.tensor([True]),
        "teacher_accept_mask": torch.tensor([[True, False, False]]),
    }

    loss = teacher_accepted_set_loss(logits, batch)
    assert loss.item() > 9.0
Test 5. Overfit gate still passes with accepted-set loss

Run the existing reviewed-only train-on-eval gate, but use:

--reviewed-accept-set-loss-weight 1.0
--reviewed-teacher-loss-weight 0.0
--reviewed-hard-loss-weight 0.0
--unreviewed-hard-loss-weight 0.0
--value-loss-weight 0.0

Success:

original_relaxed_accuracy >= 0.99
original_play_relaxed_accuracy >= 0.99
candidate_truncation_count = 0
reviewed_without_teacher_distribution = 0
Hu below 8 fan = 0

If this fails, the accepted-set mask is wrong.

Next L40 experiment after the accepted-set loss passes

Use the same data, but change the loss, not the model size.

Recommended first run:

Bash
python scripts/train_transformer_candidate.py \
  --raw data/processed/high_elo2300/tziakcha_chaga0208_elo2300_train.jsonl \
  --raw data/processed/high_elo2300/tziakcha_human256_elo2300_train.jsonl \
  --eval-raw data/processed/high_elo2300/tziakcha_chaga0208_elo2300_eval.jsonl \
  --review-audit-jsonl runs/chaga_review_alignment_audit_all.jsonl \
  --max-candidates 235 \
  --reviewed-batch-fraction 0.7 \
  --reviewed-hard-loss-weight 0.0 \
  --reviewed-teacher-loss-weight 0.0 \
  --reviewed-accept-set-loss-weight 1.0 \
  --unreviewed-hard-loss-weight 0.1 \
  --value-loss-weight 0.0 \
  --teacher-temperature 0.2 \
  --monitor-metric val_teacher_relaxed_accuracy \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.00003 \
  --weight-decay 0.03 \
  --dropout 0.2 \
  --d-model 1024 \
  --nhead 16 \
  --num-layers 16 \
  --dim-feedforward 4096

Rationale:

The current job used reviewed teacher CE, unreviewed hard CE, and value loss.

mcr-transformer-l40-rowaware-20â€?
 For a CHAGA-candidate precision experiment, value loss is noise unless you have evidence it helps. Turn it off for this ablation. Keep unreviewed hard loss small because the reviewed target is CHAGA, not human action. Increase regularization slightly because the current run overfits sharply after epoch 9.

transformer_candidate_highelo23â€?
Success gate:

candidate_truncation_count = 0
reviewed_without_teacher_distribution = 0
original_relaxed_accuracy > 0.65
original_play_relaxed_accuracy > 0.65
original_top1_accuracy improves or does not materially regress
best epoch occurs before full memorization

This is not yet the 85% gate. It is the gate that proves the loss now points at the target metric.

Add diagnostic slices before the next data expansion

The turn graph is useful, but â€œturnâ€?alone is not explanatory. The current turn summary shows high-mismatch turns, but many buckets have only 3â€?3 states, so the turn-level counts are noisy. For example, turn 65 has 7 total states and 5 relaxed mismatches, while turn 68 has 5 total states and 4 relaxed mismatches.

transformer_candidate_highelo23â€?
Extend collect_original_prediction_rows() in scripts/evaluate_transformer_chaga_review.py. It currently records turn, player, response, predicted action, top candidates, accept-top3 flag, and match booleans.

evaluate_transformer_chaga_reviâ€?
 Add:

teacher_top1_family
predicted_family
human_family
is_play_row
is_relaxed_region
candidate_count
legal_action_count
top1_score
top2_score
top1_margin
predicted_in_top5
predicted_rank_in_chaga
record_id
session_id
state_turn

You may need to add record_id/session_id to TransformerExample if not already present. If you avoid changing the dataclass, at least add teacher_top1_family, predicted_family, is_relaxed_region, and predicted_rank_in_chaga.

Then aggregate by:

decision family: PLAY / PASS / CHI / PENG / GANG / HU
relaxed region: first-six PLAY vs all other
CHAGA margin: top1-top2 buckets
candidate count bucket
turn bucket: 0-10, 11-30, 31-60, 61+

Verification gate:

Mismatch report must explain at least 80% of errors by family and margin bucket.

This matters because if many mismatches are low-margin top1/top2 disagreements outside the relaxed region, then 85% top1-like accuracy will require more CHAGA labels, not more architecture.

After metric-set loss: likely necessary data expansion

Even with the corrected loss, I would not expect 85% held-out relaxed accuracy from the current reviewed set. The training run used only 57 CHAGA train records and 19 CHAGA eval records, producing 1,517 reviewed train targets and 560 reviewed eval targets.

transformer_candidate_highelo23â€?


transformer_candidate_highelo23â€?
 The model already memorizes train and degrades on validation, which is a data scarcity pattern rather than a capacity pattern.

The repo already has the right review-fetching mechanism: audit_chaga_review_alignment.py calls the CHAGA review endpoint and caches session_id + seat review JSON.

audit_chaga_review_alignment

 It also rebuilds model states, aligns rows, and writes candidate-bearing audit entries.

audit_chaga_review_alignment

 Use that pipeline to expand reviewed targets.

Minimum next target size:

25k aligned reviewed states: minimum serious target
100k aligned reviewed states: likely needed for stable >85% relaxed

Data split rule:

Split by session_id, not row.
No same session_id across train/eval/test.
Keep eval/test fixed once created.

Success gate for the expanded dataset:

train reviewed states >= 25,000
eval reviewed states >= 5,000
test reviewed states >= 5,000
reviewed_without_teacher_distribution = 0
top1_in_legal_mask = 1.0 or every failure audited
candidate_truncation_count = 0

Only after that should you interpret â€?5%â€?as a stable number.

Lower-priority but concrete model issue: history-token collisions

Do not make this the next change before the metric-set loss, but it should be fixed soon.

encode_history_event() compresses tile identity with:

Python
Run
tile_mod = TILE_IDS.get(tile, 0) % 12
return 1 + int(player) * 96 + action_type * 12 + tile_mod

train_transformer_candidate

That aliases many distinct tiles. For Mahjong, suit and honor identity matter. This is especially likely to hurt later-turn decisions where the model needs discard and visible-tile history. Replace it with full 34-tile identity plus a no-tile sentinel:

Python
Run
tile_id = TILE_IDS.get(tile, 34)
return 1 + int(player) * (len(BOTZONE_ACTION_TYPES) * 35) + action_type * 35 + tile_id

Then set:

history_vocab_size >= 4 * len(BOTZONE_ACTION_TYPES) * 35 + 1

With 10 action types, this is 1401; use 2048.

Test:

Python
Run
def test_history_event_tokens_do_not_alias_tile_suits():
    assert encode_history_event(0, "2 W1", "Play W1") != encode_history_event(0, "2 T1", "Play T1")
    assert encode_history_event(0, "2 W1", "Play W1") != encode_history_event(0, "2 B1", "Play B1")

This change may help late-turn generalization, but it will not by itself solve the current 60.5% ceiling.

Prioritized plan
Step 1. Implement metric-aligned accepted-set loss

Success:

unit tests pass
reviewed-only train-on-eval relaxed >= 0.99
mixed run improves held-out relaxed above 0.65

Failure means the accepted mask is wrong or the training loop is not using it.

Step 2. Add richer mismatch diagnostics

Success:

mismatches sliced by family, margin, relaxed region, candidate count, and turn bucket
predicted_rank_in_chaga available for every reviewed row

Failure means you still do not know whether errors are low-margin ambiguity, wrong action family, or late-game context failure.

Step 3. Expand CHAGA reviewed data

Success:

25k+ aligned reviewed states
session-level split
fixed held-out test

Failure means 85% is not a meaningful target yet.

Step 4. Fix history-token identity collisions

Success:

no suit/honor aliasing in history tokens
same overfit gate still passes
held-out late-turn slices improve
Step 5. Add family head only after the above

A family head can help, but it is not the next highest-leverage change. The current model already receives action type features and candidate embeddings.

train_transformer_candidate

 A family head is useful if diagnostics show that many mismatches are PLAY versus PASS/CHI/PENG/HU family errors. If most mismatches are wrong discard tile within PLAY, the family head will not solve the main problem.

Bottom line: implement the metric-aligned accepted-set loss next. The current system can memorize the reviewed target, but mixed training is still optimizing a soft distillation proxy rather than the exact relaxed CHAGA-candidate metric. After that, expand the reviewed CHAGA data; the current 1,517 train reviewed states are too few to expect a stable jump from 60.5% to 85%.
