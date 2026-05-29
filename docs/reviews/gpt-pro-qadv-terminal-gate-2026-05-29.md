# GPT Pro QADV Terminal Gate Review - 2026-05-29

Source conversation: https://chatgpt.com/c/6a167dc8-942c-83e8-810c-cf16c97c2c92

Prompt context: QADV validation improved CHAGA relaxed accuracy safely, but exact official-judge fixed-league scans found `terminal_slice_count=0`, `terminal_signal_count=0`, `target_terminal_signal_count=0`, `max_average_point_delta=0.0`, and no promotable lambda.

GPT Pro diagnosis:

- The gate failed because there was no usable terminal return signal, not because QADV was unsafe.
- The current scanner is a signal detector, not a termination detector. Huang hands without exposed `canHu` or score deltas can correctly look like zero signal.
- The signal definition was too narrow: it counted only Hu, end-wait from `display.canHu`, target Hu/end-wait, or nonzero target point delta.
- End-wait may be unobservable if the official judge's Huang output does not include `display.canHu`.
- The sample-bot league and one-game offset slices are probably too draw-prone/sparse for MCR terminal statistics.
- There is a scanner bug: using `max_average_point_delta` can hide negative point deltas because `max(0, negative)` remains zero.

Required next implementation:

1. Add terminal observability metrics to `scripts/evaluate_fixed_league_feasibility.py`:
   - terminal action counts
   - terminal reason counts
   - display key counts
   - `canHu` present/nonempty counts
   - nonzero-score game count
   - absolute score-signal sum
   - finish and turn-limit counts
2. Fix point-signal detection in `scripts/scan_qadv_terminal_slices.py`:
   - count nonzero point deltas directly
   - track max absolute point delta
   - use point-signal count for `has_signal`
3. Add policy-seat rotation:
   - `scripts/official_judge_match.py` gets `--policy-seat`
   - target policy can be placed in any seat 0..3
   - summaries include `policy_seat`
   - fixed-league feasibility should default target player from `policy_seat` when possible
4. Pass policy seat through:
   - `scripts/sweep_qadv_fixed_league.py`
   - `scripts/scan_qadv_terminal_slices.py`
   - optional `--policy-seats` in scanner for multi-seat scan keys

Required probes before return training:

- 64 games against sample, seat 0.
- 64 games against `lawlorentz_effective`, seat 0.
- Seat-rotation probes against the best signal opponent.

Diagnostic gates before any Q-return training:

- `finish_count == games`
- `turn_limit_count == 0`
- terminal action counts are not only unknown
- display key counts show which official fields are available
- low-fan Hu counts are zero
- action-outside-mask and illegal-prediction counts are zero

Signal-rich data requires at least one of:

- `hu_count >= 5` per 64 games
- `nonzero_score_games >= 5` per 64 games
- `target_end_wait_count >= 5` per 64 games
- `can_hu_nonempty_count >= 5` per 64 games

If all remain zero, do not train Q returns from this official league. The next exact-data path should be historical tziakcha terminal outcomes rather than non-official `scripts/selfplay_sim.py`.
