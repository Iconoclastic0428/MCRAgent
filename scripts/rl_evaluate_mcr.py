#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_AGENT_DIR = (
    WORKSPACE_ROOT
    / "third_party"
    / "mahjong-agent-2025-aug12-excludespecial-3pkl"
    / "feature-agent"
)
if str(WORKSPACE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))
if str(FEATURE_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_AGENT_DIR))

from benchmark_json_policies import load_initdata, placement_rewards_from_scores  # noqa: E402
from rl_collect_mcr import LoggingFeaturePolicy, run_game  # noqa: E402
from rl_model_utils import load_selfvec_model  # noqa: E402


DEFAULT_CHECKPOINT = (
    WORKSPACE_ROOT
    / "models"
    / "feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a"
    / "16.pkl"
)


def infer_hu_kind(scores: list[int | float], winner: int) -> tuple[str, int | None]:
    winner_score = scores[winner]
    loser_scores = [scores[index] for index in range(4) if index != winner]
    if len(set(loser_scores)) == 1 and winner_score == -3 * loser_scores[0]:
        return "self_draw", None
    deal_in = min((index for index in range(4) if index != winner), key=lambda index: scores[index])
    return "discard", deal_in


def ci95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    delta = 1.96 * math.sqrt(var / len(values))
    return mean - delta, mean + delta


def promotion_decision(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reasons: list[str] = []
    if bool(getattr(args, "use_paired_delta", False)):
        score_ci = payload.get("average_score_delta_ci95") or (None, None)
        lower_ci = score_ci[0] if isinstance(score_ci, (list, tuple)) and score_ci else None
        score_margin = float(args.promotion_score_margin)
        if lower_ci is None:
            reasons.append("missing_score_delta_ci95")
        elif float(lower_ci) <= score_margin:
            reasons.append("score_delta_ci95_lower_not_above_margin")

        hu_delta = payload.get("hu_rate_delta")
        if hu_delta is None:
            reasons.append("missing_hu_rate_delta")
        elif float(hu_delta) < -float(args.promotion_hu_tolerance):
            reasons.append("hu_rate_delta_below_tolerance")

        deal_in_delta = payload.get("deal_in_rate_delta")
        if deal_in_delta is None:
            reasons.append("missing_deal_in_rate_delta")
        elif float(deal_in_delta) > float(args.promotion_deal_in_tolerance):
            reasons.append("deal_in_rate_delta_above_tolerance")

        blocked_hu = payload.get("blocked_hu_rate")
        if blocked_hu is None:
            reasons.append("missing_blocked_hu_rate")
        elif float(blocked_hu) > float(args.promotion_blocked_hu_max):
            reasons.append("blocked_hu_rate_above_limit")

        invalid_response = payload.get("invalid_response_rate")
        if invalid_response is None:
            reasons.append("missing_invalid_response_rate")
        elif float(invalid_response) > float(args.promotion_invalid_response_max):
            reasons.append("invalid_response_rate_above_limit")

        return {
            "decision": "promote" if not reasons else "reject",
            "reasons": reasons,
            "gates": {
                "score_delta_ci95_lower_must_exceed": score_margin,
                "hu_rate_delta_min": -float(args.promotion_hu_tolerance),
                "deal_in_rate_delta_max": float(args.promotion_deal_in_tolerance),
                "blocked_hu_rate_max": float(args.promotion_blocked_hu_max),
                "invalid_response_rate_max": float(args.promotion_invalid_response_max),
            },
        }

    score_ci = payload.get("average_score_ci95") or (None, None)
    lower_ci = score_ci[0] if isinstance(score_ci, (list, tuple)) and score_ci else None
    score_margin = float(args.promotion_score_margin)
    if lower_ci is None:
        reasons.append("missing_score_ci95")
    elif float(lower_ci) <= score_margin:
        reasons.append("score_ci95_lower_not_above_margin")

    deal_in = payload.get("deal_in_rate")
    champion_deal_in = payload.get("champion_deal_in_rate")
    if deal_in is None or champion_deal_in is None:
        reasons.append("missing_deal_in_rate")
    elif float(deal_in) > float(champion_deal_in) + float(args.promotion_deal_in_tolerance):
        reasons.append("deal_in_rate_above_champion_tolerance")

    hu_rate = payload.get("hu_rate")
    champion_hu_rate = payload.get("champion_hu_rate")
    if hu_rate is None or champion_hu_rate is None:
        reasons.append("missing_hu_rate")
    elif float(hu_rate) < float(champion_hu_rate) - float(args.promotion_hu_tolerance):
        reasons.append("hu_rate_below_champion_tolerance")

    blocked_hu = payload.get("blocked_hu_rate")
    if blocked_hu is None:
        reasons.append("missing_blocked_hu_rate")
    elif float(blocked_hu) > float(args.promotion_blocked_hu_max):
        reasons.append("blocked_hu_rate_above_limit")

    invalid_response = payload.get("invalid_response_rate")
    if invalid_response is None:
        reasons.append("missing_invalid_response_rate")
    elif float(invalid_response) > float(args.promotion_invalid_response_max):
        reasons.append("invalid_response_rate_above_limit")

    agreement = payload.get("top1_agreement_with_base")
    if agreement is None:
        reasons.append("missing_top1_agreement")
    elif float(agreement) < float(args.promotion_min_top1_agreement):
        reasons.append("top1_agreement_below_minimum")

    return {
        "decision": "promote" if not reasons else "reject",
        "reasons": reasons,
        "gates": {
            "score_ci95_lower_must_exceed": score_margin,
            "deal_in_rate_max": (
                None
                if champion_deal_in is None
                else float(champion_deal_in) + float(args.promotion_deal_in_tolerance)
            ),
            "hu_rate_min": None if champion_hu_rate is None else float(champion_hu_rate) - float(args.promotion_hu_tolerance),
            "blocked_hu_rate_max": float(args.promotion_blocked_hu_max),
            "invalid_response_rate_max": float(args.promotion_invalid_response_max),
            "top1_agreement_min": float(args.promotion_min_top1_agreement),
        },
    }


def model_policy_stats(
    rows: list[dict[str, Any]],
    *,
    candidate_checkpoint: Path,
    base_checkpoint: Path,
    legal_dueling_mean: bool,
) -> dict[str, float | None]:
    if not rows:
        return {"top1_agreement_with_base": None, "mean_kl_to_base": None}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = load_selfvec_model(candidate_checkpoint, device=device, mixed_kernel_input=True, dueling_head=True)
    base = load_selfvec_model(base_checkpoint, device=device, mixed_kernel_input=True, dueling_head=True)
    candidate.eval()
    base.eval()
    agreements: list[float] = []
    kls: list[float] = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            obs = torch.from_numpy(np.stack([row["obs"] for row in chunk]).astype(np.float32)).to(device)
            vec = torch.from_numpy(np.stack([row["vec"] for row in chunk]).astype(np.float32)).to(device)
            mask = torch.from_numpy(np.stack([row["mask"] for row in chunk]).astype(np.float32)).to(device)
            payload = {
                "is_training": False,
                "return_raw_logits": True,
                "legal_dueling_mean": legal_dueling_mean,
                "obs": {"observation": obs, "vec": vec, "action_mask": mask},
            }
            cq = candidate(payload)["masked_q"]
            bq = base(payload)["masked_q"]
            agreements.append(float((cq.argmax(dim=1) == bq.argmax(dim=1)).float().mean().cpu().item()))
            pi_base = torch.softmax(bq, dim=1)
            log_pi_candidate = torch.log_softmax(cq, dim=1)
            kls.append(float(torch.nn.functional.kl_div(log_pi_candidate, pi_base, reduction="batchmean").cpu().item()))
    return {
        "top1_agreement_with_base": sum(agreements) / len(agreements),
        "mean_kl_to_base": sum(kls) / len(kls),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--champion-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--paired-baseline-checkpoint")
    parser.add_argument("--use-paired-delta", action="store_true")
    parser.add_argument("--raw", default=str(WORKSPACE_ROOT / "data" / "eval" / "botzone_mcr_first64_suit_permuted_384.jsonl"))
    parser.add_argument("--judge", default=str(WORKSPACE_ROOT / "build" / "official_judge" / "mcr_judge.exe"))
    parser.add_argument("--games-per-seat", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legal-dueling-mean", action="store_true")
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--promotion-score-margin", type=float, default=0.0)
    parser.add_argument("--promotion-deal-in-tolerance", type=float, default=0.005)
    parser.add_argument("--promotion-hu-tolerance", type=float, default=0.005)
    parser.add_argument("--promotion-blocked-hu-max", type=float, default=0.001)
    parser.add_argument("--promotion-invalid-response-max", type=float, default=0.0)
    parser.add_argument("--promotion-min-top1-agreement", type=float, default=0.75)
    args = parser.parse_args()

    initdata_items = load_initdata(Path(args.raw), limit=int(args.games_per_seat))
    if not initdata_items:
        raise RuntimeError(f"no initdata loaded from {args.raw}")
    start = time.time()
    candidate_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    paired_results: list[dict[str, Any]] = []
    candidate_blocked = 0
    candidate_decisions = 0
    for candidate_seat in range(4):
        policies: list[LoggingFeaturePolicy] = []
        candidate_policy: LoggingFeaturePolicy | None = None
        for seat in range(4):
            if seat == candidate_seat:
                policy = LoggingFeaturePolicy(
                    args.candidate_checkpoint,
                    record=True,
                    policy_name="candidate",
                    legal_dueling_mean=args.legal_dueling_mean,
                    seed=3000 + seat,
                )
                candidate_policy = policy
            else:
                policy = LoggingFeaturePolicy(
                    args.champion_checkpoint,
                    record=False,
                    policy_name="champion",
                    legal_dueling_mean=False,
                    seed=4000 + seat,
                )
            policies.append(policy)
        assert candidate_policy is not None
        paired_policies: list[LoggingFeaturePolicy] = []
        paired_policy: LoggingFeaturePolicy | None = None
        if args.use_paired_delta:
            paired_checkpoint = args.paired_baseline_checkpoint or args.champion_checkpoint
            for seat in range(4):
                if seat == candidate_seat:
                    policy = LoggingFeaturePolicy(
                        paired_checkpoint,
                        record=False,
                        policy_name="paired_baseline",
                        legal_dueling_mean=args.legal_dueling_mean,
                        seed=3000 + seat,
                    )
                    paired_policy = policy
                else:
                    policy = LoggingFeaturePolicy(
                        args.champion_checkpoint,
                        record=False,
                        policy_name="champion",
                        legal_dueling_mean=False,
                        seed=4000 + seat,
                    )
                paired_policies.append(policy)
            assert paired_policy is not None
        for game_index, initdata in enumerate(initdata_items):
            game_id = candidate_seat * len(initdata_items) + game_index
            rows, result = run_game(
                initdata=initdata,
                policies=policies,
                trainee=candidate_policy,
                game_id=game_id,
                judge=Path(args.judge),
                reward_scale=64.0,
                reward_clip=4.0,
                max_turns=int(args.max_turns),
            )
            candidate_rows.extend(rows)
            candidate_blocked += candidate_policy.blocked_hu_count
            candidate_policy.blocked_hu_count = 0
            candidate_decisions += len(rows)
            scores = [float(value) for value in result["scores"]]
            display = result["display"]
            action = str(display.get("action") or "UNKNOWN")
            winner = int(display["player"]) if action == "HU" and display.get("player") is not None else None
            kind = None
            deal_in = None
            if winner is not None:
                kind, deal_in = infer_hu_kind(scores, winner)
            rewards = placement_rewards_from_scores(scores)
            item = {
                "candidate_seat": candidate_seat,
                "game_index": game_index,
                "scores": scores,
                "candidate_score": scores[candidate_seat],
                "candidate_reward_4_2_1_0": rewards[candidate_seat],
                "terminal_action": action,
                "winner": winner,
                "hu_kind": kind,
                "deal_in": deal_in,
            }
            results.append(item)
            if args.use_paired_delta:
                assert paired_policy is not None
                _, baseline_result = run_game(
                    initdata=initdata,
                    policies=paired_policies,
                    trainee=paired_policy,
                    game_id=100_000_000 + game_id,
                    judge=Path(args.judge),
                    reward_scale=64.0,
                    reward_clip=4.0,
                    max_turns=int(args.max_turns),
                )
                baseline_scores = [float(value) for value in baseline_result["scores"]]
                baseline_display = baseline_result["display"]
                baseline_action = str(baseline_display.get("action") or "UNKNOWN")
                baseline_winner = (
                    int(baseline_display["player"])
                    if baseline_action == "HU" and baseline_display.get("player") is not None
                    else None
                )
                baseline_kind = None
                baseline_deal_in = None
                if baseline_winner is not None:
                    baseline_kind, baseline_deal_in = infer_hu_kind(baseline_scores, baseline_winner)
                baseline_first = (
                    baseline_scores[candidate_seat] == max(baseline_scores)
                    and baseline_scores[candidate_seat] > 0
                )
                candidate_first = scores[candidate_seat] == max(scores) and scores[candidate_seat] > 0
                paired_results.append(
                    {
                        "candidate_seat": candidate_seat,
                        "game_index": game_index,
                        "candidate_score": scores[candidate_seat],
                        "baseline_score": baseline_scores[candidate_seat],
                        "score_delta": scores[candidate_seat] - baseline_scores[candidate_seat],
                        "candidate_hu": 1 if winner == candidate_seat else 0,
                        "baseline_hu": 1 if baseline_winner == candidate_seat else 0,
                        "hu_delta": (1 if winner == candidate_seat else 0)
                        - (1 if baseline_winner == candidate_seat else 0),
                        "candidate_deal_in": 1 if kind == "discard" and deal_in == candidate_seat else 0,
                        "baseline_deal_in": (
                            1 if baseline_kind == "discard" and baseline_deal_in == candidate_seat else 0
                        ),
                        "deal_in_delta": (1 if kind == "discard" and deal_in == candidate_seat else 0)
                        - (1 if baseline_kind == "discard" and baseline_deal_in == candidate_seat else 0),
                        "candidate_first_place": 1 if candidate_first else 0,
                        "baseline_first_place": 1 if baseline_first else 0,
                        "first_place_delta": (1 if candidate_first else 0) - (1 if baseline_first else 0),
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "eval_progress",
                        "candidate_seat": candidate_seat,
                        "game_index": game_index,
                        "total_games": len(results),
                        "candidate_score": scores[candidate_seat],
                        "terminal_action": action,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    scores = [float(item["candidate_score"]) for item in results]
    rewards = [float(item["candidate_reward_4_2_1_0"]) for item in results]
    hu = 0
    self_draw = 0
    discard_hu = 0
    deal_in = 0
    champion_hu = 0
    champion_self_draw = 0
    champion_discard_hu = 0
    champion_deal_in = 0
    first_place = 0
    huang = 0
    terminal_actions = Counter()
    for item in results:
        seat = int(item["candidate_seat"])
        terminal_actions[str(item["terminal_action"])] += 1
        if item["terminal_action"] == "HUANG":
            huang += 1
        if item["scores"][seat] == max(item["scores"]) and item["scores"][seat] > 0:
            first_place += 1
        if item["terminal_action"] == "HU":
            if item["winner"] == seat:
                hu += 1
                if item["hu_kind"] == "self_draw":
                    self_draw += 1
                else:
                    discard_hu += 1
            if item["winner"] is not None and item["winner"] != seat:
                champion_hu += 1
                if item["hu_kind"] == "self_draw":
                    champion_self_draw += 1
                else:
                    champion_discard_hu += 1
            if item["hu_kind"] == "discard" and item["deal_in"] == seat:
                deal_in += 1
            if item["hu_kind"] == "discard" and item["deal_in"] is not None and item["deal_in"] != seat:
                champion_deal_in += 1
    n = len(results)
    champion_appearances = n * 3
    policy_stats = model_policy_stats(
        candidate_rows,
        candidate_checkpoint=Path(args.candidate_checkpoint),
        base_checkpoint=Path(args.champion_checkpoint),
        legal_dueling_mean=bool(args.legal_dueling_mean),
    )
    score_ci = ci95(scores)
    payload = {
        "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
        "champion_checkpoint": str(Path(args.champion_checkpoint).resolve()),
        "paired_baseline_checkpoint": (
            None
            if not args.use_paired_delta
            else str(Path(args.paired_baseline_checkpoint or args.champion_checkpoint).resolve())
        ),
        "use_paired_delta": bool(args.use_paired_delta),
        "raw": str(Path(args.raw).resolve()),
        "games": n,
        "games_per_seat_loaded": len(initdata_items),
        "average_score": sum(scores) / n if n else None,
        "average_score_ci95": score_ci,
        "average_placement_reward_4_2_1_0": sum(rewards) / n if n else None,
        "first_place_rate": first_place / n if n else None,
        "hu_rate": hu / n if n else None,
        "self_draw_rate": self_draw / n if n else None,
        "discard_hu_rate": discard_hu / n if n else None,
        "deal_in_rate": deal_in / n if n else None,
        "champion_appearances": champion_appearances,
        "champion_hu_rate": champion_hu / champion_appearances if champion_appearances else None,
        "champion_self_draw_rate": champion_self_draw / champion_appearances if champion_appearances else None,
        "champion_discard_hu_rate": champion_discard_hu / champion_appearances if champion_appearances else None,
        "champion_deal_in_rate": champion_deal_in / champion_appearances if champion_appearances else None,
        "huang_rate": huang / n if n else None,
        "blocked_hu_rate": candidate_blocked / candidate_decisions if candidate_decisions else 0.0,
        "invalid_response_rate": terminal_actions.get("WA", 0) / n if n else None,
        "terminal_actions": dict(terminal_actions),
        **policy_stats,
        "elapsed_s": time.time() - start,
        "results": results,
    }
    if args.use_paired_delta:
        score_deltas = [float(item["score_delta"]) for item in paired_results]
        hu_deltas = [float(item["hu_delta"]) for item in paired_results]
        deal_in_deltas = [float(item["deal_in_delta"]) for item in paired_results]
        first_place_deltas = [float(item["first_place_delta"]) for item in paired_results]
        paired_n = len(paired_results)
        payload.update(
            {
                "paired_games": paired_n,
                "average_score_delta": sum(score_deltas) / paired_n if paired_n else None,
                "average_score_delta_ci95": ci95(score_deltas),
                "hu_rate_delta": sum(hu_deltas) / paired_n if paired_n else None,
                "deal_in_rate_delta": sum(deal_in_deltas) / paired_n if paired_n else None,
                "first_place_rate_delta": sum(first_place_deltas) / paired_n if paired_n else None,
                "paired_results": paired_results,
            }
        )
    payload["promotion_decision"] = promotion_decision(payload, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
