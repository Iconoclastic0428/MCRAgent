#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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

from rl_model_utils import load_selfvec_model  # noqa: E402


def action_family(action: int) -> str:
    if action == 0:
        return "Pass"
    if action == 1:
        return "Hu"
    if 2 <= action < 36:
        return "Discard"
    if 36 <= action < 138:
        return "Kong"
    if 138 <= action < 172:
        return "Pong"
    if 172 <= action < 235:
        return "Chow"
    return "Unknown"


def iter_shards(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.npz"))


def as_float(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legal-dueling-mean", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    replay = Path(args.replay)
    shards = iter_shards(replay)
    if not shards:
        raise RuntimeError(f"no replay shards found under {replay}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = load_selfvec_model(args.candidate_checkpoint, device=device, mixed_kernel_input=True, dueling_head=True)
    base = load_selfvec_model(args.base_checkpoint, device=device, mixed_kernel_input=True, dueling_head=True)
    candidate.eval()
    base.eval()

    total = 0
    changed = 0
    agreements: list[float] = []
    kls: list[float] = []
    family_confusion: Counter[str] = Counter()
    changed_terminal_scores: list[float] = []
    unchanged_terminal_scores: list[float] = []
    changed_hu: list[float] = []
    unchanged_hu: list[float] = []
    changed_deal_in: list[float] = []
    unchanged_deal_in: list[float] = []
    changed_top1_minus_base_action: list[float] = []
    changed_top1_minus_second: list[float] = []
    base_action_margins_on_changed: list[float] = []

    remaining = None if args.max_rows is None else int(args.max_rows)
    with torch.no_grad():
        for shard in shards:
            data = np.load(shard)
            size = int(data["action"].shape[0])
            if remaining is not None:
                size = min(size, remaining)
            for start in range(0, size, int(args.batch_size)):
                stop = min(size, start + int(args.batch_size))
                obs_np = data["obs"][start:stop].astype(np.float32)
                vec_np = data["vec"][start:stop].astype(np.float32)
                mask_np = data["mask"][start:stop].astype(np.float32)
                terminal_score_np = data["terminal_score"][start:stop].astype(np.float32)
                terminal_tag_np = data["terminal_tag"][start:stop].astype(np.int64)
                obs = torch.from_numpy(obs_np).to(device)
                vec = torch.from_numpy(vec_np).to(device)
                mask = torch.from_numpy(mask_np).to(device)
                payload = {
                    "is_training": False,
                    "return_raw_logits": True,
                    "legal_dueling_mean": bool(args.legal_dueling_mean),
                    "obs": {"observation": obs, "vec": vec, "action_mask": mask},
                }
                cq = candidate(payload)["masked_q"]
                bq = base(payload)["masked_q"]
                candidate_action = cq.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                base_action = bq.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                same = candidate_action == base_action
                agreements.append(float(np.mean(same)))
                pi_base = torch.softmax(bq, dim=1)
                log_pi_candidate = torch.log_softmax(cq, dim=1)
                kls.append(float(torch.nn.functional.kl_div(log_pi_candidate, pi_base, reduction="batchmean").cpu().item()))

                cq_np = cq.detach().cpu().numpy()
                bq_np = bq.detach().cpu().numpy()
                for row_index, is_same in enumerate(same):
                    terminal_score = float(terminal_score_np[row_index])
                    terminal_tag = int(terminal_tag_np[row_index])
                    hu_value = 1.0 if terminal_tag in (1, 2) else 0.0
                    deal_in_value = 1.0 if terminal_tag == 3 else 0.0
                    if is_same:
                        unchanged_terminal_scores.append(terminal_score)
                        unchanged_hu.append(hu_value)
                        unchanged_deal_in.append(deal_in_value)
                    else:
                        changed += 1
                        changed_terminal_scores.append(terminal_score)
                        changed_hu.append(hu_value)
                        changed_deal_in.append(deal_in_value)
                        base_act = int(base_action[row_index])
                        cand_act = int(candidate_action[row_index])
                        family_confusion[f"{action_family(base_act)}->{action_family(cand_act)}"] += 1
                        cand_values = cq_np[row_index]
                        base_values = bq_np[row_index]
                        top_two = np.partition(cand_values, -2)[-2:]
                        changed_top1_minus_second.append(float(top_two[-1] - top_two[-2]))
                        changed_top1_minus_base_action.append(float(cand_values[cand_act] - cand_values[base_act]))
                        base_top = int(np.argmax(base_values))
                        base_sorted = np.partition(base_values, -2)[-2:]
                        if base_top == base_act:
                            base_action_margins_on_changed.append(float(base_sorted[-1] - base_sorted[-2]))
                        else:
                            base_action_margins_on_changed.append(float(base_values[base_act] - base_values[base_top]))
                batch_count = stop - start
                total += batch_count
                if remaining is not None:
                    remaining -= batch_count
                    if remaining <= 0:
                        break
            data.close()
            if remaining is not None and remaining <= 0:
                break

    unchanged = total - changed
    payload: dict[str, Any] = {
        "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "replay": str(replay.resolve()),
        "rows": total,
        "top1_agreement": 1.0 - (changed / total) if total else None,
        "top1_agreement_batch_mean": as_float(agreements),
        "mean_kl": as_float(kls),
        "changed_state_count": changed,
        "unchanged_state_count": unchanged,
        "changed_fraction": changed / total if total else None,
        "family_confusion": dict(sorted(family_confusion.items())),
        "changed_terminal_score_mean": as_float(changed_terminal_scores),
        "unchanged_terminal_score_mean": as_float(unchanged_terminal_scores),
        "changed_hu_rate": as_float(changed_hu),
        "unchanged_hu_rate": as_float(unchanged_hu),
        "changed_deal_in_rate": as_float(changed_deal_in),
        "unchanged_deal_in_rate": as_float(unchanged_deal_in),
        "changed_candidate_top1_minus_base_action_q_mean": as_float(changed_top1_minus_base_action),
        "changed_candidate_top1_minus_second_q_mean": as_float(changed_top1_minus_second),
        "base_action_margin_on_changed_states_mean": as_float(base_action_margins_on_changed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
