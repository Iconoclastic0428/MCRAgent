#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from rl_dataset import MCRReplayDataset  # noqa: E402
from rl_model_utils import freeze_for_mode, load_selfvec_model, save_state_dict, write_config  # noqa: E402


def default_checkpoint() -> Path:
    env_path = os.environ.get("MCR_FEATURE_AGENT_BASE16")
    if env_path:
        return Path(env_path)
    relative = Path("models") / "feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a" / "16.pkl"
    for root in (THIS_DIR, *THIS_DIR.parents):
        candidate = root / relative
        if candidate.exists():
            return candidate
    return relative


DEFAULT_CHECKPOINT = default_checkpoint()


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


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def top1_agreement(q_a: torch.Tensor, q_b: torch.Tensor) -> float:
    return float((q_a.argmax(dim=1) == q_b.argmax(dim=1)).float().mean().detach().cpu().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", nargs="+", required=True)
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha-cql", type=float, default=2.0)
    parser.add_argument("--beta-kl", type=float, default=0.5)
    parser.add_argument("--beta-bc", type=float, default=0.1)
    parser.add_argument("--temperature-kl", type=float, default=1.0)
    parser.add_argument("--legal-dueling-mean", action="store_true")
    parser.add_argument("--freeze-mode", choices=("head", "fc", "none"), default="head")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every-updates", type=int, default=1000)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    device = torch.device(args.device)

    dataset = MCRReplayDataset([Path(item) for item in args.replay])
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    train_model = load_selfvec_model(args.init_checkpoint, device=device, mixed_kernel_input=True, dueling_head=True)
    base_model = load_selfvec_model(args.init_checkpoint, device=device, mixed_kernel_input=True, dueling_head=True)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad_(False)
    trainable = freeze_for_mode(train_model, args.freeze_mode)
    optimizer = torch.optim.AdamW(
        [param for param in train_model.parameters() if param.requires_grad],
        lr=float(args.lr),
    )

    config = vars(args).copy()
    config.update(
        {
            "dataset_size": len(dataset),
            "trainable_parameters": trainable,
            "device": str(device),
        }
    )
    write_config(out_dir / "config.json", config)

    update = 0
    start = time.time()
    train_model.train()
    with metrics_path.open("a", encoding="utf-8") as metrics:
        for epoch in range(int(args.epochs)):
            for batch_index, batch in enumerate(loader):
                update += 1
                batch = batch_to_device(batch, device)
                obs = batch["obs"]
                vec = batch["vec"]
                mask = batch["mask"].bool()
                action = batch["action"].long()
                reward = batch["reward"].float()
                steps = batch["steps_to_done"].float()
                target = (float(args.gamma) ** steps) * reward

                out = train_model(
                    {
                        "is_training": True,
                        "return_raw_logits": True,
                        "legal_dueling_mean": bool(args.legal_dueling_mean),
                        "obs": {
                            "observation": obs,
                            "vec": vec,
                            "action_mask": mask.float(),
                        },
                    }
                )
                q = out["masked_q"]
                q_taken = q[torch.arange(q.shape[0], device=device), action]
                loss_q = F.smooth_l1_loss(q_taken, target)
                q_legal = q.masked_fill(~mask, -1e38)
                loss_cql = torch.logsumexp(q_legal, dim=1).mean() - q_taken.mean()

                with torch.no_grad():
                    base_out = base_model(
                        {
                            "is_training": False,
                            "return_raw_logits": True,
                            "legal_dueling_mean": bool(args.legal_dueling_mean),
                            "obs": {
                                "observation": obs,
                                "vec": vec,
                                "action_mask": mask.float(),
                            },
                        }
                    )
                    base_q = base_out["masked_q"]
                    pi_base = torch.softmax(base_q / float(args.temperature_kl), dim=1)
                log_pi_train = torch.log_softmax(q / float(args.temperature_kl), dim=1)
                loss_kl = F.kl_div(log_pi_train, pi_base, reduction="batchmean")
                loss_bc = F.cross_entropy(q, action)
                loss = (
                    loss_q
                    + float(args.alpha_cql) * loss_cql
                    + float(args.beta_kl) * loss_kl
                    + float(args.beta_bc) * loss_bc
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [param for param in train_model.parameters() if param.requires_grad],
                    float(args.grad_clip),
                )
                optimizer.step()

                if update % max(1, int(args.log_every)) == 0:
                    families = Counter(action_family(int(x)) for x in action.detach().cpu().tolist())
                    with torch.no_grad():
                        train_eval_q = q.detach()
                        base_eval_q = base_q.detach()
                    row: dict[str, Any] = {
                        "event": "metrics",
                        "epoch": epoch,
                        "batch": batch_index,
                        "update": update,
                        "loss_total": float(loss.detach().cpu().item()),
                        "loss_q": float(loss_q.detach().cpu().item()),
                        "loss_cql": float(loss_cql.detach().cpu().item()),
                        "loss_kl": float(loss_kl.detach().cpu().item()),
                        "loss_bc": float(loss_bc.detach().cpu().item()),
                        "q_taken_mean": float(q_taken.detach().mean().cpu().item()),
                        "q_legal_max_mean": float(q_legal.detach().max(dim=1).values.mean().cpu().item()),
                        "target_mean": float(target.detach().mean().cpu().item()),
                        "target_std": float(target.detach().std(unbiased=False).cpu().item()),
                        "top1_agreement_with_base": top1_agreement(train_eval_q, base_eval_q),
                        "mean_kl_to_base": float(loss_kl.detach().cpu().item()),
                        "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                        "action_category_distribution": dict(families),
                        "elapsed_s": time.time() - start,
                    }
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    metrics.write(json.dumps(row, ensure_ascii=False) + "\n")
                    metrics.flush()

                if int(args.save_every_updates) > 0 and update % int(args.save_every_updates) == 0:
                    save_state_dict(train_model, ckpt_dir / f"update_{update:09d}.pkl")

                if int(args.max_updates) > 0 and update >= int(args.max_updates):
                    save_state_dict(train_model, out_dir / "final.pkl")
                    print(json.dumps({"event": "finished", "updates": update, "elapsed_s": time.time() - start}))
                    return 0

    save_state_dict(train_model, out_dir / "final.pkl")
    print(json.dumps({"event": "finished", "updates": update, "elapsed_s": time.time() - start}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
