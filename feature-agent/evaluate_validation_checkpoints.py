import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from feature import FeatureAgent
from model import SelfVecModel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate feature-agent checkpoints on the validation split.")
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--fan-features-folder", default="")
    parser.add_argument("--load-workers", type=int, default=24)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def checkpoint_sort_key(path):
    if path.name == "final.pkl":
        return (10**9, path.name)
    try:
        return (int(path.stem), path.name)
    except ValueError:
        return (10**8, path.name)


def checkpoint_label(path):
    if path.name == "final.pkl":
        return {
            "checkpoint": path.name,
            "saved_point": "after_epoch_29",
            "completed_epoch": 29,
        }
    epoch = int(path.stem)
    return {
        "checkpoint": path.name,
        "saved_point": f"before_epoch_{epoch}",
        "completed_epoch": epoch - 1,
    }


def load_validation_arrays(data_folder, split_ratio, load_workers):
    data_folder = Path(data_folder)
    counts = json.loads((data_folder / "count.json").read_text(encoding="utf-8"))
    begin = int(len(counts) * split_ratio)
    end = len(counts)
    match_ids = list(range(begin, end))

    def load_one(match_id):
        with np.load(data_folder / f"{match_id}.npz") as data:
            return (
                data["obs"],
                data["mask"],
                data["vec"],
                data["act"],
            )

    arrays = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_workers) as executor:
        for item in tqdm(
            executor.map(load_one, match_ids),
            total=len(match_ids),
            desc="loading validation".ljust(20),
            bar_format="{l_bar:20}{bar:40}{r_bar}",
        ):
            arrays.append(item)

    obs = np.concatenate([item[0] for item in arrays], axis=0)
    mask = np.concatenate([item[1] for item in arrays], axis=0)
    vec = np.concatenate([item[2] for item in arrays], axis=0)
    act = np.concatenate([item[3] for item in arrays], axis=0)
    return obs, mask, vec, act


def evaluate_one(model, arrays, batch_size, device, max_val_batches=0):
    model.eval()
    obs_array, mask_array, vec_array, act_array = arrays
    total_loss = 0.0
    total = 0
    correct = 0
    sample_count = len(act_array)
    batch_starts = range(0, sample_count, batch_size)
    with torch.inference_mode():
        for batch_index, start in enumerate(
            tqdm(batch_starts, desc="validation".ljust(20), bar_format="{l_bar:20}{bar:40}{r_bar}")
        ):
            end = min(start + batch_size, sample_count)
            input_dict = {
                "is_training": False,
                "obs": {
                    "observation": torch.from_numpy(obs_array[start:end]).to(device),
                    "action_mask": torch.from_numpy(mask_array[start:end]).to(device),
                    "vec": torch.from_numpy(vec_array[start:end]).to(device),
                },
            }
            target = torch.from_numpy(act_array[start:end]).long().to(device)
            logits = model(input_dict)
            loss = F.cross_entropy(logits, target)
            current_batch_size = end - start
            total_loss += loss.item() * current_batch_size
            pred = logits.argmax(dim=1)
            correct += int(pred.eq(target).sum().item())
            total += current_batch_size
            if max_val_batches and batch_index + 1 >= max_val_batches:
                break
    return {
        "validation_loss": total_loss / total,
        "validation_acc": correct / total,
        "validation_samples": total,
    }


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    arrays = load_validation_arrays(
        args.data_folder,
        args.split_ratio,
        args.load_workers,
    )
    vec_size = arrays[2].shape[1]
    print(
        "validation_cache "
        + json.dumps(
            {
                "samples": int(arrays[3].shape[0]),
                "obs_shape": list(arrays[0].shape),
                "mask_shape": list(arrays[1].shape),
                "vec_shape": list(arrays[2].shape),
                "vec_size": int(vec_size),
                "bytes": int(sum(array.nbytes for array in arrays)),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*.pkl"), key=checkpoint_sort_key)
    checkpoints = [path for path in checkpoints if path.name == "final.pkl" or path.stem.isdigit()]
    if args.max_checkpoints:
        checkpoints = checkpoints[: args.max_checkpoints]

    model = SelfVecModel(
        obs_dim=FeatureAgent.OBS_SIZE,
        vec_dim=vec_size,
        hidden=args.hidden,
        num_blocks=args.num_blocks,
    ).to(device)

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for path in checkpoints:
        state = torch.load(path, map_location=device)
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state)
        row = checkpoint_label(path)
        row.update(evaluate_one(model, arrays, args.batch_size, device, args.max_val_batches))
        rows.append(row)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("checkpoint_metrics " + json.dumps(row, sort_keys=True), flush=True)

    best_acc = max(rows, key=lambda item: item["validation_acc"]) if rows else None
    best_loss = min(rows, key=lambda item: item["validation_loss"]) if rows else None
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "data_folder": args.data_folder,
        "rows": rows,
        "best_validation_acc": best_acc,
        "best_validation_loss": best_loss,
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("validation_sweep_summary " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"wrote {out_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
