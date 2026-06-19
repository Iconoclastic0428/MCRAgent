#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


FEATURE_FILES = [
    ("paixing", "paixing.npz"),
    ("hupaizuhe", "hupaizuhe.npz"),
    ("wumenqi", "\u4e94\u95e8\u9f50.npz"),
    ("hunyise", "\u6df7\u4e00\u8272.npz"),
    ("quandaiyao", "\u5168\u5e26\u5e7a.npz"),
    ("qingyise", "\u6e05\u4e00\u8272.npz"),
    ("wufanhu", "\u65e0\u756a\u548c.npz"),
    ("dayu5", "\u5927\u4e8e\u4e94.npz"),
    ("xiaoyu5", "\u5c0f\u4e8e\u4e94.npz"),
    ("tuibudao", "\u63a8\u4e0d\u5012.npz"),
    ("other_hupaizuhe", "other_hupaizuhe.npz"),
]

HAND_CHANNEL = 22

TARGETS = None
ARGS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute compact fan-searcher shanten features for feature-agent vectors."
    )
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--fan-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-chunk", type=int, default=8192)
    parser.add_argument("--pool-chunksize", type=int, default=4)
    parser.add_argument("--limit-matches", type=int, default=0)
    return parser.parse_args()


def load_targets(fan_data):
    targets = []
    fan_data = Path(fan_data)
    for feature_name, filename in FEATURE_FILES:
        path = fan_data / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        arr = np.load(path)["array"][:, 2:36].astype(np.int16, copy=False)
        targets.append((feature_name, filename, arr))
    return targets


def init_worker(args_dict):
    global TARGETS, ARGS
    ARGS = args_dict
    TARGETS = load_targets(args_dict["fan_data"])


def min_shortage_distance(hand_counts, target_counts, target_chunk):
    best = np.full((hand_counts.shape[0],), 127, dtype=np.int16)
    hands = hand_counts.astype(np.int16, copy=False)
    for start in range(0, target_counts.shape[0], target_chunk):
        target = target_counts[start : start + target_chunk]
        diff = target[None, :, :] - hands[:, None, :]
        np.maximum(diff, 0, out=diff)
        dist = diff.sum(axis=2, dtype=np.int16)
        best = np.minimum(best, dist.min(axis=1))
    return best


def compute_features(hand_counts):
    feature_columns = []
    for _, _, target_counts in TARGETS:
        feature_columns.append(
            min_shortage_distance(hand_counts, target_counts, ARGS["target_chunk"])
        )
    return np.stack(feature_columns, axis=1).astype(np.float16)


def process_match(match_id):
    data_folder = ARGS["data_folder"]
    output_dir = ARGS["output_dir"]
    batch_size = ARGS["batch_size"]

    with np.load(os.path.join(data_folder, f"{match_id}.npz")) as d:
        obs = d["obs"]
        hand_counts = obs[:, HAND_CHANNEL].reshape(obs.shape[0], 36)[:, :34].astype(
            np.int16,
            copy=False,
        )

    parts = []
    for start in range(0, hand_counts.shape[0], batch_size):
        parts.append(compute_features(hand_counts[start : start + batch_size]))
    fan_vec = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(TARGETS)))
    np.savez(os.path.join(output_dir, f"{match_id}.npz"), fan_vec=fan_vec)
    return match_id, int(fan_vec.shape[0])


def write_meta(output_dir, counts, args):
    output = Path(output_dir)
    feature_names = [name for name, _, _ in load_targets(args.fan_data)]
    meta = {
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "source_data_folder": args.data_folder,
        "fan_data": args.fan_data,
        "match_count": len(counts),
        "sample_count": sum(counts),
        "dtype": "float16",
        "meaning": "minimum positive tile shortage to fan-searcher target shapes",
    }
    (output / "count.json").write_text(json.dumps(counts), encoding="utf8")
    (output / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    (output / "DONE").write_text("ok\n", encoding="utf8")
    return meta


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.data_folder, "count.json"), "r", encoding="utf8") as f:
        counts = json.load(f)
    if args.limit_matches:
        counts = counts[: args.limit_matches]

    worker_args = {
        "data_folder": args.data_folder,
        "fan_data": args.fan_data,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "target_chunk": args.target_chunk,
    }
    match_ids = list(range(len(counts)))
    workers = max(1, min(args.workers, len(match_ids) or 1))

    if workers == 1:
        init_worker(worker_args)
        iterator = (process_match(match_id) for match_id in match_ids)
        results = list(
            tqdm(iterator, total=len(match_ids), desc="fan shanten".ljust(20))
        )
    else:
        with mp.Pool(workers, initializer=init_worker, initargs=(worker_args,)) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        process_match,
                        match_ids,
                        chunksize=max(1, args.pool_chunksize),
                    ),
                    total=len(match_ids),
                    desc="fan shanten".ljust(20),
                )
            )

    observed = [0] * len(counts)
    for match_id, sample_count in results:
        observed[match_id] = sample_count
    if observed != counts:
        raise RuntimeError({"expected": counts[:10], "observed": observed[:10]})
    meta = write_meta(args.output_dir, counts, args)
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
