#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import collections
import json
import os

import numpy as np
from tqdm import tqdm


ACTION_CATEGORIES = (
    ("pass", 0, 1),
    ("hu", 1, 2),
    ("discard", 2, 36),
    ("kong", 36, 138),
    ("pong", 138, 172),
    ("chow", 172, 235),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit feature-agent vector npz data.")
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--split-mode", choices=("contiguous", "random"), default="contiguous")
    parser.add_argument("--seed", type=int, default=6088)
    parser.add_argument("--max-matches", type=int, default=0)
    return parser.parse_args()


def action_family(action):
    for name, start, end in ACTION_CATEGORIES:
        if start <= action < end:
            return name
    return "unknown"


def build_splits(counts, split_ratio, test_ratio, split_mode, seed):
    validation_end = 1.0 - test_ratio
    if not 0 < split_ratio < validation_end <= 1.0:
        raise RuntimeError(
            f"Invalid split: split_ratio={split_ratio}, test_ratio={test_ratio}"
        )
    match_ids = np.arange(len(counts), dtype=np.int64)
    if split_mode == "random":
        rng = np.random.default_rng(seed)
        match_ids = rng.permutation(match_ids)
    train_end = int(split_ratio * len(match_ids))
    validation_stop = int(validation_end * len(match_ids))
    return {
        "train": match_ids[:train_end].tolist(),
        "validate": match_ids[train_end:validation_stop].tolist(),
        "test": match_ids[validation_stop:].tolist(),
    }


def empty_stats():
    return {
        "matches": 0,
        "samples": 0,
        "single_action_samples": 0,
        "label_outside_mask": collections.Counter(),
        "action_family": collections.Counter(),
        "candidate_count": collections.Counter(),
        "samples_per_match": collections.Counter(),
        "shape_errors": [],
    }


def audit_split(data_folder, split_name, match_ids, max_matches):
    stats = empty_stats()
    selected = match_ids[:max_matches] if max_matches else match_ids
    for match_id in tqdm(selected, desc=f"audit {split_name}".ljust(20)):
        path = os.path.join(data_folder, f"{match_id}.npz")
        with np.load(path) as d:
            obs = d["obs"]
            mask = d["mask"]
            act = d["act"]
            vec = d["vec"]
        if tuple(obs.shape[1:]) != (85, 4, 9):
            stats["shape_errors"].append(
                {"match_id": int(match_id), "key": "obs", "shape": list(obs.shape)}
            )
        if mask.ndim != 2 or mask.shape[1] != 235:
            stats["shape_errors"].append(
                {"match_id": int(match_id), "key": "mask", "shape": list(mask.shape)}
            )
        if vec.ndim != 2 or vec.shape[1] != 117:
            stats["shape_errors"].append(
                {"match_id": int(match_id), "key": "vec", "shape": list(vec.shape)}
            )
        sample_count = int(act.shape[0])
        stats["matches"] += 1
        stats["samples"] += sample_count
        stats["samples_per_match"][sample_count] += 1
        candidate_counts = mask.sum(axis=1).astype(np.int64)
        for candidate_count, count in collections.Counter(candidate_counts.tolist()).items():
            stats["candidate_count"][int(candidate_count)] += int(count)
        stats["single_action_samples"] += int((candidate_counts <= 1).sum())
        for action in act.astype(np.int64).tolist():
            stats["action_family"][action_family(int(action))] += 1
        rows = np.arange(sample_count)
        outside = mask[rows, act.astype(np.int64)] <= 0
        if np.any(outside):
            for action in act[outside].astype(np.int64).tolist():
                stats["label_outside_mask"][action_family(int(action))] += 1
    return stats


def serializable_stats(stats):
    out = dict(stats)
    for key in (
        "label_outside_mask",
        "action_family",
        "candidate_count",
        "samples_per_match",
    ):
        out[key] = dict(sorted((str(k), int(v)) for k, v in stats[key].items()))
    return out


def main():
    args = parse_args()
    count_path = os.path.join(args.data_folder, "count.json")
    with open(count_path, "r", encoding="utf8") as f:
        counts = json.load(f)
    splits = build_splits(
        counts,
        args.split_ratio,
        args.test_ratio,
        args.split_mode,
        args.seed,
    )
    result = {
        "data_folder": args.data_folder,
        "total_matches": len(counts),
        "total_samples": int(sum(counts)),
        "split_mode": args.split_mode,
        "split_seed": args.seed,
        "split_ratio": args.split_ratio,
        "test_ratio": args.test_ratio,
        "max_matches": args.max_matches,
        "splits": {},
    }
    for name, match_ids in splits.items():
        if not match_ids:
            continue
        stats = audit_split(args.data_folder, name, match_ids, args.max_matches)
        result["splits"][name] = {
            "match_id_head": [int(x) for x in match_ids[:5]],
            "match_id_tail": [int(x) for x in match_ids[-5:]],
            **serializable_stats(stats),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
