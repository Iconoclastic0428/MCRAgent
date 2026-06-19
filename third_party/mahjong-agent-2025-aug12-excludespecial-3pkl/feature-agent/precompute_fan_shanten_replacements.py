#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from precompute_fan_shanten_features import HAND_CHANNEL, load_targets


UNKNOWN_CHANNEL = 21
VEC_DIM = 117
VEC_SHAN_EXP_PASS = 47
VEC_SHAN_EXP_PLAY = 48
VEC_SHAN_DIS_PASS = 82
VEC_SHAN_DIS_PLAY = 83

TARGET_COUNTS = None
ARGS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute fan-searcher replacements for SHAN_EXP/DIS vector fields."
    )
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--fan-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-chunk", type=int, default=2048)
    parser.add_argument("--pool-chunksize", type=int, default=4)
    parser.add_argument("--limit-matches", type=int, default=0)
    return parser.parse_args()


def init_worker(args_dict):
    global TARGET_COUNTS, ARGS
    ARGS = args_dict
    targets = load_targets(args_dict["fan_data"])
    TARGET_COUNTS = np.concatenate([arr for _, _, arr in targets], axis=0).astype(
        np.int16,
        copy=False,
    )


def distance_and_useful(states, target_counts, target_chunk):
    states = states.astype(np.int16, copy=False)
    best = np.full((states.shape[0],), 127, dtype=np.int16)
    useful = np.zeros((states.shape[0], 34), dtype=bool)
    for start in range(0, target_counts.shape[0], target_chunk):
        target = target_counts[start : start + target_chunk]
        diff = target[None, :, :] - states[:, None, :]
        np.maximum(diff, 0, out=diff)
        dist = diff.sum(axis=2, dtype=np.int16)
        chunk_best = dist.min(axis=1)
        new_best = np.minimum(best, chunk_best)
        active = chunk_best <= best
        if np.any(active):
            exact = dist[active] == new_best[active, None]
            deficits = target[None, :, :] > states[active, None, :]
            chunk_useful = np.any(deficits & exact[:, :, None], axis=1)
            lower = active.copy()
            lower[active] = chunk_best[active] < best[active]
            equal = active & ~lower
            if np.any(lower):
                useful[lower] = chunk_useful[chunk_best[active] < best[active]]
            if np.any(equal):
                useful[equal] |= chunk_useful[chunk_best[active] == best[active]]
        best = new_best
    return best, useful


def replacement_batch(hand_counts, unknown_counts, base_vec):
    out = base_vec.astype(np.float16, copy=True)
    out[:, VEC_SHAN_EXP_PASS:] = 10.0
    states = []
    refs = []
    pass_valid = base_vec[:, VEC_SHAN_DIS_PASS] < 10
    for row in np.flatnonzero(pass_valid):
        states.append(hand_counts[row])
        refs.append((int(row), "pass", -1))
    play_valid = base_vec[:, VEC_SHAN_DIS_PLAY : VEC_SHAN_DIS_PLAY + 34] < 10
    for row, tile in zip(*np.nonzero(play_valid)):
        state = hand_counts[row].copy()
        if state[tile] <= 0:
            continue
        state[tile] -= 1
        states.append(state)
        refs.append((int(row), "play", int(tile)))
    if not states:
        return out

    state_array = np.stack(states).astype(np.int16, copy=False)
    dist, useful = distance_and_useful(
        state_array,
        TARGET_COUNTS,
        int(ARGS["target_chunk"]),
    )
    for state_index, (row, kind, tile) in enumerate(refs):
        denom = float(np.sum(unknown_counts[row]))
        useful_prob = 0.0
        if denom > 0:
            useful_prob = float(np.sum(unknown_counts[row] * useful[state_index])) / denom
        expected = float(dist[state_index]) - useful_prob
        if kind == "pass":
            out[row, VEC_SHAN_DIS_PASS] = dist[state_index]
            if dist[state_index] > 0:
                out[row, VEC_SHAN_EXP_PASS] = expected
        else:
            out[row, VEC_SHAN_DIS_PLAY + tile] = dist[state_index]
            if dist[state_index] > 0:
                out[row, VEC_SHAN_EXP_PLAY + tile] = expected
    return out


def process_match(match_id):
    data_folder = ARGS["data_folder"]
    output_dir = ARGS["output_dir"]
    batch_size = int(ARGS["batch_size"])
    with np.load(os.path.join(data_folder, f"{match_id}.npz")) as d:
        obs = d["obs"]
        base_vec = d["vec"]
        hand_counts = obs[:, HAND_CHANNEL].reshape(obs.shape[0], 36)[:, :34].astype(
            np.int16,
            copy=False,
        )
        unknown_counts = obs[:, UNKNOWN_CHANNEL].reshape(obs.shape[0], 36)[:, :34].astype(
            np.float32,
            copy=False,
        )

    parts = []
    for start in range(0, hand_counts.shape[0], batch_size):
        parts.append(
            replacement_batch(
                hand_counts[start : start + batch_size],
                unknown_counts[start : start + batch_size],
                base_vec[start : start + batch_size],
            )
        )
    vec = np.concatenate(parts, axis=0) if parts else np.zeros((0, VEC_DIM), dtype=np.float16)
    np.savez(os.path.join(output_dir, f"{match_id}.npz"), vec=vec)
    return match_id, int(vec.shape[0])


def write_meta(output_dir, counts, args):
    output = Path(output_dir)
    meta = {
        "format": "fan_searcher_vec_replacement_v1",
        "vec_dim": VEC_DIM,
        "source_data_folder": args.data_folder,
        "fan_data": args.fan_data,
        "match_count": len(counts),
        "sample_count": sum(counts),
        "dtype": "float16",
        "meaning": "full 117-dim vec with fan-searcher min-shortage values replacing SHAN_EXP_PASS, SHAN_EXP_PLAY, SHAN_DIS_PASS, SHAN_DIS_PLAY",
        "slots": {
            "shan_exp_pass": [VEC_SHAN_EXP_PASS, VEC_SHAN_EXP_PASS + 1],
            "shan_exp_play": [VEC_SHAN_EXP_PLAY, VEC_SHAN_EXP_PLAY + 34],
            "shan_dis_pass": [VEC_SHAN_DIS_PASS, VEC_SHAN_DIS_PASS + 1],
            "shan_dis_play": [VEC_SHAN_DIS_PLAY, VEC_SHAN_DIS_PLAY + 34],
        },
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
            tqdm(iterator, total=len(match_ids), desc="fan shan replace".ljust(20))
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
                    desc="fan shan replace".ljust(20),
                )
            )

    observed = [0] * len(counts)
    for match_id, sample_count in results:
        observed[match_id] = sample_count
    if observed != counts:
        raise RuntimeError({"expected": counts[:10], "observed": observed[:10]})
    meta = write_meta(Path(args.output_dir), counts, args)
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
