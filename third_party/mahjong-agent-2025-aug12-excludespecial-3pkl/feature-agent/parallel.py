#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import json
import math
import multiprocessing
import os
import subprocess
import sys


def find_match_lines(file_path):
    match_lines = []
    total_lines = -1
    with open(file_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if line.startswith("Match"):
                match_lines.append(i)
            total_lines = i
    total_lines += 1
    return match_lines, total_lines


def parallel_process(
    python_exe,
    file_path,
    start_line,
    end_line,
    offset,
    cpu_id,
    output_dir,
    match_per_process,
):
    subprocess.run(
        [
            python_exe,
            "preprocess.py",
            file_path,
            str(start_line),
            "None" if end_line is None else str(end_line),
            str(offset),
            str(cpu_id),
            output_dir,
            str(match_per_process),
        ],
        check=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Vectorize a feature-agent match transcript.")
    parser.add_argument("--input", default="./data/sample.txt")
    parser.add_argument("--output-dir", default="./data-vec")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.input
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    match_lines, total_lines = find_match_lines(file_path)
    total_matches = len(match_lines)
    if total_matches == 0:
        raise RuntimeError(f"No Match lines found in {file_path}")
    print(total_matches, total_lines)

    num_cpus = args.workers or multiprocessing.cpu_count()
    num_cpus = max(1, min(num_cpus, total_matches))
    if total_matches <= 20:
        num_cpus = 1

    match_per_process = math.ceil(total_matches / num_cpus)
    chunk_size = num_cpus * match_per_process

    total_chunks = total_matches // chunk_size + (
        1 if total_matches % chunk_size != 0 else 0
    )

    for chunk_id in range(total_chunks):
        start_index = chunk_id * chunk_size
        processes = []

        for i in range(num_cpus):
            match_index = start_index + i * match_per_process
            if match_index >= total_matches:
                break
            start_line = match_lines[match_index]
            next_match_index = start_index + (i + 1) * match_per_process
            end_line = (
                match_lines[next_match_index] - 1
                if next_match_index < total_matches
                else total_lines
            )

            cpu_id = chunk_id * num_cpus + i
            p = multiprocessing.Process(
                target=parallel_process,
                args=(
                    args.python,
                    file_path,
                    start_line,
                    end_line,
                    match_index,
                    cpu_id,
                    output_dir,
                    match_per_process,
                ),
            )
            processes.append((cpu_id, p))
            p.start()

        for cpu_id, p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker {cpu_id} failed with exit code {p.exitcode}")
            print(f"process {cpu_id} finished")

    total_out = []
    for chunk_id in range(total_chunks):
        for i in range(num_cpus):
            try:
                with open(f"{output_dir}/count-{chunk_id * num_cpus + i}.json", "r") as f:
                    total_out += json.load(f)
            except FileNotFoundError:
                continue

    with open(f"{output_dir}/count.json", "w") as f:
        json.dump(total_out, f)
