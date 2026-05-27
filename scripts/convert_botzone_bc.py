#!/usr/bin/env python3
"""Convert Botzone MCR replay logs into behavior-cloning examples."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def action_type(response: Any) -> str:
    if response is None:
        return "MISSING"
    if not isinstance(response, str):
        response = str(response)
    response = response.strip()
    if not response:
        return "EMPTY"
    return response.split()[0].upper()


def iter_examples(record: dict) -> list[dict]:
    logs = record.get("logs") or []
    examples: list[dict] = []
    final_scores = record.get("scores")
    histories: dict[str, list[str]] = {}

    for turn in range(0, len(logs) - 1, 2):
        request_log = logs[turn]
        response_log = logs[turn + 1]
        output = request_log.get("output") or {}
        requests = output.get("content") or {}
        display = output.get("display") or {}

        for player, request in requests.items():
            player = str(player)
            history = histories.setdefault(player, [])
            response_item = response_log.get(str(player), {})
            response = response_item.get("raw") or response_item.get("response")
            examples.append(
                {
                    "match_id": record["match_id"],
                    "turn_index": turn // 2,
                    "player": int(player),
                    "input_text": "\n".join([*history, f"REQ {request}"]),
                    "request": request,
                    "response": response,
                    "action_type": action_type(response),
                    "judge_action": display.get("action"),
                    "judge_player": display.get("player"),
                    "judge_tile": display.get("tile"),
                    "can_hu": display.get("canHu"),
                    "tile_count": display.get("tileCnt"),
                    "final_score": None if final_scores is None else final_scores.get(str(player)),
                }
            )

        for player, request in requests.items():
            player = str(player)
            response_item = response_log.get(str(player), {})
            response = response_item.get("raw") or response_item.get("response")
            history = histories.setdefault(player, [])
            history.append(f"REQ {request}")
            if response is not None:
                history.append(f"RES {response}")
    return examples


def convert(in_path: Path, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matches = 0
    examples = 0
    actions: collections.Counter[str] = collections.Counter()
    judge_actions: collections.Counter[str] = collections.Counter()
    parse_errors: list[dict] = []

    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as out:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                match_examples = iter_examples(record)
            except Exception as exc:
                parse_errors.append({"line": line_no, "error": str(exc)})
                continue

            matches += 1
            for example in match_examples:
                out.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
                examples += 1
                actions[example["action_type"]] += 1
                judge_actions[str(example["judge_action"])] += 1

    return {
        "input": str(in_path),
        "output": str(out_path),
        "matches": matches,
        "examples": examples,
        "action_counts": dict(actions.most_common()),
        "judge_action_counts": dict(judge_actions.most_common()),
        "parse_errors": parse_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="data/raw/botzone_mcr_sample.jsonl")
    parser.add_argument("--out", default="data/processed/botzone_mcr_bc_sample.jsonl")
    args = parser.parse_args()

    summary = convert(Path(args.infile), Path(args.out))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["parse_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
