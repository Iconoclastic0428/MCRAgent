"""Replay-audit the Tjong Botzone JSON wrapper against Botzone-format logs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .policy_bot import TjongCheckpointPredictor, respond_json_with_predictor


def response_family(response: str | None) -> str:
    return str(response or "PASS").strip().split()[0].upper() if str(response or "").strip() else "PASS"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for encoding in ("utf-8", "utf-16"):
        try:
            with path.open("r", encoding=encoding) as src:
                for line in src:
                    if line.strip():
                        yield json.loads(line)
            return
        except UnicodeError:
            continue


def audit_botzone_replay(
    *,
    checkpoint: Path | str,
    raw_path: Path | str,
    max_states: int = 200,
    nonpass_only: bool = False,
    device: str | None = None,
    require_encoding_version: str | None = "tjong_cit2_12298_v3_hidden_concealed_kong",
    require_paper_config: bool = False,
    example_limit: int = 50,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    raw = Path(raw_path)
    predictor = TjongCheckpointPredictor(
        checkpoint_path,
        device=device,
        require_encoding_version=require_encoding_version,
        require_paper_config=require_paper_config,
    )
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    start = time.time()
    records_read = 0

    for record in iter_jsonl(raw):
        records_read += 1
        histories = {str(player): {"requests": [], "responses": []} for player in range(4)}
        logs = record.get("logs") or []
        for index in range(0, len(logs) - 1, 2):
            output = logs[index].get("output") or {}
            content = output.get("content") or {}
            response_log = logs[index + 1]
            for player in ("0", "1", "2", "3"):
                request = content.get(player)
                response_item = response_log.get(player) or {}
                actual = str(response_item.get("response", "PASS")).strip()
                if not request:
                    continue
                histories[player]["requests"].append(str(request))
                request_kind = str(request).split()[0] if str(request).split() else ""
                if request_kind in {"2", "3"} and (not nonpass_only or response_family(actual) != "PASS"):
                    payload = {
                        "requests": list(histories[player]["requests"]),
                        "responses": list(histories[player]["responses"]),
                    }
                    try:
                        predicted = respond_json_with_predictor(payload, predictor)
                    except Exception as exc:  # Keep auditing later states while surfacing failures.
                        errors.append(
                            {
                                "match_id": record.get("match_id"),
                                "player": int(player),
                                "request": request,
                                "actual": actual,
                                "error": str(exc),
                            }
                        )
                    else:
                        counts["states"] += 1
                        actual_family = response_family(actual)
                        predicted_family = response_family(predicted)
                        counts[f"actual:{actual_family}"] += 1
                        counts[f"predicted:{predicted_family}"] += 1
                        if predicted == actual:
                            counts["exact_match"] += 1
                        if predicted_family == actual_family:
                            counts["family_match"] += 1
                        if len(examples) < example_limit or predicted_family != actual_family:
                            examples.append(
                                {
                                    "match_id": record.get("match_id"),
                                    "player": int(player),
                                    "request": request,
                                    "actual": actual,
                                    "predicted": predicted,
                                    "exact_match": predicted == actual,
                                    "family_match": predicted_family == actual_family,
                                }
                            )
                    if counts["states"] >= max_states:
                        break
                histories[player]["responses"].append(actual)
            if counts["states"] >= max_states:
                break
        if counts["states"] >= max_states:
            break

    states = int(counts["states"])
    return {
        "format": "tjong_botzone_wrapper_training_replay_audit_v1",
        "checkpoint": str(checkpoint_path),
        "raw": str(raw),
        "records_read": records_read,
        "max_states": int(max_states),
        "nonpass_only": bool(nonpass_only),
        "states": states,
        "exact_match": int(counts["exact_match"]),
        "exact_accuracy": counts["exact_match"] / states if states else None,
        "family_match": int(counts["family_match"]),
        "family_accuracy": counts["family_match"] / states if states else None,
        "counts": dict(sorted(counts.items())),
        "errors": errors[:example_limit],
        "error_count": len(errors),
        "examples": examples[:example_limit],
        "elapsed_s": time.time() - start,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--max-states", type=int, default=200)
    parser.add_argument("--nonpass-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--require-encoding-version",
        default="tjong_cit2_12298_v3_hidden_concealed_kong",
    )
    parser.add_argument("--require-paper-config", action="store_true")
    parser.add_argument("--example-limit", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    summary = audit_botzone_replay(
        checkpoint=args.checkpoint,
        raw_path=args.raw,
        max_states=args.max_states,
        nonpass_only=args.nonpass_only,
        device=args.device,
        require_encoding_version=args.require_encoding_version or None,
        require_paper_config=args.require_paper_config,
        example_limit=args.example_limit,
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
