"""Audit whether local Tziakcha sources are covered by the Botzone merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REAL_BOTZONE_INPUTS = (
    "data/raw/tziakcha_all_plus_live_elo2300_botzone.jsonl",
    "data/raw/tziakcha_human_botzone_raw_32.jsonl",
    "data/raw/tziakcha_human_botzone_raw_517.jsonl",
    "data/raw/tziakcha_chaga0208_botzone_raw.jsonl",
)

SMALL_RAW_PAIRS = (
    ("data/raw/tziakcha_human_records_32.jsonl", "data/raw/tziakcha_human_botzone_raw_32.jsonl"),
    ("data/raw/tziakcha_human_records_256.jsonl", "data/raw/tziakcha_human_botzone_raw_517.jsonl"),
    ("data/raw/tziakcha_human_records_512.jsonl", "data/raw/tziakcha_human_botzone_raw_517.jsonl"),
    ("data/raw/tziakcha_chaga0208_records.jsonl", "data/raw/tziakcha_chaga0208_botzone_raw.jsonl"),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _jsonl_ids(path: Path, key: str) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get(key) is not None:
                ids.add(str(record[key]))
    return ids


def _input_by_path(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inputs = summary.get("inputs", [])
    if not isinstance(inputs, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in inputs:
        if isinstance(item, dict) and item.get("path") is not None:
            result[str(item["path"]).replace("\\", "/")] = item
    return result


def audit_coverage(
    *,
    repo_root: Path,
    merge_summary: Path,
    validation_summary: Path | None,
    summary_out: Path | None = None,
) -> dict[str, Any]:
    merge = _load_json(merge_summary)
    validation = _load_json(validation_summary) if validation_summary and validation_summary.exists() else None
    merged_inputs = _input_by_path(merge)

    required_input_checks = []
    for expected_path in REAL_BOTZONE_INPUTS:
        input_summary = merged_inputs.get(expected_path)
        required_input_checks.append(
            {
                "path": expected_path,
                "present_in_merge": input_summary is not None,
                "records_seen": int(input_summary.get("records_seen", 0)) if input_summary else None,
                "records_written": int(input_summary.get("records_written", 0)) if input_summary else None,
                "duplicates_skipped": int(input_summary.get("duplicates_skipped", 0)) if input_summary else None,
            }
        )

    small_pair_checks = []
    for raw_rel, botzone_rel in SMALL_RAW_PAIRS:
        raw = repo_root / raw_rel
        botzone = repo_root / botzone_rel
        check: dict[str, Any] = {
            "raw": raw_rel,
            "botzone": botzone_rel,
            "raw_exists": raw.exists(),
            "botzone_exists": botzone.exists(),
        }
        if raw.exists() and botzone.exists():
            raw_ids = _jsonl_ids(raw, "id")
            botzone_ids = _jsonl_ids(botzone, "match_id")
            missing = sorted(raw_ids - botzone_ids)
            check.update(
                {
                    "raw_unique_ids": len(raw_ids),
                    "botzone_unique_ids": len(botzone_ids),
                    "missing_from_botzone": len(missing),
                    "missing_examples": missing[:10],
                }
            )
        small_pair_checks.append(check)

    synthetic = repo_root / "data/raw/tziakcha_human_botzone_raw_517_suitaug6.jsonl"
    synthetic_summary: dict[str, Any] = {
        "path": "data/raw/tziakcha_human_botzone_raw_517_suitaug6.jsonl",
        "exists": synthetic.exists(),
        "included_in_merge": "data/raw/tziakcha_human_botzone_raw_517_suitaug6.jsonl" in merged_inputs,
        "reason_excluded": "suit-permuted augmentation is synthetic, not a real replay source",
    }
    if synthetic.exists():
        synthetic_summary["bytes"] = int(synthetic.stat().st_size)

    validation_ok = True
    if validation is not None:
        validation_ok = bool(validation.get("valid_for_paper_supervised")) and int(validation.get("error_count", -1)) == 0

    passed = (
        all(item["present_in_merge"] for item in required_input_checks)
        and all(
            item.get("raw_exists")
            and item.get("botzone_exists")
            and int(item.get("missing_from_botzone", 1)) == 0
            for item in small_pair_checks
        )
        and not synthetic_summary["included_in_merge"]
        and validation_ok
        and int(merge.get("error_count", -1)) == 0
    )

    summary = {
        "format": "tjong_tziakcha_botzone_coverage_audit_v1",
        "merge_summary": str(merge_summary),
        "validation_summary": str(validation_summary) if validation_summary else None,
        "passed": bool(passed),
        "records_written": int(merge.get("records_written", 0)),
        "duplicates_skipped": int(merge.get("duplicates_skipped", 0)),
        "merge_error_count": int(merge.get("error_count", 0)),
        "validation_error_count": int(validation.get("error_count", 0)) if validation else None,
        "required_real_botzone_inputs": required_input_checks,
        "small_raw_pair_coverage": small_pair_checks,
        "synthetic_exclusion": synthetic_summary,
    }
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--merge-summary", required=True, type=Path)
    parser.add_argument("--validation-summary", default=None, type=Path)
    parser.add_argument("--summary-out", default=None, type=Path)
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()
    summary = audit_coverage(
        repo_root=args.repo_root,
        merge_summary=args.merge_summary,
        validation_summary=args.validation_summary,
        summary_out=args.summary_out,
    )
    if args.fail_on_missing and not summary["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
