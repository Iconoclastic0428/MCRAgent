"""Validate official-judge initdata JSONL before Tjong self-play."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


INITDATA_TILE_COUNTS = Counter(
    [f"{suit}{rank}" for suit in ("W", "B", "T") for rank in range(1, 10) for _ in range(4)]
    + [f"F{rank}" for rank in range(1, 5) for _ in range(4)]
    + [f"J{rank}" for rank in range(1, 4) for _ in range(4)]
    + [f"H{rank}" for rank in range(1, 9)]
)


def validate_initdata_corpus(
    path: Path,
    *,
    expected_records: int,
    summary_out: Path | None = None,
    max_error_examples: int = 20,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"initdata corpus not found: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"initdata corpus is empty: {path}")

    records = 0
    nonempty_lines = 0
    error_count = 0
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as src:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            nonempty_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                error_count += 1
                _append_error(errors, line_no, "json_decode_error", str(exc), max_error_examples)
                continue
            if not isinstance(record, dict):
                error_count += 1
                _append_error(errors, line_no, "record_not_object", type(record).__name__, max_error_examples)
                continue
            initdata = record.get("initdata")
            if not isinstance(initdata, dict):
                error_count += 1
                _append_error(errors, line_no, "missing_initdata_object", "", max_error_examples)
                continue
            record_errors = validate_initdata(initdata)
            for error in record_errors:
                error_count += 1
                _append_error(errors, line_no, error, "", max_error_examples)
            records += 1

    summary = {
        "format": "tjong_initdata_corpus_validation_v1",
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "expected_records": int(expected_records),
        "records": int(records),
        "nonempty_lines": int(nonempty_lines),
        "records_match_expected": int(records) == int(expected_records),
        "error_count": int(error_count),
        "errors": errors,
        "valid_for_selfplay_initdata": int(records) == int(expected_records) and error_count == 0,
    }
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not summary["valid_for_selfplay_initdata"]:
        raise ValueError(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def validate_initdata(initdata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        quan = int(initdata.get("quan", 0))
    except (TypeError, ValueError):
        errors.append("invalid_quan")
    else:
        if quan < 0 or quan > 3:
            errors.append("invalid_quan")
    walltiles = str(initdata.get("walltiles") or "").split()
    if len(walltiles) != 144:
        errors.append("walltiles_count_not_144")
    elif Counter(walltiles) != INITDATA_TILE_COUNTS:
        errors.append("walltiles_illegal_tile_counts")
    return errors


def _append_error(
    errors: list[dict[str, Any]],
    line_no: int,
    code: str,
    detail: str,
    max_error_examples: int,
) -> None:
    if len(errors) < max_error_examples:
        errors.append({"line": int(line_no), "code": code, "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()
    validate_initdata_corpus(
        Path(args.infile),
        expected_records=args.expected_records,
        summary_out=Path(args.summary_out) if args.summary_out else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
