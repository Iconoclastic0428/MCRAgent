"""Merge Botzone-format JSONL corpora with validation and duplicate accounting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .validate_paper_corpus import terminal_action, validate_record_shape


def merge_botzone_corpora(
    *,
    inputs: list[Path],
    out_path: Path,
    summary_out: Path | None = None,
    dedupe_key: str | None = "match_id",
    overwrite: bool = False,
    min_written: int = 0,
    fail_on_error: bool = False,
    max_error_examples: int = 20,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input corpus is required")
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"input corpus not found: {path}")
        if path.resolve() == out_path.resolve():
            raise ValueError(f"output path cannot also be an input: {path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    seen_keys: set[str] = set()
    input_summaries: list[dict[str, Any]] = []
    first_record_keys: list[str] = []
    terminal_actions: dict[str, int] = {}
    records_seen = 0
    records_written = 0
    duplicate_count = 0
    error_count = 0
    errors: list[dict[str, Any]] = []

    with tmp_path.open("w", encoding="utf-8", newline="\n") as dst:
        for input_index, path in enumerate(inputs):
            input_records = 0
            input_written = 0
            input_duplicates = 0
            input_errors = 0
            with path.open("r", encoding="utf-8-sig") as src:
                for line_no, line in enumerate(src, start=1):
                    if not line.strip():
                        continue
                    records_seen += 1
                    input_records += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        error_count += 1
                        input_errors += 1
                        _append_error(errors, path, line_no, "json_decode_error", str(exc), max_error_examples)
                        continue
                    if not isinstance(record, dict):
                        error_count += 1
                        input_errors += 1
                        _append_error(errors, path, line_no, "record_not_object", type(record).__name__, max_error_examples)
                        continue
                    if not first_record_keys:
                        first_record_keys = sorted(str(key) for key in record.keys())
                    record_errors = validate_record_shape(record)
                    for code in record_errors:
                        error_count += 1
                        input_errors += 1
                        _append_error(errors, path, line_no, code, "", max_error_examples)
                    if record_errors and fail_on_error:
                        continue
                    if dedupe_key:
                        key_value = record.get(dedupe_key)
                        if key_value is not None:
                            key = str(key_value)
                            if key in seen_keys:
                                duplicate_count += 1
                                input_duplicates += 1
                                continue
                            seen_keys.add(key)
                    terminal = terminal_action(record)
                    terminal_actions[terminal] = terminal_actions.get(terminal, 0) + 1
                    dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                    dst.write("\n")
                    records_written += 1
                    input_written += 1
            input_summaries.append(
                {
                    "index": int(input_index),
                    "path": str(path),
                    "bytes": int(path.stat().st_size),
                    "records_seen": int(input_records),
                    "records_written": int(input_written),
                    "duplicates_skipped": int(input_duplicates),
                    "error_count": int(input_errors),
                }
            )

    tmp_path.replace(out_path)
    summary = {
        "format": "tjong_botzone_corpus_merge_v1",
        "target_format": "botzone_like_mcr_jsonl",
        "inputs": input_summaries,
        "output": str(out_path),
        "output_bytes": int(out_path.stat().st_size),
        "dedupe_key": dedupe_key,
        "records_seen": int(records_seen),
        "records_written": int(records_written),
        "duplicates_skipped": int(duplicate_count),
        "min_written": int(min_written),
        "min_written_met": int(records_written) >= int(min_written),
        "first_record_keys": first_record_keys,
        "terminal_actions": dict(sorted(terminal_actions.items())),
        "error_count": int(error_count),
        "errors": errors,
    }
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if fail_on_error and error_count:
        raise ValueError(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not summary["min_written_met"]:
        raise ValueError(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def _append_error(
    errors: list[dict[str, Any]],
    path: Path,
    line_no: int,
    code: str,
    detail: str,
    max_error_examples: int,
) -> None:
    if len(errors) < max_error_examples:
        errors.append({"path": str(path), "line": int(line_no), "code": code, "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inputs", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--dedupe-key", default="match_id")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-written", type=int, default=0)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    try:
        merge_botzone_corpora(
            inputs=[Path(path) for path in args.inputs],
            out_path=Path(args.out),
            summary_out=Path(args.summary_out) if args.summary_out else None,
            dedupe_key=None if args.no_dedupe else args.dedupe_key,
            overwrite=args.overwrite,
            min_written=args.min_written,
            fail_on_error=args.fail_on_error,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
