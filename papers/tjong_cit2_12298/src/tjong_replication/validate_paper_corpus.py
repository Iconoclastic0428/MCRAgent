"""Validate supervised Botzone-format logs before Tjong training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


PAPER_SUPERVISED_GAMES = 519_338


def validate_corpus(
    path: Path,
    *,
    expected_records: int = PAPER_SUPERVISED_GAMES,
    sha256: str | None = None,
    summary_out: Path | None = None,
    max_error_examples: int = 20,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"corpus is empty: {path}")

    digest = hashlib.sha256()
    records = 0
    nonempty_lines = 0
    first_record_keys: list[str] = []
    errors: list[dict[str, Any]] = []
    error_count = 0
    terminal_actions: dict[str, int] = {}
    with open(path, "rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()

    with _open_text(path) as src:
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
            if not first_record_keys:
                first_record_keys = sorted(str(key) for key in record.keys())
            record_errors = validate_record_shape(record)
            for error in record_errors:
                error_count += 1
                _append_error(errors, line_no, error, "", max_error_examples)
            terminal = terminal_action(record)
            terminal_actions[terminal] = terminal_actions.get(terminal, 0) + 1
            records += 1

    summary = {
        "format": "tjong_paper_corpus_validation_v1",
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "paper_supervised_games": PAPER_SUPERVISED_GAMES,
        "expected_records": int(expected_records),
        "records": int(records),
        "nonempty_lines": int(nonempty_lines),
        "records_match_expected": int(records) == int(expected_records),
        "records_match_original_paper_count": int(records) == PAPER_SUPERVISED_GAMES,
        "records_match_paper": int(records) == PAPER_SUPERVISED_GAMES,
        "expected_records_source": (
            "paper_reported_botzone_count"
            if int(expected_records) == PAPER_SUPERVISED_GAMES
            else "override_same_format_dataset"
        ),
        "sha256": actual_sha256,
        "expected_sha256": sha256,
        "sha256_match": None if sha256 is None else actual_sha256.lower() == sha256.lower(),
        "first_record_keys": first_record_keys,
        "terminal_actions": dict(sorted(terminal_actions.items())),
        "error_count": int(error_count),
        "errors": errors,
        "valid_for_paper_supervised": (
            int(records) == int(expected_records)
            and error_count == 0
            and (sha256 is None or actual_sha256.lower() == sha256.lower())
        ),
    }
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not summary["valid_for_paper_supervised"]:
        raise ValueError(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def validate_record_shape(record: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(record.get("initdata"), dict) and not has_log_init_deal(record):
        errors.append("missing_initdata_object")
    logs = record.get("logs", record.get("log"))
    if not isinstance(logs, list) or not logs:
        errors.append("missing_logs_list")
    elif not any(isinstance(item, dict) and "output" in item for item in logs):
        errors.append("logs_missing_output_entries")
    return errors


def has_log_init_deal(record: dict[str, Any]) -> bool:
    """Accept converted Botzone-style logs whose INIT/DEAL are embedded in logs."""

    logs = record.get("logs", record.get("log"))
    if not isinstance(logs, list):
        return False
    saw_init = False
    saw_deal = False
    for item in logs:
        if not isinstance(item, dict):
            continue
        output = item.get("output") or {}
        if not isinstance(output, dict):
            continue
        content = output.get("content") or {}
        display = output.get("display") or {}
        if not isinstance(content, dict) or not isinstance(display, dict):
            continue
        action = str(display.get("action") or "").upper()
        if action == "INIT" and any(str(value).split()[:1] == ["0"] for value in content.values()):
            saw_init = True
        if action == "DEAL" and any(str(value).split()[:1] == ["1"] for value in content.values()):
            saw_deal = True
    return saw_init and saw_deal


def terminal_action(record: dict[str, Any]) -> str:
    final_output = record.get("final_output") or {}
    display = final_output.get("display") if isinstance(final_output, dict) else None
    if isinstance(display, dict) and display.get("action"):
        return str(display.get("action")).upper()
    for item in reversed(record.get("logs", record.get("log", [])) or []):
        if not isinstance(item, dict):
            continue
        output = item.get("output") or {}
        display = output.get("display") if isinstance(output, dict) else None
        if isinstance(display, dict) and display.get("action"):
            action = str(display.get("action")).upper()
            if action in {"HU", "HUANG"}:
                return action
            return action
    return "UNKNOWN"


def _append_error(
    errors: list[dict[str, Any]],
    line_no: int,
    code: str,
    detail: str,
    max_error_examples: int,
) -> None:
    if len(errors) < max_error_examples:
        errors.append({"line": int(line_no), "code": code, "detail": detail})


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--expected-records", type=int, default=PAPER_SUPERVISED_GAMES)
    parser.add_argument("--sha256", default=None)
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()
    validate_corpus(
        Path(args.infile),
        expected_records=args.expected_records,
        sha256=args.sha256,
        summary_out=Path(args.summary_out) if args.summary_out else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
