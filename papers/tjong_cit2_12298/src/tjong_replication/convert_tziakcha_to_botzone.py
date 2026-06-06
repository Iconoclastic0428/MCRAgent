"""Convert Tziakcha record-miner JSONL into Botzone-style logs for Tjong."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def add_repo_scripts_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))


def convert_tziakcha_file(
    *,
    in_path: Path,
    out_path: Path,
    summary_out: Path | None = None,
    limit: int | None = None,
    min_written: int = 0,
    fail_on_error: bool = False,
) -> dict[str, Any]:
    add_repo_scripts_to_path()
    from tziakcha_records import convert_path  # noqa: PLC0415

    summary = convert_path(in_path, out_path, limit=limit)
    summary = {
        "format": "tjong_tziakcha_to_botzone_v1",
        "source_format": "tziakcha_record_miner_decoded_or_encoded_jsonl",
        "target_format": "botzone_like_mcr_jsonl",
        "min_written": int(min_written),
        **summary,
    }
    if int(summary.get("records_written", 0)) < int(min_written):
        summary["min_written_met"] = False
    else:
        summary["min_written_met"] = True
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if fail_on_error and int(summary.get("error_count", 0)) > 0:
        raise ValueError(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not summary["min_written_met"]:
        raise ValueError(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-written", type=int, default=0)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    try:
        convert_tziakcha_file(
            in_path=Path(args.in_path),
            out_path=Path(args.out),
            summary_out=Path(args.summary_out) if args.summary_out else None,
            limit=args.limit,
            min_written=args.min_written,
            fail_on_error=args.fail_on_error,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
