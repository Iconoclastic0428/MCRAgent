"""Record live advisor snapshots while a Tziakcha game is being observed."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def fetch_json(base_url: str, path: str, timeout: float = 3.0) -> Any:
    with urlopen(f"{base_url.rstrip('/')}/{path.lstrip('/')}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "seat": state.get("seat"),
        "turn": state.get("turn"),
        "wall_count": state.get("wall_count"),
        "event_count": state.get("event_count"),
        "unknown_event_count": state.get("unknown_event_count"),
        "hand_display": state.get("hand_display"),
        "available_actions": state.get("available_actions"),
        "last_discard_display": state.get("last_discard_display"),
        "last_result": state.get("last_result"),
        "result_stats": state.get("result_stats"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--heartbeat", type=float, default=30.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_key = ""
    last_heartbeat = 0.0

    with out_path.open("a", encoding="utf-8") as out:
        while True:
            now = datetime.now(timezone.utc).isoformat()
            try:
                state = fetch_json(args.base_url, "/state")
                recommendation = fetch_json(args.base_url, "/recommendation")
                results = fetch_json(args.base_url, "/results")
                events = fetch_json(args.base_url, "/events")
                errors = fetch_json(args.base_url, "/errors")
                record = {
                    "ts": now,
                    "ok": True,
                    "state": state,
                    "state_compact": compact_state(state),
                    "recommendation": recommendation,
                    "results": results,
                    "recent_events": events[-10:] if isinstance(events, list) else events,
                    "recent_errors": errors[-10:] if isinstance(errors, list) else errors,
                }
            except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                record = {"ts": now, "ok": False, "error": str(exc)}

            key = json.dumps(
                {
                    "ok": record.get("ok"),
                    "state": record.get("state_compact"),
                    "recommendation": record.get("recommendation"),
                    "results": record.get("results"),
                    "error": record.get("error"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            current = time.monotonic()
            if key != last_key or current - last_heartbeat >= args.heartbeat:
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()
                last_key = key
                last_heartbeat = current

            time.sleep(max(args.interval, 0.1))


if __name__ == "__main__":
    main()
