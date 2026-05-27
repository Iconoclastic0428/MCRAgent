"""Local HTTP server for the read-only Tziakcha advisor."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .advisor import recommend
from .messages import decode_observed_payload
from .model_advisor import DEFAULT_MODEL, TziakchaModelAdvisor
from .state import AdvisorState


class AdvisorRuntime:
    def __init__(
        self,
        use_aleo: bool = False,
        use_model: bool = True,
        model_path: Path | str | None = DEFAULT_MODEL,
        result_log_path: Path | str | None = Path("runs/tziakcha_live_results.jsonl"),
    ) -> None:
        self.use_aleo = use_aleo
        self.model_advisor = TziakchaModelAdvisor(model_path=model_path) if use_model else None
        self.state = AdvisorState()
        self.result_log_path = Path(result_log_path) if result_log_path else None
        self._persisted_result_count = 0
        self.errors: list[str] = []
        self._recommendation_key = ""
        self._recommendation: dict[str, Any] = {"action": "wait", "text": "Waiting", "source": "local-advisor"}

    def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            message = decode_observed_payload(payload)
            self.state.ingest(message)
            self._persist_new_results()
            snapshot = self.state.snapshot()
            return {"ok": True, "state": snapshot, "recommendation": self.recommendation()}
        except Exception as exc:  # Keep the page observer resilient while logging bad shapes.
            self.errors.append(str(exc))
            return {"ok": False, "error": str(exc)}

    def recommendation(self) -> dict[str, Any]:
        snapshot = self.state.snapshot()
        key = json.dumps(snapshot, sort_keys=True)
        if key != self._recommendation_key:
            self._recommendation_key = key
            self._recommendation = recommend(
                snapshot,
                use_aleo=self.use_aleo,
                model_advisor=self.model_advisor,
            )
        return self._recommendation

    def results(self) -> dict[str, Any]:
        return self.state.results_snapshot()

    def reset(self) -> None:
        self.state.reset()
        self._persisted_result_count = 0
        self._recommendation_key = ""
        self._recommendation = {"action": "wait", "text": "Waiting", "source": "local-advisor"}

    def _persist_new_results(self) -> None:
        if self.result_log_path is None:
            return
        results = self.state.result_history
        if len(results) <= self._persisted_result_count:
            return
        self.result_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.result_log_path.open("a", encoding="utf-8") as out:
            for result in results[self._persisted_result_count :]:
                out.write(
                    json.dumps(
                        {"result": result, "stats": self.state.results_snapshot()["stats"]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        self._persisted_result_count = len(results)


def make_handler(runtime: AdvisorRuntime):
    class Handler(BaseHTTPRequestHandler):
        def _headers(self, status: int = 200, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()

        def do_OPTIONS(self) -> None:
            self._headers()

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/reset":
                runtime.reset()
                self._json({"ok": True})
                return
            if path != "/observe":
                self._json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError as exc:
                self._json({"ok": False, "error": f"invalid json: {exc}"}, 400)
                return
            result = runtime.observe(payload)
            self._json(result, 200 if result.get("ok") else 400)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._json(
                    {
                        "ok": True,
                        "model_loaded": bool(runtime.model_advisor),
                        "read_only": True,
                    }
                )
                return
            if path == "/state":
                self._json(runtime.state.snapshot())
                return
            if path == "/recommendation":
                self._json(runtime.recommendation())
                return
            if path == "/results":
                self._json(runtime.results())
                return
            if path == "/events":
                self._json(runtime.state.raw_events[-100:])
                return
            if path == "/errors":
                self._json(runtime.errors[-100:])
                return
            if path in {"/", "/dashboard"}:
                html = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
                self._headers(200, "text/html; charset=utf-8")
                self.wfile.write(html.encode("utf-8"))
                return
            self._json({"error": "not found"}, 404)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._headers(status, "application/json; charset=utf-8")
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def run_server_in_thread(
    port: int = 8765,
    use_aleo: bool = False,
    use_model: bool = True,
    model_path: Path | str | None = DEFAULT_MODEL,
    result_log_path: Path | str | None = None,
):
    runtime = AdvisorRuntime(
        use_aleo=use_aleo,
        use_model=use_model,
        model_path=model_path,
        result_log_path=result_log_path,
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(runtime))
    server.runtime = runtime  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    actual = server.server_address[1]
    return server, f"http://127.0.0.1:{actual}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only Tziakcha model advisor service.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=None, help="Optional legacy policy model path.")
    parser.add_argument("--no-model", action="store_true", help="Use only the simple local fallback advisor.")
    parser.add_argument("--local-only", action="store_true", help="Disable the optional Aleo fallback bridge.")
    parser.add_argument(
        "--results-log",
        default="runs/tziakcha_live_results.jsonl",
        help="Append completed observed game results to this JSONL file. Use an empty value to disable.",
    )
    args = parser.parse_args()
    runtime = AdvisorRuntime(
        use_aleo=not args.local_only,
        use_model=not args.no_model,
        model_path=args.model,
        result_log_path=args.results_log or None,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runtime))
    print(f"Tziakcha model advisor listening on http://127.0.0.1:{args.port}/dashboard", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
