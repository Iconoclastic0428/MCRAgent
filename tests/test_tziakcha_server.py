import json
from urllib import error, request

from advisor_service.server import run_server_in_thread


def test_observe_state_recommendation_routes():
    server, base = run_server_in_thread(port=0, use_model=False)
    try:
        payload = json.dumps(
            {"kind": "text", "data": json.dumps({"m": 2, "r": 7, "v": 1, "t": 36})}
        ).encode("utf-8")
        req = request.Request(
            f"{base}/observe",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200
        with request.urlopen(f"{base}/state", timeout=2) as resp:
            state = json.loads(resp.read().decode("utf-8"))
        assert state["last_discard_display"] == "1s"
        with request.urlopen(f"{base}/recommendation", timeout=2) as resp:
            rec = json.loads(resp.read().decode("utf-8"))
        assert rec["action"] in {"wait", "pass", "waive", "discard", "hu", "pung", "chow", "kong"}
    finally:
        server.shutdown()
        server.server_close()


def test_action_route_is_not_available():
    server, base = run_server_in_thread(port=0, use_model=False)
    try:
        req = request.Request(
            f"{base}/action",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            request.urlopen(req, timeout=2)
        except error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("/action must not exist in observer-only mode")
    finally:
        server.shutdown()
        server.server_close()


def test_reset_route_clears_observed_state():
    server, base = run_server_in_thread(port=0, use_model=False)
    try:
        payload = json.dumps(
            {"kind": "text", "data": json.dumps({"m": 2, "r": 7, "v": 1, "t": 36})}
        ).encode("utf-8")
        request.urlopen(
            request.Request(
                f"{base}/observe",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=2,
        )
        request.urlopen(
            request.Request(f"{base}/reset", data=b"{}", method="POST"),
            timeout=2,
        )
        with request.urlopen(f"{base}/state", timeout=2) as resp:
            state = json.loads(resp.read().decode("utf-8"))
        assert state["event_count"] == 0
        assert state["last_discard"] is None
    finally:
        server.shutdown()
        server.server_close()
