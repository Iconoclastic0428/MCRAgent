"""Verify Tjong platform wrappers against Botzone and Tziakcha inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .audit_policy_replay import audit_botzone_replay
from .policy_bot import TjongCheckpointPredictor, respond_json_with_predictor


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from advisor_service.model_advisor import TziakchaModelAdvisor  # noqa: E402
from advisor_service.tiles import tile_id_from_botzone_symbol  # noqa: E402


DEFAULT_ENCODING_VERSION = "tjong_cit2_12298_v3_hidden_concealed_kong"


class _NoHuFanChecker:
    def evaluate(self, **kwargs) -> dict[str, Any]:
        return {"fan": 0, "can_hu": False}


class _TracingPredictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.predictor, name)

    def predict_legal_response(self, input_text, hand, player_id, request, candidates):
        response = self.predictor.predict_legal_response(
            input_text,
            hand,
            player_id,
            request,
            candidates,
        )
        self.calls.append(
            {
                "input_text": str(input_text),
                "hand": dict(Counter(hand)),
                "player_id": player_id,
                "request": str(request),
                "candidates": list(candidates),
                "response": str(response),
            }
        )
        return response

    def predict_response(self, input_text: str) -> str:
        response = self.predictor.predict_response(input_text)
        self.calls.append({"input_text": str(input_text), "response": str(response)})
        return response


class _FixedIllegalPredictor:
    requires_botzone_history = True
    kind = "legal_action_ranker"

    def __init__(self, response: str):
        self.response = response

    def predict_legal_response(self, input_text, hand, player_id, request, candidates):
        return self.response


def run_platform_wrapper_verification(
    *,
    checkpoint: Path | str,
    raw_path: Path | str,
    max_states: int = 200,
    nonpass_only: bool = True,
    device: str | None = None,
    require_encoding_version: str | None = DEFAULT_ENCODING_VERSION,
    require_paper_config: bool = False,
    min_states: int = 1,
    min_exact_accuracy: float | None = None,
    min_family_accuracy: float | None = None,
    example_limit: int = 50,
) -> dict[str, Any]:
    predictor = TjongCheckpointPredictor(
        checkpoint,
        device=device,
        require_encoding_version=require_encoding_version,
        require_paper_config=require_paper_config,
    )
    equivalence = verify_tziakcha_matches_botzone_draw(predictor)
    illegal_draw = verify_illegal_draw_fallback_matches()
    illegal_reaction = verify_illegal_reaction_fallback_matches()
    replay = audit_botzone_replay(
        checkpoint=checkpoint,
        raw_path=raw_path,
        max_states=max_states,
        nonpass_only=nonpass_only,
        device=device,
        require_encoding_version=require_encoding_version,
        require_paper_config=require_paper_config,
        example_limit=example_limit,
    )

    checks: dict[str, bool] = {
        "tziakcha_matches_botzone_draw": bool(equivalence["ok"]),
        "illegal_draw_fallback_matches_botzone": bool(illegal_draw["ok"]),
        "illegal_reaction_fallback_matches_botzone": bool(illegal_reaction["ok"]),
        "training_replay_no_errors": int(replay.get("error_count", 0) or 0) == 0,
        "training_replay_min_states": int(replay.get("states", 0) or 0) >= int(min_states),
    }
    if min_exact_accuracy is not None:
        exact_accuracy = replay.get("exact_accuracy")
        checks["training_replay_min_exact_accuracy"] = (
            isinstance(exact_accuracy, (int, float)) and float(exact_accuracy) >= float(min_exact_accuracy)
        )
    if min_family_accuracy is not None:
        family_accuracy = replay.get("family_accuracy")
        checks["training_replay_min_family_accuracy"] = (
            isinstance(family_accuracy, (int, float)) and float(family_accuracy) >= float(min_family_accuracy)
        )

    return {
        "format": "tjong_platform_wrapper_verification_v1",
        "checkpoint": str(checkpoint),
        "raw": str(raw_path),
        "require_encoding_version": require_encoding_version,
        "require_paper_config": bool(require_paper_config),
        "device": device,
        "ok": all(checks.values()),
        "checks": checks,
        "botzone_tziakcha_equivalence": equivalence,
        "illegal_draw_fallback": illegal_draw,
        "illegal_reaction_fallback": illegal_reaction,
        "training_replay": replay,
    }


def verify_tziakcha_matches_botzone_draw(predictor) -> dict[str, Any]:
    traced = _TracingPredictor(predictor)
    initial = "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
    payload = {"requests": ["0 0 1", initial, "2 W5"], "responses": ["PASS", "PASS"]}
    botzone_response = respond_json_with_predictor(payload, traced)
    botzone_call = dict(traced.calls[-1]) if traced.calls else {}

    snapshot = _draw_snapshot(initial)
    tziakcha_response = TziakchaModelAdvisor(
        predictor=traced,
        fan_checker=_NoHuFanChecker(),
    ).recommend(snapshot)
    tziakcha_call = dict(traced.calls[-1]) if traced.calls else {}

    comparisons = {
        "response": botzone_response == tziakcha_response.get("raw_response"),
        "input_text": tziakcha_call.get("input_text") == botzone_call.get("input_text"),
        "request": tziakcha_call.get("request") == botzone_call.get("request"),
        "player_id": tziakcha_call.get("player_id") == botzone_call.get("player_id"),
        "hand": tziakcha_call.get("hand") == botzone_call.get("hand"),
        "candidates": tziakcha_call.get("candidates") == botzone_call.get("candidates"),
    }
    return {
        "ok": all(comparisons.values()),
        "comparisons": comparisons,
        "botzone_response": botzone_response,
        "tziakcha_response": tziakcha_response.get("raw_response"),
        "request": botzone_call.get("request"),
        "candidate_count": len(botzone_call.get("candidates") or []),
    }


def verify_illegal_draw_fallback_matches() -> dict[str, Any]:
    predictor = _FixedIllegalPredictor("PASS")
    initial = "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
    payload = {"requests": ["0 0 1", initial, "2 W5"], "responses": ["PASS", "PASS"]}
    botzone_response = respond_json_with_predictor(payload, predictor)
    tziakcha_response = TziakchaModelAdvisor(
        predictor=predictor,
        fan_checker=_NoHuFanChecker(),
    ).recommend(_draw_snapshot(initial))
    return {
        "ok": botzone_response == tziakcha_response.get("raw_response") == "PLAY W5",
        "botzone_response": botzone_response,
        "tziakcha_response": tziakcha_response.get("raw_response"),
    }


def verify_illegal_reaction_fallback_matches() -> dict[str, Any]:
    predictor = _FixedIllegalPredictor("PENG W1")
    initial = "1 0 0 0 0 W2 W4 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3"
    payload = {"requests": ["0 0 1", initial, "3 3 PLAY W3"], "responses": ["PASS", "PASS"]}
    botzone_response = respond_json_with_predictor(payload, predictor)
    tziakcha_response = TziakchaModelAdvisor(
        predictor=predictor,
        fan_checker=_NoHuFanChecker(),
    ).recommend(_reaction_snapshot(initial))
    return {
        "ok": botzone_response == tziakcha_response.get("raw_response") == "PASS",
        "botzone_response": botzone_response,
        "tziakcha_response": tziakcha_response.get("raw_response"),
    }


def _draw_snapshot(initial: str) -> dict[str, Any]:
    return {
        "seat": 0,
        "turn": 0,
        "botzone_history_complete": True,
        "botzone_history": ["REQ 0 0 1", "RES PASS", f"REQ {initial}", "RES PASS"],
        "available_actions": {"discard": [tile_id_from_botzone_symbol("W5")]},
        "hand": [
            tile_id_from_botzone_symbol(tile)
            for tile in "W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 W5".split()
        ],
        "last_draw": {"seat": 0, "tile": tile_id_from_botzone_symbol("W5")},
    }


def _reaction_snapshot(initial: str) -> dict[str, Any]:
    return {
        "seat": 0,
        "turn": 3,
        "botzone_history_complete": True,
        "botzone_history": ["REQ 0 0 1", "RES PASS", f"REQ {initial}", "RES PASS"],
        "available_actions": {"chow": [tile_id_from_botzone_symbol("W4")], "pass": [0]},
        "hand": [
            tile_id_from_botzone_symbol(tile)
            for tile in "W2 W4 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2 J3".split()
        ],
        "last_discard": {"seat": 3, "tile": tile_id_from_botzone_symbol("W3")},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--max-states", type=int, default=200)
    parser.add_argument("--nonpass-only", action="store_true", default=True)
    parser.add_argument("--include-pass", dest="nonpass_only", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-encoding-version", default=DEFAULT_ENCODING_VERSION)
    parser.add_argument("--require-paper-config", action="store_true")
    parser.add_argument("--min-states", type=int, default=1)
    parser.add_argument("--min-exact-accuracy", type=float, default=None)
    parser.add_argument("--min-family-accuracy", type=float, default=None)
    parser.add_argument("--example-limit", type=int, default=50)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    summary = run_platform_wrapper_verification(
        checkpoint=args.checkpoint,
        raw_path=args.raw,
        max_states=args.max_states,
        nonpass_only=args.nonpass_only,
        device=args.device,
        require_encoding_version=args.require_encoding_version or None,
        require_paper_config=args.require_paper_config,
        min_states=args.min_states,
        min_exact_accuracy=args.min_exact_accuracy,
        min_family_accuracy=args.min_family_accuracy,
        example_limit=args.example_limit,
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 1 if args.fail_on_error and not summary["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
