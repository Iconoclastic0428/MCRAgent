"""Model-backed recommendation logic for the read-only Tziakcha advisor."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .tiles import botzone_symbol, display_name, tile_id_from_botzone_symbol

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from legal_actions import generate_chi_responses  # noqa: E402
from policy_bot import SklearnPredictor  # noqa: E402

DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "ensemble_draw_public1000_2026_030_070_reaction1000_prefer_hu.pkl"


class TziakchaModelAdvisor:
    """Rank Tziakcha decision prompts with the local Botzone-trained policy.

    The browser extension only observes state. This class returns text advice and
    never sends an action back to Tziakcha.
    """

    def __init__(
        self,
        model_path: Path | str | None = DEFAULT_MODEL,
        predictor: Any | None = None,
        fan_checker: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.predictor = predictor
        if self.predictor is None and self.model_path and self.model_path.exists():
            self.predictor = SklearnPredictor(self.model_path)
        self.fan_checker = fan_checker if fan_checker is not None else self._default_fan_checker()

    def _default_fan_checker(self) -> Any | None:
        try:
            from official_fan import OfficialFanChecker
        except ImportError:
            return None
        return OfficialFanChecker.default()

    def recommend(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        hand = _hand_counter(snapshot)
        decision = self._decision(snapshot, hand)
        if decision is None:
            return {"action": "wait", "text": "Waiting for decision prompt", "source": "local-model"}

        request, candidates, hu_result = decision
        if not candidates:
            return self._fallback(snapshot, hu_result)
        if self.predictor is None:
            response = _heuristic_response(candidates, hand)
        elif getattr(self.predictor, "prefer_hu", False) and "HU" in candidates:
            response = "HU"
        elif hasattr(self.predictor, "predict_legal_response"):
            response = self.predictor.predict_legal_response(
                f"REQ {request}",
                hand,
                _optional_int(snapshot.get("seat"), 0),
                request,
                candidates,
            ).strip()
        else:
            response = str(self.predictor.predict_response(f"REQ {request}")).strip()
        if response not in candidates:
            response = _heuristic_response(candidates, hand)
        return self._format_response(response, snapshot, hu_result)

    def _decision(
        self, snapshot: dict[str, Any], hand: Counter[str]
    ) -> tuple[str, list[str], dict[str, Any] | None] | None:
        actions = snapshot.get("available_actions") or {}
        seat = _optional_int(snapshot.get("seat"), None)
        turn = _optional_int(snapshot.get("turn"), None)
        last_discard = snapshot.get("last_discard") or {}
        last_discard_seat = _optional_int(last_discard.get("seat"), None)

        reaction_prompt = (
            seat is not None
            and last_discard.get("tile") is not None
            and last_discard_seat is not None
            and last_discard_seat != seat
            and any(actions.get(name) for name in ("hu", "pung", "chow", "kong", "pass", "waive"))
        )
        if reaction_prompt:
            event_tile = botzone_symbol(int(last_discard["tile"]))
            request = f"3 {last_discard_seat} PLAY {event_tile}"
            hu_result = self._hu_result(snapshot, self_drawn=False) if actions.get("hu") else None
            return request, self._reaction_candidates(snapshot, hand, event_tile, hu_result), hu_result

        draw_prompt = (
            seat is not None
            and seat == turn
            and bool(hand)
            and (bool(actions.get("discard")) or not actions or sum(hand.values()) % 3 == 2)
        )
        if draw_prompt:
            last_draw = snapshot.get("last_draw") or {}
            drawn_tile = (
                botzone_symbol(int(last_draw["tile"]))
                if isinstance(last_draw.get("tile"), int)
                else _first_symbol(hand)
            )
            request = f"2 {drawn_tile}"
            hu_result = self._hu_result(snapshot, self_drawn=True) if actions.get("hu") else None
            return request, self._draw_candidates(snapshot, hand, hu_result), hu_result
        return None

    def _draw_candidates(
        self,
        snapshot: dict[str, Any],
        hand: Counter[str],
        hu_result: dict[str, Any] | None,
    ) -> list[str]:
        candidates: list[str] = []
        if _can_hu(hu_result):
            candidates.append("HU")
        candidates.extend(f"PLAY {tile}" for tile in sorted(hand) if hand[tile] > 0)
        actions = snapshot.get("available_actions") or {}
        for tile in _action_symbols(actions.get("kong") or []):
            if hand[tile] >= 4:
                candidates.append(f"GANG {tile}")
        return _dedupe(candidates)

    def _reaction_candidates(
        self,
        snapshot: dict[str, Any],
        hand: Counter[str],
        event_tile: str,
        hu_result: dict[str, Any] | None,
    ) -> list[str]:
        actions = snapshot.get("available_actions") or {}
        candidates = ["PASS"]
        if _can_hu(hu_result):
            candidates.append("HU")
        if actions.get("kong") and hand[event_tile] >= 3:
            candidates.append("GANG")
        if actions.get("pung") and hand[event_tile] >= 2:
            after_peng = Counter(hand)
            after_peng[event_tile] -= 2
            _cleanup(after_peng)
            candidates.extend(f"PENG {tile}" for tile in sorted(after_peng) if after_peng[tile] > 0)
        if actions.get("chow"):
            seat = _optional_int(snapshot.get("seat"), None)
            actor = _optional_int((snapshot.get("last_discard") or {}).get("seat"), None)
            if seat is not None and actor is not None:
                candidates.extend(generate_chi_responses(event_tile, hand))
        return _dedupe(candidates)

    def _hu_result(self, snapshot: dict[str, Any], self_drawn: bool) -> dict[str, Any] | None:
        if self.fan_checker is None:
            return {"can_hu": False, "fan": None, "reason": "official fan checker unavailable"}
        context = _hu_context(snapshot, self_drawn)
        if context is None:
            return {"can_hu": False, "fan": None, "reason": "missing win tile"}
        try:
            if hasattr(self.fan_checker, "evaluate"):
                result = dict(self.fan_checker.evaluate(**context))
                result.setdefault("can_hu", bool(result.get("fan", -3) >= 8))
                return result
            return {"can_hu": bool(self.fan_checker.can_hu(**context)), "fan": None}
        except Exception as exc:
            return {"can_hu": False, "fan": None, "reason": str(exc)}

    def _fallback(
        self, snapshot: dict[str, Any], hu_result: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        actions = snapshot.get("available_actions") or {}
        hand = _hand_counter(snapshot)
        if actions.get("pass") or actions.get("waive"):
            rec = {"action": "pass", "text": "Pass", "source": "local-model"}
            _add_hu_note(rec, hu_result)
            return rec
        if hand:
            tile = _first_symbol(hand)
            return self._format_response(f"PLAY {tile}", snapshot, hu_result)
        return {"action": "wait", "text": "Waiting for decision prompt", "source": "local-model"}

    def _format_response(
        self,
        response: str,
        snapshot: dict[str, Any],
        hu_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parts = response.split()
        action = parts[0].upper() if parts else "PASS"
        if action == "HU":
            fan = None if hu_result is None else hu_result.get("fan")
            text = f"Hu ({fan} fan)" if fan is not None else "Hu"
            return {"action": "hu", "text": text, "fan": fan, "source": "local-model", "raw_response": response}
        if action == "PASS":
            rec = {"action": "pass", "text": "Pass", "source": "local-model", "raw_response": response}
            _add_hu_note(rec, hu_result)
            return rec
        if action == "PLAY" and len(parts) >= 2:
            return _tile_action("discard", "Discard", parts[1], snapshot, response)
        if action == "PENG" and len(parts) >= 2:
            return _tile_action("pung", "Pung; discard", parts[1], snapshot, response)
        if action == "CHI" and len(parts) >= 3:
            rec = _tile_action("chow", "Chow; discard", parts[2], snapshot, response)
            rec["meld_tile"] = parts[1]
            rec["text"] = f"Chow {display_name(_tile_id_for_symbol(parts[1], snapshot))}; discard {rec['tile_display']}"
            return rec
        if action in {"GANG", "BUGANG"}:
            tile = parts[1] if len(parts) >= 2 else _first_symbol(_hand_counter(snapshot))
            verb = "Add Kong" if action == "BUGANG" else "Kong"
            return _tile_action("kong", verb, tile, snapshot, response)
        rec = {"action": "wait", "text": "Waiting for decision prompt", "source": "local-model", "raw_response": response}
        _add_hu_note(rec, hu_result)
        return rec


def _hand_counter(snapshot: dict[str, Any]) -> Counter[str]:
    return Counter(botzone_symbol(int(tile)) for tile in snapshot.get("hand") or [])


def _action_symbols(values: list[int]) -> list[str]:
    return [botzone_symbol(int(value)) for value in values if isinstance(value, int)]


def _first_symbol(hand: Counter[str]) -> str:
    for tile in sorted(hand):
        if hand[tile] > 0:
            return tile
    return "W1"


def _heuristic_response(candidates: list[str], hand: Counter[str]) -> str:
    if "HU" in candidates:
        return "HU"
    plays = [candidate for candidate in candidates if candidate.startswith("PLAY ")]
    if plays:
        return plays[0]
    non_pass = [candidate for candidate in candidates if candidate != "PASS"]
    return non_pass[0] if non_pass else "PASS"


def _tile_action(
    action: str, verb: str, symbol: str, snapshot: dict[str, Any], raw_response: str
) -> dict[str, Any]:
    tile = _tile_id_for_symbol(symbol, snapshot)
    return {
        "action": action,
        "tile": tile,
        "tile_symbol": symbol,
        "tile_display": display_name(tile),
        "text": f"{verb} {display_name(tile)}",
        "source": "local-model",
        "raw_response": raw_response,
    }


def _tile_id_for_symbol(symbol: str, snapshot: dict[str, Any]) -> int:
    for tile in snapshot.get("hand") or []:
        if botzone_symbol(int(tile)) == symbol:
            return int(tile)
    last_discard = snapshot.get("last_discard") or {}
    if isinstance(last_discard.get("tile"), int) and botzone_symbol(int(last_discard["tile"])) == symbol:
        return int(last_discard["tile"])
    return tile_id_from_botzone_symbol(symbol)


def _hu_context(snapshot: dict[str, Any], self_drawn: bool) -> dict[str, Any] | None:
    seat = _optional_int(snapshot.get("seat"), 0)
    if self_drawn:
        last_draw = snapshot.get("last_draw") or {}
        win_tile_id = last_draw.get("tile")
    else:
        last_discard = snapshot.get("last_discard") or {}
        win_tile_id = last_discard.get("tile")
    if not isinstance(win_tile_id, int):
        return None
    win_tile = botzone_symbol(win_tile_id)
    hand = [botzone_symbol(int(tile)) for tile in snapshot.get("hand") or []]
    if self_drawn:
        try:
            hand.remove(win_tile)
        except ValueError:
            pass
    wall_count = snapshot.get("wall_count")
    return {
        "packs": [_pack_to_official_meld(pack, seat) for pack in snapshot.get("melds") or []],
        "hand": hand,
        "win_tile": win_tile,
        "flower_count": int(snapshot.get("flowers") or 0),
        "is_self_draw": self_drawn,
        "is_4th_tile": False,
        "is_about_kong": False,
        "is_last": isinstance(wall_count, int) and wall_count <= 0,
        "seat_wind": seat,
        "prevalent_wind": int(snapshot.get("prevalent_wind") or 0),
        "player": seat,
    }


def _pack_to_official_meld(pack: int, player: int) -> dict[str, Any]:
    pack_type = (int(pack) >> 8) & 3
    tile = botzone_symbol((int(pack) & 63) << 2)
    offer = (int(pack) >> 6) & 3
    if pack_type == 0:
        return {"type": "CHI", "tile": tile, "offer": max(0, offer - 1)}
    if pack_type == 1:
        return {"type": "PENG", "tile": tile, "offer": (player + offer) % 4}
    return {"type": "GANG", "tile": tile, "offer": (player + offer) % 4}


def _can_hu(hu_result: dict[str, Any] | None) -> bool:
    return bool(hu_result and hu_result.get("can_hu"))


def _add_hu_note(rec: dict[str, Any], hu_result: dict[str, Any] | None) -> None:
    if not hu_result or hu_result.get("can_hu"):
        return
    fan = hu_result.get("fan")
    if fan is not None:
        rec["fan"] = fan
        rec["note"] = f"Hu is {fan} fan, below 8"
        if rec["action"] == "pass":
            rec["text"] = f"{rec['text']} (Hu is {fan} fan, below 8)"
    elif hu_result.get("reason"):
        rec["note"] = f"Hu not recommended: {hu_result['reason']}"


def _cleanup(hand: Counter[str]) -> None:
    for tile in list(hand):
        if hand[tile] <= 0:
            del hand[tile]


def _dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _optional_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
