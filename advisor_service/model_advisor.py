"""Lawlorentz effective-tile recommendation logic for the read-only Tziakcha advisor."""

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
from lawlorentz_policy import LawlorentzEffectiveScorer  # noqa: E402

DEFAULT_TRANSFORMER_MODEL = (
    WORKSPACE_ROOT / "models" / "transformer_candidate_allhighelo_chagaeval_med_l40_20260527d_plateau_e2b1000.pt"
)
DEFAULT_MODEL = DEFAULT_TRANSFORMER_MODEL if DEFAULT_TRANSFORMER_MODEL.exists() else None


class TziakchaModelAdvisor:
    """Rank Tziakcha decision prompts with the Lawlorentz effective-tile rule.

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
        self.predictor = predictor if predictor is not None else self._load_predictor(self.model_path)
        self.fan_checker = fan_checker if fan_checker is not None else self._default_fan_checker()

    def _load_predictor(self, model_path: Path | None) -> Any | None:
        if model_path is None:
            return None
        if model_path.suffix.lower() in {".pt", ".pth"}:
            return _build_transformer_predictor(model_path)
        return None

    def model_info(self) -> dict[str, Any]:
        if self.predictor is None:
            return {"type": "lawlorentz-effective", "path": str(self.model_path) if self.model_path else None}
        if hasattr(self.predictor, "info"):
            return dict(self.predictor.info())
        return {"type": type(self.predictor).__name__, "path": str(self.model_path) if self.model_path else None}

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
            response = _effective_response(candidates, hand, request, snapshot)
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
        reaction_event = _reaction_event(snapshot, seat, actions)

        if reaction_event is not None:
            event_tile = botzone_symbol(int(reaction_event["tile"]))
            event_seat = int(reaction_event["seat"])
            op = "BUGANG" if reaction_event["source"] == "bugang" else "PLAY"
            request = f"3 {event_seat} {op} {event_tile}"
            hu_result = self._hu_result(snapshot, self_drawn=False) if actions.get("hu") else None
            return (
                request,
                self._reaction_candidates(
                    snapshot,
                    hand,
                    event_tile,
                    hu_result,
                    allow_claims=reaction_event["source"] == "discard",
                    actor_seat=event_seat,
                ),
                hu_result,
            )

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
        *,
        allow_claims: bool = True,
        actor_seat: int | None = None,
    ) -> list[str]:
        actions = snapshot.get("available_actions") or {}
        candidates = ["PASS"]
        if _can_hu(hu_result):
            candidates.append("HU")
        if not allow_claims:
            return _dedupe(candidates)
        if actions.get("kong") and hand[event_tile] >= 3:
            candidates.append("GANG")
        if actions.get("pung") and hand[event_tile] >= 2:
            after_peng = Counter(hand)
            after_peng[event_tile] -= 2
            _cleanup(after_peng)
            candidates.extend(f"PENG {tile}" for tile in sorted(after_peng) if after_peng[tile] > 0)
        if actions.get("chow"):
            seat = _optional_int(snapshot.get("seat"), None)
            actor = actor_seat
            if seat is not None and actor is not None:
                chi_responses = generate_chi_responses(event_tile, hand)
                allowed_middles = _allowed_chi_middles(actions.get("chow") or [], event_tile)
                if allowed_middles is not None:
                    chi_responses = [
                        response
                        for response in chi_responses
                        if len(response.split()) >= 2 and response.split()[1] in allowed_middles
                    ]
                candidates.extend(chi_responses)
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
                gate_fan = _fan_gate_value(result)
                result["can_hu"] = isinstance(gate_fan, int) and gate_fan >= 8 and bool(
                    result.get("can_hu", True)
                )
                return result
            accepted = bool(self.fan_checker.can_hu(**context))
            return {
                "can_hu": False,
                "fan": None,
                "reason": "fan count unavailable" if accepted else "official fan checker rejected Hu",
            }
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
            base_fan = None if hu_result is None else hu_result.get("base_fan")
            fan_items = [] if hu_result is None else list(hu_result.get("fan_items") or [])
            base_fan_items = [] if hu_result is None else list(hu_result.get("base_fan_items") or [])
            fan_text = _format_fan_items(fan_items)
            text = f"Hu ({fan} fan)" if fan is not None else "Hu"
            if base_fan is not None and base_fan != fan:
                text = f"{text}, base {base_fan}"
            if fan_text:
                if fan is not None:
                    text = f"Hu ({fan} fan: {fan_text})"
                    if base_fan is not None and base_fan != fan:
                        text = f"{text}, base {base_fan}"
                else:
                    text = f"Hu ({fan_text})"
            return {
                "action": "hu",
                "text": text,
                "fan": fan,
                "base_fan": base_fan,
                "fan_items": fan_items,
                "base_fan_items": base_fan_items,
                "fan_text": fan_text,
                "source": "local-model",
                "raw_response": response,
            }
        if action == "PASS":
            rec = {"action": "pass", "text": "Pass", "source": "local-model", "raw_response": response}
            _add_hu_note(rec, hu_result)
            return rec
        if action == "PLAY" and len(parts) >= 2:
            rec = _tile_action("discard", "Discard", parts[1], snapshot, response)
            _add_hu_note(rec, hu_result)
            return rec
        if action == "PENG" and len(parts) >= 2:
            rec = _tile_action("pung", "Pung; discard", parts[1], snapshot, response)
            _add_hu_note(rec, hu_result)
            return rec
        if action == "CHI" and len(parts) >= 3:
            rec = _tile_action("chow", "Chow; discard", parts[2], snapshot, response)
            middle = parts[1]
            shape = _chi_shape(middle)
            shape_display = [display_name(tile_id_from_botzone_symbol(tile)) for tile in shape]
            rec["meld_tile"] = middle
            rec["meld_shape"] = shape
            rec["meld_shape_display"] = shape_display
            rec["text"] = f"Chi {middle} ({' '.join(shape_display)}); discard {rec['tile_display']}"
            _add_hu_note(rec, hu_result)
            return rec
        if action in {"GANG", "BUGANG"}:
            tile = parts[1] if len(parts) >= 2 else _first_symbol(_hand_counter(snapshot))
            verb = "Add Kong" if action == "BUGANG" else "Kong"
            rec = _tile_action("kong", verb, tile, snapshot, response)
            _add_hu_note(rec, hu_result)
            return rec
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


def _reaction_event(
    snapshot: dict[str, Any],
    seat: int | None,
    actions: dict[str, list[int]],
) -> dict[str, Any] | None:
    if seat is None or not any(actions.get(name) for name in ("hu", "pung", "chow", "kong", "pass", "waive")):
        return None
    event = snapshot.get("last_win_event") or {}
    source = event.get("source")
    event_seat = _optional_int(event.get("seat"), None)
    if (
        source in {"discard", "bugang"}
        and isinstance(event.get("tile"), int)
        and event_seat is not None
        and event_seat != seat
    ):
        return {"source": source, "seat": event_seat, "tile": int(event["tile"])}
    if source in {"draw", "discard", "bugang"}:
        return None

    last_discard = snapshot.get("last_discard") or {}
    last_discard_seat = _optional_int(last_discard.get("seat"), None)
    if (
        isinstance(last_discard.get("tile"), int)
        and last_discard_seat is not None
        and last_discard_seat != seat
    ):
        return {"source": "discard", "seat": last_discard_seat, "tile": int(last_discard["tile"])}
    return None


def _effective_response(
    candidates: list[str],
    hand: Counter[str],
    request: str,
    snapshot: dict[str, Any],
) -> str:
    if "HU" in candidates:
        return "HU"
    plays = [candidate for candidate in candidates if candidate.startswith("PLAY ")]
    if plays:
        scorer = LawlorentzEffectiveScorer(
            packs=(),
            shown_tiles={},
            seat_wind=_optional_int(snapshot.get("seat"), 0) or 0,
            prevalent_wind=_optional_int(snapshot.get("prevalent_wind"), 0) or 0,
            levels=1,
        )
        hand_tiles = list(hand.elements())
        return max(
            plays,
            key=lambda response: scorer.discard_key(hand_tiles, response.split()[1]),
        )
    if request.startswith("3 ") and "PASS" in candidates:
        return "PASS"
    return _heuristic_response(candidates, hand)


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
    event = snapshot.get("last_win_event") or {}
    event_source = event.get("source")
    event_tile_id = event.get("tile")
    if isinstance(event_tile_id, int) and event_source in {"draw", "discard", "bugang"}:
        win_tile_id = event_tile_id
        self_drawn = bool(event.get("is_self_draw"))
        is_about_kong = bool(event.get("is_about_kong"))
    elif self_drawn:
        last_draw = snapshot.get("last_draw") or {}
        win_tile_id = last_draw.get("tile")
        is_about_kong = False
    else:
        last_discard = snapshot.get("last_discard") or {}
        win_tile_id = last_discard.get("tile")
        is_about_kong = False
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
    visible_counts = snapshot.get("visible_counts") or {}
    visible_count = _visible_count_for_tile(visible_counts, win_tile_id)
    is_4th_tile = event_source != "bugang" and visible_count == 3
    return {
        "packs": [_pack_to_official_meld(pack, seat) for pack in snapshot.get("melds") or []],
        "hand": hand,
        "win_tile": win_tile,
        "flower_count": int(snapshot.get("flowers") or 0),
        "is_self_draw": self_drawn,
        "is_4th_tile": is_4th_tile,
        "is_about_kong": is_about_kong,
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
    if not hu_result or not hu_result.get("can_hu"):
        return False
    fan = _fan_gate_value(hu_result)
    return isinstance(fan, int) and fan >= 8


def _fan_gate_value(hu_result: dict[str, Any]) -> Any:
    return hu_result.get("base_fan", hu_result.get("fan"))


def _allowed_chi_middles(values: list[int], event_tile: str) -> set[str] | None:
    middles: set[str] = set()
    for value in values:
        if not isinstance(value, int) or value <= 0:
            continue
        symbol = botzone_symbol(value)
        if _is_chi_middle_for_event(symbol, event_tile):
            middles.add(symbol)
    return middles or None


def _is_chi_middle_for_event(middle: str, event_tile: str) -> bool:
    try:
        middle_rank = int(middle[1])
        event_rank = int(event_tile[1])
    except (IndexError, ValueError):
        return False
    return (
        len(middle) == 2
        and len(event_tile) == 2
        and middle[0] == event_tile[0]
        and middle[0] in {"W", "B", "T"}
        and 2 <= middle_rank <= 8
        and middle_rank - 1 <= event_rank <= middle_rank + 1
    )


def _chi_shape(middle: str) -> list[str]:
    try:
        rank = int(middle[1])
    except (IndexError, ValueError):
        return [middle]
    return [f"{middle[0]}{rank - 1}", middle, f"{middle[0]}{rank + 1}"]


def _visible_count_for_tile(visible_counts: dict[Any, Any], tile_id: int) -> int:
    kind = tile_id >> 2
    for key in (kind, str(kind), botzone_symbol(tile_id)):
        value = visible_counts.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _add_hu_note(rec: dict[str, Any], hu_result: dict[str, Any] | None) -> None:
    if not hu_result or hu_result.get("can_hu"):
        return
    fan = hu_result.get("fan")
    base_fan = hu_result.get("base_fan")
    gate_fan = base_fan if base_fan is not None else fan
    if gate_fan is not None:
        rec["fan"] = fan
        if base_fan is not None:
            rec["base_fan"] = base_fan
            rec["note"] = f"Hu base fan is {base_fan}, below 8"
        else:
            rec["note"] = f"Hu is {fan} fan, below 8"
        if rec["action"] == "pass":
            rec["text"] = f"{rec['text']} ({rec['note']})"
    elif hu_result.get("reason"):
        rec["note"] = f"Hu not recommended: {hu_result['reason']}"


def _format_fan_items(items: list[dict[str, Any]], limit: int = 6) -> str:
    parts = []
    for item in sorted(items, key=_fan_item_sort_key)[:limit]:
        name = item.get("name")
        total = item.get("total", item.get("fan"))
        if name is None or total is None:
            continue
        parts.append(f"{name} {total}")
    if len(items) > limit:
        parts.append(f"+{len(items) - limit} more")
    return " + ".join(parts)


def _fan_item_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    try:
        total = int(item.get("total", item.get("fan", 0)))
    except (TypeError, ValueError):
        total = 0
    return (-total, str(item.get("name") or ""))


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


def _build_transformer_predictor(model_path: Path):
    from .transformer_predictor import TransformerCheckpointPredictor

    return TransformerCheckpointPredictor(model_path)
