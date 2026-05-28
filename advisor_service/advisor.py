"""Safe fallback recommendation logic for the read-only Tziakcha advisor."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .aleo_bridge import calculate_hu_fan, recommend_with_aleo
from .tiles import display_name, kind_from_tile_id


def recommend(
    snapshot: dict[str, Any],
    use_aleo: bool = False,
    model_advisor: Any | None = None,
) -> dict[str, Any]:
    if model_advisor is not None:
        return model_advisor.recommend(snapshot)

    actions = snapshot.get("available_actions") or {}
    hu_result = None
    if actions.get("hu") and use_aleo:
        hu_result = calculate_hu_fan(snapshot)
        if hu_result and hu_result.get("source") == "aleo" and _can_hu(hu_result):
            fan = int(hu_result["fan"])
            base_fan = hu_result.get("base_fan")
            fan_items = list(hu_result.get("fan_items") or [])
            fan_text = _format_fan_items(fan_items)
            text = f"Hu ({fan} fan: {fan_text})" if fan_text else f"Hu ({fan} fan)"
            result = {
                "action": "hu",
                "text": text,
                "fan": fan,
                "fan_items": fan_items,
                "base_fan_items": list(hu_result.get("base_fan_items") or []),
                "fan_text": fan_text,
                "source": "aleo",
            }
            if base_fan is not None:
                result["base_fan"] = int(base_fan)
            return result

    if use_aleo and _can_ask_aleo(snapshot):
        aleo_recommendation = recommend_with_aleo(snapshot)
        if aleo_recommendation and aleo_recommendation.get("source") == "aleo":
            if aleo_recommendation.get("action") != "hu":
                _add_low_fan_note(aleo_recommendation, hu_result)
                return aleo_recommendation

    if actions.get("kong"):
        tile = int(actions["kong"][0])
        return _tile_action("kong", "Kong", tile)
    if actions.get("pung"):
        tile = int(actions["pung"][0])
        return _tile_action("pung", "Pung", tile)
    if actions.get("chow"):
        tile = int(actions["chow"][0])
        return _tile_action("chow", "Chow around", tile)
    if actions.get("discard"):
        tile = int(actions["discard"][0])
        return _tile_action("discard", "Discard", tile)
    if _should_choose_discard(snapshot):
        tile = choose_discard(snapshot.get("hand") or [])
        if tile is not None:
            return _tile_action("discard", "Discard", tile)
    if actions.get("pass") or actions.get("waive"):
        rec = {"action": "pass", "text": "Pass", "source": "local-advisor"}
        _add_low_fan_note(rec, hu_result)
        return rec
    return {"action": "wait", "text": "Waiting for decision prompt", "source": "local-advisor"}


def choose_discard(hand: list[int]) -> int | None:
    if not hand:
        return None
    counts = Counter(kind_from_tile_id(tile) for tile in hand)

    def score(tile: int) -> tuple[int, int, int]:
        kind = kind_from_tile_id(tile)
        duplicate_score = counts[kind] * 10
        if kind >= 27:
            neighbor_score = 0
            honor_penalty = -2 if counts[kind] == 1 else 2
        else:
            suit_start = (kind // 9) * 9
            suit_end = suit_start + 8
            neighbors = 0
            for offset in (-2, -1, 1, 2):
                neighbor = kind + offset
                if suit_start <= neighbor <= suit_end:
                    neighbors += counts[neighbor]
            neighbor_score = neighbors * 3
            honor_penalty = 0
        return (duplicate_score + neighbor_score + honor_penalty, -kind, tile)

    return min(hand, key=score)


def _tile_action(action: str, verb: str, tile: int) -> dict[str, Any]:
    return {
        "action": action,
        "tile": tile,
        "tile_display": display_name(tile),
        "text": f"{verb} {display_name(tile)}",
        "source": "local-advisor",
    }


def _should_choose_discard(snapshot: dict[str, Any]) -> bool:
    hand = snapshot.get("hand") or []
    return bool(hand) and snapshot.get("seat") is not None and snapshot.get("seat") == snapshot.get("turn")


def _can_ask_aleo(snapshot: dict[str, Any]) -> bool:
    actions = snapshot.get("available_actions") or {}
    return bool(snapshot.get("hand")) and snapshot.get("seat") is not None and (bool(actions) or _should_choose_discard(snapshot))


def _add_low_fan_note(rec: dict[str, Any], hu_result: dict[str, Any] | None) -> None:
    if not hu_result or hu_result.get("source") != "aleo" or "fan" not in hu_result:
        return
    fan = int(hu_result["fan"])
    base_fan = hu_result.get("base_fan")
    gate_fan = int(base_fan) if base_fan is not None else fan
    if gate_fan >= 8:
        return
    rec["fan"] = fan
    if base_fan is not None:
        rec["base_fan"] = int(base_fan)
        rec["note"] = f"Hu base fan is {int(base_fan)}, below 8"
    else:
        rec["note"] = f"Hu is {fan} fan, below 8"
    if rec["action"] in {"pass", "waive"}:
        rec["text"] = f"{rec['text']} ({rec['note']})"


def _can_hu(hu_result: dict[str, Any]) -> bool:
    gate_fan = hu_result.get("base_fan", hu_result.get("fan"))
    try:
        return int(gate_fan) >= 8
    except (TypeError, ValueError):
        return False


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
