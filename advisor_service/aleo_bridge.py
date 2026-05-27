"""Optional bridge to Aleo's C++ bot CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .tiles import aleo_kind_from_tile_id, display_name, tile_id_from_aleo_kind

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALEO_BINARY = WORKSPACE_ROOT / "aleo_bridge" / "bin" / "aleo_advisor"


def physical_to_aleo_id(tile_id: int) -> int:
    return aleo_kind_from_tile_id(tile_id) + 1


def aleo_id_to_physical(aleo_id: int, hand: list[int] | None = None) -> int:
    aleo_kind = aleo_id - 1
    for tile in hand or []:
        if aleo_kind_from_tile_id(tile) == aleo_kind:
            return tile
    return tile_id_from_aleo_kind(aleo_kind)


def snapshot_to_aleo_args(snapshot: dict[str, Any]) -> list[str]:
    last_discard = snapshot.get("last_discard") or {}
    last_player = last_discard.get("seat")
    last_tile = last_discard.get("tile")
    wall_count = snapshot.get("wall_count")
    args = [
        "--seat",
        str(snapshot.get("seat", 0)),
        "--turn",
        str(snapshot.get("turn", snapshot.get("seat", 0))),
        "--wall",
        str(70 if wall_count is None else wall_count),
        "--hand",
        ",".join(str(physical_to_aleo_id(tile)) for tile in snapshot.get("hand") or []),
    ]
    if last_player is not None and last_tile is not None:
        args.extend(["--last-player", str(last_player), "--last-tile", str(physical_to_aleo_id(last_tile))])
    return args


def snapshot_to_hu_fan_args(snapshot: dict[str, Any]) -> list[str] | None:
    context = _hu_win_context(snapshot)
    if context is None:
        return None
    win_tile, self_drawn = context
    standing_hand = _standing_hand_for_hu(snapshot.get("hand") or [], win_tile, self_drawn)
    wall_count = snapshot.get("wall_count")
    args = [
        "--mode",
        "fan",
        "--seat",
        str(snapshot.get("seat", 0)),
        "--turn",
        str(snapshot.get("turn", snapshot.get("seat", 0))),
        "--wall",
        str(70 if wall_count is None else wall_count),
        "--prevalent-wind",
        str(int(snapshot.get("prevalent_wind") or 0)),
        "--hand",
        ",".join(str(physical_to_aleo_id(tile)) for tile in standing_hand),
        "--win-tile",
        str(physical_to_aleo_id(win_tile)),
        "--self-drawn",
        "1" if self_drawn else "0",
        "--flowers",
        str(int(snapshot.get("flowers") or 0)),
    ]
    melds = [_pack_to_aleo_meld(pack) for pack in snapshot.get("melds") or []]
    if melds:
        args.extend(["--melds", ",".join(melds)])
    return args


def calculate_hu_fan(snapshot: dict[str, Any], timeout: float = 8.0) -> dict[str, Any] | None:
    command = find_aleo_command()
    args = snapshot_to_hu_fan_args(snapshot)
    if not command or args is None:
        return None
    try:
        completed = subprocess.run(
            command + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"source": "aleo-error", "error": f"Aleo fan calculation timed out after {timeout:g}s"}
    if completed.returncode != 0:
        return {"source": "aleo-error", "error": completed.stderr.strip() or completed.stdout.strip()}
    try:
        return {"fan": _parse_fan_output(completed.stdout), "source": "aleo"}
    except ValueError as exc:
        return {"source": "aleo-error", "error": str(exc)}


def parse_aleo_output(output: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    parts = _last_action_parts(output)
    if not parts:
        raise ValueError("empty Aleo output")
    action = parts[0].upper()
    hand = [int(tile) for tile in snapshot.get("hand") or []]
    if action == "HU":
        return {"action": "hu", "text": "Hu", "source": "aleo"}
    if action == "PASS":
        return {"action": "pass", "text": "Pass", "source": "aleo"}
    if action == "PLAY" and len(parts) >= 2:
        return _tile_recommendation("discard", "Discard", int(parts[1]), hand)
    if action == "PENG" and len(parts) >= 2:
        return _tile_recommendation("pung", "Pung; discard", int(parts[1]), hand)
    if action == "CHI" and len(parts) >= 3:
        return _tile_recommendation("chow", "Chow; discard", int(parts[2]), hand)
    if action == "GANG":
        if len(parts) >= 2:
            return _tile_recommendation("kong", "Kong", int(parts[1]), hand)
        return {"action": "kong", "text": "Kong", "source": "aleo"}
    if action == "BUGANG":
        return {"action": "kong", "text": "Add Kong", "source": "aleo"}
    raise ValueError(f"unsupported Aleo output: {output!r}")


def _last_action_parts(output: str) -> list[str]:
    known = {"HU", "PASS", "PLAY", "PENG", "CHI", "GANG", "BUGANG"}
    for line in reversed(output.strip().splitlines()):
        parts = line.strip().split()
        if parts and parts[0].upper() in known:
            return parts
    return output.strip().split()


def recommend_with_aleo(snapshot: dict[str, Any], timeout: float = 8.0) -> dict[str, Any] | None:
    command = find_aleo_command()
    if not command:
        return None
    try:
        completed = subprocess.run(
            command + snapshot_to_aleo_args(snapshot),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "action": "wait",
            "text": f"Aleo timed out after {timeout:g}s; using local advisor",
            "source": "aleo-error",
        }
    if completed.returncode != 0:
        return {
            "action": "wait",
            "text": "Aleo unavailable; using local advisor",
            "source": "aleo-error",
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    try:
        return parse_aleo_output(completed.stdout, snapshot)
    except (TypeError, ValueError) as exc:
        return {
            "action": "wait",
            "text": "Aleo returned invalid advice; using local advisor",
            "source": "aleo-error",
            "error": str(exc),
        }


def find_aleo_command() -> list[str] | None:
    explicit = os.environ.get("ALEO_ADVISOR_CMD")
    if explicit:
        return explicit.split()
    if DEFAULT_ALEO_BINARY.exists() and shutil.which("wsl.exe"):
        distro = os.environ.get("ALEO_ADVISOR_WSL_DISTRO", "CodexUbuntu2404")
        return ["wsl.exe", "-d", distro, "--", windows_path_to_wsl(DEFAULT_ALEO_BINARY)]
    return None


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive[:1].lower()
    rest = str(resolved)[3:].replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _tile_recommendation(action: str, verb: str, aleo_id: int, hand: list[int]) -> dict[str, Any]:
    tile = aleo_id_to_physical(aleo_id, hand)
    return {
        "action": action,
        "tile": tile,
        "tile_display": display_name(tile),
        "text": f"{verb} {display_name(tile)}",
        "source": "aleo",
    }


def _hu_win_context(snapshot: dict[str, Any]) -> tuple[int, bool] | None:
    seat = snapshot.get("seat")
    turn = snapshot.get("turn")
    if seat is not None and seat == turn:
        last_draw = snapshot.get("last_draw") or {}
        tile = last_draw.get("tile")
        if isinstance(tile, int):
            return tile, True
    last_discard = snapshot.get("last_discard") or {}
    tile = last_discard.get("tile")
    if isinstance(tile, int):
        return tile, False
    return None


def _standing_hand_for_hu(hand: list[int], win_tile: int, self_drawn: bool) -> list[int]:
    standing = [int(tile) for tile in hand]
    if self_drawn:
        win_aleo_id = physical_to_aleo_id(win_tile)
        for index, tile in enumerate(standing):
            if physical_to_aleo_id(tile) == win_aleo_id:
                del standing[index]
                break
    return standing


def _pack_to_aleo_meld(pack: int) -> str:
    pack_type = (int(pack) >> 8) & 3
    fan_type = {0: 1, 1: 2, 2: 3, 3: 3}.get(pack_type, 2)
    physical_tile = (int(pack) & 63) << 2
    offer = (int(pack) >> 6) & 3
    return f"{fan_type}:{physical_to_aleo_id(physical_tile)}:{offer}"


def _parse_fan_output(output: str) -> int:
    for line in reversed(output.strip().splitlines()):
        parts = line.strip().split()
        if len(parts) == 2 and parts[0].upper() == "FAN":
            return int(parts[1])
    raise ValueError(f"missing fan output: {output!r}")
