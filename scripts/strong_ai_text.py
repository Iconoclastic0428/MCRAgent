#!/usr/bin/env python3
"""Convert organizer strong-AI MCR text logs to Botzone-like raw JSONL."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable, Iterator


PLAYER_RE = re.compile(r"^Player ([0-3]) ([A-Za-z]+)(?: (.*))?$")
IGNORE_RE = re.compile(r"Ignore Player ([0-3]) ([A-Za-z]+) ([A-Z][0-9])")


def make_log(content: dict[str, str], responses: dict[str, str], action: str | None = None) -> list[dict]:
    display = {"action": action} if action else {}
    return [
        {"output": {"content": content, "display": display}},
        {
            player: {"response": response, "raw": response, "verdict": "OK"}
            for player, response in responses.items()
        },
    ]


def parse_player_line(line: str) -> tuple[int, str, list[str]]:
    match = PLAYER_RE.match(line.strip())
    if not match:
        raise ValueError(f"not a player action line: {line}")
    player = int(match.group(1))
    action = match.group(2)
    rest = (match.group(3) or "").split()
    ignore_index = rest.index("Ignore") if "Ignore" in rest else len(rest)
    return player, action, rest[:ignore_index]


def ignored_players(line: str) -> set[int]:
    return {int(match.group(1)) for match in IGNORE_RE.finditer(line)}


def score_dict(line: str) -> dict[str, int]:
    parts = line.split()
    if len(parts) != 5 or parts[0] != "Score":
        raise ValueError(f"invalid score line: {line}")
    return {str(index): int(parts[index + 1]) for index in range(4)}


def pass_reaction_log(actor: int, event_action: str, tile: str) -> list[dict]:
    content = {
        str(other): f"3 {actor} {event_action} {tile}"
        for other in range(4)
        if other != actor
    }
    return make_log(content, {player: "PASS" for player in content}, "PASS")


def response_for_draw_action(action: str, args: list[str]) -> str:
    normalized = action.upper()
    if normalized == "PLAY" and len(args) == 1:
        return f"PLAY {args[0]}"
    if normalized == "HU":
        return "HU"
    if normalized == "ANGANG" and len(args) == 1:
        return f"GANG {args[0]}"
    if normalized == "BUGANG" and len(args) == 1:
        return f"BUGANG {args[0]}"
    if normalized == "GANG" and len(args) == 1:
        return f"GANG {args[0]}"
    raise ValueError(f"unsupported draw response: {action} {' '.join(args)}")


def reaction_response(action: str, args: list[str], next_line: str | None) -> tuple[str, bool]:
    normalized = action.upper()
    if normalized == "HU":
        return "HU", False
    if normalized == "GANG":
        return "GANG", False
    if normalized in {"PENG", "CHI"}:
        if not next_line:
            raise ValueError(f"{action} requires a following discard line")
        next_player, next_action, next_args = parse_player_line(next_line)
        if next_action.upper() != "PLAY" or len(next_args) != 1:
            raise ValueError(f"{action} must be followed by a Play line")
        if normalized == "PENG":
            return f"PENG {next_args[0]}", True
        if len(args) != 1:
            raise ValueError(f"Chi requires middle tile: {action} {' '.join(args)}")
        return f"CHI {args[0]} {next_args[0]}", True
    raise ValueError(f"unsupported reaction response: {action} {' '.join(args)}")


def is_terminal_line(line: str) -> bool:
    return line.startswith("Fan ") or line == "Huang" or line.startswith("Score ")


def parse_round_lines(lines: list[str]) -> dict:
    clean = [line.strip() for line in lines if line.strip()]
    if len(clean) < 7 or not clean[0].startswith("Match ") or not clean[1].startswith("Wind "):
        raise ValueError("round must start with Match and Wind lines")

    match_id = clean[0].split(maxsplit=1)[1]
    wind = int(clean[1].split()[1])
    deal_lines = clean[2:6]
    logs: list[dict] = []

    init_content = {str(player): f"0 {player} {wind}" for player in range(4)}
    init_responses = {str(player): "PASS" for player in range(4)}
    logs.extend(make_log(init_content, init_responses, "INIT"))

    deal_content: dict[str, str] = {}
    for line in deal_lines:
        player, action, tiles = parse_player_line(line)
        if action != "Deal" or len(tiles) != 13:
            raise ValueError(f"invalid deal line: {line}")
        deal_content[str(player)] = "1 0 0 0 0 " + " ".join(tiles)
    deal_responses = {player: "PASS" for player in deal_content}
    logs.extend(make_log(deal_content, deal_responses, "DEAL"))

    scores: dict[str, int] | None = None
    index = 6
    pending_event: tuple[int, str, str] | None = None
    while index < len(clean):
        line = clean[index]
        if pending_event is not None:
            action = ""
            if line.startswith("Player "):
                _, action, _ = parse_player_line(line)
            if is_terminal_line(line) or action.upper() not in {"CHI", "PENG", "GANG", "HU"}:
                logs.extend(pass_reaction_log(*pending_event))
                pending_event = None

        if line.startswith("Score "):
            scores = score_dict(line)
            index += 1
            continue
        if is_terminal_line(line):
            index += 1
            continue

        player, action, args = parse_player_line(line)
        normalized = action.upper()

        if normalized == "DRAW":
            if len(args) != 1:
                raise ValueError(f"invalid draw line: {line}")
            if index + 1 >= len(clean):
                raise ValueError("draw line missing response")
            response_player, response_action, response_args = parse_player_line(clean[index + 1])
            if response_player != player:
                raise ValueError(f"draw response belongs to another player: {clean[index + 1]}")
            response = response_for_draw_action(response_action, response_args)
            logs.extend(make_log({str(player): f"2 {args[0]}"}, {str(player): response}, normalized))
            if response.startswith("PLAY "):
                pending_event = (player, "PLAY", response.split()[1])
                index += 2
            elif response.startswith("BUGANG "):
                pending_event = (player, "BUGANG", response.split()[1])
                index += 2
            else:
                pending_event = None
                index += 2
            continue

        if normalized == "PLAY":
            if len(args) != 1:
                raise ValueError(f"invalid play line: {line}")
            pending_event = (player, "PLAY", args[0])
            index += 1
            continue

        if normalized in {"CHI", "PENG", "GANG", "HU"}:
            if pending_event is None:
                raise ValueError(f"reaction without prior discard: {line}")
            actor, event_action, tile = pending_event
            consumed_next = False
            response, consumed_next = reaction_response(
                action,
                args,
                clean[index + 1] if index + 1 < len(clean) else None,
            )
            ignored = ignored_players(line)
            responders = {
                str(other): f"3 {actor} {event_action} {tile}"
                for other in range(4)
                if other != actor and other not in ignored
            }
            responses = {other: "PASS" for other in responders}
            responders[str(player)] = f"3 {actor} {event_action} {tile}"
            responses[str(player)] = response
            logs.extend(make_log(responders, responses, normalized))
            if response.startswith(("PENG ", "CHI ")) and consumed_next:
                pending_event = (player, "PLAY", response.split()[-1])
                index += 2
            else:
                pending_event = None
                index += 1
            continue

        raise ValueError(f"unsupported action line: {line}")

    if scores is None:
        raise ValueError(f"round {match_id} missing Score line")
    return {
        "match_id": match_id,
        "game": "Chinese-Standard-Mahjong",
        "source": "botzone_organizer_strong_ai_text",
        "scores": scores,
        "log_count": len(logs),
        "turn_count": max(0, len(logs) // 2),
        "logs": logs,
    }


def iter_rounds(lines: Iterable[str]) -> Iterator[list[str]]:
    current: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Match ") and current:
            yield current
            current = [line]
        else:
            current.append(line)
    if current:
        yield current


def convert_zip(zip_path: Path, out_path: Path, limit: int | None = None) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rounds = 0
    written = 0
    errors: list[dict] = []
    with zipfile.ZipFile(zip_path) as archive, archive.open("data.txt") as raw, out_path.open(
        "w", encoding="utf-8"
    ) as out:
        text_lines = (line.decode("utf-8", errors="replace") for line in raw)
        for round_lines in iter_rounds(text_lines):
            rounds += 1
            if limit is not None and written >= limit:
                break
            try:
                record = parse_round_lines(round_lines)
            except Exception as exc:
                errors.append({"round_index": rounds - 1, "error": str(exc)})
                continue
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    return {
        "zip": str(zip_path),
        "out": str(out_path),
        "rounds_seen": rounds,
        "records_written": written,
        "errors": errors[:20],
        "error_count": len(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="data/raw/botzone_2026_strong_ai_data.zip")
    parser.add_argument("--out", default="data/raw/botzone_2026_strong_ai_raw.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = convert_zip(Path(args.zip), Path(args.out), limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
