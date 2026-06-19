#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path


INVALID_TERMINALS = {"WA", "TLE", "RE"}


def _response_text(payload):
    if not isinstance(payload, dict):
        return ""
    value = payload.get("response") or payload.get("raw") or payload.get("content") or ""
    return str(value).strip().upper()


def _raw_claim_text(payload):
    if not isinstance(payload, dict):
        return ""
    raw = str(payload.get("raw") or "").strip().upper()
    if raw and raw != "PASS":
        return raw
    return _response_text(payload)


def _responses_by_player(response_log):
    if not isinstance(response_log, dict) or "output" in response_log:
        return {}
    out = {}
    for key, payload in response_log.items():
        if str(key).isdigit():
            out[int(key)] = _response_text(payload)
    return out


def _accepted_response(response_log, player):
    return _responses_by_player(response_log).get(player, "")


def _nonpass_response(response_log, player):
    response = _accepted_response(response_log, player)
    return response if response and response != "PASS" else ""


def _response_source_for_player(primary_response_log, fallback_response_log, player):
    if _nonpass_response(primary_response_log, player):
        return primary_response_log
    if _nonpass_response(fallback_response_log, player):
        return fallback_response_log
    return primary_response_log if isinstance(primary_response_log, dict) else fallback_response_log


def _ignore_tokens(response_log, accepted_player, claimed_tile):
    tokens = []
    if not isinstance(response_log, dict) or "output" in response_log:
        return tokens
    responses = {
        int(key): _raw_claim_text(payload)
        for key, payload in response_log.items()
        if str(key).isdigit()
    }
    for player, response in sorted(responses.items()):
        if player == accepted_player or not response or response == "PASS":
            continue
        parts = response.split()
        action = parts[0]
        if action == "CHI" and len(parts) >= 3:
            tokens += ["Ignore", "Player", str(player), "Chi", parts[1]]
        elif action == "PENG":
            tokens += ["Ignore", "Player", str(player), "Peng", claimed_tile]
        elif action == "GANG":
            tokens += ["Ignore", "Player", str(player), "Gang", claimed_tile]
        elif action == "HU":
            tokens += ["Ignore", "Player", str(player), "Hu", claimed_tile]
    return tokens


def _score_line(record, display):
    score = display.get("score") or record.get("scores")
    if isinstance(score, dict):
        return "Score " + " ".join(str(score[str(i)]) for i in range(4))
    if isinstance(score, list) and len(score) == 4:
        return "Score " + " ".join(str(score[i]) for i in range(4))
    return "Score 0 0 0 0"


def _output_content(out):
    content = out.get("content") if isinstance(out, dict) else None
    return content if isinstance(content, dict) else {}


def _next_response_log(logs, index):
    if index + 1 >= len(logs):
        return None
    candidate = logs[index + 1]
    if isinstance(candidate, dict) and "output" not in candidate:
        return candidate
    return None


def _next_output(logs, index):
    for candidate in logs[index + 1 :]:
        if not isinstance(candidate, dict):
            continue
        out = candidate.get("output")
        if isinstance(out, dict):
            return out
    return None


def _accepted_nonpass(response_log):
    for player, response in sorted(_responses_by_player(response_log).items()):
        if response and response != "PASS":
            return player, response
    return None, ""


def _wind_from_content(content):
    for value in content.values():
        parts = str(value).split()
        if len(parts) >= 3 and parts[0] == "0":
            return parts[2]
    return None


def _deal_hands(display, content):
    hands = display.get("hand")
    if isinstance(hands, list) and len(hands) == 4:
        return hands

    parsed = []
    for player in range(4):
        parts = str(content.get(str(player), "")).split()
        if len(parts) < 18 or parts[0] != "1":
            return None
        parsed.append(parts[5:])
    return parsed


def _draw_player_tile(display, content):
    if "player" in display and "tile" in display:
        return int(display["player"]), display["tile"]
    for player, value in content.items():
        parts = str(value).split()
        if len(parts) >= 2 and parts[0] == "2":
            return int(player), parts[1]
    return None, None


def _reaction_event(content):
    for value in content.values():
        parts = str(value).split()
        if len(parts) >= 4 and parts[0] == "3":
            return int(parts[1]), parts[2].upper(), parts[3]
    return None, None, None


def _play_player_tile(display, content):
    if "player" in display and "tile" in display:
        return int(display["player"]), display["tile"]
    player, event_action, tile = _reaction_event(content)
    if event_action == "PLAY":
        return player, tile
    return None, None


def _next_output_is_play(logs, index, player, tile):
    out = _next_output(logs, index)
    if not isinstance(out, dict):
        return False
    display = out.get("display") or {}
    action = str(display.get("action") or "").upper()
    if action != "PLAY":
        return False
    next_player, next_tile = _play_player_tile(display, _output_content(out))
    return next_player == player and next_tile == tile


def convert_record(record):
    logs = record.get("logs")
    if not isinstance(logs, list):
        return None, "missing_logs", Counter()

    match_id = record.get("match_id") or record.get("id") or "unknown"
    lines = [f"Match {match_id}"]
    train_players = [
        str(player)
        for player in (record.get("train_players") or [])
        if str(player).isdigit()
    ]
    if train_players:
        lines.append("TrainPlayers " + " ".join(train_players))
    stats = Counter()
    saw_init = False
    saw_deal = False
    terminal = None
    prev_response_log = None
    last_discard = None
    skip_next_play = None
    skip_next_meld = None

    for index, log in enumerate(logs):
        out = log.get("output") if isinstance(log, dict) else None
        if not isinstance(out, dict):
            prev_response_log = log
            continue

        display = out.get("display") or {}
        content = _output_content(out)
        response_log = _next_response_log(logs, index)
        action = str(display.get("action") or "").upper()
        if not action:
            continue
        stats[f"action:{action}"] += 1

        if action in INVALID_TERMINALS:
            return None, f"invalid_terminal:{action}", stats
        if action == "BUHUA":
            return None, "unsupported_buhua", stats

        if action == "INIT":
            quan = display.get("quan", _wind_from_content(content))
            if quan is None:
                quan = (record.get("initdata") or {}).get("quan", 0)
            lines.append(f"Wind {quan}")
            saw_init = True
        elif action == "DEAL":
            hands = _deal_hands(display, content)
            if not isinstance(hands, list) or len(hands) != 4:
                return None, "bad_deal", stats
            for player, hand in enumerate(hands):
                lines.append(f"Player {player} Deal {' '.join(hand)}")
            saw_deal = True
        elif action == "DRAW":
            player, tile = _draw_player_tile(display, content)
            if player is None or tile is None:
                return None, "bad_draw", stats
            lines.append(f"Player {player} Draw {tile}")
            response = _accepted_response(response_log, player)
            parts = response.split()
            if len(parts) >= 2 and parts[0] == "PLAY":
                discard = parts[1]
                lines.append(f"Player {player} Play {discard}")
                if _next_output_is_play(logs, index, player, discard):
                    skip_next_play = (player, discard)
                last_discard = discard
            elif len(parts) >= 2 and parts[0] == "BUGANG":
                lines.append(f"Player {player} BuGang {parts[1]}")
                skip_next_meld = ("BUGANG", player, parts[1])
            elif len(parts) >= 2 and parts[0] == "GANG":
                lines.append(f"Player {player} AnGang {parts[1]}")
                skip_next_meld = ("GANG", player, parts[1])
            elif response == "HU":
                lines.append(f"Player {player} Hu {tile}")
                lines.append(_score_line(record, display))
                terminal = "HU"
        elif action == "PLAY":
            player, tile = _play_player_tile(display, content)
            if player is None or tile is None:
                return None, "bad_play", stats
            if skip_next_play == (player, tile):
                skip_next_play = None
            else:
                lines.append(f"Player {player} Play {tile}")
            last_discard = tile
        elif action == "PENG":
            if "player" in display:
                player = int(display["player"])
                response_source = _response_source_for_player(prev_response_log, response_log, player)
                response = _accepted_response(response_source, player)
            else:
                response_source = response_log
                player, response = _accepted_nonpass(response_source)
            parts = response.split()
            if len(parts) != 2 or parts[0] != "PENG" or last_discard is None:
                return None, "bad_peng_response", stats
            line = ["Player", str(player), "Peng", last_discard]
            line += _ignore_tokens(response_source, player, last_discard)
            lines.append(" ".join(line))
            discard = parts[1]
            lines.append(f"Player {player} Play {discard}")
            last_discard = discard
        elif action == "CHI":
            if "player" in display:
                player = int(display["player"])
                response_source = _response_source_for_player(prev_response_log, response_log, player)
                response = _accepted_response(response_source, player)
            else:
                response_source = response_log
                player, response = _accepted_nonpass(response_source)
            parts = response.split()
            if len(parts) != 3 or parts[0] != "CHI":
                return None, "bad_chi_response", stats
            center_tile = parts[1]
            discard = parts[2]
            lines.append(f"Player {player} Chi {center_tile}")
            lines.append(f"Player {player} Play {discard}")
            last_discard = discard
        elif action == "GANG":
            if "player" in display:
                player = int(display["player"])
                response_source = _response_source_for_player(prev_response_log, response_log, player)
                response = _accepted_response(response_source, player)
            else:
                response_source = response_log
                player, response = _accepted_nonpass(response_source)
            parts = response.split()
            if len(parts) == 2 and parts[0] == "GANG":
                if skip_next_meld == ("GANG", player, parts[1]):
                    skip_next_meld = None
                else:
                    lines.append(f"Player {player} AnGang {parts[1]}")
            elif response == "GANG" and last_discard is not None:
                line = ["Player", str(player), "Gang", last_discard]
                line += _ignore_tokens(response_source, player, last_discard)
                lines.append(" ".join(line))
            else:
                return None, "bad_gang_response", stats
        elif action == "BUGANG":
            player, tile = _draw_player_tile(display, content)
            if player is None or tile is None:
                player = int(display["player"])
                tile = display["tile"]
            if skip_next_meld == ("BUGANG", player, tile):
                skip_next_meld = None
            else:
                lines.append(f"Player {player} BuGang {tile}")
        elif action == "HU":
            response_source = None
            if "player" in display:
                player = int(display["player"])
                response_source = _response_source_for_player(prev_response_log, response_log, player)
            else:
                player, response = _accepted_nonpass(response_log)
                response_source = response_log
                if player is None:
                    player, response = _accepted_nonpass(prev_response_log)
                    response_source = prev_response_log
            if player is None:
                return None, "bad_hu_response", stats
            claimed_tile = display.get("tile") or last_discard
            line = ["Player", str(player), "Hu"]
            if claimed_tile:
                line.append(str(claimed_tile))
                line += _ignore_tokens(response_source, player, str(claimed_tile))
            lines.append(" ".join(line))
            lines.append(_score_line(record, display))
            terminal = "HU"
        elif action == "HUANG":
            lines.append(_score_line(record, display))
            terminal = "HUANG"
        elif action == "PASS":
            continue

    if not saw_init:
        return None, "missing_init", stats
    if not saw_deal:
        return None, "missing_deal", stats
    if terminal is None:
        lines.append(_score_line(record, {}))
        terminal = "HUANG"
    return lines, terminal, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Botzone JSONL logs to feature-agent transcript text.")
    parser.add_argument("--input", action="append", required=True, help="Botzone JSONL path. Can be repeated.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--allow-tziakcha-input",
        action="store_true",
        help="Allow paths containing 'tziakcha'. Use only after the records have been converted to Botzone-style JSONL.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = [Path(path) for path in args.input]
    for path in inputs:
        if "tziakcha" in str(path).lower() and not args.allow_tziakcha_input:
            raise RuntimeError(f"Refusing non-Botzone/tziakcha input path: {path}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "records_seen": 0,
        "matches_written": 0,
        "skip_reasons": Counter(),
        "terminal_counts": Counter(),
        "action_counts": Counter(),
    }

    with output.open("w", encoding="utf-8", newline="\n") as out:
        for path in inputs:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if args.max_records and summary["records_seen"] >= args.max_records:
                        break
                    summary["records_seen"] += 1
                    record = json.loads(line)
                    converted, status, stats = convert_record(record)
                    summary["action_counts"].update(stats)
                    if converted is None:
                        summary["skip_reasons"][status] += 1
                        continue
                    summary["terminal_counts"][status] += 1
                    summary["matches_written"] += 1
                    out.write("\n".join(converted))
                    out.write("\n")

    serializable = {
        **summary,
        "skip_reasons": dict(summary["skip_reasons"]),
        "terminal_counts": dict(summary["terminal_counts"]),
        "action_counts": dict(summary["action_counts"]),
    }
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(serializable, ensure_ascii=False))


if __name__ == "__main__":
    main()
