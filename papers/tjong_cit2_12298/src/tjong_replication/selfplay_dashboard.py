"""Dashboard-style reports for Tjong self-play records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FAN_GROUPS: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = (
    (
        "1番",
        (
            ("一般高", 1),
            ("喜相逢", 1),
            ("连六", 1),
            ("老少副", 1),
            ("幺九刻", 1),
            ("明杠", 1),
            ("缺一门", 1),
            ("无字", 1),
            ("独听・边张", 1),
            ("独听・嵌张", 1),
            ("独听・单钓", 1),
            ("自摸", 1),
        ),
    ),
    (
        "2番",
        (
            ("箭刻", 2),
            ("圈风刻", 2),
            ("门风刻", 2),
            ("门前清", 2),
            ("平和", 2),
            ("四归一", 2),
            ("双同刻", 2),
            ("双暗刻", 2),
            ("暗杠", 2),
            ("断幺", 2),
        ),
    ),
    ("4番", (("全带幺", 4), ("不求人", 4), ("双明杠", 4), ("和绝张", 4))),
    ("5番（增补）", (("明暗杠", 5),)),
    (
        "6番",
        (
            ("碰碰和", 6),
            ("混一色", 6),
            ("三色三步高", 6),
            ("五门齐", 6),
            ("全求人", 6),
            ("双暗杠", 6),
            ("双箭刻", 6),
        ),
    ),
    (
        "8番",
        (
            ("花龙", 8),
            ("推不倒", 8),
            ("三色三同顺", 8),
            ("三色三节高", 8),
            ("无番和", 8),
            ("妙手回春", 8),
            ("海底捞月", 8),
            ("杠上开花", 8),
            ("抢杠和", 8),
            ("※ 天和", 8),
            ("※ 地和", 8),
            ("※ 人和Ⅰ", 8),
            ("※ 人和Ⅱ", 8),
        ),
    ),
    (
        "12番",
        (("全不靠", 12), ("组合龙", 12), ("大于五", 12), ("小于五", 12), ("三风刻", 12)),
    ),
    (
        "16番",
        (
            ("清龙", 16),
            ("三色双龙会", 16),
            ("一色三步高", 16),
            ("一色三连环", 16),
            ("全带五", 16),
            ("三同刻", 16),
            ("三暗刻", 16),
        ),
    ),
    (
        "24番",
        (
            ("七对", 24),
            ("七星不靠", 24),
            ("全双刻", 24),
            ("清一色", 24),
            ("一色三同顺", 24),
            ("一色三节高", 24),
            ("全大", 24),
            ("全中", 24),
            ("全小", 24),
        ),
    ),
    ("32番", (("一色四步高", 32), ("一色四连环", 32), ("三杠", 32), ("混幺九", 32))),
    ("48番", (("一色四同顺", 48), ("一色四节高", 48))),
    (
        "64番",
        (
            ("清幺九", 64),
            ("小四喜", 64),
            ("小三元", 64),
            ("字一色", 64),
            ("四暗刻", 64),
            ("一色双龙会", 64),
        ),
    ),
    (
        "88番",
        (
            ("大四喜", 88),
            ("大三元", 88),
            ("绿一色", 88),
            ("九莲宝灯", 88),
            ("四杠", 88),
            ("连七对", 88),
            ("十三幺", 88),
        ),
    ),
)

FAN_NAME_ALIASES = {
    "边张": "独听・边张",
    "嵌张": "独听・嵌张",
    "单钓将": "独听・单钓",
    "单钓": "独听・单钓",
}

COUFAN_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("半凑番", ("不求人", "双明杠", "和绝张")),
    ("银", ("箭刻", "圈风刻", "门风刻", "门前清", "平和", "四归一", "双同刻", "双暗刻", "暗杠", "断幺")),
    ("金", ("纯正式", "双幺九式")),
    ("基础", ("门清平和", "门清平和断幺")),
    ("其他", ("2*2", "2*3", "2*4", "2*5+")),
)


def canonical_fan_name(name: str) -> str:
    return FAN_NAME_ALIASES.get(str(name).strip(), str(name).strip())


def display_action(record: dict[str, Any]) -> str:
    display = ((record.get("final_output") or {}).get("display") or {})
    return str(display.get("action") or record.get("terminal_reason") or "UNKNOWN").upper()


def display_player(display: dict[str, Any]) -> int | None:
    try:
        player = int(display.get("player"))
    except (TypeError, ValueError):
        return None
    return player if 0 <= player < 4 else None


def numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def response_text(response_record: Any) -> str:
    if isinstance(response_record, dict):
        return str(response_record.get("response") or response_record.get("raw") or "")
    return str(response_record or "")


def record_scores(record: dict[str, Any]) -> list[float]:
    scores = record.get("scores")
    if isinstance(scores, dict):
        return [float(scores.get(str(player), 0.0) or 0.0) for player in range(4)]
    if isinstance(scores, list):
        padded = [*scores, 0.0, 0.0, 0.0, 0.0]
        return [float(padded[player] or 0.0) for player in range(4)]
    display_scores = ((record.get("final_output") or {}).get("display") or {}).get("score")
    if isinstance(display_scores, list):
        padded = [*display_scores, 0.0, 0.0, 0.0, 0.0]
        return [float(padded[player] or 0.0) for player in range(4)]
    return [0.0, 0.0, 0.0, 0.0]


def infer_deal_in_loser(scores: list[float], winner: int | None) -> int | None:
    if winner is None:
        return None
    others = [(player, score) for player, score in enumerate(scores[:4]) if player != winner]
    if not others:
        return None
    minimum = min(score for _, score in others)
    losers = [player for player, score in others if score == minimum]
    return losers[0] if len(losers) == 1 else None


def request_event_tile(tokens: list[str]) -> str | None:
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2].upper() in {"PLAY", "PENG", "CHI"}:
        return tokens[-1]
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2].upper() == "BUGANG":
        return tokens[3]
    return None


def request_turn(tokens: list[str], seat: int, draw_rounds: list[int]) -> float | None:
    if not tokens:
        return None
    if tokens[0] == "2":
        return float(max(1, draw_rounds[seat]))
    if tokens[0] == "3" and len(tokens) >= 2:
        try:
            actor = int(tokens[1]) % 4
        except ValueError:
            return None
        return float(max(1, draw_rounds[actor]))
    return None


def scan_record_timing(record: dict[str, Any]) -> dict[str, Any]:
    draw_rounds = [0, 0, 0, 0]
    first_ready_turns: list[float | None] = [None, None, None, None]
    open_meld_counts = [0, 0, 0, 0]
    meld_group_turns: list[list[float]] = [[], [], [], []]
    per_player_actions = [Counter() for _ in range(4)]
    hu_context: dict[str, Any] | None = None
    logs = record.get("logs") or []

    for index in range(0, len(logs) - 1, 2):
        output = (logs[index] or {}).get("output") or {}
        display = output.get("display") or {}
        action = str(display.get("action") or "").upper()
        actor = display_player(display)
        if action == "DRAW" and actor is not None:
            draw_rounds[actor] += 1

        display_turn = float(max(1, draw_rounds[actor])) if actor is not None else None
        can_hu = display.get("canHu")
        if isinstance(can_hu, list):
            for player, value in enumerate(can_hu[:4]):
                fan = numeric(value)
                if fan is not None and fan >= 8 and first_ready_turns[player] is None:
                    first_ready_turns[player] = display_turn

        requests = output.get("content") or {}
        response_log = logs[index + 1] if isinstance(logs[index + 1], dict) else {}
        request_tokens_by_seat: dict[int, list[str]] = {}
        turn_by_seat: dict[int, float | None] = {}
        for seat_text, request in requests.items():
            try:
                seat = int(seat_text)
            except ValueError:
                continue
            tokens = str(request).strip().split()
            request_tokens_by_seat[seat] = tokens
            turn_by_seat[seat] = request_turn(tokens, seat, draw_rounds)

        for seat_text, response_record in response_log.items():
            try:
                seat = int(seat_text)
            except ValueError:
                continue
            response = response_text(response_record).strip()
            if not response:
                continue
            parts = response.split()
            head = parts[0].upper()
            tokens = request_tokens_by_seat.get(seat, [])
            turn = turn_by_seat.get(seat)
            if head == "HU" and hu_context is None:
                hu_context = {
                    "player": seat,
                    "turn": turn,
                    "self_draw": bool(tokens and tokens[0] == "2"),
                    "from_player": int(tokens[1]) % 4 if len(tokens) >= 2 and tokens[0] == "3" and tokens[1].isdigit() else None,
                    "win_tile": tokens[1] if len(tokens) >= 2 and tokens[0] == "2" else request_event_tile(tokens),
                    "request": " ".join(tokens),
                }
            if head == "CHI":
                per_player_actions[seat]["chi_count"] += 1
            elif head == "PENG":
                per_player_actions[seat]["pong_count"] += 1
            elif head == "BUGANG":
                per_player_actions[seat]["kang_count"] += 1
                per_player_actions[seat]["bukong_count"] += 1
            elif head == "GANG":
                per_player_actions[seat]["kang_count"] += 1
                if tokens and tokens[0] == "2":
                    per_player_actions[seat]["ankong_count"] += 1
                else:
                    per_player_actions[seat]["mingkong_count"] += 1

            if head in {"CHI", "PENG"} or (head == "GANG" and (not tokens or tokens[0] != "2")):
                if turn is not None:
                    open_meld_counts[seat] += 1
                    ordinal = open_meld_counts[seat]
                    if 1 <= ordinal <= 4:
                        meld_group_turns[ordinal - 1].append(float(turn))

    return {
        "hu_context": hu_context,
        "first_ready_turns": first_ready_turns,
        "meld_group_turns": meld_group_turns,
        "per_player_actions": per_player_actions,
    }


def fan_items(display: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in display.get("fan") or []:
        name = canonical_fan_name(str(item.get("name") or "UNKNOWN"))
        value = numeric(item.get("value"))
        count = numeric(item.get("cnt"))
        items.append({"name": name, "value": int(value or 0), "cnt": int(count or 1)})
    return items


def add_name_stat(stats: dict[str, dict[str, Any]], name: str, turn: float | None) -> None:
    item = stats.setdefault(name, {"count": 0, "turn_sum": 0.0, "turn_count": 0})
    item["count"] += 1
    if turn is not None:
        item["turn_sum"] += float(turn)
        item["turn_count"] += 1


def update_coufan(stats: dict[str, dict[str, Any]], items: list[dict[str, Any]], turn: float | None) -> bool:
    non_flower = [item for item in items if item["name"] != "花牌"]
    if not non_flower or max(item["value"] for item in non_flower) > 4:
        return False
    if sum(item["value"] * item["cnt"] for item in non_flower) < 8:
        return False

    by_name = {item["name"]: item for item in non_flower}
    for name in ("不求人", "双明杠", "和绝张"):
        if name in by_name:
            add_name_stat(stats, name, turn)
    for name in COUFAN_SECTIONS[1][1]:
        if name in by_name:
            add_name_stat(stats, name, turn)

    one_fan_count = sum(item["cnt"] for item in non_flower if item["value"] == 1)
    two_fan_count = sum(item["cnt"] for item in non_flower if item["value"] == 2)
    if two_fan_count == 0 and one_fan_count >= 8:
        add_name_stat(stats, "双幺九式" if by_name.get("幺九刻", {}).get("cnt", 0) >= 2 else "纯正式", turn)

    if "门前清" in by_name and "平和" in by_name:
        add_name_stat(stats, "门清平和断幺" if "断幺" in by_name else "门清平和", turn)

    if two_fan_count == 2:
        add_name_stat(stats, "2*2", turn)
    elif two_fan_count == 3:
        add_name_stat(stats, "2*3", turn)
    elif two_fan_count == 4:
        add_name_stat(stats, "2*4", turn)
    elif two_fan_count >= 5:
        add_name_stat(stats, "2*5+", turn)
    return True


def average(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def max_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(max(values)) if values else None


def analyze_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    games = 0
    terminal_actions: Counter[str] = Counter()
    turn_lists: dict[str, list[float]] = defaultdict(list)
    fan_lists: dict[str, list[float]] = defaultdict(list)
    fan_stats: dict[str, dict[str, Any]] = {}
    coufan_stats: dict[str, dict[str, Any]] = {}
    coufan_hands = 0
    meld_group_turns: list[list[float]] = [[], [], [], []]
    per_player_actions = [Counter() for _ in range(4)]
    per_player_action_games = [Counter() for _ in range(4)]

    for record in records:
        games += 1
        action = display_action(record)
        terminal_actions[action] += 1
        timing = scan_record_timing(record)
        for index, turns in enumerate(timing["meld_group_turns"]):
            meld_group_turns[index].extend(turns)
        for player, counts in enumerate(timing["per_player_actions"]):
            per_player_actions[player].update(counts)
            for key, count in counts.items():
                if count > 0:
                    per_player_action_games[player][key.replace("_count", "_games")] += 1

        if action != "HU":
            continue
        display = ((record.get("final_output") or {}).get("display") or {})
        winner = display_player(display)
        context = timing["hu_context"] or {}
        if winner is None:
            winner = context.get("player")
        turn = context.get("turn")
        if turn is None:
            turn = numeric(record.get("turn_count"))
        self_draw = context.get("self_draw")
        if self_draw is None:
            scores = record_scores(record)
            loser = infer_deal_in_loser(scores, winner)
            self_draw = loser is None

        fan_total = numeric(display.get("fanCnt"))
        if fan_total is None:
            fan_total = 0.0
        if turn is not None:
            turn_lists["和牌巡数"].append(float(turn))
            if self_draw:
                turn_lists["自摸巡数"].append(float(turn))
                turn_lists["被摸巡数"].extend([float(turn)] * 3)
            else:
                turn_lists["点和巡数"].append(float(turn))
                turn_lists["点炮巡数"].append(float(turn))
        if winner is not None:
            ready = timing["first_ready_turns"][winner]
            if ready is not None:
                turn_lists["听牌巡数"].append(float(ready))

        fan_lists["和牌番"].append(float(fan_total))
        if self_draw:
            fan_lists["自摸番"].append(float(fan_total))
            fan_lists["被摸番"].extend([float(fan_total)] * 3)
        else:
            fan_lists["点和番"].append(float(fan_total))
            fan_lists["点炮番"].append(float(fan_total))
        fan_lists["见证番"].extend([float(fan_total)] * 3)

        items = fan_items(display)
        seen = set()
        for item in items:
            if item["name"] in seen or item["name"] == "花牌":
                continue
            add_name_stat(fan_stats, item["name"], float(turn) if turn is not None else None)
            seen.add(item["name"])
        if update_coufan(coufan_stats, items, float(turn) if turn is not None else None):
            coufan_hands += 1

    fan_table = {}
    for group, fans in FAN_GROUPS:
        fan_table[group] = []
        for name, value in fans:
            stats = fan_stats.get(name, {})
            count = int(stats.get("count", 0))
            turns = int(stats.get("turn_count", 0))
            turn_sum = float(stats.get("turn_sum", 0.0))
            fan_table[group].append(
                {
                    "name": name,
                    "value": value,
                    "count": count,
                    "rate_among_hu": count / terminal_actions["HU"] if terminal_actions["HU"] else 0.0,
                    "average_hu_turn": turn_sum / turns if turns else 0.0,
                }
            )

    coufan_table = {}
    for group, names in COUFAN_SECTIONS:
        coufan_table[group] = []
        for name in names:
            stats = coufan_stats.get(name, {})
            count = int(stats.get("count", 0))
            turns = int(stats.get("turn_count", 0))
            turn_sum = float(stats.get("turn_sum", 0.0))
            coufan_table[group].append(
                {
                    "name": name,
                    "count": count,
                    "rate_among_hu": count / terminal_actions["HU"] if terminal_actions["HU"] else 0.0,
                    "average_hu_turn": turn_sum / turns if turns else 0.0,
                }
            )

    return {
        "format": "tjong_selfplay_dashboard_v1",
        "games": games,
        "terminal_actions": dict(terminal_actions),
        "hu_count": int(terminal_actions["HU"]),
        "huang_count": int(terminal_actions["HUANG"]),
        "turns": {
            "和牌巡数": average(turn_lists["和牌巡数"]),
            "听牌巡数": average(turn_lists["听牌巡数"]),
            "点和巡数": average(turn_lists["点和巡数"]),
            "自摸巡数": average(turn_lists["自摸巡数"]),
            "点炮巡数": average(turn_lists["点炮巡数"]),
            "被摸巡数": average(turn_lists["被摸巡数"]),
            "鸣第一组巡数": average(meld_group_turns[0]),
            "鸣第二组巡数": average(meld_group_turns[1]),
            "鸣第三组巡数": average(meld_group_turns[2]),
            "鸣第四组巡数": average(meld_group_turns[3]),
        },
        "fans": {
            "平均和牌番": average(fan_lists["和牌番"]),
            "平均点和番": average(fan_lists["点和番"]),
            "最大点和番": max_or_none(fan_lists["点和番"]),
            "平均自摸番": average(fan_lists["自摸番"]),
            "最大自摸番": max_or_none(fan_lists["自摸番"]),
            "平均点炮番": average(fan_lists["点炮番"]),
            "最大点炮番": max_or_none(fan_lists["点炮番"]),
            "平均被摸番": average(fan_lists["被摸番"]),
            "最大被摸番": max_or_none(fan_lists["被摸番"]),
            "平均见证番": average(fan_lists["见证番"]),
            "最大见证番": max_or_none(fan_lists["见证番"]),
        },
        "fan_table": fan_table,
        "coufan_hands": coufan_hands,
        "coufan_table": coufan_table,
        "meld_actions": [
            {
                "player": player,
                "chi_count": int(per_player_actions[player]["chi_count"]),
                "pong_count": int(per_player_actions[player]["pong_count"]),
                "kang_count": int(per_player_actions[player]["kang_count"]),
                "mingkong_count": int(per_player_actions[player]["mingkong_count"]),
                "bukong_count": int(per_player_actions[player]["bukong_count"]),
                "ankong_count": int(per_player_actions[player]["ankong_count"]),
                "chi_game_rate": per_player_action_games[player]["chi_games"] / games if games else 0.0,
                "pong_game_rate": per_player_action_games[player]["pong_games"] / games if games else 0.0,
                "kang_game_rate": per_player_action_games[player]["kang_games"] / games if games else 0.0,
            }
            for player in range(4)
        ],
        "definitions": {
            "巡数": "1-based table round from draw/discard logs; a Hu on the first discard round is turn 1.",
            "听牌巡数": "first official canHu>=8 opportunity observed for the eventual winner.",
            "凑番技术": "Hu hands whose non-flower fan items contain no single fan value above 4.",
            "达成番种率": "hands containing the fan divided by HU hands, not divided by all games.",
        },
    }


def read_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def shard_paths(shard_dir: Path, shards: int, pattern: str) -> list[Path]:
    return [shard_dir / pattern.format(index=index) for index in range(int(shards))]


def fmt_number(value: Any) -> str:
    if value is None:
        return "0.000"
    return f"{float(value):.3f}"


def fmt_count_rate_turn(item: dict[str, Any]) -> str:
    return f"{int(item['count'])} ({float(item['rate_among_hu']) * 100:.3f}%)\t{fmt_number(item['average_hu_turn'])}"


def pair_lines(items: list[tuple[str, str]]) -> list[str]:
    lines = []
    for index in range(0, len(items), 2):
        left = items[index]
        right = items[index + 1] if index + 1 < len(items) else None
        if right is None:
            lines.append(f"{left[0]}\t{left[1]}")
        else:
            lines.append(f"{left[0]}\t{left[1]}\t\t{right[0]}\t{right[1]}")
    return lines


def format_dashboard_text(summary: dict[str, Any]) -> str:
    lines = ["巡数相关"]
    turns = summary["turns"]
    lines.extend(
        pair_lines(
            [
                ("和牌巡数", fmt_number(turns["和牌巡数"])),
                ("听牌巡数", fmt_number(turns["听牌巡数"])),
                ("点和巡数", fmt_number(turns["点和巡数"])),
                ("自摸巡数", fmt_number(turns["自摸巡数"])),
                ("点炮巡数", fmt_number(turns["点炮巡数"])),
                ("被摸巡数", fmt_number(turns["被摸巡数"])),
                ("鸣第一组巡数", fmt_number(turns["鸣第一组巡数"])),
                ("鸣第二组巡数", fmt_number(turns["鸣第二组巡数"])),
                ("鸣第三组巡数", fmt_number(turns["鸣第三组巡数"])),
                ("鸣第四组巡数", fmt_number(turns["鸣第四组巡数"])),
            ]
        )
    )
    fans = summary["fans"]
    lines.append("番数相关")
    lines.extend(
        pair_lines(
            [
                ("平均和牌番", fmt_number(fans["平均和牌番"])),
                ("平均点和番", fmt_number(fans["平均点和番"])),
                ("最大点和番", fmt_number(fans["最大点和番"])),
                ("平均自摸番", fmt_number(fans["平均自摸番"])),
                ("最大自摸番", fmt_number(fans["最大自摸番"])),
                ("平均点炮番", fmt_number(fans["平均点炮番"])),
                ("最大点炮番", fmt_number(fans["最大点炮番"])),
                ("平均被摸番", fmt_number(fans["平均被摸番"])),
                ("最大被摸番", fmt_number(fans["最大被摸番"])),
                ("平均见证番", fmt_number(fans["平均见证番"])),
                ("最大见证番", fmt_number(fans["最大见证番"])),
            ]
        )
    )
    lines.append("达成番种")
    for group, items in summary["fan_table"].items():
        lines.append(group)
        lines.extend(pair_lines([(item["name"], fmt_count_rate_turn(item)) for item in items]))
    lines.append("凑番技术")
    for group, items in summary["coufan_table"].items():
        lines.append(group)
        lines.extend(pair_lines([(item["name"], fmt_count_rate_turn(item)) for item in items]))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, action="append", default=[])
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--shards", type=int, default=None)
    parser.add_argument("--raw-pattern", default="tjong_selfplay_raw_shard_{index:04d}.jsonl")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-text", type=Path, default=None)
    args = parser.parse_args()

    paths = list(args.raw)
    if args.shard_dir is not None:
        if args.shards is None:
            raise ValueError("--shards is required with --shard-dir")
        paths.extend(shard_paths(args.shard_dir, args.shards, args.raw_pattern))
    if not paths:
        raise ValueError("provide --raw or --shard-dir")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = analyze_records(read_records(paths))
    text = format_dashboard_text(summary)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_text is not None:
        args.out_text.parent.mkdir(parents=True, exist_ok=True)
        args.out_text.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
