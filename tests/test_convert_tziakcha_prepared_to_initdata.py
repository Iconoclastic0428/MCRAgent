import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from convert_tziakcha_prepared_to_initdata import (  # noqa: E402
    SEGMENT_LEN,
    convert_file,
    record_to_initdata,
    reconstruct_walltiles,
)
from official_judge_match import load_initdata  # noqa: E402


HANDS = [
    "W1 W2 W3 W4 W5 W6 W7 W8 W9 B1 B2 B3 B4".split(),
    "B5 B6 B7 B8 B9 T1 T2 T3 T4 T5 T6 T7 T8".split(),
    "T9 F1 F2 F3 F4 J1 J2 J3 W1 B1 T1 W2 B2".split(),
    "T2 W3 B3 T3 W4 B4 T4 W5 B5 T5 W6 B6 T6".split(),
]


def _deal_content(flowers_by_seat=None):
    flowers_by_seat = flowers_by_seat or [[], [], [], []]
    all_flowers = [tile for seat_flowers in flowers_by_seat for tile in seat_flowers]
    content = {}
    counts = " ".join(str(len(flowers_by_seat[seat])) for seat in range(4))
    suffix = " " + " ".join(all_flowers) if all_flowers else ""
    for seat in range(4):
        content[str(seat)] = f"1 {counts} {' '.join(HANDS[seat])}{suffix}"
    return content


def _record(flowers_by_seat=None):
    return {
        "match_id": "synthetic",
        "source_record_id": "synthetic",
        "logs": [
            {
                "output": {
                    "command": "request",
                    "content": {str(seat): f"0 {seat} 2" for seat in range(4)},
                    "display": {"action": "INIT"},
                }
            },
            {str(seat): {"response": "PASS", "verdict": "OK"} for seat in range(4)},
            {"output": {"command": "request", "content": _deal_content(flowers_by_seat), "display": {"action": "DEAL"}}},
            {str(seat): {"response": "PASS", "verdict": "OK"} for seat in range(4)},
            {
                "output": {
                    "command": "request",
                    "content": {"0": "2 W7", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                    "display": {"action": "DRAW", "player": 0, "tile": "W7"},
                }
            },
            {"0": {"response": "PLAY W7", "verdict": "OK"}, "1": {"response": "PASS", "verdict": "OK"}, "2": {"response": "PASS", "verdict": "OK"}, "3": {"response": "PASS", "verdict": "OK"}},
            {
                "output": {
                    "command": "request",
                    "content": {"0": "3 1 DRAW", "1": "2 J1", "2": "3 1 DRAW", "3": "3 1 DRAW"},
                    "display": {"action": "DRAW", "player": 1, "tile": "J1"},
                }
            },
            {"0": {"response": "PASS", "verdict": "OK"}, "1": {"response": "PLAY J1", "verdict": "OK"}, "2": {"response": "PASS", "verdict": "OK"}, "3": {"response": "PASS", "verdict": "OK"}},
        ],
    }


def _pop_prefix(walltiles, seat, count):
    segment = walltiles[seat * SEGMENT_LEN : (seat + 1) * SEGMENT_LEN]
    return list(reversed(segment))[:count]


def test_reconstructed_wall_preserves_initial_hand_and_draw_prefix():
    record = _record()

    walltiles = reconstruct_walltiles(record)

    assert len(walltiles) == 144
    assert _pop_prefix(walltiles, 0, 14) == HANDS[0] + ["W7"]
    assert _pop_prefix(walltiles, 1, 14) == HANDS[1] + ["J1"]
    assert _pop_prefix(walltiles, 2, 13) == HANDS[2]
    assert _pop_prefix(walltiles, 3, 13) == HANDS[3]


def test_reconstructed_wall_places_initial_flowers_before_standing_tiles():
    record = _record(flowers_by_seat=[["H1"], [], ["H2", "H3"], []])

    walltiles = reconstruct_walltiles(record)

    assert _pop_prefix(walltiles, 0, 15) == ["H1"] + HANDS[0] + ["W7"]
    assert _pop_prefix(walltiles, 2, 15) == ["H2", "H3"] + HANDS[2]


def test_convert_file_outputs_loadable_initdata(tmp_path):
    prepared = tmp_path / "prepared.jsonl"
    out = tmp_path / "initdata.jsonl"
    summary = tmp_path / "summary.json"
    prepared.write_text(json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")

    result = convert_file(
        type(
            "Args",
            (),
            {
                "prepared": str(prepared),
                "out_jsonl": str(out),
                "summary_out": str(summary),
                "limit": None,
                "min_written": 1,
                "srand": 123,
                "fail_on_error": True,
            },
        )()
    )

    assert result["records_written"] == 1
    loaded = load_initdata(out)
    assert loaded[0]["quan"] == 2
    assert loaded[0]["srand"] == 123
    assert len(loaded[0]["walltiles"].split()) == 144


def test_convert_file_skips_unrepresentable_overlong_records(tmp_path):
    prepared = tmp_path / "prepared.jsonl"
    out = tmp_path / "initdata.jsonl"
    bad = _record()
    for turn in range(40):
        bad["logs"].extend(
            [
                {
                    "output": {
                        "command": "request",
                        "content": {"0": "2 W1", "1": "3 0 DRAW", "2": "3 0 DRAW", "3": "3 0 DRAW"},
                        "display": {"action": "DRAW", "player": 0, "tile": "W1"},
                    }
                },
                {"0": {"response": "PLAY W1", "verdict": "OK"}, "1": {"response": "PASS", "verdict": "OK"}, "2": {"response": "PASS", "verdict": "OK"}, "3": {"response": "PASS", "verdict": "OK"}},
            ]
        )
    prepared.write_text(
        json.dumps(bad, ensure_ascii=False) + "\n" + json.dumps(_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = convert_file(
        type(
            "Args",
            (),
            {
                "prepared": str(prepared),
                "out_jsonl": str(out),
                "summary_out": None,
                "limit": None,
                "min_written": 1,
                "srand": 123,
                "fail_on_error": False,
            },
        )()
    )

    assert result["records_written"] == 1
    assert len(result["errors"]) == 1
    assert load_initdata(out)[0]["quan"] == 2
