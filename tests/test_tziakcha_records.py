import json
import sys
import base64
import zlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from convert_botzone_bc import iter_examples
from train_legal_action_ranker import iter_legal_action_candidates
from tziakcha_records import convert_record, tile_id_to_botzone_symbol


EXAMPLE_RECORD_WITHOUT_WALL = (
    ROOT / "external" / "tziakcha_record_miner" / "test" / "unit" / "assets" / "example_record.json"
)


def response_values(record):
    values = []
    for index in range(1, len(record["logs"]), 2):
        values.extend(item["response"] for item in record["logs"][index].values())
    return values


def make_wall_for_dealt_hands(hands):
    rotated = [143] * 144
    cursors = [0, 0, 0, 0]
    index = 0
    for _ in range(3):
        for player in range(4):
            rotated[index : index + 4] = hands[player][cursors[player] : cursors[player] + 4]
            cursors[player] += 4
            index += 4
    for player in range(4):
        rotated[index] = hands[player][cursors[player]]
        cursors[player] += 1
        index += 1
    rotated[index] = hands[0][cursors[0]]
    original = rotated[108:] + rotated[:108]
    return "".join(f"{tile:02x}" for tile in original)


def synthetic_full_record():
    hands = [
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 108],
        [109, 110, 72, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32],
        [76, 80, 36, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52],
        [36, 37, 38, 56, 57, 58, 60, 61, 62, 64, 65, 66, 68],
    ]
    return {
        "id": "synthetic-rich",
        "belongs": "unit",
        "script": "<Decoded>",
        "step": {
            "w": make_wall_for_dealt_hands(hands),
            "d": 0,
            "i": 8,
            "p": [{"n": f"p{player}"} for player in range(4)],
            "a": [
                [0, 0, 0],
                [2, 108, 100],
                [20, 219, 200],
                [18, 72, 300],
                [35, 19, 400],
                [34, 36, 500],
                [53, 9, 600],
                [7, 17, 700],
                [6, 16, 800],
            ],
            "s": [24, -8, -8, -8],
        },
    }


def encode_step(step):
    raw = json.dumps(step, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(zlib.compress(raw)).decode("ascii")


def test_tile_id_to_botzone_symbol_matches_existing_tziakcha_mapping():
    assert tile_id_to_botzone_symbol(0) == "W1"
    assert tile_id_to_botzone_symbol(36) == "T1"
    assert tile_id_to_botzone_symbol(72) == "B1"
    assert tile_id_to_botzone_symbol(108) == "F1"
    assert tile_id_to_botzone_symbol(132) == "J3"
    assert tile_id_to_botzone_symbol(136) == "H1"


def test_convert_decoded_example_record_to_botzone_like_raw_log():
    source = synthetic_full_record()

    record = convert_record(source)

    assert record["match_id"] == "synthetic-rich"
    assert record["source"] == "tziakcha_record_miner"
    assert record["scores"] == {"0": 24, "1": -8, "2": -8, "3": -8}
    assert record["logs"][0]["output"]["content"]["0"] == "0 0 2"
    assert record["logs"][2]["output"]["content"]["0"].startswith("1 ")
    assert len(record["logs"]) % 2 == 0

    responses = response_values(record)
    assert any(response.startswith("PLAY ") for response in responses)
    assert any(response.startswith("PENG ") for response in responses)
    assert any(response.startswith("CHI ") for response in responses)
    assert "GANG" in responses
    assert "HU" in responses
    assert "ABANDON" not in responses

    examples = iter_examples(record)
    assert len(examples) > 20
    assert any(example["action_type"] == "HU" for example in examples)


def test_converted_example_yields_legal_candidates_for_human_nonpass_actions():
    source = synthetic_full_record()
    record = convert_record(source)

    actual_nonpass = {
        item["actual_response"].split()[0]
        for item in iter_legal_action_candidates(record)
        if item["label"] == 1 and item["actual_response"] != "PASS"
    }

    assert {"PLAY", "PENG", "CHI", "GANG", "HU"}.issubset(actual_nonpass)


def test_recorded_hu_below_eight_fan_is_suppressed():
    wall = "".join(f"{tile:02x}" for tile in range(144))
    low_fan_record = {
        "id": "lowfan",
        "belongs": "synthetic",
        "script": "<Decoded>",
        "step": {
            "w": wall,
            "d": 0,
            "i": 0,
            "p": [{"n": f"p{player}"} for player in range(4)],
            "a": [
                [0, 0, 0],
                [7, 0, 100],
                [6, 14, 200],
            ],
            "s": [0, 0, 0, 0],
        },
    }

    record = convert_record(low_fan_record)

    assert "HU" not in response_values(record)


def test_record_without_wall_data_is_rejected_with_clear_error():
    source = json.loads(EXAMPLE_RECORD_WITHOUT_WALL.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="step.w"):
        convert_record(source)


def test_convert_record_decodes_tziakcha_script_when_step_is_absent():
    source = synthetic_full_record()
    encoded = {
        "id": "encoded-synthetic",
        "belongs": "unit",
        "script": encode_step(source["step"]),
    }

    record = convert_record(encoded)

    assert record["match_id"] == "encoded-synthetic"
    assert record["scores"] == {"0": 24, "1": -8, "2": -8, "3": -8}
    assert "HU" in response_values(record)
