import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_suit_permuted_initdata import (
    generate_permuted_records,
    permute_record_tiles,
    permute_text_tiles,
    permute_tile_symbol,
    permute_walltiles,
)
from official_judge_match import load_initdata


def test_permute_tile_symbol_changes_numbered_suits_only():
    mapping = {"W": "T", "T": "B", "B": "W"}

    assert permute_tile_symbol("W1", mapping) == "T1"
    assert permute_tile_symbol("T9", mapping) == "B9"
    assert permute_tile_symbol("B5", mapping) == "W5"
    assert permute_tile_symbol("F1", mapping) == "F1"
    assert permute_tile_symbol("J3", mapping) == "J3"
    assert permute_tile_symbol("H8", mapping) == "H8"


def test_permute_walltiles_preserves_tile_order_and_honors():
    mapping = {"W": "B", "B": "T", "T": "W"}

    assert permute_walltiles("W1 T2 B3 F1 J2 H3", mapping) == "B1 W2 T3 F1 J2 H3"


def test_generate_permuted_records_are_loadable_raw_jsonl(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "match_id": "m1",
                "initdata": {
                    "quan": 0,
                    "srand": 123,
                    "walltiles": "W1 T2 B3 F1",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "permuted.jsonl"

    records = list(
        generate_permuted_records(
            raw,
            include_identity=False,
            limit=1,
            permutations=[("W", "B", "T")],
        )
    )
    out.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    assert len(records) == 1
    assert records[0]["match_id"] == "m1__suit_WBT"
    assert records[0]["source_match_id"] == "m1"
    assert records[0]["suit_permutation"] == {"W": "W", "T": "B", "B": "T"}
    assert records[0]["initdata"]["walltiles"] == "W1 B2 T3 F1"
    assert load_initdata(out, limit=1) == [
        {"quan": 0, "srand": 123, "walltiles": "W1 B2 T3 F1"}
    ]


def test_permute_text_tiles_rewrites_only_standalone_tile_tokens():
    mapping = {"W": "T", "T": "B", "B": "W"}

    assert permute_text_tiles("PLAY W1 then CHI T2 B3", mapping) == "PLAY T1 then CHI B2 W3"
    assert permute_text_tiles("https://example.test/W1 and FW1 stay", mapping) == (
        "https://example.test/W1 and FW1 stay"
    )


def test_permute_record_tiles_rewrites_logs_and_initdata_but_keeps_metadata():
    mapping = {"W": "T", "T": "B", "B": "W"}
    record = {
        "match_id": "W1_source_id",
        "initdata": {"walltiles": "W1 T2 B3 F1"},
        "logs": [
            {"output": {"content": {"0": "2 W1"}, "display": {"tile": "B3"}}},
            {"0": {"response": "PLAY T2"}},
        ],
    }

    permuted = permute_record_tiles(record, mapping)

    assert permuted["match_id"] == "W1_source_id"
    assert permuted["initdata"]["walltiles"] == "T1 B2 W3 F1"
    assert permuted["logs"][0]["output"]["content"]["0"] == "2 T1"
    assert permuted["logs"][0]["output"]["display"]["tile"] == "W3"
    assert permuted["logs"][1]["0"]["response"] == "PLAY B2"
