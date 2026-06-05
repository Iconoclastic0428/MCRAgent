import argparse
import json
import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from convert_tziakcha_prepared_to_initdata import TILE_COUNTS  # noqa: E402
from generate_random_mcr_initdata import generate_file, initdata_for_seed, random_walltiles  # noqa: E402
from official_judge_match import load_initdata  # noqa: E402


def test_random_walltiles_are_deterministic_and_legal():
    first = random_walltiles(20260605)
    second = random_walltiles(20260605)
    other = random_walltiles(20260606)

    assert first == second
    assert first != other
    assert len(first) == 144
    assert Counter(first) == TILE_COUNTS


def test_initdata_for_seed_contains_legal_wall_and_quan():
    initdata = initdata_for_seed(20260605)

    assert initdata["srand"] == 20260605
    assert initdata["quan"] == 20260605 % 4
    assert Counter(initdata["walltiles"].split()) == TILE_COUNTS


def test_generate_file_writes_loadable_initdata(tmp_path):
    out = tmp_path / "random_initdata.jsonl"
    summary_out = tmp_path / "summary.json"
    summary = generate_file(
        argparse.Namespace(
            out_jsonl=str(out),
            summary_out=str(summary_out),
            games=3,
            start_seed=100,
            quan=None,
        )
    )

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    loaded = load_initdata(out)
    saved_summary = json.loads(summary_out.read_text(encoding="utf-8"))

    assert summary["games"] == 3
    assert saved_summary["end_seed"] == 102
    assert [record["match_id"] for record in records] == [
        "random-seed-0000000100",
        "random-seed-0000000101",
        "random-seed-0000000102",
    ]
    assert [item["srand"] for item in loaded] == [100, 101, 102]
    assert all(len(item["walltiles"].split()) == 144 for item in loaded)
