import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_bc_policy import derive_action_type, load_examples, split_by_match_id


def test_split_by_match_id_keeps_matches_exclusive():
    examples = [
        {"match_id": "m1", "input_text": "a", "response": "PASS"},
        {"match_id": "m1", "input_text": "b", "response": "PLAY W1"},
        {"match_id": "m2", "input_text": "c", "response": "PASS"},
        {"match_id": "m3", "input_text": "d", "response": "HU"},
    ]

    train, test = split_by_match_id(examples, test_fraction=0.34)
    train_ids = {item["match_id"] for item in train}
    test_ids = {item["match_id"] for item in test}

    assert train_ids
    assert test_ids
    assert train_ids.isdisjoint(test_ids)


def test_derive_action_type_reads_first_response_token():
    assert derive_action_type("PLAY W1") == "PLAY"
    assert derive_action_type("  chi W1 W2  ") == "CHI"
    assert derive_action_type(None) == "MISSING"


def test_load_examples_filters_request_prefix(tmp_path):
    data = tmp_path / "examples.jsonl"
    data.write_text(
        "\n".join(
            [
                '{"match_id":"m1","request":"2 W1","input_text":"REQ 2 W1","response":"PLAY W1"}',
                '{"match_id":"m1","request":"3 0 DRAW","input_text":"REQ 3 0 DRAW","response":"PASS"}',
            ]
        ),
        encoding="utf-8",
    )

    examples = load_examples(data, target="response", request_prefix="2 ")

    assert [example["label"] for example in examples] == ["PLAY W1"]
