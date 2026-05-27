import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from convert_botzone_bc import iter_examples


def test_iter_examples_builds_player_history_text_before_current_response():
    record = {
        "match_id": "abc123",
        "scores": {"0": 10, "1": -10},
        "logs": [
            {
                "output": {
                    "content": {"0": "0 0 3", "1": "0 1 3"},
                    "display": {"action": "INIT"},
                }
            },
            {
                "0": {"response": "PASS"},
                "1": {"response": "PASS"},
            },
            {
                "output": {
                    "content": {"0": "2 W1", "1": "3 0 DRAW"},
                    "display": {"action": "DRAW", "player": 0, "tile": "W1"},
                }
            },
            {
                "0": {"response": "PLAY B1"},
                "1": {"response": "PASS"},
            },
        ],
    }

    examples = iter_examples(record)
    player_zero_draw = [
        example
        for example in examples
        if example["turn_index"] == 1 and example["player"] == 0
    ][0]

    assert player_zero_draw["input_text"] == "REQ 0 0 3\nRES PASS\nREQ 2 W1"
    assert "PLAY B1" not in player_zero_draw["input_text"]
    assert player_zero_draw["response"] == "PLAY B1"
