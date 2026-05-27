import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_discard_ranker import candidate_text, iter_discard_candidates


def test_iter_discard_candidates_labels_actual_draw_discard_from_hand():
    record = {
        "match_id": "m1",
        "logs": [
            {
                "output": {
                    "content": {
                        "0": "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
                    }
                }
            },
            {"0": {"response": "PASS"}},
            {"output": {"content": {"0": "2 W4"}}},
            {"0": {"response": "PLAY W2"}},
        ],
    }

    candidates = list(iter_discard_candidates(record))
    positive = [item for item in candidates if item["label"] == 1]

    assert len(candidates) == 14
    assert [item["candidate_tile"] for item in positive] == ["W2"]
    assert all(item["match_id"] == "m1" for item in candidates)
    assert any(item["candidate_tile"] == "W4" for item in candidates)


def test_candidate_text_includes_post_discard_shanten_features():
    item = {
        "input_text": "REQ 2 F2",
        "candidate_tile": "F2",
        "hand_tiles": "W1 W1 W1 W2 W3 W4 B1 B2 B3 T1 T2 T3 F1 F2".split(),
    }

    text = candidate_text(item)

    assert "REQ 2 F2" in text
    assert "CAND F2" in text
    assert "REG_SHANTEN 0" in text
    assert "MIN_SHANTEN 0" in text
