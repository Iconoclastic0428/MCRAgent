import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prefetch_chaga_reviews import ReviewTarget, collect_review_targets  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_collect_review_targets_uses_train_players_and_session_zero_api_seats(tmp_path):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(
        raw,
        [
            {
                "belongs": "s1",
                "id": "r0",
                "step": {
                    "i": 0,
                    "p": [{"n": "Human"}, {"n": "CHAGA02"}, {"n": "CHAGA03"}, {"n": "CHAGA04"}],
                },
                "train_players": ["1", "3"],
            },
            {
                "belongs": "s1",
                "id": "r1",
                "step": {
                    "i": 1,
                    "p": [{"n": "CHAGA04"}, {"n": "Human"}, {"n": "CHAGA02"}, {"n": "CHAGA03"}],
                },
                "train_players": ["0", "2"],
            },
        ],
    )

    targets = collect_review_targets(raw, use_train_players=True)

    assert targets == [
        ReviewTarget(session_id="s1", api_seat=1, player_name="CHAGA02"),
        ReviewTarget(session_id="s1", api_seat=3, player_name="CHAGA04"),
    ]
