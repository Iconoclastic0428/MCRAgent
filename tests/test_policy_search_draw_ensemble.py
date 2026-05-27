import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from policy_search_draw_ensemble import (
    candidate_reward,
    normalize_weights,
    parse_weight_vector,
    rank_candidate_summaries,
    write_candidate_policy,
)


def test_parse_and_normalize_weight_vector():
    assert parse_weight_vector("0.2,0.5,0.3") == [0.2, 0.5, 0.3]

    weights = normalize_weights([2, 3, 5])

    assert weights == [0.2, 0.3, 0.5]
    assert sum(weights) == 1.0


def test_normalize_rejects_zero_total_weights():
    try:
        normalize_weights([0, 0, 0])
    except ValueError as exc:
        assert "positive total" in str(exc)
    else:
        raise AssertionError("zero-total weights should be rejected")


def test_write_candidate_policy_preserves_components_and_prefer_hu(tmp_path):
    draw_a = tmp_path / "draw_a.pkl"
    draw_b = tmp_path / "draw_b.pkl"
    reaction = tmp_path / "reaction.pkl"
    out_path = tmp_path / "candidate.pkl"
    draw_a_payload = {"kind": "feature_action_ranker", "name": "a"}
    draw_b_payload = {"kind": "feature_action_ranker", "name": "b"}
    reaction_payload = {"kind": "legal_action_ranker", "name": "reaction"}
    for path, payload in [
        (draw_a, draw_a_payload),
        (draw_b, draw_b_payload),
        (reaction, reaction_payload),
    ]:
        with path.open("wb") as out:
            pickle.dump(payload, out)

    payload = write_candidate_policy(
        draw_model_paths=[draw_a, draw_b],
        draw_weights=[0.25, 0.75],
        reaction_model_path=reaction,
        out_path=out_path,
        prefer_hu=True,
    )

    assert out_path.exists()
    assert payload["kind"] == "draw_ensemble_composite_policy"
    assert payload["draw_payloads"] == [draw_a_payload, draw_b_payload]
    assert payload["draw_weights"] == [0.25, 0.75]
    assert payload["reaction_payload"] == reaction_payload
    assert payload["prefer_hu"] is True
    assert payload["components"]["draw_models"] == [str(draw_a), str(draw_b)]


def test_rank_candidate_summaries_uses_win_rate_then_average_score():
    weak = {"games": 8, "wins": [3, 0, 0, 0], "average_scores": [40.0, 0, 0, 0]}
    better_wins = {"games": 8, "wins": [4, 0, 0, 0], "average_scores": [0.0, 0, 0, 0]}
    better_score = {"games": 8, "wins": [4, 0, 0, 0], "average_scores": [12.0, 0, 0, 0]}

    assert candidate_reward(weak) == (0.375, 40.0)
    assert rank_candidate_summaries([weak, better_wins, better_score]) == [
        better_score,
        better_wins,
        weak,
    ]
