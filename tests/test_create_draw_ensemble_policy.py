import pickle
import subprocess
import sys
from pathlib import Path


def test_create_draw_ensemble_policy_writes_weighted_components(tmp_path):
    draw_a = tmp_path / "draw_a.pkl"
    draw_b = tmp_path / "draw_b.pkl"
    reaction = tmp_path / "reaction.pkl"
    out_path = tmp_path / "ensemble.pkl"
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

    subprocess.run(
        [
            sys.executable,
            "scripts/create_draw_ensemble_policy.py",
            "--draw-model",
            str(draw_a),
            "--draw-model",
            str(draw_b),
            "--draw-weight",
            "0.4",
            "--draw-weight",
            "0.6",
            "--reaction-model",
            str(reaction),
            "--out",
            str(out_path),
            "--prefer-hu",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    with out_path.open("rb") as src:
        payload = pickle.load(src)

    assert payload["kind"] == "draw_ensemble_composite_policy"
    assert payload["draw_payloads"] == [draw_a_payload, draw_b_payload]
    assert payload["draw_weights"] == [0.4, 0.6]
    assert payload["reaction_payload"] == reaction_payload
    assert payload["prefer_hu"] is True
    assert payload["components"]["draw_models"] == [str(draw_a), str(draw_b)]
