import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_payload(path: str):
    with (ROOT / path).open("rb") as src:
        return pickle.load(src)


def test_exported_hist_gradient_boosting_scores_match_sklearn():
    from botzone_pure_runtime import score_hgb_export
    from export_botzone_pure_python import export_hist_gradient_boosting

    payload = load_payload("models/feature_draw_ranker_1000.pkl")
    model = payload["model"]
    exported = export_hist_gradient_boosting(model)
    rows = np.asarray(
        [
            np.zeros(model.n_features_in_, dtype=np.float32),
            np.linspace(0.0, 1.0, model.n_features_in_, dtype=np.float32),
        ]
    )

    expected = [float(value) for value in model.predict_proba(rows)[:, 1]]
    actual = [score_hgb_export(exported, row.tolist()) for row in rows]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_exported_tfidf_sgd_scores_match_sklearn_pipeline():
    from botzone_pure_runtime import score_tfidf_sgd_export
    from export_botzone_pure_python import export_tfidf_sgd_pipeline

    payload = load_payload("models/reaction_ranker_nonpass50_1000.pkl")
    pipeline = payload["pipeline"]
    exported = export_tfidf_sgd_pipeline(pipeline)
    texts = [
        "REQ 3 2 PLAY W1\nRESP PASS\nACTION PASS\nHAND_MIN_SHANTEN 2",
        "REQ 3 1 PLAY B5\nRESP PENG W1\nACTION PENG\nCAND W1\nREG_SHANTEN 1",
    ]

    expected = [float(value) for value in pipeline.predict_proba(texts)[:, 1]]
    actual = [score_tfidf_sgd_export(exported, text) for text in texts]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_exported_botzone_zip_outputs_json_without_sklearn(tmp_path):
    out_dir = tmp_path / "pure_pkg"
    out_zip = tmp_path / "pure_pkg.zip"

    subprocess.run(
        [
            sys.executable,
            "scripts/export_botzone_pure_python.py",
            "--model",
            "models/ensemble_draw_public1000_2026_050_050_reaction1000.pkl",
            "--out-dir",
            str(out_dir),
            "--zip",
            str(out_zip),
        ],
        check=True,
        cwd=ROOT,
    )

    assert (out_dir / "__main__.py").exists()
    assert out_zip.exists()
    assert out_zip.stat().st_size < 4 * 1024 * 1024

    payload = {
        "requests": [
            "0 0 1",
            "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
            "2 W4",
        ],
        "responses": ["PASS", "PASS"],
    }
    proc = subprocess.run(
        [sys.executable, str(out_zip)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )

    output = json.loads(proc.stdout)
    assert set(output) == {"response"}
    assert isinstance(output["response"], str)


def test_exported_single_file_outputs_json_without_sklearn(tmp_path):
    out_dir = tmp_path / "pure_pkg"
    out_file = tmp_path / "pure_bot.py"

    subprocess.run(
        [
            sys.executable,
            "scripts/export_botzone_pure_python.py",
            "--model",
            "models/ensemble_draw_public1000_2026_050_050_reaction1000.pkl",
            "--out-dir",
            str(out_dir),
            "--single-file",
            str(out_file),
        ],
        check=True,
        cwd=ROOT,
    )

    assert out_file.exists()
    assert out_file.stat().st_size < 4 * 1024 * 1024
    text = out_file.read_text(encoding="utf-8")
    assert "MODEL =" in text
    assert "raise SystemExit(main_with_model(MODEL))" in text

    payload = {
        "requests": [
            "0 0 1",
            "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
            "2 W4",
        ],
        "responses": ["PASS", "PASS"],
    }
    proc = subprocess.run(
        [sys.executable, str(out_file)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )

    output = json.loads(proc.stdout)
    assert set(output) == {"response"}
    assert isinstance(output["response"], str)


def test_exported_bootstrap_file_outputs_json_without_sklearn(tmp_path):
    out_dir = tmp_path / "pure_pkg"
    out_file = tmp_path / "pure_bootstrap.py"
    out_zip = tmp_path / "pure_pkg.zip"

    subprocess.run(
        [
            sys.executable,
            "scripts/export_botzone_pure_python.py",
            "--model",
            "models/ensemble_draw_public1000_2026_050_050_reaction1000.pkl",
            "--out-dir",
            str(out_dir),
            "--zip",
            str(out_zip),
            "--bootstrap-file",
            str(out_file),
        ],
        check=True,
        cwd=ROOT,
    )

    assert out_file.exists()
    assert out_file.stat().st_size < 1024 * 1024
    text = out_file.read_text(encoding="utf-8")
    assert "_ZIP_B64" in text
    assert "runtime_ns['main_with_model'](model_ns['MODEL'])" in text
    assert "zipfile.ZipFile(io.BytesIO(payload))" in text
    assert "os.getcwd()" not in text
    assert "with open(path, 'wb')" not in text
    assert "tempfile.gettempdir()" not in text

    payload = {
        "requests": [
            "0 0 1",
            "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
            "2 W4",
        ],
        "responses": ["PASS", "PASS"],
    }
    proc = subprocess.run(
        [sys.executable, str(out_file)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )

    output = json.loads(proc.stdout)
    assert set(output) == {"response"}
    assert isinstance(output["response"], str)


def test_pure_runtime_suppresses_sub_8_fan_hu_without_official_checker():
    from botzone_pure_runtime import PurePolicy

    policy = PurePolicy({})
    policy.player_id = 0
    policy.hand.update(
        [
            "W1",
            "W2",
            "W4",
            "W5",
            "W6",
            "B2",
            "B3",
            "B4",
            "T2",
            "T3",
            "T4",
            "F1",
            "F1",
        ]
    )

    responses = policy.legal_responses("3 1 PLAY W3")

    assert "HU" not in responses


def test_pure_runtime_allows_unambiguous_high_fan_hu():
    from botzone_pure_runtime import PurePolicy, conservative_high_fan_lower_bound

    tiles = "W1 W1 W1 W2 W2 W2 W3 W3 W3 W4 W4 W4 W5 W5".split()
    policy = PurePolicy({})
    policy.player_id = 0
    policy.hand.update(tiles)

    assert conservative_high_fan_lower_bound(tiles) >= 8
    assert "HU" in policy.legal_responses("2 W5")


def test_pure_runtime_conservative_high_fan_matches_official_acceptance():
    from botzone_pure_runtime import conservative_high_fan_lower_bound
    from official_fan import OfficialFanChecker

    tiles = "W1 W1 W1 W2 W2 W2 W3 W3 W3 W4 W4 W4 W5 W5".split()
    checker = OfficialFanChecker(Path("build/official_judge/mcr_fan_check.exe"))

    result = checker.evaluate(
        packs=[],
        hand=tiles[:-1],
        win_tile=tiles[-1],
        flower_count=0,
        is_self_draw=True,
        is_4th_tile=False,
        is_about_kong=False,
        is_last=False,
        seat_wind=0,
        prevalent_wind=0,
        player=0,
    )

    assert conservative_high_fan_lower_bound(tiles) >= 8
    assert result["fan"] >= 8
    assert result["can_hu"]


def test_pure_runtime_prefer_hu_still_requires_conservative_fan_gate():
    from botzone_pure_runtime import PurePolicy

    low = PurePolicy({"prefer_hu": True})
    low.player_id = 0
    low.hand.update("W1 W2 W4 W5 W6 B2 B3 B4 T2 T3 T4 F1 F1".split())
    assert "HU" not in low.legal_responses("3 1 PLAY W3")

    high = PurePolicy({"prefer_hu": True})
    high.player_id = 0
    high.hand.update("W1 W1 W1 W2 W2 W2 W3 W3 W3 W4 W4 W4 W5 W5".split())
    assert high.choose_response("2 W5") == "HU"


@pytest.mark.parametrize(
    "tiles",
    [
        "W1 W2 W3 B4 B5 B6 T7 T8 T9 J1 J1 J1 W5 W5",
        "W2 W3 W4 B2 B3 B4 T2 T3 T4 F1 F1 F1 W5 W5",
        "W1 W2 W3 W4 W5 W6 W7 W8 W9 B2 B2 B2 T5 T5",
        "W2 W3 W4 W5 W6 W7 B2 B3 B4 T6 T7 T8 B5 B5",
    ],
)
def test_pure_runtime_conservative_gate_accepts_official_backed_common_8_fan_patterns(tiles):
    from botzone_pure_runtime import conservative_high_fan_lower_bound
    from official_fan import OfficialFanChecker

    hand = tiles.split()
    checker = OfficialFanChecker(Path("build/official_judge/mcr_fan_check.exe"))
    result = checker.evaluate(
        packs=[],
        hand=hand[:-1],
        win_tile=hand[-1],
        flower_count=0,
        is_self_draw=False,
        is_4th_tile=False,
        is_about_kong=False,
        is_last=False,
        seat_wind=0,
        prevalent_wind=0,
        player=0,
    )

    assert result["fan"] >= 8
    assert conservative_high_fan_lower_bound(hand) >= 8


def test_pure_runtime_all_chows_self_draw_hu_requires_official_backed_8_fan():
    from botzone_pure_runtime import PurePolicy, conservative_high_fan_lower_bound
    from official_fan import OfficialFanChecker

    hand = "W2 W3 W4 W5 W6 W7 B2 B3 B4 T6 T7 T8 B5 B5".split()
    checker = OfficialFanChecker(Path("build/official_judge/mcr_fan_check.exe"))
    result = checker.evaluate(
        packs=[],
        hand=hand[:-1],
        win_tile=hand[-1],
        flower_count=0,
        is_self_draw=True,
        is_4th_tile=False,
        is_about_kong=False,
        is_last=False,
        seat_wind=0,
        prevalent_wind=0,
        player=0,
    )

    assert result["fan"] >= 8
    assert conservative_high_fan_lower_bound(hand, self_draw=True) >= 8

    policy = PurePolicy({"prefer_hu": True})
    policy.player_id = 0
    policy.hand.update(hand[:-1])

    assert policy.respond("2 B5") == "HU"


def test_pure_runtime_self_draw_bonus_does_not_make_open_all_chows_hu_safe():
    from botzone_pure_runtime import conservative_high_fan_lower_bound

    hand = "W2 W3 W4 W5 W6 W7 B2 B3 B4 T6 T7 T8 B5 B5".split()

    assert conservative_high_fan_lower_bound(hand, open_meld_count=1, self_draw=True) < 8


def test_pure_runtime_draw_scoring_matches_local_draw_ensemble_semantics():
    from botzone_pure_runtime import PurePolicy
    from export_botzone_pure_python import export_payload

    model_data = export_payload(
        load_payload("models/ensemble_draw_public1000_2026_030_070_reaction1000_prefer_hu.pkl")
    )
    policy = PurePolicy(model_data)
    policy.player_id = 0
    policy.hand.update("B2 B5 B6 B9 J3 J3 T8 T8 T8 W1 W2 W4 W6".split())

    assert policy.respond("2 B3") == "PLAY B9"


@pytest.mark.parametrize(
    "pre_hand,bot_request,official_kwargs,wall_counts",
    [
        (
            "B3 W6 W6 B2 W4 W3 B5 T7 B1 T6 B6 W5 B4",
            "2 T5",
            {"win_tile": "T5", "is_self_draw": True, "is_last": False, "prevalent_wind": 0},
            None,
        ),
        (
            "T5 W5 W5 J2 J2 T6 F3 F3 F3 W6 W6 W6 T4",
            "3 3 PLAY J2",
            {"win_tile": "J2", "is_self_draw": False, "is_last": True, "prevalent_wind": 1},
            [0, 1, 1, 1],
        ),
        (
            "J1 J1 T1 T1 T2 T2 T3 T3 W2 W3 W4 W7 W7",
            "2 J1",
            {"win_tile": "J1", "is_self_draw": True, "is_last": False, "prevalent_wind": 1},
            None,
        ),
        (
            "W7 W7 T4 B2 B2 B2 B8 T6 B6 B6 T5 B4 B7",
            "2 B5",
            {"win_tile": "B5", "is_self_draw": True, "is_last": False, "prevalent_wind": 3},
            None,
        ),
    ],
)
def test_pure_runtime_accepts_official_backed_missed_hu_patterns(
    pre_hand, bot_request, official_kwargs, wall_counts
):
    from botzone_pure_runtime import PurePolicy
    from official_fan import OfficialFanChecker

    checker = OfficialFanChecker(Path("build/official_judge/mcr_fan_check.exe"))
    result = checker.evaluate(
        packs=[],
        hand=pre_hand.split(),
        flower_count=0,
        is_4th_tile=False,
        is_about_kong=False,
        seat_wind=0,
        player=0,
        **official_kwargs,
    )
    assert result["can_hu"]
    assert result["fan"] >= 8

    policy = PurePolicy({"prefer_hu": True})
    policy.player_id = 0
    policy.hand.update(pre_hand.split())
    if bot_request.startswith("2 "):
        policy.hand[official_kwargs["win_tile"]] += 1
    if wall_counts is not None:
        policy.wall_counts = list(wall_counts)

    assert "HU" in policy.legal_responses(bot_request)


def test_pure_runtime_suppresses_official_backed_7_fan_shifted_chow_false_positive():
    from botzone_pure_runtime import PurePolicy
    from official_fan import OfficialFanChecker

    pre_hand = "B4 B5 B6 T2 T3 T4 T5 T5 T6 T7 T8 W3 W4".split()
    checker = OfficialFanChecker(Path("build/official_judge/mcr_fan_check.exe"))
    result = checker.evaluate(
        packs=[],
        hand=pre_hand,
        win_tile="W2",
        flower_count=0,
        is_self_draw=False,
        is_4th_tile=False,
        is_about_kong=False,
        is_last=False,
        seat_wind=0,
        prevalent_wind=0,
        player=0,
    )
    assert result["fan"] == 7
    assert not result["can_hu"]

    policy = PurePolicy({"prefer_hu": True})
    policy.player_id = 0
    policy.hand.update(pre_hand)

    assert "HU" not in policy.legal_responses("3 3 PLAY W2")
