import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sweep_qadv_fixed_league as sweep  # noqa: E402


def test_parse_lambdas_accepts_comma_list():
    assert sweep.parse_lambdas("0,0.05, .5") == [0.0, 0.05, 0.5]


def test_run_sweep_marks_only_safe_utility_improvements_as_promotable(monkeypatch, tmp_path):
    calls = []

    def fake_run_match_set(args):
        calls.append((args.qadv_model, args.qadv_lambda))
        score = {0.0: 0.0, 0.05: 1.0, 0.5: 2.0}[float(args.qadv_lambda)]
        low_fan = 1 if float(args.qadv_lambda) == 0.5 else 0
        return {
            "policy": args.policy,
            "model": args.model,
            "qadv_model": args.qadv_model,
            "qadv_lambda": args.qadv_lambda,
            "opponent": args.opponent,
            "opponent_model": args.opponent_model,
            "average_scores": [score, 0.0, 0.0, 0.0],
            "policy_diagnostics_totals": [{"illegal_predictions": 0}],
            "results": [
                {
                    "turns": 20,
                    "scores": [score, 0.0, 0.0, 0.0],
                    "policy_diagnostics": [{} for _ in range(4)],
                    "final_output": {
                        "display": {
                            "action": "HU",
                            "player": 0,
                            "fanCnt": 6 if low_fan else 8,
                        }
                    },
                }
            ],
        }

    monkeypatch.setattr(sweep, "run_match_set", fake_run_match_set)
    summary = sweep.run_sweep(
        argparse.Namespace(
            model="models/base.pt",
            qadv_model="models/qadv.pt",
            lambdas="0,0.05,0.50",
            policy="transformer",
            opponent="sample",
            opponent_model=None,
            opponent_qadv_model=None,
            opponent_qadv_lambda=0.0,
            raw="data/raw/example.jsonl",
            games=1,
            offset=0,
            max_turns=500,
            judge="judge.exe",
            aleo_exe="aleo.exe",
            sample_exe="sample.exe",
            lawlorentz_levels=1,
            target_player=0,
            out_dir=str(tmp_path),
            summary_out=str(tmp_path / "summary.json"),
            min_point_delta=0.0,
            max_deal_in_regression=0.02,
        )
    )

    assert calls == [(None, 0.0), ("models/qadv.pt", 0.05), ("models/qadv.pt", 0.5)]
    assert summary["baseline_lambda"] == "0.00"
    assert summary["best_promotable_lambda"] == "0.05"
    assert summary["lambda_metrics"]["0.50"]["safe"] is False
    assert (tmp_path / "summary.json").exists()
