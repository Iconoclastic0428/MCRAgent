import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import scan_qadv_terminal_slices as scan  # noqa: E402


def _summary(offset: int, *, hu_count: int = 0, avg_delta: float = 0.0, safe: bool = True):
    return {
        "format": "mcr_qadv_fixed_league_sweep_v1",
        "offset": offset,
        "best_promotable_lambda": "0.05" if avg_delta > 0 else None,
        "lambda_metrics": {
            "0.00": {
                "safe": True,
                "hu_count": hu_count,
                "target_hu_count": hu_count,
                "target_end_wait_count": 0,
                "average_point_delta": 0.0,
                "low_fan_hu_count": 0,
                "illegal_prediction_count": 0,
            },
            "0.05": {
                "safe": safe,
                "hu_count": hu_count,
                "target_hu_count": hu_count,
                "target_end_wait_count": 0,
                "average_point_delta": avg_delta,
                "low_fan_hu_count": 0,
                "illegal_prediction_count": 0,
            },
        },
    }


def test_slice_signal_requires_terminal_or_point_signal():
    no_signal = scan.summarize_slice(_summary(0))
    terminal = scan.summarize_slice(_summary(4, hu_count=1))
    point = scan.summarize_slice(_summary(8, avg_delta=2.0))

    assert no_signal["terminal_signal_count"] == 0
    assert no_signal["has_signal"] is False
    assert terminal["terminal_signal_count"] == 2
    assert terminal["has_signal"] is True
    assert point["max_average_point_delta"] == 2.0
    assert point["has_signal"] is True


def test_scan_offsets_stops_after_target_signal(monkeypatch, tmp_path):
    calls = []

    def fake_run_sweep(args):
        calls.append(args.offset)
        if args.offset == 0:
            return _summary(args.offset)
        return _summary(args.offset, hu_count=1, avg_delta=1.0)

    monkeypatch.setattr(scan, "run_sweep", fake_run_sweep)
    summary = scan.scan_offsets(
        argparse.Namespace(
            model="models/base.pt",
            qadv_model="models/qadv.pt",
            lambdas="0,0.05",
            offset_start=0,
            offset_count=4,
            offset_step=4,
            games_per_slice=2,
            target_signal_slices=1,
            scan_all=False,
            policy="transformer",
            opponent="sample",
            opponent_model=None,
            opponent_qadv_model=None,
            opponent_qadv_lambda=0.0,
            raw="data/raw/example.jsonl",
            max_turns=500,
            judge="judge.exe",
            aleo_exe="aleo.exe",
            sample_exe="sample.exe",
            lawlorentz_levels=1,
            target_player=0,
            out_dir=str(tmp_path),
            summary_out=str(tmp_path / "scan.json"),
            min_point_delta=0.0,
            max_deal_in_regression=0.02,
        )
    )

    assert calls == [0, 4]
    assert summary["terminal_slice_count"] == 1
    assert summary["terminal_offsets"] == [4]
    assert summary["best_promotable_lambda"] == "0.05"
    assert (tmp_path / "scan.json").exists()
