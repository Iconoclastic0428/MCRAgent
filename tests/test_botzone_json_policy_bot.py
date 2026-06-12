import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TJONG_SRC = ROOT / "papers" / "tjong_cit2_12298" / "src"
if str(TJONG_SRC) not in sys.path:
    sys.path.insert(0, str(TJONG_SRC))

from tjong_replication.model import TjongConfig, TjongNetwork  # noqa: E402


def test_botzone_json_policy_bot_reads_requests_and_outputs_response_json():
    payload = {
        "requests": [
            "0 0 1",
            "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
            "2 W4",
        ],
        "responses": ["PASS", "PASS"],
    }

    proc = subprocess.run(
        [sys.executable, "scripts/botzone_json_policy_bot.py"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MCR_MODEL": ""},
        check=True,
    )

    output = json.loads(proc.stdout)

    assert output == {"response": "PLAY W1"}


def test_tjong_botzone_json_policy_bot_uses_configured_checkpoint(monkeypatch):
    script = ROOT / "scripts" / "tjong_botzone_json_policy_bot.py"
    spec = importlib.util.spec_from_file_location("tjong_botzone_json_policy_bot", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    calls = []

    def fake_respond_json(payload, checkpoint, **kwargs):
        calls.append((payload, checkpoint, kwargs))
        return "PASS"

    monkeypatch.setattr(module, "respond_json", fake_respond_json)

    response = module.respond(
        {"requests": ["0 0 0"], "responses": []},
        checkpoint="models/tjong.pt",
        device="cpu",
    )

    assert response == "PASS"
    assert calls == [
        (
            {"requests": ["0 0 0"], "responses": []},
            "models/tjong.pt",
            {
                "device": "cpu",
                "require_encoding_version": "tjong_cit2_12298_v3_hidden_concealed_kong",
                "require_paper_config": False,
            },
        )
    ]


def test_tjong_botzone_json_policy_bot_subprocess_loads_checkpoint(tmp_path):
    checkpoint = tmp_path / "tiny_tjong.pt"
    config = TjongConfig(d_model=32, n_heads=4, ffn_dim=64, dropout=0.0)
    model = TjongNetwork(config)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "encoding_schema": {"version": "tjong_cit2_12298_v3_hidden_concealed_kong"},
            "tensor_encoding_version": "tjong_cit2_12298_v3_hidden_concealed_kong",
        },
        checkpoint,
    )
    payload = {
        "requests": [
            "0 0 1",
            "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
            "2 W4",
        ],
        "responses": ["PASS", "PASS"],
    }

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/tjong_botzone_json_policy_bot.py",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=True,
    )

    output = json.loads(proc.stdout)

    assert output["response"] in {
        "PLAY W1",
        "PLAY W2",
        "PLAY W3",
        "PLAY W4",
        "PLAY B1",
        "PLAY B2",
        "PLAY B3",
        "PLAY T1",
        "PLAY T2",
        "PLAY T3",
        "PLAY F1",
        "PLAY F2",
        "PLAY J1",
        "PLAY J2",
    }
