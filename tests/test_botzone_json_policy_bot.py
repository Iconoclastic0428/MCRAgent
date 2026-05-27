import json
import os
import subprocess
import sys
from pathlib import Path


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

    assert output == {"response": "PLAY W4"}
