import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from official_judge_match import (
    AleoProcessPolicy,
    BotzonePolicy,
    BotzoneJsonProcessPolicy,
    LawlorentzEffectivePolicy,
    build_policies,
    aggregate_policy_diagnostics,
    build_response_log,
    load_initdata,
    make_policy,
    placement_rewards_from_scores,
    run_json_bot_process,
    run_match,
    summarize_terminals,
)


class EchoPolicy:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def respond(self, request):
        self.requests.append(request)
        return self.response


def test_build_response_log_asks_only_requested_players():
    policies = [EchoPolicy("PASS") for _ in range(4)]
    output = {"command": "request", "content": {"1": "0 1 0", "3": "0 3 0"}}

    response_log = build_response_log(output, policies)

    assert sorted(response_log) == ["1", "3"]
    assert response_log["1"]["response"] == "PASS"
    assert response_log["1"]["verdict"] == "OK"
    assert policies[0].requests == []
    assert policies[1].requests == ["0 1 0"]
    assert policies[3].requests == ["0 3 0"]


def test_run_match_stops_on_finish_output_from_judge():
    policies = [EchoPolicy("PASS") for _ in range(4)]

    def fake_judge(log, initdata, exe_path=None):
        if not log:
            return {"command": "request", "content": {"0": "0 0 0"}}
        return {"command": "finish", "content": {"0": 3, "1": -1, "2": -1, "3": -1}}

    result = run_match(policies, initdata={}, judge_func=fake_judge)

    assert result["scores"] == [3, -1, -1, -1]
    assert result["turns"] == 1
    assert result["terminal_reason"] == "finish"


def test_load_initdata_supports_offset(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"initdata": {"srand": 1}}',
                '{"initdata": {"srand": 2}}',
                '{"initdata": {"srand": 3}}',
            ]
        ),
        encoding="utf-8",
    )

    assert load_initdata(raw, limit=1, offset=1) == [{"srand": 2}]


def test_aleo_process_policy_sends_line_history_to_runner():
    calls = []

    def fake_runner(exe_path, payload):
        calls.append((exe_path, payload))
        return "PASS\n"

    policy = AleoProcessPolicy("aleo.exe", runner=fake_runner)

    assert policy.respond("0 0 2") == "PASS"
    assert policy.respond("1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2") == "PASS"

    assert calls[0] == ("aleo.exe", "1\n0 0 2\n")
    assert calls[1] == (
        "aleo.exe",
        "2\n0 0 2\nPASS\n1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2\n",
    )


def test_aleo_process_policy_ignores_stdout_diagnostics_before_action():
    def fake_runner(exe_path, payload):
        return (
            "(faan.cpp: 328) Tile happens in both needs and discards.\n"
            "(faan.cpp: 244) Multiple recall happens in Hu.\n"
            "PLAY F4\n"
        )

    policy = AleoProcessPolicy("aleo.exe", runner=fake_runner)

    assert policy.respond("2 F4") == "PLAY F4"


def test_aleo_process_policy_fails_closed_to_pass_when_native_runner_crashes():
    def fake_runner(exe_path, payload):
        raise RuntimeError("Aleo bot failed rc=3221225477")

    policy = AleoProcessPolicy("aleo.exe", runner=fake_runner)

    assert policy.respond("3 0 PLAY F4") == "PASS"
    assert policy.responses == ["PASS"]
    assert policy.error_count == 1
    assert "3221225477" in policy.last_error


def test_run_match_reports_aleo_policy_diagnostics():
    def fake_judge(log, initdata, exe_path=None):
        if not log:
            return {"command": "request", "content": {"1": "3 0 PLAY F4"}}
        return {"command": "finish", "content": {"0": 0, "1": 0, "2": 0, "3": 0}}

    def fake_runner(exe_path, payload):
        raise RuntimeError("native crash")

    policies = [
        EchoPolicy("PASS"),
        AleoProcessPolicy("aleo.exe", runner=fake_runner),
        EchoPolicy("PASS"),
        EchoPolicy("PASS"),
    ]

    result = run_match(policies, initdata={}, judge_func=fake_judge)

    assert result["policy_diagnostics"][1]["kind"] == "aleo"
    assert result["policy_diagnostics"][1]["error_count"] == 1
    assert "native crash" in result["policy_diagnostics"][1]["last_error"]


def test_aggregate_policy_diagnostics_sums_numeric_fields_by_player():
    results = [
        {
            "policy_diagnostics": [
                {"draw_turns": 2, "last_model_response": "PLAY W1"},
                {"draw_turns": 1, "fan_check_calls": 3},
            ]
        },
        {
            "policy_diagnostics": [
                {"draw_turns": 4, "fan_check_calls": 5},
                {"draw_turns": 0, "fan_check_calls": 2},
            ]
        },
    ]

    totals = aggregate_policy_diagnostics(results)

    assert totals[0] == {"draw_turns": 6, "fan_check_calls": 5}
    assert totals[1] == {"draw_turns": 1, "fan_check_calls": 5}


def test_summarize_terminals_counts_actions_and_player0_hu_fans():
    results = [
        {
            "turns": 10,
            "final_output": {
                "display": {
                    "action": "HU",
                    "player": 0,
                    "fanCnt": 8,
                }
            }
        },
        {"turns": 15, "final_output": {"display": {"action": "HUANG"}}},
        {
            "turns": 20,
            "final_output": {
                "display": {
                    "action": "HU",
                    "player": 2,
                    "fanCnt": 12,
                }
            }
        },
    ]

    summary = summarize_terminals(results)

    assert summary["terminal_actions"] == {"HU": 2, "HUANG": 1}
    assert summary["placement_reward_schema"].startswith("score rank reward 4/2/1/0")
    assert summary["average_placement_rewards_4_2_1_0"] == [0.0, 0.0, 0.0, 0.0]
    assert summary["player0_average_final_score_reward_4_2_1_0"] == 0.0
    assert summary["player0_hu_fans"] == [8]
    assert summary["min_player0_hu_fan"] == 8
    assert summary["hu_count"] == 2
    assert summary["hu_rate"] == 2 / 3
    assert summary["average_hu_turn"] == 15.0
    assert summary["player0_hu_count"] == 1
    assert summary["player0_hu_rate"] == 1 / 3
    assert summary["player0_average_hu_turn"] == 10.0


def test_placement_rewards_from_scores_handles_ties_and_all_zero_draws():
    assert placement_rewards_from_scores([48, -16, -16, -16]) == [4.0, 1.0, 1.0, 1.0]
    assert placement_rewards_from_scores([16, -8, -4, -4]) == [4.0, 0.0, 1.5, 1.5]
    assert placement_rewards_from_scores([0, 0, 0, 0]) == [0.0, 0.0, 0.0, 0.0]


def test_make_policy_can_create_aleo_process_policy():
    policy = make_policy("aleo", aleo_exe="build/aleo_bot.exe")

    assert isinstance(policy, AleoProcessPolicy)


def test_make_policy_can_create_lawlorentz_effective_policy():
    policy = make_policy("lawlorentz_effective")

    assert isinstance(policy, LawlorentzEffectivePolicy)


def test_make_policy_can_create_transformer_checkpoint_policy(monkeypatch):
    constructed = []

    class FakeTransformerCheckpointPredictor:
        def __init__(self, model_path):
            constructed.append(model_path)
            self.kind = "legal_action_ranker"

    monkeypatch.setattr(
        "official_judge_match.TransformerCheckpointPredictor",
        FakeTransformerCheckpointPredictor,
        raising=False,
    )

    policy = make_policy("transformer", model="models/baseline.pt")

    assert constructed == ["models/baseline.pt"]
    assert policy.predictor.kind == "legal_action_ranker"


def test_make_policy_can_create_transformer_checkpoint_policy_with_qadv(monkeypatch):
    constructed = []

    class FakeTransformerCheckpointPredictor:
        def __init__(self, model_path, *, qadv_path=None, qadv_lambda=0.0):
            constructed.append((model_path, qadv_path, float(qadv_lambda)))
            self.kind = "legal_action_ranker"

    monkeypatch.setattr(
        "official_judge_match.TransformerCheckpointPredictor",
        FakeTransformerCheckpointPredictor,
        raising=False,
    )

    policy = make_policy(
        "transformer",
        model="models/baseline.pt",
        qadv_model="models/qadv.pt",
        qadv_lambda=0.2,
    )

    assert constructed == [("models/baseline.pt", "models/qadv.pt", 0.2)]
    assert policy.predictor.kind == "legal_action_ranker"


def test_botzone_json_process_policy_sends_requests_and_responses_to_runner():
    calls = []

    def fake_runner(exe_path, payload):
        calls.append((exe_path, payload))
        return '{"response":"PASS"}'

    policy = BotzoneJsonProcessPolicy("sample.exe", runner=fake_runner)

    assert policy.respond("0 0 2") == "PASS"
    assert policy.respond("1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2") == "PASS"

    assert calls[0][0] == "sample.exe"
    assert '"requests": ["0 0 2"]' in calls[0][1]
    assert '"responses": []' in calls[0][1]
    assert '"requests": ["0 0 2", "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"]' in calls[1][1]
    assert '"responses": ["PASS"]' in calls[1][1]


def test_run_json_bot_process_launches_python_zip_with_current_interpreter(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout='{"response":"PASS"}', stderr="")

    monkeypatch.setattr("official_judge_match.subprocess.run", fake_run)

    assert run_json_bot_process("bot.zip", "{}") == '{"response":"PASS"}'

    assert calls[0][0] == [sys.executable, "bot.zip"]
    assert calls[0][1]["input"] == "{}"


def test_make_policy_can_create_json_process_policy():
    policy = make_policy("json", model="bot.zip")

    assert isinstance(policy, BotzoneJsonProcessPolicy)
    assert policy.exe_path == "bot.zip"


def test_build_policies_places_target_policy_at_requested_seat():
    class Args:
        policy = "fallback"
        model = None
        qadv_model = None
        qadv_lambda = 0.0
        opponent = "sample"
        opponent_model = None
        opponent_qadv_model = None
        opponent_qadv_lambda = 0.0
        aleo_exe = "aleo.exe"
        sample_exe = "sample.exe"
        lawlorentz_levels = 1
        policy_seat = 2

    policies = build_policies(Args())

    assert isinstance(policies[2], BotzonePolicy)
    assert isinstance(policies[0], BotzoneJsonProcessPolicy)
    assert isinstance(policies[1], BotzoneJsonProcessPolicy)
    assert isinstance(policies[3], BotzoneJsonProcessPolicy)
