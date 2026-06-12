from advisor_service.state import AdvisorState


def test_reconnect_snapshot_extracts_hand_and_prompt():
    state = AdvisorState()
    state.ingest(
        {
            "m": 4,
            "i": {
                "v": 0,
                "t": 0,
                "y": 16,
                "p": [
                    {
                        "h": {
                            "s": [0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 112, 116, 124],
                            "p": [],
                        },
                        "f": 0,
                    },
                    {"h": {"s": [], "p": []}, "f": 0},
                    {"h": {"s": [], "p": []}, "f": 0},
                    {"h": {"s": [], "p": []}, "f": 0},
                ],
                "a": {"2": [44], "8": [0]},
            },
        }
    )
    snapshot = state.snapshot()
    assert snapshot["seat"] == 0
    assert snapshot["hand_display"] == [
        "1m",
        "2m",
        "3m",
        "1s",
        "2s",
        "3s",
        "1p",
        "2p",
        "3p",
        "East",
        "South",
        "West",
        "Red",
    ]
    assert snapshot["available_actions"]["discard"] == [44]
    assert snapshot["available_actions"]["pass"] == [0]
    assert snapshot["botzone_history_complete"] is False
    assert snapshot["botzone_history"] == []


def test_live_draw_and_discard_shapes_update_hand_and_prompt():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    state.ingest({"m": 2, "r": 2, "v": [0, 4, 36]})
    state.ingest({"m": 2, "r": 6, "v": 0, "t": 72, "h": 70, "a": [393216]})
    snapshot = state.snapshot()
    assert snapshot["turn"] == 0
    assert snapshot["wall_count"] == 70
    assert snapshot["last_draw"] == {"seat": 0, "tile": 72, "display": "1p"}
    assert snapshot["hand_display"] == ["1m", "2m", "1s", "1p"]
    assert snapshot["available_actions"] == {"hu": [0]}

    state.ingest({"m": 2, "r": 7, "v": 0, "t": 72, "h": 1})
    snapshot = state.snapshot()
    assert snapshot["last_discard"] == {"seat": 0, "tile": 72, "display": "1p"}
    assert snapshot["hand_display"] == ["1m", "2m", "1s"]
    assert "REQ 2 B1" in snapshot["botzone_history"]
    assert "RES PLAY B1" in snapshot["botzone_history"]


def test_live_state_exports_botzone_history_before_current_reaction():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    state.ingest(
        {
            "m": 2,
            "r": 2,
            "v": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 72],
        }
    )
    state.ingest({"m": 2, "r": 6, "v": 0, "t": 76, "h": 70, "a": {"2": [76]}})
    state.ingest({"m": 2, "r": 7, "v": 0, "t": 76})
    state.ingest({"m": 2, "r": 7, "v": 3, "t": 8, "a": {"3": [12], "8": [0]}})

    snapshot = state.snapshot()

    assert snapshot["botzone_history_complete"] is True
    assert snapshot["botzone_history"][:4] == [
        "REQ 0 0 0",
        "RES PASS",
        "REQ 1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 B1",
        "RES PASS",
    ]
    assert snapshot["botzone_history"][-2:] == ["REQ 2 B2", "RES PLAY B2"]
    assert "REQ 3 3 PLAY W3" not in snapshot["botzone_history"]
    assert snapshot["available_actions"] == {"chow": [12], "pass": [0]}


def test_live_state_leaves_opponent_bugang_hu_prompt_as_current_request():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    state.ingest(
        {
            "m": 2,
            "r": 2,
            "v": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 72],
        }
    )
    promoted_kong_pack = (3 << 8) | (40 >> 2)
    state.ingest({"m": 2, "r": 10, "v": 1, "p": promoted_kong_pack, "a": {"6": [0], "8": [0]}})

    snapshot = state.snapshot()

    assert snapshot["last_win_event"] == {
        "seat": 1,
        "tile": 40,
        "source": "bugang",
        "is_self_draw": False,
        "is_about_kong": True,
    }
    assert snapshot["available_actions"] == {"hu": [0], "pass": [0]}
    assert "REQ 3 1 BUGANG T2" not in snapshot["botzone_history"]


def test_live_state_records_opponent_claim_forced_discard_as_claim_request():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    state.ingest(
        {
            "m": 2,
            "r": 2,
            "v": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 72],
        }
    )
    state.ingest({"m": 2, "r": 7, "v": 1, "t": 0})
    pung_pack = (1 << 8) | (0 >> 2)
    state.ingest({"m": 2, "r": 9, "v": 2, "p": pung_pack})
    state.ingest({"m": 2, "r": 7, "v": 2, "t": 72})

    history = state.snapshot()["botzone_history"]

    assert "REQ 3 1 PLAY W1" in history
    assert "REQ 3 2 PENG B1" in history
    assert "REQ 3 2 PLAY B1" not in history


def test_action_list_decodes_hiword_action_and_loword_value():
    state = AdvisorState()
    state.ingest({"m": 2, "r": 7, "v": 1, "t": 36, "a": [196616, 393216, 524288]})
    assert state.snapshot()["available_actions"] == {
        "chow": [8],
        "hu": [0],
        "pass": [0],
    }


def test_action_list_normalizes_packed_claim_values():
    state = AdvisorState()
    state.ingest({"m": 2, "r": 7, "v": 1, "t": 36, "a": [(3 << 16) | 18512]})
    assert state.snapshot()["available_actions"] == {"chow": [64]}


def test_deal_filters_flower_and_applies_self_flower_replacement():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    state.ingest({"m": 2, "r": 2, "v": [0, 4, 36, 136]})

    snapshot = state.snapshot()
    assert snapshot["hand_display"] == ["1m", "2m", "1s"]
    assert snapshot["flowers"] == 0

    state.ingest({"m": 2, "r": 4, "v": 0, "h": 90, "t": 40, "tt": 0})
    snapshot = state.snapshot()
    assert snapshot["wall_count"] == 90
    assert snapshot["hand_display"] == ["1m", "2m", "1s", "2s"]
    assert snapshot["flowers"] == 1


def test_live_reconnect_snapshot_restores_local_flower_count():
    state = AdvisorState()
    state.ingest(
        {
            "m": 4,
            "v": 2,
            "u": [{"f": 0}, {"f": 1}, {"f": 2}, {"f": 3}],
            "i": {
                "t": 2,
                "w": {"f": 60, "b": 8},
                "h": [
                    {"s": [], "p": []},
                    {"s": [], "p": []},
                    {"s": [36, 120, 142], "p": []},
                    {"s": [], "p": []},
                ],
            },
        }
    )

    snapshot = state.snapshot()
    assert snapshot["seat"] == 2
    assert snapshot["hand"] == [36, 120]
    assert snapshot["flowers"] == 2


def test_self_melds_remove_tiles_and_record_pack():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    state.ingest({"m": 2, "r": 2, "v": [40, 44, 72]})
    chow_pack = 64 | (40 >> 2)
    state.ingest({"m": 2, "r": 8, "v": 0, "p": chow_pack, "tt": 35000})
    assert state.snapshot()["hand"] == [72]
    assert state.snapshot()["melds"] == [chow_pack]

    state.ingest({"m": 2, "r": 2, "v": [36, 37, 38, 39, 72]})
    kong_pack = (2 << 8) | 64 | (36 >> 2)
    state.ingest({"m": 2, "r": 10, "v": 0, "p": kong_pack, "tt": 35000})
    snapshot = state.snapshot()
    assert snapshot["hand"] == [39, 72]
    assert snapshot["melds"][-1] == kong_pack


def test_state_tracks_kong_draw_and_robbing_kong_win_context():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    state.ingest({"m": 2, "r": 2, "v": [36, 37, 38, 39, 72]})
    kong_pack = (2 << 8) | (36 >> 2)
    state.ingest({"m": 2, "r": 10, "v": 0, "p": kong_pack})
    state.ingest({"m": 2, "r": 6, "v": 0, "t": 120, "h": 12, "a": {"6": [0]}})

    snapshot = state.snapshot()

    assert snapshot["last_win_event"] == {
        "seat": 0,
        "tile": 120,
        "source": "draw",
        "is_self_draw": True,
        "is_about_kong": True,
    }

    state.ingest({"m": 2, "r": 7, "v": 1, "t": 40})
    promoted_kong_pack = (3 << 8) | (40 >> 2)
    state.ingest({"m": 2, "r": 10, "v": 1, "p": promoted_kong_pack, "a": {"6": [0], "8": [0]}})

    snapshot = state.snapshot()

    assert snapshot["last_win_event"] == {
        "seat": 1,
        "tile": 40,
        "source": "bugang",
        "is_self_draw": False,
        "is_about_kong": True,
    }


def test_state_tracks_visible_tiles_for_last_tile_hu_context():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    for seat in (1, 2, 3):
        state.ingest({"m": 2, "r": 7, "v": seat, "t": 36})
        state.ingest({"m": 2, "r": 6, "v": seat, "t": 72})

    state.ingest({"m": 2, "r": 7, "v": 1, "t": 36, "a": {"6": [0], "8": [0]}})
    snapshot = state.snapshot()

    assert snapshot["visible_counts"][36 >> 2] == 3
    assert snapshot["last_win_event"]["tile"] == 36


def test_state_preserves_after_kong_flag_through_flower_replacement():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    state.ingest({"m": 2, "r": 2, "v": [36, 37, 38, 39, 72]})
    kong_pack = (2 << 8) | (36 >> 2)
    state.ingest({"m": 2, "r": 10, "v": 0, "p": kong_pack})
    state.ingest({"m": 2, "r": 4, "v": 0, "t": 120, "h": 12, "a": {"6": [0]}})

    assert state.snapshot()["last_win_event"] == {
        "seat": 0,
        "tile": 120,
        "source": "draw",
        "is_self_draw": True,
        "is_about_kong": True,
    }


def test_state_records_ron_result_and_deal_in_rate_from_settlement_flags():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 1, "i": {"t": 1}})
    state.ingest({"m": 2, "r": 2, "v": [0, 4, 8]})
    state.ingest({"m": 2, "r": 7, "v": 1, "t": 36})
    state.ingest(
        {
            "m": 2,
            "r": 12,
            "b": (1 << 2) | (1 << (1 + 4)),
            "s": [-8, -16, 32, -8],
            "h": 8,
        }
    )

    snapshot = state.snapshot()

    assert snapshot["last_result"] == {
        "round_index": 1,
        "seat": 1,
        "winner": 2,
        "discarder": 1,
        "is_draw": False,
        "is_self_draw": False,
        "is_win": False,
        "is_deal_in": True,
        "scores": [-8, -16, 32, -8],
        "score_delta": -16,
        "fan": 8,
        "source": "settlement",
        "event_count": 4,
    }
    assert snapshot["result_stats"]["games"] == 1
    assert snapshot["result_stats"]["wins"] == 0
    assert snapshot["result_stats"]["deal_ins"] == 1
    assert snapshot["result_stats"]["win_rate"] == 0.0
    assert snapshot["result_stats"]["deal_in_rate"] == 1.0


def test_state_records_self_draw_win_and_drawn_game_rates():
    state = AdvisorState()
    state.ingest({"m": 4, "v": 0, "i": {"t": 0}})
    state.ingest({"m": 2, "r": 2, "v": [0, 4, 8]})
    state.ingest({"m": 2, "r": 6, "v": 0, "t": 36})
    state.ingest({"m": 2, "r": 12, "v": 0, "s": [48, -16, -16, -16], "h": 8})
    state.ingest({"m": 2, "r": 14, "v": 0})
    state.ingest({"m": 2, "r": 2, "v": [0, 4, 8]})
    state.ingest({"m": 2, "r": 13, "s": [0, 0, 0, 0]})

    stats = state.snapshot()["result_stats"]

    assert stats["games"] == 2
    assert stats["wins"] == 1
    assert stats["draws"] == 1
    assert stats["deal_ins"] == 0
    assert stats["win_rate"] == 0.5
    assert stats["deal_in_rate"] == 0.0
    assert state.snapshot()["result_history"][-1]["is_draw"] is True
