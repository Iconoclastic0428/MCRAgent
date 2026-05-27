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
