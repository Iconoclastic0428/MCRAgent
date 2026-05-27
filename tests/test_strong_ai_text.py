import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from strong_ai_text import parse_round_lines


DEALS = [
    "Player 0 Deal W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2",
    "Player 1 Deal W1 W1 W2 W2 W3 W3 B1 B2 B3 T1 T2 T3 F1",
    "Player 2 Deal B1 B1 B2 B2 B3 B3 W4 W5 W6 T4 T5 T6 J1",
    "Player 3 Deal T1 T1 T2 T2 T3 T3 W7 W8 W9 B7 B8 B9 J2",
]


def base_round(*body):
    return [
        "Match abcdefabcdefabcdefabcdef",
        "Wind 1",
        *DEALS,
        *body,
    ]


def response_for(record, turn_index, player):
    return record["logs"][turn_index * 2 + 1][str(player)]["response"]


def requests_for(record, turn_index):
    return (record["logs"][turn_index * 2]["output"] or {})["content"]


def test_parse_round_folds_draw_play_and_synthesizes_pass_reactions():
    record = parse_round_lines(
        base_round(
            "Player 0 Draw J3",
            "Player 0 Play T1",
            "Huang",
            "Score 0 0 0 0",
        )
    )

    assert record["match_id"] == "abcdefabcdefabcdefabcdef"
    assert record["scores"] == {"0": 0, "1": 0, "2": 0, "3": 0}
    assert requests_for(record, 1)["0"] == "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2"
    assert requests_for(record, 2) == {"0": "2 J3"}
    assert response_for(record, 2, 0) == "PLAY T1"
    assert requests_for(record, 3) == {
        "1": "3 0 PLAY T1",
        "2": "3 0 PLAY T1",
        "3": "3 0 PLAY T1",
    }
    assert response_for(record, 3, 1) == "PASS"


def test_parse_round_folds_peng_with_followup_discard():
    record = parse_round_lines(
        base_round(
            "Player 0 Draw J3",
            "Player 0 Play W1",
            "Player 1 Peng W1",
            "Player 1 Play F1",
            "Huang",
            "Score 0 0 0 0",
        )
    )

    assert requests_for(record, 3)["1"] == "3 0 PLAY W1"
    assert response_for(record, 3, 1) == "PENG F1"
    assert "0" not in requests_for(record, 3)


def test_parse_round_folds_chi_with_middle_tile_and_followup_discard():
    record = parse_round_lines(
        base_round(
            "Player 0 Draw J3",
            "Player 0 Play B7",
            "Player 1 Chi B8",
            "Player 1 Play W2",
            "Huang",
            "Score 0 0 0 0",
        )
    )

    assert requests_for(record, 3)["1"] == "3 0 PLAY B7"
    assert response_for(record, 3, 1) == "CHI B8 W2"


def test_parse_round_maps_hu_and_omits_ignored_claimers():
    record = parse_round_lines(
        base_round(
            "Player 0 Draw J3",
            "Player 0 Play B3",
            "Player 1 Hu B3 Ignore Player 2 PENG B3",
            "Fan 9 Mixed",
            "Score -8 24 -8 -8",
        )
    )

    assert requests_for(record, 3) == {
        "1": "3 0 PLAY B3",
        "3": "3 0 PLAY B3",
    }
    assert response_for(record, 3, 1) == "HU"
    assert response_for(record, 3, 3) == "PASS"
    assert record["scores"] == {"0": -8, "1": 24, "2": -8, "3": -8}


def test_parse_round_maps_self_draw_hu():
    record = parse_round_lines(
        base_round(
            "Player 0 Draw J3",
            "Player 0 Hu J3",
            "Fan 8 SelfDraw",
            "Score 24 -8 -8 -8",
        )
    )

    assert requests_for(record, 2) == {"0": "2 J3"}
    assert response_for(record, 2, 0) == "HU"


def test_parse_round_maps_robbing_bugang_hu():
    record = parse_round_lines(
        base_round(
            "Player 1 Draw B5",
            "Player 1 BuGang B5",
            "Player 0 Hu B5",
            "Fan 30 RobbingKong",
            "Score 54 -38 -8 -8",
        )
    )

    assert requests_for(record, 2) == {"1": "2 B5"}
    assert response_for(record, 2, 1) == "BUGANG B5"
    assert requests_for(record, 3)["0"] == "3 1 BUGANG B5"
    assert response_for(record, 3, 0) == "HU"
