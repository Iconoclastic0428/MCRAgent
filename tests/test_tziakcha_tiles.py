from advisor_service.tiles import (
    aleo_symbol,
    botzone_symbol,
    display_name,
    kind_from_tile_id,
    tile_id_from_botzone_symbol,
)


def test_suit_display_names():
    assert display_name(0) == "1m"
    assert display_name(35) == "9m"
    assert display_name(36) == "1s"
    assert display_name(71) == "9s"
    assert display_name(72) == "1p"
    assert display_name(107) == "9p"


def test_wind_and_dragon_display_names():
    assert display_name(108) == "East"
    assert display_name(112) == "South"
    assert display_name(116) == "West"
    assert display_name(120) == "North"
    assert display_name(124) == "Red"
    assert display_name(128) == "Green"
    assert display_name(132) == "White"


def test_kind_and_botzone_symbols():
    assert kind_from_tile_id(0) == 0
    assert kind_from_tile_id(3) == 0
    assert kind_from_tile_id(4) == 1
    assert botzone_symbol(0) == "W1"
    assert botzone_symbol(36) == "T1"
    assert botzone_symbol(72) == "B1"
    assert botzone_symbol(108) == "F1"
    assert botzone_symbol(132) == "J3"
    assert aleo_symbol(36) == "T1"
    assert tile_id_from_botzone_symbol("B1") == 72
