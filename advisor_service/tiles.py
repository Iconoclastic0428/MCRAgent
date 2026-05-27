"""Tile conversion helpers for Tziakcha and Botzone/MCR notation."""

HONOR_NAMES = {
    27: "East",
    28: "South",
    29: "West",
    30: "North",
    31: "Red",
    32: "Green",
    33: "White",
}

BOTZONE_SYMBOLS = (
    [f"W{i}" for i in range(1, 10)]
    + [f"B{i}" for i in range(1, 10)]
    + [f"T{i}" for i in range(1, 10)]
    + [f"F{i}" for i in range(1, 5)]
    + [f"J{i}" for i in range(1, 4)]
)


def kind_from_tile_id(tile_id: int) -> int:
    if not isinstance(tile_id, int):
        raise TypeError("tile_id must be int")
    if tile_id < 0 or tile_id > 143:
        raise ValueError(f"tile_id out of range: {tile_id}")
    return tile_id >> 2


def display_name(tile_id: int) -> str:
    kind = kind_from_tile_id(tile_id)
    if 0 <= kind <= 8:
        return f"{kind + 1}m"
    if 9 <= kind <= 17:
        return f"{kind - 8}s"
    if 18 <= kind <= 26:
        return f"{kind - 17}p"
    if kind in HONOR_NAMES:
        return HONOR_NAMES[kind]
    return f"flower{kind - 33}"


def botzone_symbol(tile_id: int) -> str:
    kind = botzone_kind_from_tile_id(tile_id)
    if 0 <= kind < len(BOTZONE_SYMBOLS):
        return BOTZONE_SYMBOLS[kind]
    return f"H{kind - 33}"


def botzone_kind_from_tile_id(tile_id: int) -> int:
    return tziakcha_kind_to_botzone_kind(kind_from_tile_id(tile_id))


def tile_id_from_botzone_kind(botzone_kind: int) -> int:
    return tziakcha_kind_from_botzone_kind(botzone_kind) << 2


def tile_id_from_botzone_symbol(symbol: str) -> int:
    return tile_id_from_botzone_kind(BOTZONE_SYMBOLS.index(symbol))


def aleo_symbol(tile_id: int) -> str:
    return botzone_symbol(tile_id)


def aleo_kind_from_tile_id(tile_id: int) -> int:
    return botzone_kind_from_tile_id(tile_id)


def tile_id_from_aleo_kind(aleo_kind: int) -> int:
    return tile_id_from_botzone_kind(aleo_kind)


def tziakcha_kind_to_aleo_kind(kind: int) -> int:
    return tziakcha_kind_to_botzone_kind(kind)


def tziakcha_kind_from_aleo_kind(kind: int) -> int:
    return tziakcha_kind_from_botzone_kind(kind)


def tziakcha_kind_to_botzone_kind(kind: int) -> int:
    if 9 <= kind <= 17:
        return kind + 9
    if 18 <= kind <= 26:
        return kind - 9
    return kind


def tziakcha_kind_from_botzone_kind(kind: int) -> int:
    if 9 <= kind <= 17:
        return kind + 9
    if 18 <= kind <= 26:
        return kind - 9
    return kind
