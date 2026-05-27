#!/usr/bin/env python3
"""Pure-Python runtime for exported MCR Botzone policies.

This file is intentionally Python 3.6 compatible and dependency-free so it can
run inside Botzone's uploaded Python zip environment.
"""

from __future__ import print_function

import json
import math
import re
import sys
from collections import Counter
from functools import lru_cache


SUITS = ("W", "B", "T")
TILE_ORDER = (
    ["%s%d" % (suit, rank) for suit in SUITS for rank in range(1, 10)]
    + ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]
)
TILE_TO_INDEX = dict((tile, index) for index, tile in enumerate(TILE_ORDER))
ORPHANS = set(
    TILE_TO_INDEX[tile]
    for tile in [
        "W1",
        "W9",
        "B1",
        "B9",
        "T1",
        "T9",
        "F1",
        "F2",
        "F3",
        "F4",
        "J1",
        "J2",
        "J3",
    ]
)
ORPHAN_TILES = set(TILE_ORDER[index] for index in ORPHANS)
HONOR_TILES = set(["F1", "F2", "F3", "F4", "J1", "J2", "J3"])
WIND_TILES = set(["F1", "F2", "F3", "F4"])
DRAGON_TILES = set(["J1", "J2", "J3"])
ACTIONS = ("PASS", "HU", "PLAY", "GANG", "BUGANG", "PENG", "CHI")
REQUEST_TYPES = ("init", "deal", "draw", "reaction", "other")
EVENT_ACTIONS = ("NONE", "PLAY", "PENG", "CHI", "GANG", "BUGANG")
FEATURE_NAMES = (
    ["request_%s" % name for name in REQUEST_TYPES]
    + ["event_%s" % name for name in EVENT_ACTIONS]
    + ["action_%s" % name for name in ACTIONS]
    + ["hand_%s" % tile for tile in TILE_ORDER]
    + ["discard_%s" % tile for tile in TILE_ORDER]
    + ["claim_%s" % tile for tile in TILE_ORDER]
    + ["drawn_%s" % tile for tile in TILE_ORDER]
    + [
        "hand_total",
        "candidate_tile_count",
        "current_regular_shanten",
        "current_seven_shanten",
        "current_orphan_shanten",
        "current_min_shanten",
        "after_regular_shanten",
        "after_seven_shanten",
        "after_orphan_shanten",
        "after_min_shanten",
        "delta_min_shanten",
        "candidate_is_drawn",
        "legal_count",
    ]
)
FEATURE_INDEX = dict((name, index) for index, name in enumerate(FEATURE_NAMES))


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def score_hgb_export(model_data, features):
    raw = float(model_data["baseline"])
    for tree in model_data["trees"]:
        node_index = 0
        while True:
            node = tree[node_index]
            if node[5]:
                raw += float(node[4])
                break
            value = float(features[int(node[0])])
            if math.isnan(value):
                go_left = bool(node[6])
            else:
                go_left = value <= float(node[1])
            node_index = int(node[2] if go_left else node[3])
    return sigmoid(raw)


def char_wb_ngrams(text, min_n, max_n):
    text = re.sub(r"\s+", " ", text.strip())
    ngrams = []
    for word in text.split():
        padded = " " + word + " "
        length = len(padded)
        for n in range(min_n, max_n + 1):
            if length < n:
                continue
            for index in range(length - n + 1):
                ngrams.append(padded[index : index + n])
    return ngrams


def score_tfidf_sgd_export(model_data, text):
    vocab = model_data["vocabulary"]
    counts = {}
    for gram in char_wb_ngrams(text, model_data["min_n"], model_data["max_n"]):
        index = vocab.get(gram)
        if index is not None:
            counts[index] = counts.get(index, 0.0) + 1.0
    if not counts:
        return sigmoid(float(model_data["intercept"]))

    idf = model_data["idf"]
    norm_sq = 0.0
    weighted = {}
    for index, count in counts.items():
        value = count * float(idf[index])
        weighted[index] = value
        norm_sq += value * value
    norm = math.sqrt(norm_sq) or 1.0

    score = float(model_data["intercept"])
    coef = model_data["coef"]
    for index, value in weighted.items():
        score += (value / norm) * float(coef[index])
    return sigmoid(score)


def tile_counts(tiles):
    counts = [0] * len(TILE_ORDER)
    for tile in tiles:
        index = TILE_TO_INDEX.get(tile)
        if index is not None:
            counts[index] += 1
    return tuple(counts)


def is_suited_index(index):
    return 0 <= index < 27


def same_suit_sequence_start(index):
    return is_suited_index(index) and index % 9 <= 6


@lru_cache(maxsize=200000)
def regular_shanten_counts(counts):
    @lru_cache(maxsize=None)
    def search(state, melds, taatsu, pair):
        index = None
        for i, count in enumerate(state):
            if count:
                index = i
                break
        if index is None:
            usable_taatsu = min(taatsu, 4 - melds)
            return 8 - 2 * melds - usable_taatsu - pair

        best = 8
        current = list(state)
        if current[index] >= 3:
            current[index] -= 3
            best = min(best, search(tuple(current), melds + 1, taatsu, pair))
            current[index] += 3
        if (
            same_suit_sequence_start(index)
            and current[index + 1] > 0
            and current[index + 2] > 0
        ):
            current[index] -= 1
            current[index + 1] -= 1
            current[index + 2] -= 1
            best = min(best, search(tuple(current), melds + 1, taatsu, pair))
            current[index] += 1
            current[index + 1] += 1
            current[index + 2] += 1
        if current[index] >= 2:
            current[index] -= 2
            if pair == 0:
                best = min(best, search(tuple(current), melds, taatsu, 1))
            best = min(best, search(tuple(current), melds, taatsu + 1, pair))
            current[index] += 2
        if same_suit_sequence_start(index) and current[index + 1] > 0:
            current[index] -= 1
            current[index + 1] -= 1
            best = min(best, search(tuple(current), melds, taatsu + 1, pair))
            current[index] += 1
            current[index + 1] += 1
        if is_suited_index(index) and index % 9 <= 6 and current[index + 2] > 0:
            current[index] -= 1
            current[index + 2] -= 1
            best = min(best, search(tuple(current), melds, taatsu + 1, pair))
            current[index] += 1
            current[index + 2] += 1
        current[index] -= 1
        best = min(best, search(tuple(current), melds, taatsu, pair))
        return best

    return search(counts, 0, 0, 0)


@lru_cache(maxsize=200000)
def seven_pairs_shanten_counts(counts):
    pairs = sum(1 for count in counts if count >= 2)
    distinct = sum(1 for count in counts if count > 0)
    return 6 - pairs + max(0, 7 - distinct)


@lru_cache(maxsize=200000)
def thirteen_orphans_shanten_counts(counts):
    unique = sum(1 for index in ORPHANS if counts[index] > 0)
    has_pair = any(counts[index] >= 2 for index in ORPHANS)
    return 13 - unique - (1 if has_pair else 0)


def regular_shanten(tiles):
    return regular_shanten_counts(tile_counts(tiles))


def seven_pairs_shanten(tiles):
    return seven_pairs_shanten_counts(tile_counts(tiles))


def thirteen_orphans_shanten(tiles):
    return thirteen_orphans_shanten_counts(tile_counts(tiles))


def min_shanten(tiles):
    counts = tile_counts(tiles)
    return min(
        regular_shanten_counts(counts),
        seven_pairs_shanten_counts(counts),
        thirteen_orphans_shanten_counts(counts),
    )


def remove_one_tile(tiles, tile):
    result = list(tiles)
    try:
        result.remove(tile)
    except ValueError:
        pass
    return result


def split_tiles(tokens):
    return [
        token
        for token in tokens
        if token and token[0] in ("W", "B", "T", "F", "J", "H")
    ]


def parse_request(request):
    return request.strip().split()


def is_suited(tile):
    return len(tile) == 2 and tile[0] in ("W", "B", "T") and tile[1].isdigit()


def request_event_tile(tokens):
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2] in ("PLAY", "PENG", "CHI"):
        return tokens[-1]
    if len(tokens) >= 4 and tokens[0] == "3" and tokens[2] == "BUGANG":
        return tokens[3]
    return None


def can_complete_hand(hand, tile=None):
    tiles = list(hand.elements())
    if tile:
        tiles.append(tile)
    return min_shanten(tiles) <= -1


@lru_cache(maxsize=200000)
def meld_decompositions_counts(counts):
    index = None
    for i, count in enumerate(counts):
        if count:
            index = i
            break
    if index is None:
        return ((),)

    results = []
    current = list(counts)
    tile = TILE_ORDER[index]
    if current[index] >= 3:
        current[index] -= 3
        for rest in meld_decompositions_counts(tuple(current)):
            results.append((("TRIP", tile),) + rest)
        current[index] += 3
    if (
        same_suit_sequence_start(index)
        and current[index + 1] > 0
        and current[index + 2] > 0
    ):
        current[index] -= 1
        current[index + 1] -= 1
        current[index + 2] -= 1
        for rest in meld_decompositions_counts(tuple(current)):
            results.append((("SEQ", tile[0], int(tile[1])),) + rest)
        current[index] += 1
        current[index + 1] += 1
        current[index + 2] += 1
    return tuple(results[:200])


@lru_cache(maxsize=200000)
def regular_decompositions_counts(counts):
    results = []
    for index, count in enumerate(counts):
        if count < 2:
            continue
        current = list(counts)
        current[index] -= 2
        pair = ("PAIR", TILE_ORDER[index])
        for melds in meld_decompositions_counts(tuple(current)):
            if len(melds) == 4:
                results.append((pair,) + melds)
    return tuple(results[:500])


def _is_all_simples(tiles):
    return all(tile[0] in SUITS and tile[1] not in ("1", "9") for tile in tiles)


def _is_one_void(tiles):
    suits = set(tile[0] for tile in tiles if tile[0] in SUITS)
    return 1 <= len(suits) <= 2


def _has_mixed_shifted_chows(seq_set):
    for step in (1, 2):
        for start in range(1, 8 - 2 * step):
            if step == 2 and start % 2 == 0:
                continue
            ranks = (start, start + step, start + 2 * step)
            for suit_a in SUITS:
                for suit_b in SUITS:
                    for suit_c in SUITS:
                        if len(set([suit_a, suit_b, suit_c])) == 3 and all(
                            key in seq_set
                            for key in ((suit_a, ranks[0]), (suit_b, ranks[1]), (suit_c, ranks[2]))
                        ):
                            return True
    return False


def _has_wait_fan(decomp, win_tile):
    if not win_tile:
        return False
    pair = [part for part in decomp if part[0] == "PAIR"][0]
    if pair[1] == win_tile:
        return True
    if not is_suited(win_tile):
        return False
    win_suit = win_tile[0]
    win_rank = int(win_tile[1])
    for part in decomp:
        if part[0] != "SEQ" or part[1] != win_suit:
            continue
        start = part[2]
        if win_rank == start + 1:
            return True
        if start == 1 and win_rank == 3:
            return True
        if start == 7 and win_rank == 7:
            return True
    return False


def _decomposition_lower_bound(decomp, tiles, open_meld_count, self_draw, win_tile=None):
    pair = [part for part in decomp if part[0] == "PAIR"][0]
    seqs = [part for part in decomp if part[0] == "SEQ"]
    trips = [part for part in decomp if part[0] == "TRIP"]

    score = 0
    high = 0
    seq_keys = [(seq[1], seq[2]) for seq in seqs]
    seq_set = set(seq_keys)

    for suit in SUITS:
        if (suit, 1) in seq_set and (suit, 4) in seq_set and (suit, 7) in seq_set:
            high = max(high, 16)

    for ranks in ((1, 4, 7),):
        for suit_a in SUITS:
            for suit_b in SUITS:
                for suit_c in SUITS:
                    if len(set([suit_a, suit_b, suit_c])) == 3 and all(
                        key in seq_set for key in ((suit_a, ranks[0]), (suit_b, ranks[1]), (suit_c, ranks[2]))
                    ):
                        high = max(high, 8)

    for rank in range(1, 8):
        if all((suit, rank) in seq_set for suit in SUITS):
            high = max(high, 8)

    if not seqs:
        score += 6
    if _is_all_simples(tiles):
        score += 2
    if _is_one_void(tiles):
        score += 1
    if len(seqs) == 4 and pair[1] not in HONOR_TILES:
        score += 2
    if open_meld_count == 0:
        score += 4 if self_draw else 2

    if _has_mixed_shifted_chows(seq_set):
        score += 6

    dragon_triplets = sum(1 for trip in trips if trip[1] in DRAGON_TILES)
    score += 2 * dragon_triplets

    rank_to_suits = {}
    for suit, rank in seq_keys:
        suits = rank_to_suits.get(rank)
        if suits is None:
            suits = rank_to_suits[rank] = set()
        suits.add(suit)
    if len(set(seq_keys)) < len(seq_keys) or any(len(suits) >= 2 for suits in rank_to_suits.values()):
        score += 1
    if any((suit, start) in seq_set and (suit, start + 3) in seq_set for suit in SUITS for start in range(1, 4)):
        score += 1
    if _has_wait_fan(decomp, win_tile):
        score += 1

    return max(high, score)


def conservative_high_fan_lower_bound(
    tiles, open_meld_count=0, self_draw=False, win_tile=None, is_last=False
):
    """Return a safe lower bound for a small set of unambiguous MCR high fans.

    This intentionally does not try to be a full MCR fan calculator. Botzone's
    source-only Python runtime cannot call the official MahjongGB helper, so HU
    is allowed only when a standalone high-fan pattern proves at least 8 fan.
    """

    tiles = [tile for tile in tiles if tile in TILE_TO_INDEX]
    if len(tiles) != 14:
        return 0
    hand = Counter(tiles)
    if min_shanten(tiles) > -1:
        return 0

    if is_last:
        return 8

    tile_set = set(tiles)
    if tile_set.issubset(ORPHAN_TILES) and all(hand[tile] > 0 for tile in ORPHAN_TILES):
        return 88
    if sum(1 for count in hand.values() if count == 2) == 7:
        return 24
    if tile_set.issubset(HONOR_TILES):
        return 64
    if tile_set.issubset(ORPHAN_TILES):
        return 32
    if all(tile not in HONOR_TILES and tile[1] in ("1", "9") for tile in tiles):
        return 64

    suits = set(tile[0] for tile in tiles if tile[0] in SUITS)
    has_honor = any(tile in HONOR_TILES for tile in tiles)
    if len(suits) == 1 and not has_honor:
        return 24

    wind_triplets = sum(1 for tile in WIND_TILES if hand[tile] >= 3)
    wind_pairs = sum(1 for tile in WIND_TILES if hand[tile] == 2)
    if wind_triplets == 4:
        return 88
    if wind_triplets == 3 and wind_pairs >= 1:
        return 64

    dragon_triplets = sum(1 for tile in DRAGON_TILES if hand[tile] >= 3)
    dragon_pairs = sum(1 for tile in DRAGON_TILES if hand[tile] == 2)
    if dragon_triplets == 3:
        return 88
    if dragon_triplets == 2 and dragon_pairs >= 1:
        return 64

    counts = tile_counts(tiles)
    best = 0
    for decomp in regular_decompositions_counts(counts):
        best = max(
            best,
            _decomposition_lower_bound(
                decomp, tiles, open_meld_count, self_draw, win_tile=win_tile
            ),
        )
    return best


def conservative_can_hu(hand, request, open_meld_count=0, is_last=False):
    tokens = parse_request(request)
    tiles = list(hand.elements())
    self_draw = bool(tokens and tokens[0] == "2")
    win_tile = tokens[1] if self_draw and len(tokens) >= 2 else None
    if tokens and tokens[0] == "3":
        event_tile = request_event_tile(tokens)
        if not event_tile:
            return False
        tiles.append(event_tile)
        win_tile = event_tile
    return (
        conservative_high_fan_lower_bound(
            tiles,
            open_meld_count=open_meld_count,
            self_draw=self_draw,
            win_tile=win_tile,
            is_last=is_last,
        )
        >= 8
    )


def dedupe(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def cleanup(hand):
    for tile in list(hand):
        if hand[tile] <= 0:
            del hand[tile]


def generate_draw_responses(hand):
    responses = ["PASS"] if not hand else []
    if can_complete_hand(hand):
        responses.append("HU")
    for tile in sorted(tile for tile, count in hand.items() if count > 0):
        responses.append("PLAY %s" % tile)
    for tile in sorted(tile for tile, count in hand.items() if count >= 4):
        responses.append("GANG %s" % tile)
    return dedupe(responses)


def can_chi(player_id, actor_id, event_tile):
    return player_id is not None and player_id == (actor_id + 1) % 4 and is_suited(event_tile)


def generate_chi_responses(event_tile, hand):
    suit = event_tile[0]
    rank = int(event_tile[1])
    responses = []
    for middle_rank in range(rank - 1, rank + 2):
        if middle_rank < 2 or middle_rank > 8:
            continue
        sequence = [
            "%s%d" % (suit, middle_rank - 1),
            "%s%d" % (suit, middle_rank),
            "%s%d" % (suit, middle_rank + 1),
        ]
        needed = Counter(sequence)
        needed[event_tile] -= 1
        if all(hand[tile] >= count for tile, count in needed.items() if count > 0):
            after_chi = Counter(hand)
            for tile, count in needed.items():
                after_chi[tile] -= count
            cleanup(after_chi)
            for discard in sorted(tile for tile, count in after_chi.items() if count > 0):
                responses.append("CHI %s%d %s" % (suit, middle_rank, discard))
    return responses


def generate_reaction_responses(player_id, tokens, hand):
    responses = ["PASS"]
    event_tile = request_event_tile(tokens)
    if not event_tile:
        return responses
    if player_id is not None and len(tokens) >= 2 and player_id == int(tokens[1]):
        return responses
    if can_complete_hand(hand, event_tile):
        responses.append("HU")
    if tokens[2] in ("PLAY", "PENG", "CHI"):
        if hand[event_tile] >= 3:
            responses.append("GANG")
        if hand[event_tile] >= 2:
            after_peng = Counter(hand)
            after_peng[event_tile] -= 2
            cleanup(after_peng)
            for discard in sorted(tile for tile, count in after_peng.items() if count > 0):
                responses.append("PENG %s" % discard)
        if can_chi(player_id, int(tokens[1]), event_tile):
            responses.extend(generate_chi_responses(event_tile, hand))
    return dedupe(responses)


def generate_legal_responses(player_id, request, hand):
    tokens = parse_request(request)
    if not tokens:
        return ["PASS"]
    if tokens[0] in ("0", "1"):
        return ["PASS"]
    if tokens[0] == "2" and len(tokens) >= 2:
        return generate_draw_responses(hand)
    if tokens[0] == "3" and len(tokens) >= 3:
        return generate_reaction_responses(player_id, tokens, hand)
    return ["PASS"]


def response_action(response):
    parts = response.strip().split()
    action = parts[0].upper() if parts else "PASS"
    return action if action in ACTIONS else "PASS"


def response_discard_tile(response):
    parts = response.split()
    if len(parts) == 2 and parts[0] in ("PLAY", "PENG"):
        return parts[1]
    if len(parts) == 3 and parts[0] == "CHI":
        return parts[2]
    return None


def request_type(tokens):
    if not tokens:
        return "other"
    return {"0": "init", "1": "deal", "2": "draw", "3": "reaction"}.get(tokens[0], "other")


def event_action(tokens):
    if len(tokens) >= 3 and tokens[0] == "3":
        action = tokens[2].upper()
        return action if action in EVENT_ACTIONS else "NONE"
    return "NONE"


def set_one_hot(values, prefix, key):
    if key is None:
        return
    index = FEATURE_INDEX.get("%s_%s" % (prefix, key))
    if index is not None:
        values[index] = 1.0


def hand_shanten_values(tiles):
    regular = float(regular_shanten(tiles))
    seven = float(seven_pairs_shanten(tiles))
    orphan = float(thirteen_orphans_shanten(tiles))
    return regular, seven, orphan, float(min(regular, seven, orphan))


def featurize_response(input_text, request, response, hand, legal_count=None):
    values = [0.0] * len(FEATURE_NAMES)
    hand_counts = Counter(hand)
    tokens = parse_request(request)
    action = response_action(response)
    discard = response_discard_tile(response)
    claim_tile = request_event_tile(tokens)
    drawn_tile = tokens[1] if len(tokens) >= 2 and tokens[0] == "2" else None

    set_one_hot(values, "request", request_type(tokens))
    set_one_hot(values, "event", event_action(tokens))
    set_one_hot(values, "action", action)
    set_one_hot(values, "discard", discard)
    set_one_hot(values, "claim", claim_tile)
    set_one_hot(values, "drawn", drawn_tile)

    for tile, count in hand_counts.items():
        index = FEATURE_INDEX.get("hand_%s" % tile)
        if index is not None:
            values[index] = min(float(count), 4.0) / 4.0

    hand_tiles = list(hand_counts.elements())
    current_regular, current_seven, current_orphan, current_min = hand_shanten_values(hand_tiles)
    after_tiles = remove_one_tile(hand_tiles, discard) if discard else hand_tiles
    after_regular, after_seven, after_orphan, after_min = hand_shanten_values(after_tiles)
    scalar_values = {
        "hand_total": min(float(sum(hand_counts.values())), 14.0) / 14.0,
        "candidate_tile_count": min(float(hand_counts[discard]), 4.0) / 4.0
        if discard
        else 0.0,
        "current_regular_shanten": current_regular,
        "current_seven_shanten": current_seven,
        "current_orphan_shanten": current_orphan,
        "current_min_shanten": current_min,
        "after_regular_shanten": after_regular,
        "after_seven_shanten": after_seven,
        "after_orphan_shanten": after_orphan,
        "after_min_shanten": after_min,
        "delta_min_shanten": after_min - current_min,
        "candidate_is_drawn": 1.0 if discard and discard == drawn_tile else 0.0,
        "legal_count": min(float(legal_count or 0), 32.0) / 32.0,
    }
    for name, value in scalar_values.items():
        values[FEATURE_INDEX[name]] = value
    return values


def response_candidate_text(input_text, response, hand, request):
    parts = response.split()
    action = parts[0] if parts else "PASS"
    discard = response_discard_tile(response)
    features = [input_text, "RESP %s" % response, "ACTION %s" % action]
    if discard:
        drawn = parse_request(request)[1] if request.startswith("2 ") else None
        hand_tiles = list(hand.elements())
        after = remove_one_tile(hand_tiles, discard)
        counts = Counter(after)
        reg = regular_shanten(after)
        seven = seven_pairs_shanten(after)
        orphan = thirteen_orphans_shanten(after)
        minimum = min(reg, seven, orphan)
        features.extend(
            [
                "CAND %s" % discard,
                "REG_SHANTEN %s" % reg,
                "SEVEN_SHANTEN %s" % seven,
                "ORPHAN_SHANTEN %s" % orphan,
                "MIN_SHANTEN %s" % minimum,
                "PAIR_COUNT %s" % sum(1 for count in counts.values() if count >= 2),
                "TRIPLE_COUNT %s" % sum(1 for count in counts.values() if count >= 3),
                " ".join(
                    "CNT_%s_%s" % (tile, counts[tile])
                    for tile in sorted(counts)
                    if counts[tile]
                ),
            ]
        )
        if drawn is not None:
            candidate_min = min_shanten(remove_one_tile(hand_tiles, discard))
            drawn_min = min_shanten(remove_one_tile(hand_tiles, drawn))
            features.extend(
                [
                    "DRAWN_TILE %s" % drawn,
                    "CAND_IS_DRAWN %s" % (1 if discard == drawn else 0),
                    "DRAWN_MIN_SHANTEN %s" % drawn_min,
                    "SHANTEN_DELTA_VS_DRAWN %s" % (candidate_min - drawn_min),
                ]
            )
    else:
        features.append("HAND_MIN_SHANTEN %s" % min_shanten(hand.elements()))
    return "\n".join(features)


def apply_response(hand, request, response):
    tokens = parse_request(request)
    parts = response.strip().split()
    if not tokens or not parts:
        return
    action = parts[0].upper()
    if tokens[0] == "2" and len(tokens) >= 2:
        if action == "PLAY" and len(parts) == 2:
            hand[parts[1]] -= 1
        elif action == "GANG" and len(parts) == 2:
            hand[parts[1]] -= 4
        elif action == "BUGANG" and len(parts) == 2:
            hand[parts[1]] -= 1
    elif tokens[0] == "3":
        event_tile = request_event_tile(tokens)
        if action == "PENG" and event_tile and len(parts) == 2:
            hand[event_tile] -= 2
            hand[parts[1]] -= 1
        elif action == "GANG" and event_tile:
            hand[event_tile] -= 3
        elif action == "CHI" and event_tile and len(parts) == 3:
            middle = parts[1]
            sequence = [
                "%s%d" % (middle[0], int(middle[1]) - 1),
                middle,
                "%s%d" % (middle[0], int(middle[1]) + 1),
            ]
            needed = Counter(sequence)
            needed[event_tile] -= 1
            for tile, count in needed.items():
                hand[tile] -= count
            hand[parts[2]] -= 1
    cleanup(hand)


class PurePolicy(object):
    def __init__(self, model_data):
        self.model_data = model_data
        self.prefer_hu = bool(model_data.get("prefer_hu", False))
        self.history = []
        self.hand = Counter()
        self.player_id = None
        self.quan = 0
        self.flower_counts = [0, 0, 0, 0]
        self.wall_counts = [23, 23, 23, 23]

    def respond(self, request):
        tokens = request.strip().split()
        if not tokens:
            response = "PASS"
        elif tokens[0] == "0" and len(tokens) >= 2:
            self.player_id = int(tokens[1])
            if len(tokens) >= 3:
                self.quan = int(tokens[2])
            response = "PASS"
        elif tokens[0] == "1":
            self.load_initial_hand(tokens)
            response = "PASS"
        elif tokens[0] == "2" and len(tokens) >= 2:
            if self.player_id is not None:
                self.record_wall_draw(self.player_id)
            self.hand[tokens[1]] += 1
            response = self.choose_response(request)
            apply_response(self.hand, request, response)
        elif tokens[0] == "3":
            self.record_public_event(tokens)
            response = self.choose_response(request)
            apply_response(self.hand, request, response)
        else:
            response = "PASS"
        self.history.append("REQ %s" % request)
        self.history.append("RES %s" % response)
        return response

    def load_initial_hand(self, tokens):
        self.hand.clear()
        if len(tokens) >= 5:
            self.flower_counts = [int(value) for value in tokens[1:5]]
            self.wall_counts = [21 - count for count in self.flower_counts]
        for tile in split_tiles(tokens[5:18]):
            self.hand[tile] += 1

    def record_wall_draw(self, player):
        if 0 <= player < 4:
            self.wall_counts[player] -= 1

    def record_public_event(self, tokens):
        if len(tokens) < 3:
            return
        try:
            actor = int(tokens[1])
        except ValueError:
            return
        if tokens[2] == "DRAW":
            self.record_wall_draw(actor)
        elif tokens[2] == "BUHUA":
            self.record_wall_draw(actor)
            if 0 <= actor < 4:
                self.flower_counts[actor] += 1

    def input_text(self, request):
        return "\n".join(self.history + ["REQ %s" % request])

    def legal_responses(self, request):
        responses = generate_legal_responses(self.player_id, request, self.hand)
        return [
            response
            for response in responses
            if self.passes_wall_legality(request, response)
            and self.passes_basic_hu_legality(request, response)
        ]

    def passes_wall_legality(self, request, response):
        action = response.split()[0].upper() if response else "PASS"
        if action not in ("GANG", "BUGANG", "PENG", "CHI"):
            return True
        tokens = request.strip().split()
        if not tokens:
            return True
        if tokens[0] == "2":
            if action not in ("GANG", "BUGANG") or self.player_id is None:
                return True
            return self.wall_counts[self.player_id] > 0 and self.wall_counts[
                (self.player_id + 1) % 4
            ] > 0
        if tokens[0] == "3" and len(tokens) >= 2:
            try:
                actor = int(tokens[1])
            except ValueError:
                return True
            return self.wall_counts[(actor + 1) % 4] > 0
        return True

    def passes_basic_hu_legality(self, request, response):
        action = response.split()[0].upper() if response else "PASS"
        if action != "HU":
            return True
        return conservative_can_hu(self.hand, request, is_last=self.is_last_tile_context(request))

    def is_last_tile_context(self, request):
        tokens = request.strip().split()
        if not tokens:
            return False
        actor = self.player_id
        if tokens[0] == "3" and len(tokens) >= 2:
            try:
                actor = int(tokens[1])
            except ValueError:
                actor = self.player_id
        if actor is None:
            return False
        return self.wall_counts[(actor + 1) % 4] <= 0

    def choose_response(self, request):
        candidates = self.legal_responses(request)
        if not candidates:
            return "PASS"
        if self.prefer_hu and "HU" in candidates:
            return "HU"
        input_text = self.input_text(request)
        if request.startswith("2 "):
            return self.choose_draw(input_text, request, candidates)
        return self.choose_reaction(input_text, request, candidates)

    def choose_draw(self, input_text, request, candidates):
        weights = self.model_data["draw_weights"]
        scores = [0.0] * len(candidates)
        total = sum(float(weight) for weight in weights) or 1.0
        scoring_request = "2 _"
        for model_data, weight in zip(self.model_data["draw_models"], weights):
            for index, response in enumerate(candidates):
                features = featurize_response(
                    input_text, scoring_request, response, self.hand, legal_count=len(candidates)
                )
                scores[index] += (float(weight) / total) * score_hgb_export(
                    model_data, features
                )
        return candidates[max(range(len(candidates)), key=lambda index: scores[index])]

    def choose_reaction(self, input_text, request, candidates):
        reaction = self.model_data["reaction_model"]
        scores = [
            score_tfidf_sgd_export(
                reaction, response_candidate_text(input_text, response, self.hand, request)
            )
            for response in candidates
        ]
        return candidates[max(range(len(candidates)), key=lambda index: scores[index])]


def respond_payload(payload, model_data):
    requests = [str(item) for item in payload.get("requests", [])]
    responses = [str(item) for item in payload.get("responses", [])]
    policy = PurePolicy(model_data)
    for request, _expected_response in zip(requests[:-1], responses):
        policy.respond(request)
    if not requests:
        return "PASS"
    return policy.respond(requests[-1])


def main_with_model(model_data):
    payload = json.loads(sys.stdin.read() or "{}")
    print(json.dumps({"response": respond_payload(payload, model_data)}))
    return 0
