#include "jsoncpp/json.h"
#include "MahjongGB/fan_calculator.cpp"

#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

static unordered_map<string, mahjong::tile_t> str2tile;

static void init_tiles() {
    for (int i = 1; i <= 9; i++) {
        str2tile["W" + to_string(i)] = mahjong::make_tile(TILE_SUIT_CHARACTERS, i);
        str2tile["B" + to_string(i)] = mahjong::make_tile(TILE_SUIT_BAMBOO, i);
        str2tile["T" + to_string(i)] = mahjong::make_tile(TILE_SUIT_DOTS, i);
    }
    for (int i = 1; i <= 4; i++) {
        str2tile["F" + to_string(i)] = mahjong::make_tile(TILE_SUIT_HONORS, i);
    }
    for (int i = 1; i <= 3; i++) {
        str2tile["J" + to_string(i)] = mahjong::make_tile(TILE_SUIT_HONORS, i + 4);
    }
}

static mahjong::tile_t parse_tile(const Json::Value &value) {
    string tile = value.asString();
    auto it = str2tile.find(tile);
    if (it == str2tile.end()) {
        return 0;
    }
    return it->second;
}

int main() {
    init_tiles();
    Json::Value input;
    cin >> input;

    int player = input.get("player", 0).asInt();
    mahjong::calculate_param_t calculate_param;
    mahjong::fan_table_t fan_table;
    memset(&calculate_param, 0, sizeof(mahjong::calculate_param_t));
    memset(&fan_table, 0, sizeof(mahjong::fan_table_t));

    const Json::Value &hand = input["hand"];
    calculate_param.hand_tiles.tile_count = (int) hand.size();
    for (Json::ArrayIndex i = 0; i < hand.size(); i++) {
        calculate_param.hand_tiles.standing_tiles[i] = parse_tile(hand[i]);
    }

    const Json::Value &packs = input["packs"];
    calculate_param.hand_tiles.pack_count = (int) packs.size();
    for (Json::ArrayIndex i = 0; i < packs.size(); i++) {
        string type = packs[i]["type"].asString();
        string tile = packs[i]["tile"].asString();
        int offer = packs[i]["offer"].asInt();
        mahjong::pack_t &pack = calculate_param.hand_tiles.fixed_packs[i];
        if (type == "PENG") {
            pack = mahjong::make_pack((offer - player + 4) % 4, PACK_TYPE_PUNG, str2tile[tile]);
        } else if (type == "GANG") {
            pack = mahjong::make_pack((offer - player + 4) % 4, PACK_TYPE_KONG, str2tile[tile]);
        } else if (type == "CHI") {
            pack = mahjong::make_pack(offer + 1, PACK_TYPE_CHOW, str2tile[tile]);
        }
    }

    calculate_param.win_tile = parse_tile(input["win_tile"]);
    calculate_param.flower_count = (uint8_t) input.get("flower_count", 0).asInt();
    if (input.get("is_self_draw", false).asBool()) {
        calculate_param.win_flag |= WIN_FLAG_SELF_DRAWN;
    }
    if (input.get("is_4th_tile", false).asBool()) {
        calculate_param.win_flag |= WIN_FLAG_4TH_TILE;
    }
    if (input.get("is_about_kong", false).asBool()) {
        calculate_param.win_flag |= WIN_FLAG_ABOUT_KONG;
    }
    if (input.get("is_last", false).asBool()) {
        calculate_param.win_flag |= WIN_FLAG_WALL_LAST;
    }
    calculate_param.seat_wind = (mahjong::wind_t) input.get("seat_wind", player).asInt();
    calculate_param.prevalent_wind = (mahjong::wind_t) input.get("prevalent_wind", 0).asInt();

    int fan = mahjong::calculate_fan(&calculate_param, &fan_table);
    Json::Value output;
    output["fan"] = fan;
    output["can_hu"] = fan >= 8;
    cout << output;
    return 0;
}
