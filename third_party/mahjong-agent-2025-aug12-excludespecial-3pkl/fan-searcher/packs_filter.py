import pickle
import itertools
import os
import numpy as np
from collections import defaultdict
from MahjongGB import MahjongFanCalculator
from tqdm import tqdm

with open("packs.pickle", "rb") as fs:
    data = pickle.load(fs)


new_data = {}
for name, packs_list in data.items():
    new_packs_list = set()
    for packs in packs_list:
        tile_count = defaultdict(int)
        for pack in packs:
            for tile in pack:
                tile_count[tile] += 1
        for count in tile_count.values():
            if count > 4:
                break
        else:
            new_packs_list.add(tuple(sorted(packs)))
    print(name, len(packs_list), "-->", len(new_packs_list))
    new_data[name] = [list(i) for i in sorted(new_packs_list)]


# 对无番和做单独处理。对于给定的4组面子和1组对子，可以过滤掉以下情况：
# 老少副、连六、喜相逢、一般高、双同刻、四归一、全带幺

def 一般高_双同刻(packs):
    return len(set(packs)) != len(packs)

def 连六(packs):
    for p1, p2 in itertools.permutations(packs, 2):
        if p1[0][0] == p2[0][0] \
            and int(p1[0][1]) - int(p2[0][1]) == 3 \
            and int(p1[1][1]) - int(p2[1][1]) == 3 \
            and int(p1[2][1]) - int(p2[2][1]) == 3:
                return True
    return False

def 老少副(packs):
    for p1, p2 in itertools.permutations(packs, 2):
        if p1[0][0] == p2[0][0] \
            and int(p1[0][1]) == 1 and int(p1[1][1]) == 2 and int(p1[2][1]) == 3 \
            and int(p2[0][1]) == 7 and int(p2[1][1]) == 8 and int(p2[2][1]) == 9:
                return True
    return False

def 喜相逢(packs):
    for p1, p2 in itertools.permutations(packs, 2):
        if p1[0][0] == p2[0][0] \
            and int(p1[0][1]) - int(p2[0][1]) == 0 \
            and int(p1[1][1]) - int(p2[1][1]) == 0 \
            and int(p1[2][1]) - int(p2[2][1]) == 0:
                return True
    return False

def 四归一(packs):
    for p1, p2 in itertools.permutations(packs, 2):
        if len(set(p1)) == 1 and p1[0] in p2:
            return True
    return False

YAOJIU_TILE_LIST = "W1 W9 B1 B9 T1 T9 F1 F2 F3 F4 J1 J2 J3".split(" ")

def 全带幺(packs):
    for pack in packs:
        yaojiu = False
        for tile in pack:
            if tile in YAOJIU_TILE_LIST:
                yaojiu = True
        if not yaojiu:
            return False
    return True

def 碰碰和(packs):
    for pack in packs:
        if not len(set(pack)) == 1:  # 对子、刻子一定是1，顺子一定是3，不为1就是有顺子
            return False
    return True

def 平和(packs):
    for pack in packs:
        if not len(set(pack)) == 3 and not len(pack) == 3:  # 对子、刻子一定是1，顺子一定是3，不为3且长度不为3的一定是刻子
            return False
    return True


def 兜底(packs):
    pack = []
    hand = ()
    winTile = ""
    
    for p in packs:
        if len(p) == 2:  # 对子
            hand += p
        elif len(p) == 3 and p[0] == p[1] == p[2]:  # 刻子
            pack.append(("PENG", p[0], 1))
        else:  # 顺子
            if winTile:  # 已经有获胜的张
                hand += p
            else:  # 还没有获胜的张，从顺子里选一张出来
                winTile = p[0]
                h = (p[1], p[2])
                if winTile[-1] == "3" or winTile[-1] == "7":  # 这个顺子有概率为123或789
                    winTile = p[2]
                    h = (p[0], p[1])
                hand += h

    fans = MahjongFanCalculator(
        pack=tuple(pack),
        hand=hand,
        winTile=winTile,
        flowerCount=0,
        isSelfDrawn=False,
        is4thTile=False,
        isAboutKong=False,
        isWallLast=False,
        seatWind=3,
        prevalentWind=3
    )

    if (8, '无番和') in fans:
        return False
    return True


filter_functions = [
    # 一般高_双同刻,
    # 连六,
    # 老少副,
    # 喜相逢,
    # 四归一,
    # 全带幺,
    # 碰碰和,
    # 平和,
    兜底,
]

packs_list = new_data["无番和"]
new_packs_list = packs_list

for function in filter_functions:
    print("无番和 filter by", function.__name__)
    new_packs_list = [i for i in tqdm(new_packs_list) if not function(i)]

print("无番和", len(packs_list), "-->", len(new_packs_list))
new_data["无番和"] = new_packs_list

count = sum(len(i) for i in new_data.values())

print("总共有", len(new_data), "种主流大番种")

for name, data in new_data.items():
    print(name, len(data))

print("总共有", count, "种牌型")

with open("packs_filter.pickle", "wb") as fs:
    pickle.dump(new_data, fs)

with open("packs_filter.pickle", "rb") as fs:
    new_data = pickle.load(fs)
    count = sum(len(i) for i in new_data.values())


FAN = {
    '大四喜': 88,
    '大三元': 88,
    '绿一色': 88,
    '九莲宝灯': 88,
    '四杠': 88,
    '连七对': 88,
    '十三幺': 88,
    '清幺九': 64,
    '小四喜': 64,
    '小三元': 64,
    '字一色': 64,
    '四暗刻': 64,
    '一色双龙会': 64,
    '一色四同顺': 48,
    '一色四节高': 48,
    '一色四步高': 32,
    '三杠': 32,
    '混幺九': 32,
    '七对': 24,
    '七星不靠': 24,
    '组合不靠': 24,
    '全双刻': 24,
    '清一色': 24,
    '一色三同顺': 24,
    '一色三节高': 24,
    '全大': 24,
    '全中': 24,
    '全小': 24,
    '清龙': 16,
    '三色双龙会': 16,
    '一色三步高': 16,
    '全带五': 16,
    '三同刻': 16,
    '三暗刻': 16,
    '全不靠': 12,
    '组合龙': 12,
    '大于五': 12,
    '小于五': 12,
    '三风刻': 12,
    '花龙': 8,
    '推不倒': 8,
    '三色三同顺': 8,
    '三色三节高': 8,
    '无番和': 8,
    '妙手回春': 8,
    '海底捞月': 8,
    '杠上开花': 8,
    '抢杠和': 8,
    '碰碰和': 6,
    '混一色': 6,
    '三色三步高': 6,
    '五门齐': 6,
    '全求人': 6,
    '双暗杠': 6,
    '双箭刻': 6,
    '明暗杠': 5,
    '全带幺': 4,
    '不求人': 4,
    '双明杠': 4,
    '和绝张': 4,
    '箭刻': 2,
    '圈风刻': 2,
    '门风刻': 2,
    '门前清': 2,
    '平和': 2,
    '四归一': 2,
    '双同刻': 2,
    '双暗刻': 2,
    '暗杠': 2,
    '断幺': 2,
    '一般高': 1,
    '喜相逢': 1,
    '连六': 1,
    '老少副': 1,
    '幺九刻': 1,
    '明杠': 1,
    '缺一门': 1,
    '无字': 1,
    '边张': 1,
    '坎张': 1,
    '单钓将': 1,
    '自摸': 1,
    '花牌': 1,
    '荒庄': 0
 }
FAN_NAMES = list(FAN.keys())
# 牌名列表
TILE_LIST = [
    *("W%d" % (i + 1) for i in range(9)),  # 万
    *("T%d" % (i + 1) for i in range(9)),  # 条
    *("B%d" % (i + 1) for i in range(9)),  # 饼
    *("F%d" % (i + 1) for i in range(4)),  # 风
    *("J%d" % (i + 1) for i in range(3)),  # 箭
]

# 牌名 -> 索引 的一个字典，这个索引的范围是 OBS 的第二个维度 36（实际上最后有空余）
OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}
OFFSET_TILE["CONCEALED"] = 34

# 番种名称id + 番种牌数量 + 每种牌各多少牌 + 听牌类型（基本型3*4+2/七对型2*7/一组型14） + 至多14张牌

class OffsetPoint:
    NAME_ID = 0  # +1
    HAND_COUNT = 1  # +1
    TILE_COUNT = 2  # +34
    PACK_TYPE = 36  # +1
    TILES = 37  # +14

    LENGTH = 51


class PackType:
    NONE = 0
    TOTAL = 1  # 只有一组牌，整体型
    BASIC = 2  # 有2~5组牌，基本型
    BASIC_WITH_PAIR = 3  # 有2~5组牌，基本型
    QIDUI = 4  # 有7组牌，七对型


def save_data_to_npz(hupai_data, npz_path):
    count = sum(len(i) for i in hupai_data.values())
    print("=" * 25)
    print(f"{npz_path} 总共保存 {count} 个牌型/胡牌组合")
    hand_values = np.full((count, OffsetPoint.LENGTH), 255, dtype=np.uint8)
    i = 0
    for name, data in hupai_data.items():
        for packs in data:
            hand = sum(packs, ())
            hand = sorted(hand)

            hand_values[i, OffsetPoint.NAME_ID] = FAN_NAMES.index(name)
            hand_values[i, OffsetPoint.HAND_COUNT] = len(hand)

            for tile in set(hand):
                hand_values[i, OffsetPoint.TILE_COUNT + OFFSET_TILE[tile]] = hand.count(tile)
            
            if len(packs) == 1 and len(packs[0]) >= 9:  # 只有一组牌，且特别大，属于整体类
                hand_values[i, OffsetPoint.PACK_TYPE] = PackType.TOTAL
            elif 2 <= len(packs) <= 5:  # 有至少2组牌，属于基本胡型
                hand_values[i, OffsetPoint.PACK_TYPE] = PackType.BASIC
                if any(len(pack) == 2 for pack in packs):
                    hand_values[i, OffsetPoint.PACK_TYPE] = PackType.BASIC_WITH_PAIR
            elif len(packs) == 7:
                hand_values[i, OffsetPoint.PACK_TYPE] = PackType.QIDUI
            else:
                hand_values[i, OffsetPoint.PACK_TYPE] = PackType.NONE
                print(name, hand, len(packs))
                raise RuntimeError("未知的牌型")
            idx = 0
            for pack in sorted(packs, key=lambda x:len(x)):  # 整体型和七对型不受影响，基本型固定为2+3+3+3+3
                for item in pack:
                    hand_values[i, OffsetPoint.TILES + idx] = TILE_LIST.index(item)
                    idx += 1
            i += 1

    np.savez(npz_path, array=hand_values)


paixing_fan_names = [
    '三色三步高', '三色三同顺', '花龙', '清龙', '一色三步高', '组合龙', '双箭刻', '三色三节高', '一色三节高', '小三元', '三同刻',
    '一色四步高', '三风刻', '一色三同顺', '大三元', '小四喜', '一色四节高', '大四喜', '一色四同顺']
hupaizuhe_fan_names = [
    '三色双龙会', '一色双龙会', '连七对', '五门齐', '混一色', '全带幺', '清一色', '无番和', '大于五', '小于五', '全不靠',
    '七星不靠', '组合不靠', '推不倒', '全带五', '全大', '全小', '全中', '混幺九', '全双刻', '十三幺', '字一色', '绿一色', '清幺九']

os.makedirs("data", exist_ok=True)
save_data_to_npz({key: new_data[key] for key in paixing_fan_names}, "./data/paixing.npz")
save_data_to_npz({key: new_data[key] for key in hupaizuhe_fan_names}, "./data/hupaizuhe.npz")

important_fan_names = ['五门齐', '混一色', '全带幺', '清一色', '无番和', '大于五', '小于五', '推不倒']
other_fan_names = [i for i in hupaizuhe_fan_names if i not in important_fan_names]

for fan_name in important_fan_names:
    save_data_to_npz({fan_name: new_data[fan_name]}, f'./data/{fan_name}.npz')
save_data_to_npz({fan_name: new_data[fan_name] for fan_name in other_fan_names}, './data/other_hupaizuhe.npz')

print(FAN_NAMES)


