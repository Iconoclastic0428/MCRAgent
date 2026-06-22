#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   preprocess.py
# @Time    :   2024/06/30 18:34:35
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
preprocess.py: 数据预处理，对于每一局数据，建四个FeatureAgent（作为四个玩家视角），将对局中的事件传给相应的agent，整理出每个人每个决策点下的特征表示以及实际选择的动作，保存到data文件夹下。
"""

import json
import os
import re
import time
import traceback

import numpy as np
from feature import FeatureAgent


def filterData():
    global obs
    global actions
    newobs = [[] for i in range(4)]
    newactions = [[] for i in range(4)]
    for i in range(4):
        if i not in train_players:
            continue
        for j, o in enumerate(obs[i]):
            if (
                o["action_mask"].sum() > 1
            ):  # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    obs = newobs
    actions = newactions
    
    
save_count = 0
start_time = time.time()


def saveData():
    global save_count
    assert [len(x) for x in obs] == [
        len(x) for x in actions
    ], "obs actions not matching!"
    selected_obs = [x for i in range(4) if i in train_players for x in obs[i]]
    selected_actions = [x for i in range(4) if i in train_players for x in actions[i]]
    sample_count = len(selected_obs)
    l.append(sample_count)
    if sample_count:
        obs_array = np.stack([x["observation"] for x in selected_obs]).astype(np.int8)
        mask_array = np.stack([x["action_mask"] for x in selected_obs]).astype(np.int8)
        vec_array = np.stack([x["vec"] for x in selected_obs]).astype(np.float16)
        act_array = np.array(selected_actions)
    else:
        obs_array = np.zeros((0, FeatureAgent.OBS_SIZE, 4, 9), dtype=np.int8)
        mask_array = np.zeros((0, FeatureAgent.ACT_SIZE), dtype=np.int8)
        vec_array = np.zeros((0, FeatureAgent.VEC_SIZE), dtype=np.float16)
        act_array = np.zeros((0,), dtype=np.int64)
    np.savez(
        "%s/%d.npz" % (output_dir, matchid),
        obs=obs_array,
        mask=mask_array,
        vec=vec_array,
        act=act_array,
    )
    
    save_count += 1
    cost_time = time.time() - start_time
    print(f"{round(100 * save_count / match_per_process, 2)}% ({save_count}/{match_per_process}) time: {cost_time:.3f}s    ", end="\r")
    
    for x in obs:
        x.clear()
    for x in actions:
        x.clear()


def saveEmptyData(error_info=None):
    global save_count
    l.append(0)
    np.savez(
        "%s/%d.npz" % (output_dir, matchid),
        obs=np.zeros((0, FeatureAgent.OBS_SIZE, 4, 9), dtype=np.int8),
        mask=np.zeros((0, FeatureAgent.ACT_SIZE), dtype=np.int8),
        vec=np.zeros((0, FeatureAgent.VEC_SIZE), dtype=np.float16),
        act=np.zeros((0,), dtype=np.int64),
    )
    if error_info is not None:
        preprocess_errors.append(error_info)

    save_count += 1
    cost_time = time.time() - start_time
    print(f"{round(100 * save_count / match_per_process, 2)}% ({save_count}/{match_per_process}) time: {cost_time:.3f}s    ", end="\r")

    for x in obs:
        x.clear()
    for x in actions:
        x.clear()


def process_match_lines(lines):
    global obs, actions, train_players
    obs = [[] for i in range(4)]
    actions = [[] for i in range(4)]
    train_players = set(range(4))
    agents = None
    curTile = None
    saved = False

    for line in lines:
        t = line.split()
        if len(t) == 0:
            continue
        if t[0] == "Match":
            agents = [FeatureAgent(i) for i in range(4)]
            train_players = set(range(4))
        elif t[0] == "TrainPlayers":
            parsed_players = {int(value) for value in t[1:] if value.isdigit()}
            train_players = parsed_players or set(range(4))
        elif t[0] == "Wind":
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == "Player":
            p = int(t[1])
            if t[2] == "Deal":
                agents[p].request2obs(" ".join(t[2:]))
            elif t[2] == "Draw":
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs(" ".join(t[2:])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs(" ".join(t[:3]))
            elif t[2] == "Play":
                actions[p].pop()
                actions[p].append(agents[p].response2action(" ".join(t[2:])))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(line)
                    else:
                        obs[i].append(agents[i].request2obs(line))
                        actions[i].append(0)
                curTile = t[3]
            elif t[2] == "Chi":
                actions[p].pop()
                actions[p].append(
                    agents[p].response2action("Chi %s %s" % (curTile, t[3]))
                )
                for i in range(4):
                    if i == p:
                        obs[p].append(
                            agents[p].request2obs("Player %d Chi %s" % (p, t[3]))
                        )
                        actions[p].append(0)
                    else:
                        agents[i].request2obs("Player %d Chi %s" % (p, t[3]))
            elif t[2] == "Peng":
                actions[p].pop()
                actions[p].append(agents[p].response2action("Peng %s" % t[3]))
                for i in range(4):
                    if i == p:
                        obs[p].append(
                            agents[p].request2obs("Player %d Peng %s" % (p, t[3]))
                        )
                        actions[p].append(0)
                    else:
                        agents[i].request2obs("Player %d Peng %s" % (p, t[3]))
            elif t[2] == "Gang":
                actions[p].pop()
                actions[p].append(agents[p].response2action("Gang %s" % t[3]))
                for i in range(4):
                    agents[i].request2obs("Player %d Gang %s" % (p, t[3]))
            elif t[2] == "AnGang":
                actions[p].pop()
                actions[p].append(agents[p].response2action("AnGang %s" % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs("Player %d AnGang %s" % (p, t[3]))
                    else:
                        agents[i].request2obs("Player %d AnGang" % p)
            elif t[2] == "BuGang":
                actions[p].pop()
                actions[p].append(agents[p].response2action("BuGang %s" % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs("Player %d BuGang %s" % (p, t[3]))
                    else:
                        obs[i].append(
                            agents[i].request2obs("Player %d BuGang %s" % (p, t[3]))
                        )
                        actions[i].append(0)
            elif t[2] == "Hu":
                actions[p].pop()
                actions[p].append(agents[p].response2action("Hu"))
            # Deal with Ignore clause
            if t[2] in ["Peng", "Gang", "Hu"]:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == "Chi":
                            actions[p].pop()
                            actions[p].append(
                                agents[p].response2action(
                                    "Chi %s %s" % (curTile, t[k + 3])
                                )
                            )
                        elif t[k + 2] == "Peng":
                            actions[p].pop()
                            actions[p].append(
                                agents[p].response2action("Peng %s" % t[k + 3])
                            )
                        elif t[k + 2] == "Gang":
                            actions[p].pop()
                            actions[p].append(
                                agents[p].response2action("Gang %s" % t[k + 3])
                            )
                        elif t[k + 2] == "Hu":
                            actions[p].pop()
                            actions[p].append(agents[p].response2action("Hu"))
                    else:
                        break

        elif t[0] == "Score":
            filterData()
            saveData()
            saved = True
    if not saved:
        raise RuntimeError("match ended without Score")


def process_data(file_path, start_line, end_line, offset, cpu_id):
    global obs, actions, matchid, l, train_players, preprocess_errors
    obs = [[] for i in range(4)]
    actions = [[] for i in range(4)]
    train_players = set(range(4))
    matchid = offset - 1
    l = []
    preprocess_errors = []

    with open(file_path, encoding="utf8") as f:
        for _ in range(start_line):
            f.readline()  # Skip lines until start_line
        first_line = f.readline()
        assert first_line.startswith("Match"), "Not a match start line"
        line_number = start_line
        if end_line is None:
            end_line = 1e9
        block = []
        block_start_line = line_number
        line = first_line
        while line and line_number <= end_line:
            if line.startswith("Match") and block:
                matchid += 1
                try:
                    process_match_lines(block)
                except Exception as exc:
                    saveEmptyData(
                        {
                            "match_id": matchid,
                            "start_line": block_start_line,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=6),
                        }
                    )
                block = [line]
                block_start_line = line_number
            else:
                block.append(line)
            line = f.readline()
            line_number += 1
        if block:
            matchid += 1
            try:
                process_match_lines(block)
            except Exception as exc:
                saveEmptyData(
                    {
                        "match_id": matchid,
                        "start_line": block_start_line,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=6),
                    }
                )
    with open(f"{output_dir}/count-{cpu_id}.json", "w", encoding="utf8") as f:
        json.dump(l, f)
    with open(f"{output_dir}/errors-{cpu_id}.json", "w", encoding="utf8") as f:
        json.dump(preprocess_errors, f, ensure_ascii=False, indent=2)


output_dir = "data-output"

# python preprocess.py {file_path} {start_line} {end_line} {offset} {cpu_id}
if __name__ == "__main__":
    import sys

    file_path = sys.argv[1]
    start_line = int(sys.argv[2])
    if sys.argv[3] == "None":
        end_line = None
    else:
        end_line = int(sys.argv[3])
    offset = int(sys.argv[4])
    cpu_id = int(sys.argv[5])
    output_dir = sys.argv[6]
    match_per_process = int(sys.argv[7])
    process_data(file_path, start_line, end_line, offset, cpu_id)
