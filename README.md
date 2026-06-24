# MCRAgent River-Elastic Feature Agent

## 1. 背景

该模型面向中国竞技麻将（MCR, Chinese Official Mahjong）Botzone 2026 比赛环境。训练数据主要来自 Botzone 2026 官方页面提供的强 AI 对局数据集，同时保留了一套个人清洗的雀渣平台前 80 名玩家对局记录数据集，后者用于补充训练或人工对局水平校验。

该模型的核心设计是：使用 185 个 `OBS` 平面描述牌局可见状态，使用 117 维 `VEC` 表示局面统计与向听相关特征，经 ResNet 主干、混合卷积输入层、Dueling Head 输出 235 维动作 logits，再通过合法动作 mask 将非法动作屏蔽。

## 2. 数据来源与预处理

### 2.1 数据来源

训练数据包含两部分：

| 数据源 | 用途 | 说明 |
|---|---:|---|
| Botzone 2026 官方强 AI 数据集 | 主训练数据 | 仓库 promoted checkpoint 记录为 `botzone98209`，即 98,209 局 Botzone 数据 |
| 雀渣平台前 80 名玩家对局记录 | 辅助数据 | 由个人清洗得到，可用于补充训练、对比实验或人工风格校验 |

Botzone 2026 比赛采用中国竞技麻将规则，并提供样例 bot、样例裁判程序、FanCalculator 计番库和强 AI 对局数据。比赛阶段使用复式赛制，因此本项目的评测也采用座位轮换与复式对局统计。

### 2.2 NPZ 数据格式

`feature-agent` 训练路径期望数据目录包含：

```text
count.json
0.npz
1.npz
...
```

每个 `.npz` 文件对应一局或一个 match 的预处理样本，核心字段为：

```text
obs   : observation tensor
mask  : legal action mask
vec   : vector features
act   : supervised target action
```

训练脚本通过 `count.json` 读取每局样本数，并按 match 做连续切分，而不是按单条样本随机切分。默认训练/验证切分为 90/10，`test_ratio=0` 时不额外拆分测试集。

### 2.3 12 倍数据增强

麻将中的万、条、饼三门花色在大多数结构中可互换，数牌也可以围绕中心对称。因此训练集使用 `all12` 增强：

```text
6 种花色置换 × 2 种数字镜像 = 12 倍增强
```

数字镜像时需要同步替换与大小相关的番种特征名，例如：

```text
dayu5  <-> xiaoyu5
quanda <-> quanxiao
大于五 <-> 小于五
全大   <-> 全小
```

对绿一色与推不倒相关对局，当前代码支持两种策略：

1. 保留原局，不做 12 倍增强；
2. 通过 `--exclude-special-matches` 在训练中排除这些特殊局。

当前模型的训练条件下，我们保留这些特殊局但不增强；`RIVER_ELASTIC_NOTES.md` 中的示例命令则展示了可选的排除参数。复现实验时必须记录实际使用的参数，否则特殊局处理会改变有效训练分布。最后实验结果也证明模型仍然可以做出主番的推不倒和绿一色。

## 3. 特征表示

## 3.1 牌索引

内部牌序为 34 张有效牌加两个保留位置，reshape 为 `4 × 9`：

```text
W1-W9: 万子
T1-T9: 条子
B1-B9: 饼子
F1-F4: 东南西北
J1-J3: 中发白
CONCEALED: 34
```

实际神经网络输入为 `OBS_SIZE × 4 × 9`。当前模型使用 `OBS_SIZE = 185`。

## 3.2 OBS 特征演化

### HaibaraAi baseline

原始 HaibaraAi feature-agent 使用 60 个局面平面：

| Offset | 宽度 | 含义 |
|---:|---:|---|
| `KE = 0` | 4 | 四个玩家的刻子副露 |
| `SHUN = 4` | 4 | 四个玩家的顺子副露 |
| `GANG = 8` | 4 | 四个玩家的明杠副露 |
| `ANGANG = 12` | 4 | 四个玩家的暗杠 |
| `PLAY = 16` | 4 | 四个玩家的出牌信息 |
| `LAST = 20` | 1 | 最后一张牌 |
| `UNKNOWN = 21` | 1 | 未知牌数量 |
| `HAND = 22` | 1 | 当前玩家手牌 |
| `PREVALENT_WIND = 23` | 1 | 场风 |
| `SEAT_WIND = 24` | 1 | 自风 |
| `SHANTEN_PASS = 25` | 1 | PASS 后上听或胡牌所需牌 |
| `SHANTEN_PLAY = 26` | 34 | PLAY 后上听或胡牌所需牌 |

### Vec-fix 85 平面

`GoddessLuBoYan/mahjong-agent-2025` 在原始 60 平面后添加 25 个静态牌属性平面，使 CNN 不必完全从位置模式中自行拟合点数、牌型与特殊牌集合：

| Offset | 宽度 | 含义 |
|---:|---:|---|
| `DIANSHU = 60` | 16 | 点数，数牌 1-9，字牌映射到 10-16 |
| `LEIXING = 76` | 5 | 万、条、饼、风、箭 |
| `IS_YAOJIU = 81` | 1 | 是否幺九牌 |
| `IS_ZIPAI = 82` | 1 | 是否字牌 |
| `IS_TUIBUDAO = 83` | 1 | 是否推不倒相关牌 |
| `IS_LVYISE = 84` | 1 | 是否绿一色相关牌 |

### River-elastic 185 平面

本项目在 85 平面基础上额外加入牌河属性平面：

```text
RIVER_PROPERTY = 85
TILE_PROPERTY_SIZE = 25
RIVER_PLAYER_COUNT = 4
新增平面数 = 25 × 4 = 100
最终 OBS_SIZE = 85 + 100 = 185
```

每个玩家牌河都有 25 个属性平面，其属性定义与 60-84 的 vec-fix 静态牌属性一致：

```text
16 维点数 one-hot
5 维牌型 one-hot
1 维幺九标记
1 维字牌标记
1 维推不倒标记
1 维绿一色标记
```

如果输入数据仍是旧的 85 平面 NPZ，`MahjongGBDataset` 会从 `PLAY` 平面 `16:20` 自动派生 100 个牌河属性平面。新的预处理路径则可以直接写出 185 平面 OBS。

## 3.3 VEC 特征

VEC 维度为 117。它主要记录手牌数量、未知牌数、风信息、牌山剩余、步数，以及 PASS/PLAY 后的和牌概率与向听期望。

| Offset | 宽度 | 含义 |
|---:|---:|---|
| `PLAYERS_HAND = 0` | 4 | 四个玩家的手牌数量 |
| `UNKNOWN = 4` | 1 | 未知牌数量 |
| `PREVALENT_WIND = 5` | 1 | 场风 |
| `SEAT_WIND = 6` | 1 | 自风 |
| `REST = 7` | 4 | 四个玩家的牌山剩余数量 |
| `STEP = 11` | 1 | 当前步数 |
| `HU_PROB_PASS = 12` | 1 | 听牌时 PASS 后的和牌概率 |
| `HU_PROB_PLAY = 13` | 34 | 听牌时 PLAY 后的和牌概率 |
| `SHAN_EXP_PASS = 47` | 1 | PASS 后摸下一张牌的向听期望 |
| `SHAN_EXP_PLAY = 48` | 34 | PLAY 后摸下一张牌的向听期望 |
| `SHAN_DIS_PASS = 82` | 1 | PASS 后当前牌型的最小向听距离 |
| `SHAN_DIS_PLAY = 83` | 34 | PLAY 后当前牌型的最小向听距离 |

向听期望定义为：

```text
向听期望 = 最小向听距离 - 下一巡摸到上听牌的期望
```

其中最小向听距离原始实现取以下五种向听函数的最小值：

```text
一般型
七对
十三幺
组合龙
全不靠
```

仓库当前 `feature.py` 中直接导入了 `RegularShanten`、`SevenPairsShanten`、`ThirteenOrphansShanten`、`KnittedStraightShanten` 和 `HonorsAndKnittedTilesShanten`。这对应上述五类基础向听结构。

### 关于 fan feature 与全番种向听

当前训练脚本提供 `--fan-features-folder` 扩展口，可以使用全番种组合的向听版本，并在数字镜像增强时同步交换大小类番种特征。

## 3.4 动作空间

输出动作空间为 235 维：

| Offset | 宽度 | 动作 |
|---:|---:|---|
| `Pass = 0` | 1 | 过 |
| `Hu = 1` | 1 | 胡 |
| `Play = 2` | 34 | 打牌 |
| `Gang = 36` | 34 | 明杠 |
| `AnGang = 70` | 34 | 暗杠 |
| `BuGang = 104` | 34 | 补杠 |
| `Peng = 138` | 34 | 碰 |
| `Chi = 172` | 63 | 吃 |

吃牌动作共 63 维：

```text
3 门花色 × 7 个中心位置（2-8）× 3 个被吃牌位置 = 63
```

网络输出 logits 后，使用合法动作 mask 屏蔽非法动作：

```python
inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
logits = raw_logits + inf_mask
```

合法动作对应 mask 值为 1，log 后为 0；非法动作对应 mask 值为 0，log 后趋向负无穷，再被 clamp 到极小值。

## 4. 模型架构

## 4.1 主模型：SelfVecModel

当前 promoted checkpoint 使用 `SelfVecModel`，默认参数为：

```text
obs_dim = 185
vec_dim = 117
hidden = 128
num_blocks = 20
mixed_kernel_input = True
dueling_head = True
```

整体结构如下：

```text
OBS: 185 × 4 × 9
  -> MixedKernelInputLayer
  -> 128 × 4 × 9
  -> 20-layer ResNet trunk
  -> flatten: 128 × 4 × 9 = 4608
  -> concat VEC: 4608 + 117 = 4725
  -> Dueling Head
  -> logits: 235
  -> action mask
```

## 4.2 混合卷积输入层

输入层并行使用 `3 × 3` 与 `1 × 1` 卷积：

```text
branch 1: Conv2d(obs_dim, 128, kernel=3, padding=1) + BN
branch 2: Conv2d(obs_dim, 128, kernel=1, padding=0) + BN
merge: GELU + Conv2d(128, hidden, kernel=3, padding=1) + BN + GELU
```

设计动机：

```text
1 × 1 分支保留同一牌位置上的通道组合信息；
3 × 3 分支捕获牌面相邻关系；
二者相加后再通过 3 × 3 卷积融合。
```

该设计对麻将输入有明确结构意义，因为 `4 × 9` 的空间布局同时编码数牌顺序、花色块和字牌位置。

## 4.3 ResNet 主干

主干包含 20 个 residual block。每个 block 为：

```text
Conv2d(hidden, hidden, 3, padding=1, bias=False)
BatchNorm2d(hidden)
GELU
Conv2d(hidden, hidden, 3, padding=1, bias=False)
BatchNorm2d(hidden)
```

前向传播为：

```python
x = x + block(x)
x = GELU(x)
```

该结构保持 `128 × 4 × 9` 分辨率不变，使卷积主干集中学习局面局部组合、牌型邻接关系、副露/牌河/手牌之间的平面交互。

## 4.4 Dueling Head

Dueling Head 将输出分为 Value Stream 与 Advantage Stream：

```text
shared trunk:       4725 -> 1736 -> 632
value stream:       632 -> 1
advantage stream:   632 -> 235
Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
```

这使模型可以把当前局面整体价值和各动作相对优势分开表达。这一分解有利于处理多数合法动作都较差，但少数动作相对较优的局面，也能降低不同动作类型之间 logits 标度互相干扰的风险。

按 `obs_dim=185, vec_dim=117, hidden=128, num_blocks=20, mixed_kernel_input=True, dueling_head=True` 计算，该模型约有 15.75M 参数。

## 4.5 可选 SlideStyleModel

仓库中还保留了 `SlideStyleModel`。它同样使用输入层和 residual blocks，但在输出前通过 `1 × 1` 卷积投影到较少的 `out_planes`，再与压缩后的 vec 拼接：

```text
Conv projection: hidden -> slide_out_planes
vec adapter: vec_dim -> slide_vec_dim
head input: slide_out_planes × 4 × 9 + slide_vec_dim
```

该模型是可选实验结构，不是当前 promoted checkpoint 的主模型。

## 5. 推理阶段规则搜索与动作格式

## 5.1 Botzone JSON/text wrapper

导出的 bot 使用：

```text
scripts/feature_agent_checkpoint_json_bot.py
```

默认检查点为：

```text
models/feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a/16.pkl
```

可通过环境变量覆盖：

```powershell
$env:MCR_FEATURE_AGENT_DIR="D:\MCR_Agent\feature-agent"
$env:MCR_FEATURE_AGENT_CHECKPOINT="D:\MCR_Agent\models\feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a\16.pkl"
$env:MCR_FEATURE_AGENT_RIVER="1"
python D:\MCR_Agent\scripts\feature_agent_checkpoint_json_bot.py --protocol text
```

如果 `MCR_FEATURE_AGENT_RIVER=1`，wrapper 会以 `mixed_kernel_input=True` 和 `dueling_head=True` 初始化 `SelfVecModel`。

## 5.2 摸牌与响应动作

摸牌阶段，模型直接在当前 observation 上预测动作，并将内部动作格式转换为 Botzone 输出：

```text
Hu      -> HU
Play X  -> PLAY X
Gang X  -> GANG X
AnGang X -> GANG X
BuGang X -> BUGANG X
其他    -> PASS
```

## 5.3 吃碰后的二阶段决策

对他人打牌后的反应，模型先预测是否 `Hu`、`Peng`、`Chi`、`Gang` 或 `Pass`。若预测 `Peng` 或 `Chi`，runtime 会临时模拟己方吃碰后的局面，再调用模型预测吃碰后的打牌：

```text
反应模型预测 Peng/Chi
  -> request2obs(Player self Peng/Chi ...)
  -> 再预测一个 Play 动作作为吃碰后的出牌
  -> request2obs(Player self UnPeng/UnChi ...) 回滚临时状态
  -> 输出 PENG/CHI ... discard
```

这一操作是因为235 维动作空间本身只选择是否吃碰，但Botzone 实际响应却需要同时给出吃碰后的弃牌。

## 5.4 胡牌保护

本地 benchmark wrapper 会使用官方 judge 输出的 `canHu` 信息做胡牌保护：

```text
如果模型输出 HU，但 judge 显示 fan < 8，则强制改为 PASS 或打出摸到牌；
如果考虑花牌后 base_fan < 8，也会阻断 HU。
```

该保护只用于本地评测和防止非法胡牌，不改变训练标签本身。

## 6. 训练配置

当前 promoted checkpoint 的可复现训练入口为：

```powershell
cd feature-agent
python supervised.py `
  --data-folder <botzone98209_vec_dir> `
  --augment-mode all12 `
  --special-matches <lv_yi_se_tui_bu_dao_special_matches.json> `
  --split-ratio 0.9 `
  --test-ratio 0 `
  --batch-size 8192 `
  --lr 0.0005 `
  --mixed-kernel-input `
  --dueling-head
```

`RIVER_ELASTIC_NOTES.md` 中的完整 launch shape 还包括：

```sh
python supervised.py \
  --data-folder "${VEC_DIR}" \
  --version feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a \
  --log-root "${DATA_ROOT}/models/feature_agent_checkpoints" \
  --runs-root "${DATA_ROOT}/runs/feature_agent_tensorboard" \
  --epochs 30 \
  --batch-size 8192 \
  --lr 0.0005 \
  --split-ratio 0.9 \
  --device cuda \
  --data-parallel \
  --num-workers 0 \
  --no-augment \
  --augment-mode all12 \
  --special-matches "${SPECIAL_JSON}" \
  --exclude-special-matches \
  --dueling-head \
  --mixed-kernel-input
```

注意：`--no-augment` 与 `--augment-mode all12` 同时出现时，当前代码仍会把 `augment_mode` 传入 `MahjongGBDataset`；有效行为取决于 dataset 中 `augment_mode` 的逻辑，而非 `augment` 值本身。复现实验时建议直接检查 `dataset_summary` 输出中的 `augment_mode`、`train_effective_samples`、`special_match_count`、`train_excluded_special_matches` 等字段。

## 6.1 损失函数

训练目标为 masked logits 上的交叉熵，加 Elastic Net 正则：

```text
L_total = L_cross_entropy + λ1 × L1 + λ2 × L2
λ1 = 0.01 × lr
λ2 = 0.1 × lr
```

在 `lr = 0.0005` 时：

```text
λ1 = 5e-6
λ2 = 5e-5
```

优化器为：

```python
torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
```

因此 L2 正则只来自显式 Elastic Net，不来自 AdamW 的 decoupled weight decay。当前实现还跳过了一维参数，即 bias 和 BatchNorm 的 scale/shift 参数不进入 L1/L2 统计。

## 6.2 训练过程与日志

训练脚本默认：

```text
epochs = 30
batch_size = 8192
lr = 5e-4
seed = 6088
device = cuda
num_workers = 0
DataParallel 可选
```

训练过程记录：

```text
cross_entropy_loss
regularization_loss
l1_value
l2_value
train accuracy
validation accuracy
category accuracy by action
epoch timing
```

动作类别统计按照以下区间聚合：

```text
pass
hu
discard
kong
pong
chow
```

## 6.3 checkpoint 命名细节

训练脚本在每个 epoch 开始时保存一次：

```python
for epoch in range(args.epochs):
    torch.save(state_dict, f"{epoch}.pkl")
    train_one_epoch(...)
```

## 7. 验证集准确率与复式评测

该模型在验证集上最高可以达到约 0.90 accuracy。但实验发现，验证集 accuracy 与真实对局分数并不单调一致：accuracy 更高的 checkpoint 可能在复式对局中得分更低。

一个典型对比如下：

| model | race score | raw score | 和率 | 和牌巡目 | 放铳率 |
|---:|---:|---:|---:|---:|---:|
| `16.pkl` | 1882 | 12712 | 25.42 | 11.55 | 16.28 |
| `24.pkl` | 1820 | -12712 | 22.82 | 11.52 | 18.07 |

尽管 `24.pkl` 在验证集上的 accuracy 和 loss 更低，它在复式对局中的表现弱于 `16.pkl`。这说明监督学习的分类准确率只能衡量“是否复现数据集动作”，不能直接等价于“是否最大化 MCR 对局得分”。

实际选模时建议同时使用以下标准：

```text
1. validation accuracy 不能过低。经验上低于 0.88 的模型通常无法得到好结果；
2. validation loss 继续下降但复式分数下降时，应优先选复式分数更高的 checkpoint；
3. 分析和率、放铳率、听牌率、自摸率、不和牌时放铳率、放铳时听牌率；
4. 对比不同座位、不同 initdata、不同对手脚本下的稳定性；
5. 保留 action-category accuracy，避免模型只在 discard 多数类上变好。
```

## 8. 本地复式 benchmark

仓库提供：

```text
scripts/benchmark_json_policies.py
```

其评测方式是使用 persistent text-mode bot wrappers，按座位轮换运行 head-to-head benchmark。默认 placement reward 为：

```text
1st: 4
2nd: 2
3rd: 1
4th: 0
```

脚本统计：

```text
average_score
average_placement_reward_4_2_1_0
hu_rate
self_draw_rate
discard_hu_rate
deal_in_rate
huang_rate
first_place_rate
terminal_actions
```

该评测比单纯 validation accuracy 更接近比赛目标，因为它直接模拟 Botzone 请求/响应流程、官方 judge 结算、座位轮换和胡牌合法性约束。

## 9. 强化学习

当前实验中，强化学习没有稳定提升该监督模型：

```text
PPO: 以监督模型为 base 时，多数情况下产生负面提升；
GRPO: 同样未观察到稳定收益；
DQN: 可能有轻微提升，但提升幅度不明显且不稳定。
```

这类结果并不意外。MCR 的即时 reward 稀疏且高方差，局面价值受隐藏信息、对手策略、复式初始牌、番种结构和放铳风险共同影响。若 RL 目标与监督模型已学到的动作先验冲突，PPO/GRPO 很容易破坏已校准的动作分布。后续可以考虑做离线评估、低 KL 约束、分动作类型 reward shaping，以及只在高置信局面或残局子任务上微调。

## 10. 和牌分布

总和牌数: 2316262
未达成番种数: 3

## 达成番种

### 1番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 一般高 | 89992 (3.885%) | 14.221 | 喜相逢 | 472991 (20.420%) | 13.673 |
| 连六 | 312776 (13.503%) | 12.266 | 老少副 | 92573 (3.997%) | 12.378 |
| 幺九刻 | 527583 (22.777%) | 13.718 | 明杠 | 104516 (4.512%) | 14.232 |
| 缺一门 | 337839 (14.586%) | 16.807 | 无字 | 258380 (11.155%) | 14.137 |
| 独听・边张 | 149008 (6.433%) | 14.513 | 独听・嵌张 | 455145 (19.650%) | 13.826 |
| 独听・单钓 | 138356 (5.973%) | 14.018 | 自摸 | 456639 (19.714%) | 15.311 |

### 2番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 箭刻 | 320965 (13.857%) | 12.704 | 圈风刻 | 114699 (4.952%) | 13.370 |
| 门风刻 | 115504 (4.987%) | 13.375 | 门前清 | 346636 (14.965%) | 13.877 |
| 平和 | 915295 (39.516%) | 13.897 | 四归一 | 85579 (3.695%) | 16.477 |
| 双同刻 | 61054 (2.636%) | 16.821 | 双暗刻 | 88522 (3.822%) | 13.741 |
| 暗杠 | 35357 (1.526%) | 14.946 | 断幺 | 295162 (12.743%) | 13.435 |

### 4番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 全带幺 | 67884 (2.931%) | 13.468 | 不求人 | 183279 (7.913%) | 14.069 |
| 双明杠 | 4629 (0.200%) | 17.176 | 和绝张 | 111302 (4.805%) | 15.746 |

### 5番（增补）
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 明暗杠 | 2057 (0.089%) | 16.948 |  |  |  |

### 6番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 碰碰和 | 94417 (4.076%) | 16.381 | 混一色 | 220781 (9.532%) | 13.820 |
| 三色三步高 | 524091 (22.627%) | 11.551 | 五门齐 | 246509 (10.643%) | 13.105 |
| 全求人 | 18702 (0.807%) | 17.040 | 双暗杠 | 459 (0.020%) | 18.941 |
| 双箭刻 | 20727 (0.895%) | 14.010 |  |  |  |

### 8番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 花龙 | 177877 (7.679%) | 12.833 | 推不倒 | 3385 (0.146%) | 19.801 |
| 三色三同顺 | 213183 (9.204%) | 13.877 | 三色三节高 | 3729 (0.161%) | 23.429 |
| 无番和 | 35336 (1.526%) | 9.300 | 妙手回春 | 2756 (0.119%) | 21.766 |
| 海底捞月 | 4210 (0.182%) | 22.245 | 杠上开花 | 8149 (0.352%) | 21.053 |
| 抢杠和 | 4378 (0.189%) | 18.349 | 天和 | 2 (0.000%) | 25.500 |
| 地和 | 4 (0.000%) | 25.750 | 人和Ⅰ | 72 (0.003%) | 26.639 |
| 人和Ⅱ | 8 (0.000%) | 24.000 |  |  |  |

### 12番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 全不靠 | 84793 (3.661%) | 16.791 | 组合龙 | 68043 (2.938%) | 19.326 |
| 大于五 | 39567 (1.708%) | 18.322 | 小于五 | 39815 (1.719%) | 18.438 |
| 三风刻 | 1183 (0.051%) | 25.345 |  |  |  |

### 16番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 清龙 | 124320 (5.367%) | 21.992 | 三色双龙会 | 71 (0.003%) | 20.239 |
| 一色三步高 | 82611 (3.567%) | 22.518 | 全带五 | 2532 (0.109%) | 26.086 |
| 三同刻 | 1568 (0.068%) | 31.020 | 三暗刻 | 17771 (0.767%) | 27.263 |

### 24番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 七对 | 69379 (2.995%) | 27.543 | 七星不靠 | 37726 (1.629%) | 26.256 |
| 全双刻 | 367 (0.016%) | 35.256 | 清一色 | 21326 (0.921%) | 32.019 |
| 一色三同顺 | 455 (0.020%) | 34.793 | 一色三节高 | 3093 (0.134%) | 39.512 |
| 全大 | 1742 (0.075%) | 34.065 | 全中 | 464 (0.020%) | 36.349 |
| 全小 | 2233 (0.096%) | 33.433 |  |  |  |

### 32番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 一色四步高 | 1242 (0.054%) | 41.463 | 三杠 | 152 (0.007%) | 46.520 |
| 混幺九 | 837 (0.036%) | 44.118 |  |  |  |

### 48番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 一色四同顺 | 0 (0.000%) | 0.000 | 一色四节高 | 56 (0.002%) | 62.321 |

### 64番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 清幺九 | 13 (0.001%) | 76.923 | 小四喜 | 89 (0.004%) | 80.652 |
| 小三元 | 3191 (0.138%) | 71.455 | 字一色 | 13 (0.001%) | 110.231 |
| 四暗刻 | 730 (0.032%) | 71.645 | 一色双龙会 | 7 (0.000%) | 67.429 |

### 88番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 大四喜 | 2 (0.000%) | 126.500 | 大三元 | 369 (0.016%) | 95.978 |
| 绿一色 | 9 (0.000%) | 103.556 | 九莲宝灯 | 0 (0.000%) | 0.000 |
| 四杠 | 5 (0.000%) | 97.400 | 连七对 | 0 (0.000%) | 0.000 |
| 十三幺 | 573 (0.025%) | 90.232 |  |  |  |

## 凑番技术

### 半凑番
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 不求人 | 92621 (3.999%) | 10.602 | 双明杠 | 883 (0.038%) | 11.084 |
| 和绝张 | 51777 (2.235%) | 11.057 |  |  |  |

### 银
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 箭刻 | 96 (0.004%) | 9.385 | 圈风刻 | 32 (0.001%) | 9.281 |
| 门风刻 | 29 (0.001%) | 9.621 | 门前清 | 406 (0.018%) | 9.414 |
| 平和 | 297 (0.013%) | 9.404 | 四归一 | 479 (0.021%) | 9.395 |
| 双同刻 | 1087 (0.047%) | 9.546 | 双暗刻 | 52 (0.002%) | 9.346 |
| 暗杠 | 64 (0.003%) | 9.531 | 断幺 | 1 (0.000%) | 10.000 |

### 金
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 纯正式 | 13 (0.001%) | 9.615 | 双幺九式 | 4 (0.000%) | 9.250 |

### 基础
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 门清平和 | 13061 (0.564%) | 9.538 | 门清平和断幺 | 39745 (1.716%) | 9.903 |

### 其他
| 番种 | 数量 (%) | 平均总番数 | 番种 | 数量 (%) | 平均总番数 |
|---|---:|---:|---|---:|---:|
| 2*2 | 19025 (0.821%) | 9.627 | 2*3 | 14918 (0.644%) | 9.960 |
| 2*4 | 5812 (0.251%) | 10.589 | 2*5+ | 429 (0.019%) | 12.471 |


## 13. 参考链接

```text
Botzone 2026 Mahjong Competition:
https://botzone.org.cn/static/gamecontest2026a.html

MCRAgent repository:
https://github.com/Iconoclastic0428/MCRAgent

Promoted checkpoint README:
https://github.com/Iconoclastic0428/MCRAgent/blob/main/models/feature_agent_botzone98209_aug12_river_elastic_2xgpu_20260617a/README.md

River Elastic notes:
https://github.com/Iconoclastic0428/MCRAgent/blob/main/feature-agent/RIVER_ELASTIC_NOTES.md

Feature construction:
https://github.com/Iconoclastic0428/MCRAgent/blob/main/feature-agent/feature.py

Dataset and augmentation:
https://github.com/Iconoclastic0428/MCRAgent/blob/main/feature-agent/dataset.py

Model architecture:
https://github.com/Iconoclastic0428/MCRAgent/blob/main/feature-agent/model.py

Supervised training:
https://github.com/Iconoclastic0428/MCRAgent/blob/main/feature-agent/supervised.py

Botzone runtime wrapper:
https://github.com/Iconoclastic0428/MCRAgent/blob/main/scripts/feature_repo_json_runtime.py

Promoted checkpoint wrapper:
https://github.com/Iconoclastic0428/MCRAgent/blob/main/scripts/feature_agent_checkpoint_json_bot.py

Local benchmark:
https://github.com/Iconoclastic0428/MCRAgent/blob/main/scripts/benchmark_json_policies.py
```
