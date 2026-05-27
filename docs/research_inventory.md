# MCR Mahjong AI Research Inventory

Last refreshed: 2026-05-25.

## Authoritative MCR Sources

- Botzone Chinese-Standard-Mahjong is the live online MCR/Guobiao platform target. Its own repository says Botzone evaluates game AIs through multi-agent competition and an ELO-like ranking system.
- The IJCAI 2025 Mahjong AI Competition page states the task is to develop an MCR AI competing with agents and humans on Botzone.
- Botzone currently lists `Chinese-Standard-Mahjong` as the active game and shows IJCAI 2026 simulation contests scheduled under that game.
- Botzone exposes public match data in two ways:
  - Monthly matchpack downloads at `https://extra.botzone.org.cn/matchpacks/<game>-<year>-<month>.zip`.
  - Individual replay pages under `https://en.botzone.org.cn/match/<match_id>`, which embed `_rawLogJSON`.

## Online MCR / Chinese-Standard-Mahjong Models And Toolkits

- `ailab-pku/Chinese-Standard-Mahjong`: official competition repository with rules, judge, fan calculator usage, and sample Botzone bot.
  - Local clone: `external/Chinese-Standard-Mahjong`, commit `5e81821`.
- `KiddoZhu/Aleo`: C++ Guobiao/MCR Mahjong AI toolkit with self-play simulator and Python interface; README reports a 2018 course competition win and 46.8% average winning rate.
  - Local clone: `external/Aleo`, commit `9ab737f`.
- `ailab-pku/PyMahjongGB`: official Python fan/shanten calculator. The README documents `MahjongFanCalculator` and `MahjongShanten`.
  - Local clone: `external/PyMahjongGB`, commit `93fdc2d`.
  - Install attempt on this Windows machine failed because `pip install PyMahjongGB` needs Microsoft C++ Build Tools for the extension build.
- Botzone public leaderboard/replays: current bots are generally not open-source, but their public match logs and ratings are usable evaluation targets. Recent visible examples include `MangoBotBC`, `小寻歌`, `龙虎争霸`, `TMahjong`, `QwQ`, `bot32`, and many others.
  - Fresh local rank-list snapshot: `data/metadata/botzone_ranklist_top30_2026-05-25.json`.
  - 2026-05-25 top observed target from local fetch: `gyt美女一号` rating 1313.15. The top 30 includes explicit ML/RL descriptors such as `chunjiandu` (`监督加强化微调`), `mahjong_sl_test`, `resnet`-described bots, `TMahjong` (`深度强化学习`), and `MangoBotBC`.
- IJCAI 2020 Mahjong AI Competition reports:
  - Team ALONG report describes supervised learning on organizer-provided logs and matching SL/RL models in Botzone-style play.
  - `On Policy Gradient Applied to Chinese Standard Mahjong` describes a policy-gradient approach for Chinese Standard Mahjong.
- Botzone 2020 competition page and ALONG symposium PDF:
  - Contest page: `https://www.botzone.org.cn/static/gamecontest2020a.html`.
  - ALONG paper/PPT: `https://www.botzone.org.cn/static/IJCAI2020MahjongPPT/04-ALONG-yue.pdf`.
  - The page lists the Mahjong AI final-stage table with `SuperJong` first at 1339, `ALONG` second at 1314, and `WuKong` third at 1304.
  - ALONG is directly relevant to this goal because it reports imitation learning from official match logs followed by reinforcement learning/self-play with PPO-style policy-gradient training. The reported scale is much larger than the current local prototype: about 500K pretraining games and roughly 32M RL steps before convergence.
  - This makes ALONG a concrete historical target architecture and score reference, not just background reading.
- Latest competition pages checked from Botzone:
  - 2026 page: `https://botzone.org.cn/static/gamecontest2026a.html`.
  - 2025 page: `https://botzone.org.cn/static/gamecontest2025a.html`.
  - The 2026 page is the current/latest competition page as of 2026-05-25. It schedules Round 1 for 2026-06-09 and the final round for 2026-07-07, so final 2026 rankings do not exist yet.
  - The 2026 page links a strong-AI dataset as `https://disk.pku.edu.cn/link/AA8CB7A57AFDCD48CAA7C749E04B5B6FAA` with title `data.zip`. This is now the highest-priority dataset target.
  - The 2025 final-stage table lists `SeaMan` first at 1345.5020, `dimaria` second at 1327.0010, `Jirachi` third at 1251.9993, and `哞哞哞` fourth at 1195.4980.
- IJCAI 2024 Mahjong AI Competition paper summarizes 2020/2022/2023 MCR competitions:
  - Top methods include heuristic search, supervised learning, and reinforcement learning.
  - SL uses policy cloning from match datasets; top systems use CNN/ResNet-like image features, augmentation, and high-level features such as shanten.
  - RL methods include PPO, IMPALA-style distributed actor/learner training, and self-play; high compute and instability are major constraints.
- Recent papers/tools mostly target Riichi, not MCR:
  - Suphx is strong deep RL for Japanese Mahjong, not MCR.
  - Mortal, kanachan, mjx, pymahjong, and Mahjax are Riichi environments/models, useful architecturally but not valid MCR baselines.

## Dataset Assessment

- Primary open dataset: Botzone matchpacks for `Chinese-Standard-Mahjong`, monthly JSONL ZIP files.
- Current machine result: `extra.botzone.org.cn` has a TLS certificate principal mismatch and returns `502 Bad Gateway` for tested Chinese-Standard-Mahjong matchpack URLs, including `2026-4`, `2026-04`, `2026-3`, `2025-12`, and `2024-5`.
- Fallback open dataset path: scrape public Botzone replay pages from the global match list. Each replay embeds full alternating judge request and player response logs, enough to build behavior-cloning data and replay-based evaluation corpora.
- Current local fallback corpus:
  - `data/raw/botzone_mcr_160.jsonl`: 160 public replays.
  - `data/processed/botzone_mcr_bc_160.jsonl`: 62,904 behavior-cloning examples, including 7,355 active draw decisions and all core MCR action types observed in the sample.
  - `data/raw/botzone_mcr_300.jsonl`: 300 public replays.
  - `data/processed/botzone_mcr_bc_300.jsonl`: 121,412 behavior-cloning examples, including 14,215 active draw decisions.
  - `data/raw/botzone_mcr_1000.jsonl`: 1,000 public replays.
  - `data/processed/botzone_mcr_bc_1000.jsonl`: 418,112 behavior-cloning examples, including 48,767 active draw decisions.

## Current Training Strategy

1. Build behavior-cloning data from public Botzone logs.
2. Train a small first policy on request/response examples to prove the data path.
3. Integrate an MCR legal-action/simulator layer from the official judge or Aleo-style toolkit.
4. Move the RL branch away from reward-weighted sklearn cloning and toward an ALONG-like setup: neural policy/value model, official legal masks, self-play rollouts, and PPO-style updates from the supervised checkpoint.
5. Benchmark against public baselines, official sample bots, open-source heuristic/search bots when stable, and replay-derived targets.

## Local Baseline Results

- Conservative fallback policy: on active draw requests, play the drawn tile. Replay imitation on 160 scraped matches:
  - `runs/policy_replay_eval_fallback_160.json`
  - active-draw exact accuracy: 0.4761
  - illegal prediction rate: 0
- Text response classifier:
  - `models/bc_policy_draw_response_160.pkl`
  - `runs/policy_replay_eval_draw_response_160.json`
  - active-draw exact accuracy: 0.4757 after legal fallback
  - raw illegal prediction rate before fallback: 0.1008
- Legal discard ranker:
  - `models/discard_ranker_160.pkl`
  - `runs/policy_replay_eval_discard_ranker_160.json`
  - active-draw exact accuracy: 0.2959
  - illegal prediction rate: 0
- Min-shanten heuristic:
  - `runs/policy_replay_eval_shanten_heuristic_160.json`
  - active-draw exact accuracy: 0.4411
  - illegal prediction rate: 0
- Shanten-aware legal discard rankers:
  - `models/discard_ranker_shanten_160.pkl`: active-draw exact accuracy 0.3697, illegal prediction rate 0
  - `models/discard_ranker_shanten_drawn_160.pkl`: active-draw exact accuracy 0.3744, illegal prediction rate 0
- Full legal-action reaction models:
  - `models/legal_action_ranker_160.pkl`: legal but draw decisions degraded under stateful replay.
  - `models/composite_draw_reaction_160.pkl`: combines draw ranker and reaction ranker; predicts no non-pass reactions in stateful replay.
  - `models/composite_draw_reaction_nonpass50_160.pkl`: predicts 229 non-pass reactions; 50 exact, precision 0.2183, recall 0.0576.
  - `models/composite_draw_reaction_nonpass50_300.pkl`: predicts 435 non-pass reactions on the 300-match set; 60 exact, precision 0.1379, recall 0.0364.
  - `models/composite_draw_reaction_nonpass50_1000.pkl`: predicts 1,503 non-pass reactions on the 1000-match set; 151 exact, precision 0.1005, recall 0.0249.
  - `models/composite_feature_draw_reaction_nonpass50_1000.pkl`: best current proxy artifact; numeric draw ranker plus weighted TF-IDF reaction ranker. It predicts 3,880 non-pass reactions on replay, with precision 0.0459 and recall 0.0293, but is much stronger in lightweight self-play.

These are not proven competitive online models. The best current proxy self-play artifact is `models/composite_feature_draw_reaction_nonpass50_1000.pkl`, averaging 1.6024 reward versus fallback and 0.9403 versus shanten opponents across 384 lightweight games on seeds 9700/9800/9900. Under the official C++ Botzone judge, that same checkpoint was negative until the policy wrapper added official wall-count tracking and MahjongGB fan filtering. With those filters, it averaged 4.9922 official score versus fallback and 5.2813 versus shanten across 128 games, with no `WA` or `WH` penalties. Official-reward weighted training from those trajectories regressed to all-draw behavior on held-out games, so the best artifact remains supervised rather than a successful RL model. It is still not proven competitive with live Botzone bots.

Additional official-benchmark status:

- Official sample bot built from `external/Chinese-Standard-Mahjong/sample-bot-Botzone/sample.cpp` is available at `build/official_sample_bot.exe`.
- `runs/official_judge_official_sample_vs_fallback_64.json`: sample as player 0 versus fallback seats scored 0.0 average with 64 `HUANG`.
- `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_vs_official_sample_64.json`: the current supervised source as player 0 versus three official sample bots scored 313 total, 4.8906 average, and 8 wins in 64 games.
- Aleo is built at `build/aleo_bot.exe` and `build/aleo_botzone.exe`, but it crashes with access violation on valid official-protocol payloads saved under `tmp/aleo_crash_payload.txt` and `tmp/aleo_botzone_crash_payload.txt`. It should not be treated as a completed benchmark until that runner is fixed.

## Lightweight Self-Play Status

- Added `scripts/selfplay_sim.py`, a Python Botzone-protocol simulator that produces RL-style trajectories and terminal rewards.
- Added `scripts/train_reward_weighted_policy.py`, a reward-weighted policy trainer over self-play trajectories.
- Generated `runs/selfplay_composite_nonpass50_vs_fallback_128_trajectories.json`.
- Trained `models/reward_weighted_selfplay_policy_128.pkl`.
- Negative result: reward-weighted policy averaged 0.5358 reward against fallback on seed 9400, below the source composite policy's 1.0371 and the shanten heuristic's 0.9831 under the same lightweight evaluator.
- Trained `models/reward_weighted_selfplay_policy_300src_128.pkl` from 300-match composite trajectories.
- Negative result: 300-source reward-weighted policy averaged 0.8581 reward against fallback on seed 9700, below the 300-match source composite policy's 1.0358 under the same lightweight evaluator.
- Trained `models/reward_weighted_selfplay_policy_1000src_128.pkl` from 1000-match composite trajectories.
- Negative result: 1000-source reward-weighted policy averaged 0.8451 reward against fallback and -0.4798 against shanten on seed 9700, below the 1000-match source composite policy's 1.1569 versus fallback and 0.0664 versus shanten.
- Added a player filter to reward-weighted training so only controlled player-0 decisions are imitated from self-play trajectories.
- Trained `models/reward_weighted_policy_feature_source_player0_shanten_128.pkl` from the best feature-draw source policy against shanten. It beat fallback and shanten in the proxy, but still regressed versus the source policy.
- Tested reward-tuned reaction thresholds for the best feature-draw policy. The thresholded variants underperformed the unthresholded source in the proxy, so they are diagnostic artifacts rather than current best models.
- This remains a proxy evaluator, not proof of MCR strength against current Botzone models.

## Completion Standard

The objective is not complete until a trained MCR RL model is benchmarked with statistically meaningful evidence against the current online models found. Current work is dataset/research scaffolding only.
