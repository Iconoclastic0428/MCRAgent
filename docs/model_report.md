# MCR Baseline Model Report

Generated: 2026-05-25.

## Data

- Raw public replay corpus: `data/raw/botzone_mcr_160.jsonl`
- Raw replay count: 160
- Behavior-cloning examples: `data/processed/botzone_mcr_bc_160.jsonl`
- Converted example count: 62,904
- Active draw examples: 7,355
- Observed action counts: PASS 54,665; PLAY 7,281; CHI 390; PENG 334; HU 134; GANG 59; BUGANG 25; MISSING 16.

Expanded corpus:

- Raw public replay corpus: `data/raw/botzone_mcr_300.jsonl`
- Raw replay count: 300
- Behavior-cloning examples: `data/processed/botzone_mcr_bc_300.jsonl`
- Converted example count: 121,412
- Active draw examples: 14,215
- Observed action counts: PASS 105,523; PLAY 14,089; CHI 743; PENG 642; HU 258; GANG 84; BUGANG 46; MISSING 27.

Larger corpus:

- Raw public replay corpus: `data/raw/botzone_mcr_1000.jsonl`
- Raw replay count: 1,000
- Behavior-cloning examples: `data/processed/botzone_mcr_bc_1000.jsonl`
- Converted example count: 418,112
- Active draw examples: 48,767
- Observed action counts: PASS 363,218; PLAY 48,294; CHI 2,973; PENG 2,305; HU 859; GANG 240; BUGANG 163; MISSING 51; PASS_NN 4; PASS_MODEL_OK 4; EMPTY 1.

## Models And Evaluation

All policies were evaluated through `scripts/evaluate_policy_replay.py`, which replays Botzone request logs through a stateful policy wrapper and checks exact/action-type match plus illegal-prediction/fallback rates.

| Policy | Model | Eval file | Active draw exact | Illegal prediction rate |
| --- | --- | --- | ---: | ---: |
| Drawn-tile fallback | none | `runs/policy_replay_eval_fallback_160.json` | 0.4761 | 0.0000 |
| Min-shanten heuristic | none | `runs/policy_replay_eval_shanten_heuristic_160.json` | 0.4411 | 0.0000 |
| Text response classifier | `models/bc_policy_draw_response_160.pkl` | `runs/policy_replay_eval_draw_response_160.json` | 0.4757 | 0.1008 |
| Legal discard ranker | `models/discard_ranker_160.pkl` | `runs/policy_replay_eval_discard_ranker_160.json` | 0.2959 | 0.0000 |
| Shanten-aware legal discard ranker | `models/discard_ranker_shanten_160.pkl` | `runs/policy_replay_eval_discard_ranker_shanten_160.json` | 0.3697 | 0.0000 |
| Shanten+drawn-tile legal discard ranker | `models/discard_ranker_shanten_drawn_160.pkl` | `runs/policy_replay_eval_discard_ranker_shanten_drawn_160.json` | 0.3744 | 0.0000 |
| Composite draw+reaction ranker | `models/composite_draw_reaction_160.pkl` | `runs/policy_replay_eval_composite_draw_reaction_160.json` | 0.3744 | 0.0000 |
| Composite draw+weighted-reaction ranker | `models/composite_draw_reaction_nonpass50_160.pkl` | `runs/policy_replay_eval_composite_draw_reaction_nonpass50_160.json` | 0.3765 | 0.0000 |
| 300-match composite draw+weighted-reaction ranker | `models/composite_draw_reaction_nonpass50_300.pkl` | `runs/policy_replay_eval_composite_draw_reaction_nonpass50_300.json` | 0.3906 | 0.0000 |
| 300-match reward-weighted self-play policy | `models/reward_weighted_selfplay_policy_300src_128.pkl` | `runs/policy_replay_eval_reward_weighted_selfplay_300src_128.json` | 0.3809 | 0.0000 |
| 1000-match drawn-tile fallback | none | `runs/policy_replay_eval_fallback_1000.json` | 0.4888 | 0.0000 |
| 1000-match min-shanten heuristic | none | `runs/policy_replay_eval_shanten_heuristic_1000.json` | 0.4515 | 0.0000 |
| 1000-match composite draw+weighted-reaction ranker | `models/composite_draw_reaction_nonpass50_1000.pkl` | `runs/policy_replay_eval_composite_draw_reaction_nonpass50_1000.json` | 0.3883 | 0.0000 |
| 1000-source reward-weighted self-play policy | `models/reward_weighted_selfplay_policy_1000src_128.pkl` | `runs/policy_replay_eval_reward_weighted_selfplay_1000src_128.json` | 0.3791 | 0.0000 |
| 1000-match feature-draw + weighted-reaction ranker | `models/composite_feature_draw_reaction_nonpass50_1000.pkl` | `runs/policy_replay_eval_composite_feature_draw_reaction_nonpass50_1000.json` | 0.4838 | 0.0000 |
| 1000-match feature-draw + conservative-reaction ranker | `models/composite_feature_draw_reaction_1000.pkl` | `runs/policy_replay_eval_composite_feature_draw_reaction_1000.json` | 0.4818 | 0.0000 |
| 1000-match feature-draw + feature-reaction ranker | `models/composite_feature_draw_feature_reaction_nonpass10_1000.pkl` | `runs/policy_replay_eval_composite_feature_draw_feature_reaction_nonpass10_1000.json` | 0.5169 | 0.0000 |
| Player-0 filtered reward-weighted policy | `models/reward_weighted_policy_feature_source_player0_shanten_128.pkl` | `runs/policy_replay_eval_reward_weighted_policy_feature_source_player0_shanten_128.json` | 0.3820 | 0.0000 |
| 1000-match feature-draw official-safe no-Hu/no-kong composite | `models/composite_feature_draw_reaction_nonpass50_1000_nohu_nogang.pkl` | `runs/policy_replay_eval_composite_feature_draw_reaction_nonpass50_1000_nohu_nogang.json` | 0.4841 | 0.0000 |

## Interpretation

The current supervised text models prove the data/training/evaluation path, but they are not strong Mahjong policies. The text response classifier often predicts illegal discard tiles and is saved only behind a legal fallback wrapper. The shanten-aware rankers avoid illegal actions and improve over the first text-only legal ranker, but still underperform the simple drawn-tile replay baseline.

The full legal-action work added CHI/PENG/GANG/HU candidate generation and reaction-specific evaluation. The conservative reaction model predicts no non-pass reactions. The weighted reaction model predicts 229 non-pass reactions, but only 50 are exact matches: precision 0.2183 and recall 0.0576 on replay imitation. It is legal, but not stronger than passing reactions.

The 300-match weighted-reaction composite predicts 435 non-pass reactions on the 300-match replay set, but only 60 are exact: precision 0.1379 and recall 0.0364. Scaling from 160 to 300 matches improved lightweight self-play reward for the composite policy but did not improve replay imitation enough to beat the drawn-tile fallback.

Outcome-filtered 300-match legal-action rankers were also tested. The positive-score model reached only 0.3453 active-draw replay accuracy and 0.9629 proxy reward versus fallback; the winner-only model reached 0.3541 active-draw replay accuracy and 0.9355 proxy reward versus fallback. Both underperformed the broad 300-match composite on the same seed.

The 1000-match composite improved the lightweight self-play proxy but not replay imitation. On the 1000-match replay corpus, the conservative fallback still has higher active-draw exact accuracy (0.4888) than the composite (0.3883). In lightweight self-play, however, the 1000-match composite averaged 1.1569 reward versus fallback and 0.0664 versus shanten opponents on seed 9700. This is useful progress, but the proxy is not official MCR scoring and does not prove online competitiveness.

The numeric feature-draw ranker is a material upgrade over the TF-IDF discard ranker. Its held-out draw group accuracy is 0.5806, compared with 0.3228 for `models/discard_ranker_shanten_drawn_1000.pkl`. The best current policy artifact is `models/composite_feature_draw_reaction_nonpass50_1000.pkl`, which combines numeric draw ranking with the weighted TF-IDF reaction ranker. In 384 lightweight self-play games across seeds 9700/9800/9900, it averaged 1.6024 reward and 153 wins versus fallback, and 0.9403 reward and 151 wins versus shanten opponents.

Numeric feature reaction models were weaker despite better replay active-draw numbers. The nonpass10 feature-reaction composite predicted 9,172 non-pass reactions, with precision 0.0785 and recall 0.1186, and lost in lightweight self-play. The nonpass50 feature-reaction composite was also negative in proxy self-play.

The reward-weighted trainer now supports a player filter so it can train from only the controlled seat instead of copying opponent policies. The player-0-only reward-weighted model trained from the best current source no longer collapses, but it still regresses versus the source: on seed 9700 it averaged 1.3984 versus fallback and 0.3288 versus shanten, below the source policy's 1.5833 and 0.9167.

Reaction-threshold policy tuning was tested as a small reward-optimization step. `scripts/create_composite_policy.py` can now store `reaction_thresholds`, and `policy_bot.py` gates low-margin non-pass reaction claims to PASS. Thresholds hurt the current best model in lightweight self-play: the best tested margin, 0.02, averaged 0.6484 versus shanten and 1.4167 versus fallback on seed 9700, below the unthresholded model's 0.9167 and 1.5833 on the same seed. This suggests the weak-looking aggressive reactions still help in the current proxy.

Initial official judge evaluation showed the lightweight proxy was too optimistic. The best proxy artifact, `models/composite_feature_draw_reaction_nonpass50_1000.pkl`, was not safe under official scoring because the Python legal mask admitted shape-complete Hu claims that did not meet the MCR 8-fan threshold and allowed late-wall kong attempts that the judge rejected. Suppressing Hu removed `WH` penalties but left late-wall `WA` penalties. Suppressing both Hu and kong actions produced penalty-free official games, but it also removed the policy's ability to win.

The policy wrapper now tracks official wall counts, updates concealed-kong hand state correctly, and filters Hu candidates through a compiled MahjongGB fan-check helper before offering them to the model. With those official-aware filters, the same `models/composite_feature_draw_reaction_nonpass50_1000.pkl` checkpoint is positive against local fallback and shanten baselines under the official judge. This is still far below the final requirement: it has not beaten current online Botzone models.

The next useful work is stronger neural/legal-mask RL and broader official evaluation against stronger open or replay-derived opponents. More proxy reward tuning is secondary now that the policy can make official scoring decisions.

## Current Online Targets

A fresh local fetch of the Botzone `Chinese-Standard-Mahjong` rank list on 2026-05-25 is stored at `data/metadata/botzone_ranklist_top30_2026-05-25.json`. The top observed target was `gyt美女一号` at 1313.15 rating; the top 30 also includes model-like or explicitly ML-labeled targets such as `chunjiandu` (`监督加强化微调`), `mahjong_sl_test`, `resnet`-described bots, `TMahjong` (`深度强化学习`), and `MangoBotBC`. The local artifacts are not yet competitive with these online targets.

The Botzone 2020 Mahjong competition page is also relevant evidence for the target bar. Its final-stage table lists `SuperJong` first at 1339, `ALONG` second at 1314, and `WuKong` third at 1304. The linked ALONG symposium PDF reports a combined imitation-learning and reinforcement-learning approach: pretraining from about 500K official match-log games, then PPO-style self-play RL for about 32M steps. The current project is far below that scale and architecture; the reward-weighted sklearn branch is only a diagnostic RL-style baseline, not an ALONG-class RL implementation.

Sources:

- `https://www.botzone.org.cn/static/gamecontest2020a.html`
- `https://www.botzone.org.cn/static/IJCAI2020MahjongPPT/04-ALONG-yue.pdf`

## Official Judge Evaluation

The official C++ Botzone judge is built locally at `build/official_judge/mcr_judge.exe`, using the official `Chinese-Standard-Mahjong` judge source and `PyMahjongGB` MahjongGB algorithm source with jsoncpp 1.9.7 source files. `scripts/official_judge_match.py` runs Botzone-style request/response logs and records official terminal scores.

| Policy | Opponent | Eval file | Games | Player-0 total | Player-0 average | Player-0 wins | Terminal actions |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Proxy-best feature-draw weighted-reaction before official filters | fallback | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_vs_fallback_32.json` | 32 | -213.0 | -6.6563 | 3 | HU 3; HUANG 18; WA 3; WH 8 |
| Proxy-best feature-draw weighted-reaction before official filters | shanten | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_vs_shanten_32.json` | 32 | -273.0 | -8.5313 | 3 | HU 3; HUANG 16; WA 3; WH 10 |
| No-Hu feature-draw weighted-reaction | fallback | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_vs_fallback_32.json` | 32 | -60.0 | -1.8750 | 0 | HUANG 30; WA 2 |
| No-Hu feature-draw weighted-reaction | shanten | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_vs_shanten_32.json` | 32 | -60.0 | -1.8750 | 0 | HUANG 30; WA 2 |
| No-Hu/no-kong feature-draw weighted-reaction | fallback | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_nogang_vs_fallback_32.json` | 32 | 0.0 | 0.0000 | 0 | HUANG 32 |
| No-Hu/no-kong feature-draw weighted-reaction | shanten | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_nogang_vs_shanten_32.json` | 32 | 0.0 | 0.0000 | 0 | HUANG 32 |
| No-Hu feature-draw weighted-reaction after wall filter | fallback | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_wallfix2_vs_fallback_32.json` | 32 | 0.0 | 0.0000 | 0 | HUANG 32 |
| No-Hu feature-draw weighted-reaction after wall filter | shanten | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_nohu_wallfix2_vs_shanten_32.json` | 32 | 0.0 | 0.0000 | 0 | HUANG 32 |
| Proxy-best feature-draw weighted-reaction with official filters | fallback | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_officialfilter_vs_fallback_128.json` | 128 | 639.0 | 4.9922 | 15 | HU 15; HUANG 113 |
| Proxy-best feature-draw weighted-reaction with official filters | shanten | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_officialfilter_vs_shanten_128.json` | 128 | 676.0 | 5.2813 | 16 | HU 16; HUANG 112 |
| Player-0 reward-weighted policy with official filters | fallback | `runs/official_judge_reward_weighted_policy_feature_source_player0_shanten_128_officialfilter_vs_fallback_64.json` | 64 | 203.0 | 3.1719 | 5 | HU 5; HUANG 59 |
| Player-0 reward-weighted policy with official filters | shanten | `runs/official_judge_reward_weighted_policy_feature_source_player0_shanten_128_officialfilter_vs_shanten_64.json` | 64 | 207.0 | 3.2344 | 5 | HU 5; HUANG 59 |
| Official-reward standalone policy, held-out offset 128 | fallback | `runs/official_judge_official_reward_weighted_policy_feature_source_shanten_128_vs_fallback_64_offset128.json` | 64 | 0.0 | 0.0000 | 0 | HUANG 64 |
| Official-reward standalone policy, held-out offset 128 | shanten | `runs/official_judge_official_reward_weighted_policy_feature_source_shanten_128_vs_shanten_64_offset128.json` | 64 | 0.0 | 0.0000 | 0 | HUANG 64 |
| Feature draw + official-reward reaction, held-out offset 128 | fallback | `runs/official_judge_composite_feature_draw_official_reward_reaction_shanten_128_vs_fallback_64_offset128.json` | 64 | 0.0 | 0.0000 | 0 | HUANG 64 |
| Feature draw + official-reward reaction, held-out offset 128 | shanten | `runs/official_judge_composite_feature_draw_official_reward_reaction_shanten_128_vs_shanten_64_offset128.json` | 64 | 0.0 | 0.0000 | 0 | HUANG 64 |
| Supervised source with official filters, held-out offset 128 | fallback | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_officialfilter_vs_fallback_64_offset128.json` | 64 | 382.0 | 5.9688 | 10 | HU 10; HUANG 54 |
| Supervised source with official filters, held-out offset 128 | shanten | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_officialfilter_vs_shanten_64_offset128.json` | 64 | 382.0 | 5.9688 | 10 | HU 10; HUANG 54 |
| Official sample bot | fallback | `runs/official_judge_official_sample_vs_fallback_64.json` | 64 | 0.0 | 0.0000 | 0 | HUANG 64 |
| Supervised source with official filters | official sample bot | `runs/official_judge_composite_feature_draw_reaction_nonpass50_1000_vs_official_sample_64.json` | 64 | 313.0 | 4.8906 | 8 | HU 8; HUANG 56 |

On the same first 64 `initdata` seeds, the supervised source still beats the reward-weighted policy: source average 4.3906 vs fallback and 4.8906 vs shanten, compared with 3.1719 and 3.2344 for the reward-weighted checkpoint. The current RL-style branch is therefore not the final model.

An official-trajectory reward attempt was also tested. `scripts/official_trajectories.py` generated 128 source-vs-shanten official games for reward-weighted training, and `scripts/train_reward_weighted_policy.py` now supports `--request-kind` so reaction-only variants can preserve the strong feature draw ranker. Both official-reward checkpoints collapsed to all `HUANG` on held-out offset-128 official games, while the supervised source scored 5.9688 average on the same seeds. This reinforces that the current reward-weighted approach is not a sufficient RL method.

The official sample bot is now wrapped through `BotzoneJsonProcessPolicy` in `scripts/official_judge_match.py`, and the Aleo process wrapper exists as `AleoProcessPolicy`. The official sample bot is a usable benchmark. Aleo is not yet usable as a benchmark because both the line-history build and Botzone-style merged build can crash on valid official payloads; the saved crash inputs are `tmp/aleo_crash_payload.txt` and `tmp/aleo_botzone_crash_payload.txt`.

## Lightweight Self-Play

A lightweight Python Botzone-protocol simulator was added in `scripts/selfplay_sim.py` to generate RL-style trajectories with terminal rewards based on Hu or final shanten. This remains useful for iteration, but the official C++ judge results above are the binding evidence for MCR legality and scoring.

Artifacts:

| Run | File | Player-0 average reward | Player-0 wins |
| --- | --- | ---: | ---: |
| fallback vs fallback, 64 games | `runs/selfplay_fallback_vs_fallback_64.json` | 0.0990 | 0 |
| shanten vs fallback, 128 games seed 9400 | `runs/selfplay_shanten_vs_fallback_128_seed9400.json` | 0.9831 | 0 |
| composite vs fallback, 128 games seed 9400 | `runs/selfplay_composite_nonpass50_vs_fallback_128_seed9400.json` | 1.0371 | 21 |
| reward-weighted self-play policy vs fallback, 128 games seed 9400 | `runs/selfplay_reward_weighted_policy_128_vs_fallback_128.json` | 0.5358 | 0 |
| 300-match composite vs fallback, 128 games seed 9700 | `runs/selfplay_composite_nonpass50_300_vs_fallback_128_seed9700.json` | 1.0358 | 19 |
| 300-source reward-weighted policy vs fallback, 128 games seed 9700 | `runs/selfplay_reward_weighted_policy_300src_128_vs_fallback_128_seed9700.json` | 0.8581 | 0 |
| 1000-match composite vs fallback, 128 games seed 9700 | `runs/selfplay_composite_nonpass50_1000_vs_fallback_128_seed9700.json` | 1.1569 | 21 |
| 1000-match composite vs shanten, 128 games seed 9700 | `runs/selfplay_composite_nonpass50_1000_vs_shanten_128_seed9700.json` | 0.0664 | 17 |
| 1000-source reward-weighted policy vs fallback, 128 games seed 9700 | `runs/selfplay_reward_weighted_policy_1000src_128_vs_fallback_128_seed9700.json` | 0.8451 | 0 |
| 1000-source reward-weighted policy vs shanten, 128 games seed 9700 | `runs/selfplay_reward_weighted_policy_1000src_128_vs_shanten_128_seed9700.json` | -0.4798 | 0 |
| feature-draw weighted-reaction vs fallback, 384 games seeds 9700/9800/9900 | `runs/selfplay_composite_feature_draw_reaction_nonpass50_1000_vs_fallback_128_seed*.json` | 1.6024 | 153 |
| feature-draw weighted-reaction vs shanten, 384 games seeds 9700/9800/9900 | `runs/selfplay_composite_feature_draw_reaction_nonpass50_1000_vs_shanten_128_seed*.json` | 0.9403 | 151 |
| feature-draw conservative-reaction vs fallback, 128 games seed 9700 | `runs/selfplay_composite_feature_draw_reaction_1000_vs_fallback_128_seed9700.json` | 1.2240 | 20 |
| feature-draw conservative-reaction vs shanten, 128 games seed 9700 | `runs/selfplay_composite_feature_draw_reaction_1000_vs_shanten_128_seed9700.json` | 0.4375 | 20 |
| player-0 filtered reward-weighted policy vs fallback, 128 games seed 9700 | `runs/selfplay_reward_weighted_policy_feature_source_player0_shanten_128_vs_fallback_seed9700.json` | 1.3984 | 26 |
| player-0 filtered reward-weighted policy vs shanten, 128 games seed 9700 | `runs/selfplay_reward_weighted_policy_feature_source_player0_shanten_128_vs_shanten_seed9700.json` | 0.3288 | 26 |
| feature-draw weighted-reaction margin 0.02 vs fallback, 128 games seed 9700 | `runs/selfplay_composite_feature_draw_reaction_nonpass50_margin002_1000_vs_fallback_128_seed9700.json` | 1.4167 | 34 |
| feature-draw weighted-reaction margin 0.02 vs shanten, 128 games seed 9700 | `runs/selfplay_composite_feature_draw_reaction_nonpass50_margin002_1000_vs_shanten_128_seed9700.json` | 0.6484 | 33 |

The first reward-weighted policy artifact is `models/reward_weighted_selfplay_policy_128.pkl`, trained from `runs/selfplay_composite_nonpass50_vs_fallback_128_trajectories.json`. It did not improve over the composite policy that generated the trajectories.

The second reward-weighted policy artifact is `models/reward_weighted_selfplay_policy_300src_128.pkl`, trained from `runs/selfplay_composite_nonpass50_300_vs_fallback_128_trajectories.json`. It also did not improve over the 300-match composite source policy.

The third reward-weighted policy artifact is `models/reward_weighted_selfplay_policy_1000src_128.pkl`, trained from `runs/selfplay_composite_nonpass50_1000_vs_fallback_128_trajectories.json`. It also regressed versus its 1000-match composite source policy.
