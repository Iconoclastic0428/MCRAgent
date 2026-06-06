# Extraction Notes

## Source Metadata

- Title: Tjong: A transformer-based Mahjong AI via hierarchical decision-making and fan backward.
- DOI: `10.1049/cit2.12298`.
- Journal: CAAI Transactions on Intelligence Technology, volume 9, issue 4, pages 982-995.
- Open access status: gold open access, CC BY-NC 4.0 according to Crossref/Semantic Scholar metadata.

## Paper-Specified Architecture

- Backbone: Transformer in Transformer (TIT).
- Policy input visible tile features: `22 x 34 x 1`.
- Game features: `1 x 24`.
- Hidden/global value input: `5 x 34` hidden tile matrix.
- Inner Transformer:
  - Receives feature slices plus trainable class token and trainable positional encoding.
  - No mask.
  - Uses multi-head self-attention, layer normalization, FFN.
  - The final class token `z_L[0]` represents all feature information.
- Outer Transformer:
  - Receives the historical inner-block outputs, actions, and rewards.
  - Future time steps are masked.
  - The last element is passed through an FFN/head to produce the action.
- Tjong policy network:
  - Action head chooses among 8 action types: Pass, Hu, Discard, Chow, Pong, MingKong, BuKong, AnKong.
  - If the action is a claiming action, the state is modified and passed through the Transformer again, then the Claiming head selects the tile/claim.
  - If the action is Discard, the Discard head selects the tile to discard.
- Value network:
  - Similar to Oracle Guiding.
  - Uses global features formed by other players' hand tiles and wall tiles.
- Paper stated constants:
  - Memory length: 4.
  - Inner blocks: 3 layers.
  - Outer blocks: 3 layers.
  - Approximate model size: 15M parameters.

## State Representation

Visible tile features are represented by 22 one-by-34 vectors. The figure OCR identifies the categories:

- Hands
- Discard
- Peng
- Chow
- Kong
- Available tile mask
- Remaining tiles

Game features are represented by 24 scalar entries:

- Prevailing wind x 1
- Seat wind x 4
- Opponent concealed Kong x 3
- Opponent remaining tiles x 4
- Opponent hand tile count x 4
- Current action mask x 8

Hidden/global value features are represented as a `5 x 34` tile matrix:

- Other players' hands
- Remaining wall

## Training Setup

- Supervised learning data: 519,338 Botzone battle logs.
- Loss: cross entropy.
- Offline metrics: action decision accuracy/loss, claiming decision accuracy/loss, and discard decision accuracy/loss.
- The paper discusses testing after 125 supervised epochs, but does not specify a train/validation split, held-out log list, or random seed for the offline metrics.
- Optimizer: Adam.
- Learning rate: `0.0001`.
- Batch size: `1024`.
- Reported supervised learning duration: 7 days on one server with 2 NVIDIA 2080 Ti GPUs.
- PPO policy clip: `0.2`.
- PPO value clip: `0.3`.
- PPO gradient clip: `0.5`.
- Reported RL phase: 3 days.
- PPO implementation uses hierarchical log probability: action-decision log probability plus the claiming/discard tile-head log probability when the selected action requires a tile decision.
- PPO rollout tensors require frozen old policy log probabilities, old value estimates, returns, and advantages. The rollout builder computes these from an SL checkpoint and tensorized fan-backward reward trajectories. Any run using `value_target_terminal_fallback` is a smoke/plumbing run, not a faithful RL replication.
- Paper PPO training now requires rollout metadata with `actual_reward_source=fan_backward_reward`, so terminal-score fallback rollouts are rejected before optimizer steps.

## Fan Backward

The reward shaping computes per-tile scores from the winning hand:

- `score_type[fan.tile] += fan.score`.
- `score_i = 0` if tile count `n_i = 0`, otherwise `score_type_i / n_i`.
- If Tjong wins, current-step reward is:
  `sum_i (hand_hu_i - abs(hand_hu_i - hand_now_i)) * score_i`.
- If Tjong is not the winner, discards that led to winner melds receive negative tile-score rewards.
- If Tjong dianpao, the final winning tile score and prior meld-causing discards are negated.
- For discard wins, the implementation includes the winning tile in the score denominator so the final tile can receive its fan score.
- Faithful reward population prefers explicit per-fan tile attribution and can also derive deterministic structural fan tile sets from Botzone terminal fan displays for supported fan names. Approximate final-hand fan distribution is only available as a smoke-test mode and is not used by the PPO pod manifest.

## Reproducibility Blockers

- The paper does not state `d_model`, number of attention heads, FFN width, dropout, optimizer betas, PPO rollout length, GAE lambda, or exact Botzone preprocessing format.
- The original GitHub URL listed in the paper was not publicly resolvable during this extraction pass. A 2026-06-05 re-check still returned `404 Not Found` for `https://github.com/750251985/Tjong`; the GitHub REST repository endpoint also returned `{"message":"Not Found"}`, local `git ls-remote` retries timed out, and web search still only surfaced the paper's URL reference rather than browsable source.
- The implementation therefore exposes those parameters in `TjongConfig` while preserving all constants that the paper does state. The default `d_model=384`, `n_heads=6`, and `ffn_dim=1536` yields about 16.3M parameters in this scaffold, close to the paper's approximate 15M. A local sweep found a closer 15.27M parameter count at `d_model=372` and `ffn_dim=1488`, but that width would make the attention head dimension 62; it was not used because the paper does not disclose it, and the current `384/6` setting preserves the common 64-dimensional attention head.
- The policy network now supports the paper's two-pass hierarchy explicitly: the action head reads the original encoded state, and the claiming/discard heads can read an action-conditioned sub-state that is passed through the same TIT blocks again.
- Game features are injected as an inner-Transformer token alongside the 22 tile-feature rows, matching the paper text that game features are concatenated with tile information before the Transformer.
- The value network consumes the `5 x 34` hidden/global tile matrix directly, matching the paper text that its input is global features formed by opponents' hand tiles and wall tiles.

## Tensorization Assumptions

- The tile order is `W1..W9`, `T1..T9`, `B1..B9`, `F1..F4`, `J1..J3`, matching the public Botzone/Lawlorentz action layout.
- Chow claim indices use `suit * 21 + (middle_rank - 2) * 3 + (offer_position - 1)`, producing the paper's 63 Chow choices.
- Visible remaining-tile rows are live tile counts from the acting player's visible perspective.
- The hidden `5 x 34` matrix stores three opponent hands, concealed Kong tile counts, and remaining wall tiles. This is the sole tile input to the value network and follows the paper text that hidden information includes opponents' hands, concealed Kong, and wall tiles.
- The tile-decision sub-state shares the historical memory frames with the action-decision state and replaces the current frame's action mask with the selected action family.
- Claim responses with forced discards become two supervised examples: the claim-family decision and the post-claim discard decision. The post-claim discard example uses a replayed post-meld state before the forced discard.
