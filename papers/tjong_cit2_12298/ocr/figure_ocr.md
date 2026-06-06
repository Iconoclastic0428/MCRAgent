# Figure OCR Transcript

This file records the OCR/visual transcript from rendered PDF pages accessed through the web reader. Local binary PDF/image downloads were blocked by Wiley/ResearchGate security responses, so the OCR pass was performed from rendered page images plus the PDF text layer.

## Figure 1 - TIT Network Structure

Detected labels:

- `H x W x C`, `22 x 34 x 1`
- `o`, `z0`
- `Inner Block`
- `Inner Block x L`
- Historical triplets:
  - `R_{t-1}`
  - `o_{t-1}`
  - `a_{t-1}`
  - `R_t`
  - `o_t`
  - `a_t`
- `Outer Block`
- `Outer Block x L`
- `FFN`
- `a_t`

Interpretation used in code:

- The visible Mahjong state is encoded as a 22-row by 34-column tile-feature image.
- The inner Transformer turns each per-step observation into a class-token embedding.
- The outer Transformer consumes short-term memory of per-step observation embeddings, action embeddings, and rewards.

## Figure 2 - Encoding of Mahjong State

Detected labels:

- Mahjong tiles shown above a length-34 count vector.
- Counts below tile types show values such as `0` and `1`.

Interpretation used in code:

- A tile multiset is represented as a 34-length vector.
- Each tile column stores the count of that tile type.
- Flower tiles are excluded in duplicate competition encoding.

## Figure 3 - State Representation

Detected labels:

- Tile features:
  - `Hands`
  - `Discard`
  - `Peng`
  - `Chow`
  - `Kong`
  - `Available tile mask`
  - `Remaining tiles`
- Game features:
  - `Prevailing Wind`
  - `Dealer's Wind`
  - `Remaining tiles number`
  - `Concealed Kong`
  - `Hand free number`
  - `Available action mask`
- Global features:
  - `Other's hands`
  - `Remaining wall`
- Outputs:
  - `Policy`
  - `Value`

Interpretation used in code:

- Policy network uses visible tile and game features.
- Value network additionally uses hidden/global features.

## Figure 4 - Multiple Decision Options

Detected labels:

- `multiple possible actions and tiles`
- Action options: `Pong`, `Chow`, `Pass`.
- Scenario shows previous player discarding Five of Characters and the acting player having multiple legal choices.

Interpretation used in code:

- The model must first compare action families and then choose the tile/claim within the chosen family.

## Figure 5 - Hierarchical Decision-Making Model

Detected labels:

- `Observation`
- `Tile features`, `22 x 34`
- `Game features`, `1 x 24`
- `S_t`
- `Transformer blocks`
- `Encoder`
- `Actions`
- Action list:
  - `Chow`
  - `Pong`
  - `Kong`
  - `BuKong`
  - `AnKong`
  - `Discard`
  - `Pass`
  - `Hu`
- `S_t_sub`
- `Tile decision`
- `Claiming tile`
- Claim options:
  - `Chow`
  - `Pong`
  - `MingKong`
  - `BuKong`
  - `AnKong`
- `Discard tile`
- `probability of tiles`

Interpretation used in code:

- The action head is 8-way.
- The claiming head covers the five claim families.
- The discard head covers the 34 tile types.

## Figure 6 - Structure of Tjong Network

Detected from page text around the figure:

- Model has a policy network and a value network.
- Policy network takes tile features and game features through Transformer blocks.
- Inner-block output plus stored memory goes to outer blocks.
- Class token output goes to the Action head.
- If action is a claim, the state is modified and passed through Transformer blocks again.
- Claiming head or Discard head determines the final action/tile.
- Memory length is 4.
- Inner and outer blocks each have 3 layers.
- Total network has about 15M parameters.

## Figure 7 - Fan Backward Example

Detected from page text around the figure:

- Example total: 26 fan.
- Components:
  - Three Concealed Pongs: 16 fan.
  - Two Dragon Pongs: 6 fan.
  - Fully Concealed Hand: 4 fan.
- Two Dragon Pongs assign two points each to red dragon and green dragon tiles.
- Three Concealed Pongs assigns `16/3` points each to red dragon, green dragon, and six of bamboo.
- Fully Concealed Hand distributes `4/7` points over seven tile types.

Interpretation used in code:

- Fan scores are decomposed into a 34-tile score vector, then used to reward or penalize each decision.
