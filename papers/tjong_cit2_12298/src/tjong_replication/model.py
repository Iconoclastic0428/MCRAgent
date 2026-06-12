"""Tjong network scaffold following the paper architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .actions import ACTION_NAMES, CLAIM_SIZE, DISCARD_SIZE
from .encoding import GAME_FEATURES, HIDDEN_TILE_ROWS, TILE_TYPES, VISIBLE_TILE_ROWS


@dataclass
class TjongConfig:
    tile_rows: int = VISIBLE_TILE_ROWS
    hidden_tile_rows: int = HIDDEN_TILE_ROWS
    tile_types: int = TILE_TYPES
    game_features: int = GAME_FEATURES
    action_size: int = len(ACTION_NAMES)
    claim_size: int = CLAIM_SIZE
    discard_size: int = DISCARD_SIZE
    memory_len: int = 4
    d_model: int = 384
    n_heads: int = 6
    inner_layers: int = 3
    outer_layers: int = 3
    ffn_dim: int = 1536
    dropout: float = 0.1
    action_embedding_size: int = len(ACTION_NAMES)


class InnerTransformer(nn.Module):
    """Inner TIT block over one Mahjong feature image and optional game token."""

    def __init__(self, rows: int, config: TjongConfig, *, game_feature_dim: int = 0):
        super().__init__()
        self.rows = int(rows)
        self.config = config
        self.game_feature_dim = int(game_feature_dim)
        self.row_projection = nn.Linear(config.tile_types, config.d_model)
        self.game_projection = (
            nn.Linear(self.game_feature_dim, config.d_model) if self.game_feature_dim > 0 else None
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.rows + 1 + (1 if self.game_projection is not None else 0), config.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.inner_layers)
        self.norm = nn.LayerNorm(config.d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, tile_features: torch.Tensor, game_features: torch.Tensor | None = None) -> torch.Tensor:
        if tile_features.shape[-2:] != (self.rows, self.config.tile_types):
            raise ValueError(
                f"tile features must end with {(self.rows, self.config.tile_types)}, got {tuple(tile_features.shape)}"
            )
        flat = tile_features.reshape(-1, self.rows, self.config.tile_types).float()
        tokens = self.row_projection(flat)
        if self.game_projection is not None:
            if game_features is None:
                raise ValueError("game_features must be provided for this inner transformer")
            if game_features.shape[-1] != self.game_feature_dim:
                raise ValueError(f"game_features must end with {self.game_feature_dim}")
            game = self.game_projection(game_features.reshape(-1, self.game_feature_dim).float()).unsqueeze(1)
            tokens = torch.cat([tokens, game], dim=1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embedding
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0]).reshape(*tile_features.shape[:-2], self.config.d_model)


class TITPolicyBackbone(nn.Module):
    """Policy TIT backbone with inner observation encoder and outer memory encoder."""

    def __init__(self, config: TjongConfig):
        super().__init__()
        self.config = config
        self.inner = InnerTransformer(config.tile_rows, config, game_feature_dim=config.game_features)
        self.reward_projection = nn.Linear(1, config.d_model)
        self.action_embedding = nn.Embedding(config.action_embedding_size, config.d_model)
        self.memory_pos_embedding = nn.Parameter(torch.zeros(1, config.memory_len, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.outer = nn.TransformerEncoder(layer, num_layers=config.outer_layers)
        self.norm = nn.LayerNorm(config.d_model)
        nn.init.trunc_normal_(self.memory_pos_embedding, std=0.02)

    def forward(
        self,
        visible_tiles: torch.Tensor,
        game_features: torch.Tensor,
        rewards: torch.Tensor | None = None,
        previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visible_tiles.ndim != 4:
            raise ValueError("visible_tiles must have shape [batch, memory_len, 22, 34]")
        batch, memory_len = visible_tiles.shape[:2]
        if memory_len != self.config.memory_len:
            raise ValueError(f"memory length must be {self.config.memory_len}")
        if game_features.shape[:2] != (batch, memory_len):
            raise ValueError("game_features must align with visible_tiles batch/memory dimensions")
        if game_features.shape[-1] != self.config.game_features:
            raise ValueError(f"game_features must end with {self.config.game_features}")
        if rewards is None:
            rewards = torch.zeros(batch, memory_len, device=visible_tiles.device, dtype=visible_tiles.dtype)
        if previous_actions is None:
            previous_actions = torch.zeros(batch, memory_len, device=visible_tiles.device, dtype=torch.long)

        obs = self.inner(visible_tiles, game_features)
        rew = self.reward_projection(rewards.float().unsqueeze(-1))
        act = self.action_embedding(previous_actions.long().clamp_min(0).clamp_max(self.config.action_embedding_size - 1))
        tokens = obs + rew + act + self.memory_pos_embedding
        causal_mask = torch.triu(
            torch.full((memory_len, memory_len), float("-inf"), device=visible_tiles.device),
            diagonal=1,
        )
        encoded = self.outer(tokens, mask=causal_mask)
        return self.norm(encoded[:, -1])


class TjongNetwork(nn.Module):
    """Hierarchical Tjong policy plus global-information value network."""

    def __init__(self, config: TjongConfig | None = None):
        super().__init__()
        self.config = config or TjongConfig()
        self.reuse_identical_substate = False
        self.policy_backbone = TITPolicyBackbone(self.config)
        self.value_inner = InnerTransformer(self.config.hidden_tile_rows, self.config)
        self.action_head = nn.Linear(self.config.d_model, self.config.action_size)
        self.claim_head = nn.Linear(self.config.d_model, self.config.claim_size)
        self.discard_head = nn.Linear(self.config.d_model, self.config.discard_size)
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, 1),
        )

    def forward(
        self,
        *,
        visible_tiles: torch.Tensor,
        game_features: torch.Tensor,
        rewards: torch.Tensor | None = None,
        previous_actions: torch.Tensor | None = None,
        sub_visible_tiles: torch.Tensor | None = None,
        sub_game_features: torch.Tensor | None = None,
        sub_rewards: torch.Tensor | None = None,
        sub_previous_actions: torch.Tensor | None = None,
        hidden_tiles: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        policy_state = self.policy_backbone(
            visible_tiles=visible_tiles,
            game_features=game_features,
            rewards=rewards,
            previous_actions=previous_actions,
        )
        if sub_visible_tiles is None and sub_game_features is None:
            tile_decision_state = policy_state
        else:
            if sub_visible_tiles is None or sub_game_features is None:
                raise ValueError("sub_visible_tiles and sub_game_features must be provided together")
            base_rewards = (
                rewards
                if rewards is not None
                else torch.zeros(visible_tiles.shape[:2], device=visible_tiles.device, dtype=visible_tiles.dtype)
            )
            base_previous_actions = (
                previous_actions
                if previous_actions is not None
                else torch.zeros(visible_tiles.shape[:2], device=visible_tiles.device, dtype=torch.long)
            )
            sub_rewards_input = base_rewards if sub_rewards is None else sub_rewards
            sub_previous_actions_input = base_previous_actions if sub_previous_actions is None else sub_previous_actions
            if bool(getattr(self, "reuse_identical_substate", False)):
                same_substate = _same_sequence_rows(visible_tiles, sub_visible_tiles)
                same_substate &= _same_sequence_rows(game_features, sub_game_features)
                same_substate &= _same_sequence_rows(base_rewards, sub_rewards_input)
                same_substate &= _same_sequence_rows(base_previous_actions, sub_previous_actions_input)
                if bool(same_substate.all().item()):
                    tile_decision_state = policy_state
                elif bool(same_substate.any().item()):
                    changed = ~same_substate
                    tile_decision_state = policy_state.new_empty(policy_state.shape)
                    tile_decision_state[same_substate] = policy_state[same_substate]
                    tile_decision_state[changed] = self.policy_backbone(
                        visible_tiles=sub_visible_tiles[changed],
                        game_features=sub_game_features[changed],
                        rewards=sub_rewards_input[changed],
                        previous_actions=sub_previous_actions_input[changed],
                    )
                else:
                    tile_decision_state = self.policy_backbone(
                        visible_tiles=sub_visible_tiles,
                        game_features=sub_game_features,
                        rewards=sub_rewards_input,
                        previous_actions=sub_previous_actions_input,
                    )
            else:
                tile_decision_state = self.policy_backbone(
                    visible_tiles=sub_visible_tiles,
                    game_features=sub_game_features,
                    rewards=sub_rewards_input,
                    previous_actions=sub_previous_actions_input,
                )
        output = {
            "action_logits": self.action_head(policy_state),
            "claim_logits": self.claim_head(tile_decision_state),
            "discard_logits": self.discard_head(tile_decision_state),
        }
        if hidden_tiles is not None:
            value_state = self.value_inner(hidden_tiles.float())
            output["value"] = self.value_head(value_state).squeeze(-1)
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _same_sequence_rows(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor:
    if left is None or right is None:
        tensor = left if left is not None else right
        assert tensor is not None
        return torch.zeros(tensor.shape[0], dtype=torch.bool, device=tensor.device)
    if left.shape != right.shape:
        return torch.zeros(left.shape[0], dtype=torch.bool, device=left.device)
    return (left == right).reshape(left.shape[0], -1).all(dim=1)


def masked_argmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.argmax(logits, dim=-1)
    masked = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
    return torch.argmax(masked, dim=-1)
