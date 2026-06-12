"""Slide-style CNN/ResNet34 dueling network for an isolated side run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations

import torch
from torch import nn

from .actions import ACTION_NAMES, CLAIM_OFFSETS, CLAIM_SIZE, DISCARD_SIZE
from .encoding import GAME_FEATURES, HIDDEN_TILE_ROWS, TILE_TYPES, VISIBLE_TILE_ROWS
from .tiles import SUIT_ORDER, SUIT_TO_ID, TILE_NAMES, tile_id

SLIDE_GRID_HEIGHT = 4
SLIDE_GRID_WIDTH = 9
SLIDE_V1_CHANNELS = 190
SLIDE_SEARCH_CHANNELS = 30
SLIDE_V2_CHANNELS = SLIDE_V1_CHANNELS + SLIDE_SEARCH_CHANNELS
SLIDE_GRID_SUIT_ROWS = ("T", "W", "B")
SLIDE_SYMMETRY_TRANSFORMS = tuple(
    (suit_permutation, mirror)
    for suit_permutation in permutations(SUIT_ORDER)
    for mirror in (False, True)
)
_TRANSFORM_TENSOR_CACHE: dict[tuple[tuple[int, ...], str, str], torch.Tensor] = {}


@dataclass
class SlideResNetConfig:
    """Config for the slide-replication side model.

    The channel counts follow the attached slides:
    v1 = 190 channels; v2 = v1 plus 30 search/shanten channels.
    """

    feature_version: str = "v1"
    in_channels: int | None = None
    base_channels: int = 64
    head_hidden: int = 512
    dropout: float = 0.1
    action_size: int = len(ACTION_NAMES)
    claim_size: int = CLAIM_SIZE
    discard_size: int = DISCARD_SIZE
    use_hidden_tiles: bool = False
    require_search_features: bool = False

    def resolved_in_channels(self) -> int:
        if self.in_channels is not None:
            return int(self.in_channels)
        version = self.feature_version.lower()
        if version == "v1":
            return SLIDE_V1_CHANNELS
        if version == "v2":
            return SLIDE_V2_CHANNELS
        raise ValueError(f"unknown slide feature version: {self.feature_version!r}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def tile_vectors_to_grid(tile_vectors: torch.Tensor) -> torch.Tensor:
    """Map 34-tile vectors to a 4x9 Mahjong tile grid used by the CNN."""

    if tile_vectors.shape[-1] != TILE_TYPES:
        raise ValueError(f"tile vector must end with {TILE_TYPES}, got {tuple(tile_vectors.shape)}")
    grid = tile_vectors.new_zeros(*tile_vectors.shape[:-1], SLIDE_GRID_HEIGHT, SLIDE_GRID_WIDTH)
    for row, suit in enumerate(SLIDE_GRID_SUIT_ROWS):
        start = SUIT_TO_ID[suit] * 9
        grid[..., row, :] = tile_vectors[..., start : start + 9]
    grid[..., 3, :7] = tile_vectors[..., 27:]
    return grid


def tile_grid_to_vectors(tile_grid: torch.Tensor) -> torch.Tensor:
    if tile_grid.shape[-2:] != (SLIDE_GRID_HEIGHT, SLIDE_GRID_WIDTH):
        raise ValueError(f"tile grid must end with {(SLIDE_GRID_HEIGHT, SLIDE_GRID_WIDTH)}")
    vectors = tile_grid.new_zeros(*tile_grid.shape[:-2], TILE_TYPES)
    for row, suit in enumerate(SLIDE_GRID_SUIT_ROWS):
        start = SUIT_TO_ID[suit] * 9
        vectors[..., start : start + 9] = tile_grid[..., row, :]
    vectors[..., 27:] = tile_grid[..., 3, :7]
    return vectors


def transform_tile_id(index: int, suit_permutation: tuple[str, str, str], *, mirror: bool) -> int:
    tile = TILE_NAMES[int(index)]
    if tile[0] not in SUIT_TO_ID:
        return int(index)
    new_suit = suit_permutation[SUIT_TO_ID[tile[0]]]
    rank = int(tile[1])
    new_rank = 10 - rank if mirror else rank
    return tile_id(f"{new_suit}{new_rank}")


def tile_source_indices_for_transform(
    suit_permutation: tuple[str, str, str],
    *,
    mirror: bool,
) -> list[int]:
    source_for_target = [0] * TILE_TYPES
    for source in range(TILE_TYPES):
        target = transform_tile_id(source, suit_permutation, mirror=mirror)
        source_for_target[target] = source
    return source_for_target


@lru_cache(maxsize=None)
def _tile_source_indices_tuple(
    suit_permutation: tuple[str, str, str],
    mirror: bool,
) -> tuple[int, ...]:
    return tuple(tile_source_indices_for_transform(suit_permutation, mirror=mirror))


@lru_cache(maxsize=None)
def _discard_label_mapping_tuple(
    suit_permutation: tuple[str, str, str],
    mirror: bool,
) -> tuple[int, ...]:
    return tuple(transform_tile_id(index, suit_permutation, mirror=mirror) for index in range(TILE_TYPES))


@lru_cache(maxsize=None)
def _claim_label_mapping_tuple(
    suit_permutation: tuple[str, str, str],
    mirror: bool,
) -> tuple[int, ...]:
    mapping = [0] * CLAIM_SIZE
    for old_suit_id, old_suit in enumerate(SUIT_ORDER):
        for middle_rank in range(2, 9):
            for offer_position in range(1, 4):
                old_index = CLAIM_OFFSETS["CHOW"] + old_suit_id * 21 + (middle_rank - 2) * 3 + (offer_position - 1)
                new_suit = suit_permutation[SUIT_TO_ID[old_suit]]
                new_suit_id = SUIT_TO_ID[new_suit]
                new_middle = 10 - middle_rank if mirror else middle_rank
                new_offer = 4 - offer_position if mirror else offer_position
                mapping[old_index] = CLAIM_OFFSETS["CHOW"] + new_suit_id * 21 + (new_middle - 2) * 3 + (new_offer - 1)
    for family in ("PONG", "MINGKONG", "BUKONG", "ANKONG"):
        offset = CLAIM_OFFSETS[family]
        for local in range(TILE_TYPES):
            mapping[offset + local] = offset + transform_tile_id(local, suit_permutation, mirror=mirror)
    return tuple(mapping)


def _cached_transform_tensor(
    values: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (values, str(device), str(dtype))
    cached = _TRANSFORM_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(values, device=device, dtype=dtype)
        _TRANSFORM_TENSOR_CACHE[key] = cached
    return cached


def transform_tile_tensor(
    tensor: torch.Tensor,
    suit_permutation: tuple[str, str, str],
    *,
    mirror: bool,
) -> torch.Tensor:
    source = _cached_transform_tensor(
        _tile_source_indices_tuple(suit_permutation, mirror),
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(-1, source)


def transform_discard_labels(
    labels: torch.Tensor,
    suit_permutation: tuple[str, str, str],
    *,
    mirror: bool,
) -> torch.Tensor:
    mapping = _cached_transform_tensor(
        _discard_label_mapping_tuple(suit_permutation, mirror),
        device=labels.device,
        dtype=labels.dtype,
    )
    return mapping[labels.long()]


def transform_claim_labels(
    labels: torch.Tensor,
    suit_permutation: tuple[str, str, str],
    *,
    mirror: bool,
) -> torch.Tensor:
    label_mapping = _cached_transform_tensor(
        _claim_label_mapping_tuple(suit_permutation, mirror),
        device=labels.device,
        dtype=labels.dtype,
    )
    return label_mapping[labels.long()]


def _game_features_to_planes(game_features: torch.Tensor) -> torch.Tensor:
    batch, memory_len, feature_count = game_features.shape
    if feature_count != GAME_FEATURES:
        raise ValueError(f"game_features must end with {GAME_FEATURES}, got {feature_count}")
    return game_features.reshape(batch, memory_len * feature_count, 1, 1).expand(
        -1,
        -1,
        SLIDE_GRID_HEIGHT,
        SLIDE_GRID_WIDTH,
    )


def _search_features_to_planes(
    search_features: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    require_search_features: bool,
) -> torch.Tensor:
    if search_features is None:
        if require_search_features:
            raise ValueError("v2 slide features require 30 search feature planes")
        return torch.zeros(
            batch_size,
            SLIDE_SEARCH_CHANNELS,
            SLIDE_GRID_HEIGHT,
            SLIDE_GRID_WIDTH,
            device=device,
            dtype=dtype,
        )
    if search_features.ndim == 3 and search_features.shape[1:] == (SLIDE_SEARCH_CHANNELS, TILE_TYPES):
        return tile_vectors_to_grid(search_features)
    if search_features.ndim == 4 and search_features.shape[1:] == (
        SLIDE_SEARCH_CHANNELS,
        SLIDE_GRID_HEIGHT,
        SLIDE_GRID_WIDTH,
    ):
        return search_features
    raise ValueError(
        "search_features must have shape "
        f"[batch, {SLIDE_SEARCH_CHANNELS}, {TILE_TYPES}] or "
        f"[batch, {SLIDE_SEARCH_CHANNELS}, {SLIDE_GRID_HEIGHT}, {SLIDE_GRID_WIDTH}]"
    )


def build_slide_feature_planes(
    *,
    visible_tiles: torch.Tensor,
    game_features: torch.Tensor,
    hidden_tiles: torch.Tensor | None = None,
    search_features: torch.Tensor | None = None,
    feature_version: str = "v1",
    use_hidden_tiles: bool = False,
    require_search_features: bool = False,
) -> torch.Tensor:
    """Build the 190/220-channel slide feature tensor from validated Tjong tensors."""

    if visible_tiles.ndim != 4:
        raise ValueError("visible_tiles must have shape [batch, memory_len, 22, 34]")
    batch, memory_len, visible_rows, tile_count = visible_tiles.shape
    if visible_rows != VISIBLE_TILE_ROWS or tile_count != TILE_TYPES:
        raise ValueError(f"visible_tiles must end with {(VISIBLE_TILE_ROWS, TILE_TYPES)}")
    if game_features.shape != (batch, memory_len, GAME_FEATURES):
        raise ValueError("game_features must align with visible_tiles and end with 24 features")

    visible_planes = tile_vectors_to_grid(visible_tiles.reshape(batch, memory_len * visible_rows, TILE_TYPES))
    game_planes = _game_features_to_planes(game_features)
    if use_hidden_tiles and hidden_tiles is not None:
        if hidden_tiles.shape != (batch, HIDDEN_TILE_ROWS, TILE_TYPES):
            raise ValueError(f"hidden_tiles must have shape [batch, {HIDDEN_TILE_ROWS}, {TILE_TYPES}]")
        hidden_planes = tile_vectors_to_grid(hidden_tiles)
    else:
        hidden_planes = visible_tiles.new_zeros(batch, HIDDEN_TILE_ROWS, SLIDE_GRID_HEIGHT, SLIDE_GRID_WIDTH)
    bias_plane = visible_tiles.new_ones(batch, 1, SLIDE_GRID_HEIGHT, SLIDE_GRID_WIDTH)
    v1 = torch.cat((visible_planes, game_planes, hidden_planes, bias_plane), dim=1)
    if v1.shape[1] != SLIDE_V1_CHANNELS:
        raise AssertionError(f"v1 channel construction produced {v1.shape[1]} channels")

    version = feature_version.lower()
    if version == "v1":
        return v1
    if version == "v2":
        search_planes = _search_features_to_planes(
            search_features,
            batch_size=batch,
            device=visible_tiles.device,
            dtype=visible_tiles.dtype,
            require_search_features=require_search_features,
        )
        return torch.cat((v1, search_planes.to(device=visible_tiles.device, dtype=visible_tiles.dtype)), dim=1)
    raise ValueError(f"unknown slide feature version: {feature_version!r}")


class MixedKernelInputLayer(nn.Module):
    """Parallel 3x3 and 1x1 input convolutions from the slides."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        three_by_three = out_channels // 2
        one_by_one = out_channels - three_by_three
        self.conv3 = nn.Conv2d(in_channels, three_by_three, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, one_by_one, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat((self.conv3(features), self.conv1(features)), dim=1)
        return self.activation(self.norm(mixed))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class DuelingHead(nn.Module):
    """Value plus mean-centered advantage head: Q(s,a) = V(s) + A(s,a) - mean(A)."""

    def __init__(self, in_features: int, out_features: int, hidden_features: int, dropout: float):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class SlideMahjongResNetDueling(nn.Module):
    """ResNet34-like slide architecture with hierarchical dueling outputs."""

    def __init__(self, config: SlideResNetConfig | None = None):
        super().__init__()
        self.config = config or SlideResNetConfig()
        self.in_channels = self.config.resolved_in_channels()
        base = int(self.config.base_channels)
        self.stem = MixedKernelInputLayer(self.in_channels, base)
        self.inplanes = base
        self.layer1 = self._make_layer(base, blocks=3, stride=1)
        self.layer2 = self._make_layer(base * 2, blocks=4, stride=2)
        self.layer3 = self._make_layer(base * 4, blocks=6, stride=2)
        self.layer4 = self._make_layer(base * 8, blocks=3, stride=2)
        final_channels = base * 8
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.action_head = DuelingHead(final_channels, self.config.action_size, self.config.head_hidden, self.config.dropout)
        self.claim_head = DuelingHead(final_channels, self.config.claim_size, self.config.head_hidden, self.config.dropout)
        self.discard_head = DuelingHead(final_channels, self.config.discard_size, self.config.head_hidden, self.config.dropout)

    def _make_layer(self, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock(self.inplanes, out_channels, stride=stride)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _encode(
        self,
        *,
        visible_tiles: torch.Tensor,
        game_features: torch.Tensor,
        hidden_tiles: torch.Tensor | None,
        search_features: torch.Tensor | None,
    ) -> torch.Tensor:
        features = build_slide_feature_planes(
            visible_tiles=visible_tiles,
            game_features=game_features,
            hidden_tiles=hidden_tiles,
            search_features=search_features,
            feature_version=self.config.feature_version,
            use_hidden_tiles=self.config.use_hidden_tiles,
            require_search_features=self.config.require_search_features,
        )
        if features.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} feature channels, got {features.shape[1]}")
        x = self.stem(features)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)

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
        search_features: torch.Tensor | None = None,
        sub_search_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del rewards, previous_actions, sub_rewards, sub_previous_actions
        policy_state = self._encode(
            visible_tiles=visible_tiles,
            game_features=game_features,
            hidden_tiles=hidden_tiles,
            search_features=search_features,
        )
        if sub_visible_tiles is None and sub_game_features is None:
            tile_state = policy_state
        else:
            if sub_visible_tiles is None or sub_game_features is None:
                raise ValueError("sub_visible_tiles and sub_game_features must be provided together")
            tile_state = self._encode(
                visible_tiles=sub_visible_tiles,
                game_features=sub_game_features,
                hidden_tiles=hidden_tiles,
                search_features=sub_search_features if sub_search_features is not None else search_features,
            )
        return {
            "action_logits": self.action_head(policy_state),
            "claim_logits": self.claim_head(tile_state),
            "discard_logits": self.discard_head(tile_state),
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
