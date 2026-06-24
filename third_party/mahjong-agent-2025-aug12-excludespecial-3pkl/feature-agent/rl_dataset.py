#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = (
    "obs",
    "vec",
    "mask",
    "action",
    "reward",
    "steps_to_done",
    "seat",
    "terminal_score",
    "terminal_tag",
)


class MCRReplayDataset(Dataset):
    def __init__(
        self,
        replay_dirs: str | Path | list[str | Path],
        *,
        max_shards: int | None = None,
        mmap: bool = True,
        cache_size: int = 4,
        replay_weights: list[float] | None = None,
    ) -> None:
        if isinstance(replay_dirs, (str, Path)):
            replay_dirs = [replay_dirs]
        self.replay_dirs = [Path(item) for item in replay_dirs]
        if replay_weights is not None and len(replay_weights) != len(self.replay_dirs):
            raise ValueError("--replay-weights length must match replay directory count")
        self.replay_weights = replay_weights
        shards: list[Path] = []
        shard_dir_indices: list[int] = []
        for replay_index, replay_dir in enumerate(self.replay_dirs):
            replay_shards = sorted(replay_dir.rglob("shard_*.npz"))
            shards.extend(replay_shards)
            shard_dir_indices.extend([replay_index] * len(replay_shards))
        if max_shards is not None:
            shards = shards[: int(max_shards)]
            shard_dir_indices = shard_dir_indices[: int(max_shards)]
        if not shards:
            raise FileNotFoundError(f"no replay shards found in {self.replay_dirs}")
        self.shards = shards
        self._shard_dir_indices = shard_dir_indices
        self.mmap = mmap
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._lengths: list[int] = []
        for shard in self.shards:
            with np.load(shard, mmap_mode="r" if mmap else None) as data:
                missing = [key for key in REQUIRED_KEYS if key not in data.files]
                if missing:
                    raise ValueError(f"{shard} missing keys: {missing}")
                self._lengths.append(int(data["action"].shape[0]))
        self._offsets = np.cumsum([0, *self._lengths])
        self._dir_lengths = [0 for _ in self.replay_dirs]
        for shard_index, length in enumerate(self._lengths):
            self._dir_lengths[self._shard_dir_indices[shard_index]] += int(length)

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def sample_weights(self) -> torch.Tensor:
        if self.replay_weights is None:
            raise ValueError("replay_weights were not configured")
        values: list[float] = []
        for shard_index, length in enumerate(self._lengths):
            dir_index = self._shard_dir_indices[shard_index]
            dir_length = max(1, self._dir_lengths[dir_index])
            per_sample = float(self.replay_weights[dir_index]) / dir_length
            values.extend([per_sample] * int(length))
        return torch.as_tensor(values, dtype=torch.double)

    def _load_shard(self, shard_index: int):
        if shard_index in self._cache:
            data = self._cache.pop(shard_index)
            self._cache[shard_index] = data
            return data
        data = np.load(self.shards[shard_index], mmap_mode="r" if self.mmap else None)
        self._cache[shard_index] = data
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            old.close()
        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_index = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local_index = int(idx - self._offsets[shard_index])
        data = self._load_shard(shard_index)
        return {
            "obs": torch.as_tensor(np.array(data["obs"][local_index], copy=True), dtype=torch.float32),
            "vec": torch.as_tensor(np.array(data["vec"][local_index], copy=True), dtype=torch.float32),
            "mask": torch.as_tensor(np.array(data["mask"][local_index], copy=True), dtype=torch.bool),
            "action": torch.as_tensor(int(data["action"][local_index]), dtype=torch.long),
            "reward": torch.as_tensor(float(data["reward"][local_index]), dtype=torch.float32),
            "absolute_reward": torch.as_tensor(
                float(data["absolute_reward"][local_index]) if "absolute_reward" in data.files else float(data["reward"][local_index]),
                dtype=torch.float32,
            ),
            "baseline_score": torch.as_tensor(
                float(data["baseline_score"][local_index]) if "baseline_score" in data.files else 0.0,
                dtype=torch.float32,
            ),
            "relative_reward": torch.as_tensor(
                float(data["relative_reward"][local_index]) if "relative_reward" in data.files else float(data["reward"][local_index]),
                dtype=torch.float32,
            ),
            "steps_to_done": torch.as_tensor(float(data["steps_to_done"][local_index]), dtype=torch.float32),
            "seat": torch.as_tensor(int(data["seat"][local_index]), dtype=torch.long),
            "terminal_score": torch.as_tensor(float(data["terminal_score"][local_index]), dtype=torch.float32),
            "terminal_tag": torch.as_tensor(int(data["terminal_tag"][local_index]), dtype=torch.long),
            "initdata_index": torch.as_tensor(
                int(data["initdata_index"][local_index]) if "initdata_index" in data.files else -1,
                dtype=torch.long,
            ),
            "initdata_offset": torch.as_tensor(
                int(data["initdata_offset"][local_index]) if "initdata_offset" in data.files else 0,
                dtype=torch.long,
            ),
        }
