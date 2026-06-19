#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import itertools
import json
import os
import shutil
from bisect import bisect_right
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


SUIT_PERMUTATIONS = list(itertools.permutations(range(3)))
TRANSFORMS = [
    (suit_perm, mirror_rank)
    for suit_perm in SUIT_PERMUTATIONS
    for mirror_rank in (False, True)
]


def _tile_old_to_new(suit_perm, mirror_rank):
    old_to_new = np.arange(36, dtype=np.int64)
    for old_tile in range(27):
        old_suit = old_tile // 9
        old_rank = old_tile % 9 + 1
        new_suit = suit_perm[old_suit]
        new_rank = 10 - old_rank if mirror_rank else old_rank
        old_to_new[old_tile] = new_suit * 9 + new_rank - 1
    return old_to_new


def _inverse_permutation(old_to_new):
    inverse = np.empty_like(old_to_new)
    inverse[old_to_new] = np.arange(len(old_to_new), dtype=old_to_new.dtype)
    return inverse


def _action_old_to_new(tile_old_to_new, suit_perm, mirror_rank):
    old_to_new = np.arange(235, dtype=np.int64)

    for offset in (2, 36, 70, 104, 138):
        for old_tile in range(34):
            old_to_new[offset + old_tile] = offset + tile_old_to_new[old_tile]

    for old_chi in range(63):
        old_suit = old_chi // 21
        rem = old_chi % 21
        old_center = rem // 3 + 2
        old_pos = rem % 3

        new_suit = suit_perm[old_suit]
        new_center = 10 - old_center if mirror_rank else old_center
        new_pos = 2 - old_pos if mirror_rank else old_pos
        new_chi = new_suit * 21 + (new_center - 2) * 3 + new_pos
        old_to_new[172 + old_chi] = 172 + new_chi

    return old_to_new


def _vec_old_to_new(tile_old_to_new):
    old_to_new = np.arange(117, dtype=np.int64)
    for offset in (13, 48, 83):
        for old_tile in range(34):
            old_to_new[offset + old_tile] = offset + tile_old_to_new[old_tile]
    return old_to_new


def _build_transform_tables():
    tables = []
    for suit_perm, mirror_rank in TRANSFORMS:
        tile_old_to_new = _tile_old_to_new(suit_perm, mirror_rank)
        action_old_to_new = _action_old_to_new(tile_old_to_new, suit_perm, mirror_rank)
        vec_old_to_new = _vec_old_to_new(tile_old_to_new)
        tables.append(
            {
                "tile_inverse": _inverse_permutation(tile_old_to_new),
                "action_inverse": _inverse_permutation(action_old_to_new),
                "action_old_to_new": action_old_to_new,
                "vec_inverse": _inverse_permutation(vec_old_to_new),
                "mirror_rank": mirror_rank,
            }
        )
    return tables


TRANSFORM_TABLES = _build_transform_tables()


def _build_static_obs_planes():
    tile_list = [
        *("W%d" % (i + 1) for i in range(9)),
        *("T%d" % (i + 1) for i in range(9)),
        *("B%d" % (i + 1) for i in range(9)),
        *("F%d" % (i + 1) for i in range(4)),
        *("J%d" % (i + 1) for i in range(3)),
    ]
    offset_tile = {tile: i for i, tile in enumerate(tile_list)}
    planes = np.zeros((25, 36), dtype=np.int8)
    honor_list = "F1 F2 F3 F4 J1 J2 J3".split()
    yaojiu_list = "W1 W9 B1 B9 T1 T9 F1 F2 F3 F4 J1 J2 J3".split()
    zipai_list = "F1 F2 F3 F4 J1 J2 J3".split()
    tuibudao_list = "B1 B2 B3 B4 B5 B8 B9 T2 T4 T5 T6 T8 T9 J3".split()
    lvyise_list = "T2 T3 T4 T6 T8 J2".split()

    for tile in tile_list:
        tile_id = offset_tile[tile]
        if tile[0] in "FJ":
            dianshu_offset = 9 + honor_list.index(tile)
        else:
            dianshu_offset = int(tile[1]) - 1
        planes[dianshu_offset, tile_id] = 1
        planes[16 + "WTBFJ".index(tile[0]), tile_id] = 1
    for tile in yaojiu_list:
        planes[21, offset_tile[tile]] = 1
    for tile in zipai_list:
        planes[22, offset_tile[tile]] = 1
    for tile in tuibudao_list:
        planes[23, offset_tile[tile]] = 1
    for tile in lvyise_list:
        planes[24, offset_tile[tile]] = 1
    return planes.reshape((25, 4, 9))


STATIC_OBS_PLANES = _build_static_obs_planes()


def _load_special_matches(path):
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        values = payload
    else:
        values = payload.get("special_match_indices", [])
    return {int(value) for value in values}


class MahjongGBDataset(Dataset):
    def __init__(
        self,
        folder="./data-vec",
        begin=0,
        end=1,
        tqdm_disable=0,
        augment=False,
        lazy=False,
        augment_mode="none",
        special_matches_path="",
        exclude_special_matches=False,
        fan_features_folder="",
        fan_shanten_replace_folder="",
        match_indices=None,
    ):
        with open(os.path.join(folder, "count.json"), "r", encoding="utf8") as f:
            all_match_samples = json.load(f)

        self.total_matches = len(all_match_samples)
        self.total_samples = sum(all_match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        if match_indices is None:
            self.global_match_ids = list(range(self.begin, self.end))
        else:
            self.global_match_ids = [int(match_id) for match_id in match_indices]
            self.begin = 0
            self.end = len(self.global_match_ids)
        self.base_match_samples = [all_match_samples[i] for i in self.global_match_ids]
        self.base_matches = len(self.base_match_samples)
        self.base_samples = sum(self.base_match_samples)
        self.folder = folder
        self.lazy = lazy
        self.fan_features_folder = fan_features_folder
        self.fan_shanten_replace_folder = fan_shanten_replace_folder
        self.fan_feature_dim = 0
        self.fan_feature_names = []
        self._fan_swap_pairs = []

        if augment and augment_mode == "none":
            augment_mode = "random_suit"
        self.augment_mode = augment_mode
        self.special_match_indices = _load_special_matches(special_matches_path)
        self.exclude_special_matches = exclude_special_matches
        if fan_features_folder:
            meta_path = os.path.join(fan_features_folder, "meta.json")
            with open(meta_path, "r", encoding="utf8") as f:
                meta = json.load(f)
            self.fan_feature_names = list(meta["feature_names"])
            self.fan_feature_dim = int(meta["feature_dim"])
            self._fan_swap_pairs = [
                (self.fan_feature_names.index(a), self.fan_feature_names.index(b))
                for a, b in (("dayu5", "xiaoyu5"),)
                if a in self.fan_feature_names and b in self.fan_feature_names
            ]
        if fan_shanten_replace_folder:
            meta_path = os.path.join(fan_shanten_replace_folder, "meta.json")
            with open(meta_path, "r", encoding="utf8") as f:
                meta = json.load(f)
            if int(meta["vec_dim"]) != 117:
                raise ValueError(meta)
        self.vec_size = 117 + self.fan_feature_dim

        self.expanded_matches = []
        self.match_offsets = []
        self.excluded_special_matches = 0
        self.excluded_special_samples = 0
        total = 0
        for local_match_id, sample_count in enumerate(self.base_match_samples):
            global_match_id = self.global_match_ids[local_match_id]
            if self.exclude_special_matches and global_match_id in self.special_match_indices:
                self.excluded_special_matches += 1
                self.excluded_special_samples += sample_count
                continue
            if augment_mode == "all12" and global_match_id not in self.special_match_indices:
                transform_ids = range(len(TRANSFORM_TABLES))
            else:
                transform_ids = (0,)
            for transform_id in transform_ids:
                self.match_offsets.append(total)
                self.expanded_matches.append((local_match_id, transform_id))
                total += sample_count

        self.matches = len(self.expanded_matches)
        self.samples = total
        self.cache = {"obs": [], "mask": [], "vec": [], "act": [], "fan_vec": []}

        if self.lazy:
            return

        for local_match_id in tqdm(
            range(self.base_matches),
            desc="loading data".ljust(20),
            bar_format="{l_bar}{bar:40}{r_bar}",
            disable=bool(tqdm_disable),
        ):
            global_match_id = self.global_match_ids[local_match_id]
            with np.load("%s/%d.npz" % (self.folder, global_match_id)) as d:
                for key in ("obs", "mask", "vec", "act"):
                    self.cache[key].append(d[key])
            if self.fan_features_folder:
                with np.load(
                    "%s/%d.npz" % (self.fan_features_folder, global_match_id)
                ) as d:
                    self.cache["fan_vec"].append(d["fan_vec"])
            if self.fan_shanten_replace_folder:
                with np.load(
                    "%s/%d.npz" % (self.fan_shanten_replace_folder, global_match_id)
                ) as d:
                    self.cache["vec"][-1] = d["vec"]

    def __len__(self):
        return self.samples

    def _load_sample(self, local_match_id, sample_id):
        global_match_id = self.global_match_ids[local_match_id]
        if self.lazy:
            with np.load("%s/%d.npz" % (self.folder, global_match_id)) as d:
                obs = d["obs"][sample_id]
                mask = d["mask"][sample_id]
                vec = d["vec"][sample_id]
                act = d["act"][sample_id]
            fan_vec = None
            if self.fan_features_folder:
                with np.load(
                    "%s/%d.npz" % (self.fan_features_folder, global_match_id)
                ) as d:
                    fan_vec = d["fan_vec"][sample_id]
            if self.fan_shanten_replace_folder:
                with np.load(
                    "%s/%d.npz" % (self.fan_shanten_replace_folder, global_match_id)
                ) as d:
                    vec = d["vec"][sample_id]
            return obs, mask, vec, act, fan_vec
        fan_vec = None
        if self.fan_features_folder:
            fan_vec = self.cache["fan_vec"][local_match_id][sample_id]
        return (
            self.cache["obs"][local_match_id][sample_id],
            self.cache["mask"][local_match_id][sample_id],
            self.cache["vec"][local_match_id][sample_id],
            self.cache["act"][local_match_id][sample_id],
            fan_vec,
        )

    def _random_transform_id(self, local_match_id):
        global_match_id = self.global_match_ids[local_match_id]
        if global_match_id in self.special_match_indices:
            return 0
        # Preserve the old random suit-only behavior for callers that use augment=True.
        return int(np.random.randint(0, len(SUIT_PERMUTATIONS))) * 2

    def _apply_transform(self, obs, mask, vec, act, transform_id):
        table = TRANSFORM_TABLES[transform_id]
        obs_flat = obs.reshape((obs.shape[0], 36))
        transformed_obs = obs_flat[:, table["tile_inverse"]].reshape(obs.shape).copy()
        transformed_obs[60:85] = STATIC_OBS_PLANES
        transformed_mask = mask[table["action_inverse"]].copy()
        transformed_vec = vec[table["vec_inverse"]].copy()
        transformed_act = int(table["action_old_to_new"][int(act)])
        return transformed_obs, transformed_mask, transformed_vec, transformed_act

    def _transform_fan_vec(self, fan_vec, transform_id):
        if fan_vec is None:
            return None
        transformed = fan_vec.astype(np.float16, copy=True)
        if TRANSFORM_TABLES[transform_id]["mirror_rank"]:
            for a, b in self._fan_swap_pairs:
                transformed[a], transformed[b] = transformed[b], transformed[a]
        return transformed

    def _append_fan_vec(self, vec, fan_vec):
        if fan_vec is None:
            return vec
        return np.concatenate([vec, fan_vec]).astype(np.float16, copy=False)

    def __getitem__(self, index):
        if index < 0:
            index += self.samples
        if index < 0 or index >= self.samples:
            raise IndexError(index)

        expanded_match_id = bisect_right(self.match_offsets, index, 0, self.matches) - 1
        local_match_id, transform_id = self.expanded_matches[expanded_match_id]
        sample_id = index - self.match_offsets[expanded_match_id]

        obs, mask, vec, act, fan_vec = self._load_sample(local_match_id, sample_id)
        if self.augment_mode == "random_suit":
            transform_id = self._random_transform_id(local_match_id)
        if transform_id:
            obs, mask, vec, act = self._apply_transform(obs, mask, vec, act, transform_id)
        fan_vec = self._transform_fan_vec(fan_vec, transform_id)
        vec = self._append_fan_vec(vec, fan_vec)
        return obs, mask, vec, act


def _packed_meta_path(folder):
    return Path(folder) / "meta.json"


def _packed_arrays_exist(folder):
    folder = Path(folder)
    return all((folder / name).exists() for name in ("obs.npy", "mask.npy", "vec.npy", "act.npy", "meta.json"))


def _load_npz_arrays(folder, match_id, fan_shanten_replace_folder=""):
    with np.load("%s/%d.npz" % (folder, match_id)) as d:
        obs = d["obs"]
        mask = d["mask"]
        vec = d["vec"]
        act = d["act"]
    if fan_shanten_replace_folder:
        with np.load("%s/%d.npz" % (fan_shanten_replace_folder, match_id)) as d:
            vec = d["vec"]
    return obs, mask, vec, act


def _read_count(folder):
    with open(os.path.join(folder, "count.json"), "r", encoding="utf8") as f:
        return json.load(f)


def _validate_packed_metadata(cache_dir, expected):
    meta_path = _packed_meta_path(cache_dir)
    if not meta_path.exists():
        return False
    with open(meta_path, "r", encoding="utf8") as f:
        actual = json.load(f)
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def build_packed_dataset(
    source_folder,
    cache_dir,
    match_indices,
    split_name,
    special_matches_path="",
    exclude_special_matches=False,
    fan_shanten_replace_folder="",
    force=False,
):
    """Pack many per-match npz files into reusable mmap-backed .npy arrays."""

    if special_matches_path:
        special_match_indices = _load_special_matches(special_matches_path)
    else:
        special_match_indices = set()
    counts = _read_count(source_folder)
    selected_match_ids = []
    excluded_special_matches = 0
    excluded_special_samples = 0
    for match_id in [int(value) for value in match_indices]:
        if exclude_special_matches and match_id in special_match_indices:
            excluded_special_matches += 1
            excluded_special_samples += int(counts[match_id])
            continue
        selected_match_ids.append(match_id)
    total_samples = int(sum(counts[match_id] for match_id in selected_match_ids))

    expected_meta = {
        "format": "mahjong-agent-packed-v1",
        "source_folder": os.path.abspath(source_folder),
        "split_name": split_name,
        "match_count": len(selected_match_ids),
        "sample_count": total_samples,
        "first_match_id": None if not selected_match_ids else selected_match_ids[0],
        "last_match_id": None if not selected_match_ids else selected_match_ids[-1],
        "exclude_special_matches": bool(exclude_special_matches),
        "excluded_special_matches": excluded_special_matches,
        "excluded_special_samples": excluded_special_samples,
        "fan_shanten_replace_folder": os.path.abspath(fan_shanten_replace_folder)
        if fan_shanten_replace_folder
        else "",
    }
    cache_dir = Path(cache_dir)
    if not force and _packed_arrays_exist(cache_dir) and _validate_packed_metadata(cache_dir, expected_meta):
        return str(cache_dir)

    if total_samples <= 0:
        raise RuntimeError(f"{split_name} packed dataset would be empty")

    first_match = selected_match_ids[0]
    sample_obs, sample_mask, sample_vec, sample_act = _load_npz_arrays(
        source_folder, first_match, fan_shanten_replace_folder
    )
    tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp-%d" % os.getpid())
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    obs_out = np.lib.format.open_memmap(
        tmp_dir / "obs.npy",
        mode="w+",
        dtype=sample_obs.dtype,
        shape=(total_samples,) + tuple(sample_obs.shape[1:]),
    )
    mask_out = np.lib.format.open_memmap(
        tmp_dir / "mask.npy",
        mode="w+",
        dtype=sample_mask.dtype,
        shape=(total_samples,) + tuple(sample_mask.shape[1:]),
    )
    vec_out = np.lib.format.open_memmap(
        tmp_dir / "vec.npy",
        mode="w+",
        dtype=sample_vec.dtype,
        shape=(total_samples,) + tuple(sample_vec.shape[1:]),
    )
    act_out = np.lib.format.open_memmap(
        tmp_dir / "act.npy",
        mode="w+",
        dtype=sample_act.dtype,
        shape=(total_samples,) + tuple(sample_act.shape[1:]),
    )

    offset = 0
    for match_id in tqdm(
        selected_match_ids,
        desc=("packing %s" % split_name).ljust(20),
        bar_format="{l_bar}{bar:40}{r_bar}",
    ):
        obs, mask, vec, act = _load_npz_arrays(source_folder, match_id, fan_shanten_replace_folder)
        next_offset = offset + int(act.shape[0])
        obs_out[offset:next_offset] = obs
        mask_out[offset:next_offset] = mask
        vec_out[offset:next_offset] = vec
        act_out[offset:next_offset] = act
        offset = next_offset
    if offset != total_samples:
        raise RuntimeError(f"packed {offset} samples, expected {total_samples}")

    for array in (obs_out, mask_out, vec_out, act_out):
        array.flush()
    expected_meta.update(
        {
            "obs_shape": list(obs_out.shape),
            "mask_shape": list(mask_out.shape),
            "vec_shape": list(vec_out.shape),
            "act_shape": list(act_out.shape),
            "obs_dtype": str(obs_out.dtype),
            "mask_dtype": str(mask_out.dtype),
            "vec_dtype": str(vec_out.dtype),
            "act_dtype": str(act_out.dtype),
        }
    )
    with open(tmp_dir / "meta.json", "w", encoding="utf8") as f:
        json.dump(expected_meta, f, indent=2, sort_keys=True)

    if cache_dir.exists():
        if force:
            shutil.rmtree(cache_dir)
        else:
            backup_dir = cache_dir.with_name(cache_dir.name + ".stale-%d" % os.getpid())
            cache_dir.rename(backup_dir)
    tmp_dir.rename(cache_dir)
    return str(cache_dir)


class PackedMahjongGBDataset(Dataset):
    """Memory-mapped dataset built once from the per-match npz files."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "meta.json", "r", encoding="utf8") as f:
            self.meta = json.load(f)
        self.obs = np.load(self.cache_dir / "obs.npy", mmap_mode="r")
        self.mask = np.load(self.cache_dir / "mask.npy", mmap_mode="r")
        self.vec = np.load(self.cache_dir / "vec.npy", mmap_mode="r")
        self.act = np.load(self.cache_dir / "act.npy", mmap_mode="r")
        self.samples = int(self.meta["sample_count"])
        self.base_matches = int(self.meta["match_count"])
        self.base_samples = self.samples
        self.matches = self.base_matches
        self.total_matches = self.base_matches
        self.total_samples = self.samples
        self.fan_feature_dim = 0
        self.fan_feature_names = []
        self.vec_size = int(self.vec.shape[1])
        self.special_match_indices = set()
        self.excluded_special_matches = int(self.meta.get("excluded_special_matches", 0))
        self.excluded_special_samples = int(self.meta.get("excluded_special_samples", 0))

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        if index < 0:
            index += self.samples
        if index < 0 or index >= self.samples:
            raise IndexError(index)
        return self.obs[index], self.mask[index], self.vec[index], self.act[index]
