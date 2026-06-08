"""Supervised training entry point using the paper hyperparameters."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .actions import (
    ACTION_NAMES,
    ACTION_TO_INDEX,
    CLAIM_ACTION_NAMES,
    CLAIM_GROUP_SIZES,
    CLAIM_OFFSETS,
    CLAIM_SIZE,
    DISCARD_SIZE,
)
from .model import TjongConfig, TjongNetwork
from .tensorize_botzone import SHARD_INDEX_FORMAT, TENSOR_ENCODING_VERSION, tensor_encoding_schema
from .tiles import TILE_NAMES

CLAIM_ACTION_INDICES = tuple(ACTION_TO_INDEX[name] for name in ("CHOW", "PONG", "MINGKONG", "BUKONG", "ANKONG"))
ACTION_MASK_OFFSET = 16
HAND_SELF_VISIBLE_ROW = 0
PAPER_SUPERVISED_EPOCHS = 125
PAPER_SUPERVISED_BATCH_SIZE = 1024
PAPER_SUPERVISED_LR = 1e-4
SUPERVISED_GRAD_CLIP = 0.5


def validate_tensor_encoding(data: dict, *, expected_version: str | None = None, path: Path | None = None) -> None:
    if expected_version is None:
        return
    schema = data.get("encoding_schema") or {}
    observed = schema.get("version")
    if observed != expected_version:
        location = f" in {path}" if path is not None else ""
        raise ValueError(
            f"tensor encoding version mismatch{location}: expected {expected_version!r}, observed {observed!r}"
        )


def load_tensor_payload(path: Path, *, expected_encoding_version: str | None = None) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != SHARD_INDEX_FORMAT:
            raise ValueError(f"unsupported tensor index format in {path}: {data.get('format')!r}")
        data["_index_path"] = str(path)
    else:
        data = torch.load(path, map_location="cpu")
    validate_tensor_encoding(data, expected_version=expected_encoding_version, path=path)
    return data


class ShardedTensorDataset(Dataset):
    def __init__(self, index: dict, *, index_path: Path):
        self.index = index
        self.index_path = Path(index_path)
        self.shards = list(index.get("shards") or [])
        self.cumulative_examples: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard.get("examples", 0))
            self.cumulative_examples.append(total)
        expected = int(index.get("examples", total) or 0)
        if expected != total:
            raise ValueError(f"sharded index example mismatch: index={expected} shards={total}")
        self._cached_shard_index: int | None = None
        self._cached_shard_dataset: TensorDataset | None = None

    def __len__(self) -> int:
        return self.cumulative_examples[-1] if self.cumulative_examples else 0

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    def shard_path(self, shard_index: int) -> Path:
        raw_path = Path(str(self.shards[shard_index]["path"]))
        if raw_path.is_absolute():
            return raw_path
        index_relative = self.index_path.parent / raw_path
        if index_relative.exists():
            return index_relative
        shard_dir = self.index.get("shard_dir")
        if shard_dir:
            return Path(str(shard_dir)) / raw_path
        return index_relative

    def load_shard_payload(self, shard_index: int) -> dict:
        path = self.shard_path(shard_index)
        payload = torch.load(path, map_location="cpu")
        validate_tensor_encoding(payload, expected_version=(self.index.get("encoding_schema") or {}).get("version"), path=path)
        expected_examples = int(self.shards[shard_index].get("examples", 0))
        observed_examples = int(payload.get("examples", payload["visible_tiles"].shape[0]))
        if expected_examples != observed_examples:
            raise ValueError(
                f"shard example mismatch in {path}: index={expected_examples} payload={observed_examples}"
            )
        return payload

    def load_shard_dataset(self, shard_index: int) -> TensorDataset:
        if self._cached_shard_index == shard_index and self._cached_shard_dataset is not None:
            return self._cached_shard_dataset
        dataset = tensor_dataset_from_payload(self.load_shard_payload(shard_index))
        self._cached_shard_index = shard_index
        self._cached_shard_dataset = dataset
        return dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative_examples, index)
        previous_total = 0 if shard_index == 0 else self.cumulative_examples[shard_index - 1]
        shard_dataset = self.load_shard_dataset(shard_index)
        return shard_dataset[index - previous_total]


def tensor_dataset_from_payload(data: dict) -> Dataset:
    if data.get("format") == SHARD_INDEX_FORMAT:
        index_path = Path(str(data.get("_index_path") or ".")).resolve()
        return ShardedTensorDataset(data, index_path=index_path)
    required = ["visible_tiles", "game_features", "action_label", "claim_label", "discard_label"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"dataset is missing keys: {missing}")
    rewards = data.get("rewards", torch.zeros(data["visible_tiles"].shape[:2]))
    previous_actions = data.get("previous_actions", torch.zeros(data["visible_tiles"].shape[:2], dtype=torch.long))
    sub_visible_tiles = data.get("sub_visible_tiles", data["visible_tiles"])
    sub_game_features = data.get("sub_game_features", data["game_features"])
    sub_rewards = data.get("sub_rewards", rewards)
    sub_previous_actions = data.get("sub_previous_actions", previous_actions)
    hidden_tiles = data.get(
        "hidden_tiles",
        torch.zeros(data["visible_tiles"].shape[0], TjongConfig.hidden_tile_rows, TjongConfig.tile_types),
    )
    return TensorDataset(
        data["visible_tiles"],
        data["game_features"],
        rewards,
        previous_actions,
        sub_visible_tiles,
        sub_game_features,
        sub_rewards,
        sub_previous_actions,
        hidden_tiles,
        data["action_label"],
        data["claim_label"],
        data["discard_label"],
    )


def load_tensor_dataset(path: Path, *, expected_encoding_version: str | None = None) -> Dataset:
    return tensor_dataset_from_payload(load_tensor_payload(path, expected_encoding_version=expected_encoding_version))


def dataset_data_format(dataset: Dataset) -> str:
    return SHARD_INDEX_FORMAT if isinstance(dataset, ShardedTensorDataset) else "monolithic"


def iter_supervised_batches(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int | None = None,
):
    if isinstance(dataset, ShardedTensorDataset):
        shard_order = list(range(dataset.shard_count))
        if shuffle and shard_order:
            generator = torch.Generator()
            generator.manual_seed(int(seed or 0))
            shard_order = torch.randperm(len(shard_order), generator=generator).tolist()
        for order_position, shard_index in enumerate(shard_order):
            shard_dataset = dataset.load_shard_dataset(int(shard_index))
            loader_generator = None
            if shuffle:
                loader_generator = torch.Generator()
                loader_generator.manual_seed(int(seed or 0) + int(shard_index) + order_position * 1009)
            yield from DataLoader(
                shard_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                generator=loader_generator,
            )
        return

    loader_generator = None
    if shuffle:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed or 0))
    yield from DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=loader_generator,
    )


def write_metrics_file(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temp_path.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as dst:
        dst.write(json.dumps(payload, sort_keys=True) + "\n")


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def checkpoint_directory(args: argparse.Namespace) -> Path:
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if checkpoint_dir:
        return Path(checkpoint_dir)
    checkpoint_out = getattr(args, "checkpoint_out", None)
    if not checkpoint_out:
        raise ValueError("--checkpoint-dir is required when periodic checkpoints are enabled without --checkpoint-out")
    checkpoint_path = Path(checkpoint_out)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_checkpoints")


def parse_batch_set(value: str | None) -> set[int]:
    if not value:
        return set()
    result = set()
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if part:
            result.add(int(part))
    return result


def configure_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_cuda_device(device_arg: str | None) -> None:
    if torch.cuda.is_available() and (device_arg is None or str(device_arg).startswith("cuda")):
        torch.cuda.set_device(0)


def validate_resume_config(resume_payload: dict, config: TjongConfig) -> None:
    observed = resume_payload.get("config") or {}
    expected = config.__dict__
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if key not in observed or observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume checkpoint config mismatch: {json.dumps(mismatches, sort_keys=True)}")


def validate_paper_supervised_args(args: argparse.Namespace, config: TjongConfig) -> None:
    if not bool(getattr(args, "require_paper_config", False)):
        return
    expected_config = TjongConfig()
    config_mismatches = {
        key: {"expected": value, "observed": getattr(config, key)}
        for key, value in expected_config.__dict__.items()
        if getattr(config, key) != value
    }
    expected_args = {
        "epochs": PAPER_SUPERVISED_EPOCHS,
        "batch_size": PAPER_SUPERVISED_BATCH_SIZE,
        "lr": PAPER_SUPERVISED_LR,
    }
    arg_mismatches = {
        key: {"expected": value, "observed": getattr(args, key, None)}
        for key, value in expected_args.items()
        if getattr(args, key, None) != value
    }
    if config_mismatches or arg_mismatches:
        raise ValueError(
            "paper supervised training config mismatch: "
            + json.dumps({"model": config_mismatches, "training": arg_mismatches}, sort_keys=True)
        )


def configure_cuda_attention(*, force_math_sdp: bool = False) -> dict[str, bool | None]:
    """Optionally avoid fused CUDA attention kernels for reproducible debug runs."""
    if torch.cuda.is_available() and force_math_sdp:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    status: dict[str, bool | None] = {}
    for name in ("flash", "mem_efficient", "cudnn", "math"):
        enabled = getattr(torch.backends.cuda, f"{name}_sdp_enabled", None) if torch.cuda.is_available() else None
        status[f"{name}_sdp_enabled"] = bool(enabled()) if callable(enabled) else None
    return status


def safe_clip_grad_norm_(parameters: list[torch.nn.Parameter], max_norm: float) -> float:
    """Clip gradients without foreach kernels or CUDA-wide finite scans."""
    params = [parameter for parameter in parameters if parameter.grad is not None]
    if not params:
        return 0.0
    total_sq = 0.0
    for parameter in params:
        grad = parameter.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        grad_norm = torch.linalg.vector_norm(grad, 2)
        grad_norm_value = float(grad_norm.cpu())
        if not math.isfinite(grad_norm_value):
            return float("nan")
        total_sq += grad_norm_value * grad_norm_value

    total_norm = math.sqrt(total_sq)
    if total_norm > max_norm:
        scale = float(max_norm) / (total_norm + 1e-12)
        for parameter in params:
            parameter.grad.mul_(scale)
    return total_norm


def cuda_sync_debug(args: argparse.Namespace, device: torch.device) -> None:
    if bool(getattr(args, "cuda_sync_debug", False)) and device.type == "cuda":
        torch.cuda.synchronize(device)


def action_count_context(action_label: torch.Tensor) -> dict[str, int]:
    return {name: int((action_label == index).sum().item()) for index, name in enumerate(ACTION_NAMES)}


def tensor_finite_context(tensor: torch.Tensor) -> dict[str, object]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    context: dict[str, object] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": finite_count == detached.numel(),
        "finite_count": finite_count,
        "nan_count": int(torch.isnan(detached).sum().item()),
        "posinf_count": int(torch.isposinf(detached).sum().item()),
        "neginf_count": int(torch.isneginf(detached).sum().item()),
    }
    if finite_count:
        finite_values = detached[finite].float()
        context["finite_min"] = float(finite_values.min().item())
        context["finite_max"] = float(finite_values.max().item())
        context["finite_abs_max"] = float(finite_values.abs().max().item())
    return context


def label_range_context(labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, object]:
    action_label, claim_label, discard_label = labels
    claim_mask, discard_mask = decision_masks(action_label)
    context: dict[str, object] = {
        "action": _label_range_stats(action_label, len(ACTION_NAMES)),
        "claim_selected": _label_range_stats(claim_label[claim_mask], CLAIM_SIZE),
        "discard_selected": _label_range_stats(discard_label[discard_mask], DISCARD_SIZE),
        "claim_selected_count": int(claim_mask.sum().item()),
        "discard_selected_count": int(discard_mask.sum().item()),
    }
    return context


def _label_range_stats(label: torch.Tensor, size: int) -> dict[str, int | None]:
    if label.numel() == 0:
        return {"count": 0, "min": None, "max": None, "invalid_count": 0}
    invalid = (label < 0) | (label >= size)
    return {
        "count": int(label.numel()),
        "min": int(label.min().item()),
        "max": int(label.max().item()),
        "invalid_count": int(invalid.sum().item()),
    }


def validate_supervised_labels(labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    context = label_range_context(labels)
    invalid = {
        name: stats
        for name, stats in (
            ("action", context["action"]),
            ("claim_selected", context["claim_selected"]),
            ("discard_selected", context["discard_selected"]),
        )
        if int(stats["invalid_count"]) > 0
    }
    if invalid:
        raise ValueError(f"supervised label outside head range: {json.dumps(invalid, sort_keys=True)}")


def _tensor_minmax(tensor: torch.Tensor) -> list[int | None]:
    if tensor.numel() == 0:
        return [None, None]
    return [int(tensor.min().item()), int(tensor.max().item())]


def _append_contract_issue(
    issues: list[dict[str, object]],
    issue_counts: dict[str, int],
    kind: str,
    mask: torch.Tensor | None = None,
    *,
    count: int | None = None,
    shard_index: int | None = None,
    path: Path | None = None,
    details: dict[str, object] | None = None,
    row_limit: int = 16,
) -> None:
    if count is None:
        if mask is None:
            raise ValueError("contract issue count requires mask or explicit count")
        count = int(mask.sum().item())
    if count <= 0:
        return
    issue_counts[kind] = int(issue_counts.get(kind, 0)) + int(count)
    if len(issues) >= 64:
        return
    issue: dict[str, object] = {"kind": kind, "count": int(count)}
    if shard_index is not None:
        issue["shard_index"] = int(shard_index)
    if path is not None:
        issue["path"] = str(path)
    if mask is not None and mask.ndim >= 1:
        rows = torch.nonzero(mask.detach().cpu(), as_tuple=False)
        if rows.numel():
            issue["first_rows"] = rows[:, 0].flatten()[:row_limit].tolist()
    if details:
        issue.update(details)
    issues.append(issue)


def _validate_supervised_payload_contract(
    data: dict,
    *,
    shard_index: int | None = None,
    path: Path | None = None,
    row_limit: int = 16,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    issues: list[dict[str, object]] = []
    issue_counts: dict[str, int] = {}
    required = [
        "visible_tiles",
        "game_features",
        "previous_actions",
        "sub_visible_tiles",
        "sub_game_features",
        "sub_previous_actions",
        "hidden_tiles",
        "action_label",
        "claim_label",
        "discard_label",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        _append_contract_issue(
            issues,
            issue_counts,
            "missing_keys",
            count=1,
            shard_index=shard_index,
            path=path,
            details={"missing": missing},
            row_limit=row_limit,
        )
        return issue_counts, issues

    action = data["action_label"].long().reshape(-1)
    claim = data["claim_label"].long().reshape(-1)
    discard = data["discard_label"].long().reshape(-1)
    n = int(action.numel())
    length_mismatch = False
    for key, tensor in (("claim_label", claim), ("discard_label", discard)):
        if int(tensor.numel()) != n:
            length_mismatch = True
            _append_contract_issue(
                issues,
                issue_counts,
                f"{key}_length_mismatch",
                count=1,
                shard_index=shard_index,
                path=path,
                details={"action_label": n, key: int(tensor.numel())},
                row_limit=row_limit,
            )
    if length_mismatch:
        return issue_counts, issues

    bad_action = (action < 0) | (action >= len(ACTION_NAMES))
    _append_contract_issue(
        issues,
        issue_counts,
        "bad_action_range",
        bad_action,
        shard_index=shard_index,
        path=path,
        details={"action_minmax": _tensor_minmax(action)},
        row_limit=row_limit,
    )

    claim_mask, discard_mask = decision_masks(action)
    bad_active_claim = claim_mask & ((claim < 0) | (claim >= CLAIM_SIZE))
    _append_contract_issue(
        issues,
        issue_counts,
        "bad_active_claim_range",
        bad_active_claim,
        shard_index=shard_index,
        path=path,
        details={"claim_minmax": _tensor_minmax(claim)},
        row_limit=row_limit,
    )
    bad_active_discard = discard_mask & ((discard < 0) | (discard >= DISCARD_SIZE))
    _append_contract_issue(
        issues,
        issue_counts,
        "bad_active_discard_range",
        bad_active_discard,
        shard_index=shard_index,
        path=path,
        details={"discard_minmax": _tensor_minmax(discard)},
        row_limit=row_limit,
    )
    _append_contract_issue(
        issues,
        issue_counts,
        "nonclaim_nonzero_claim_label",
        (~claim_mask) & (claim != 0),
        shard_index=shard_index,
        path=path,
        row_limit=row_limit,
    )
    _append_contract_issue(
        issues,
        issue_counts,
        "nondiscard_nonzero_discard_label",
        (~discard_mask) & (discard != 0),
        shard_index=shard_index,
        path=path,
        row_limit=row_limit,
    )
    for name in CLAIM_ACTION_NAMES:
        lower = CLAIM_OFFSETS[name]
        upper = lower + CLAIM_GROUP_SIZES[name]
        family_bad = (action == ACTION_TO_INDEX[name]) & ((claim < lower) | (claim >= upper))
        _append_contract_issue(
            issues,
            issue_counts,
            f"claim_family_mismatch:{name}",
            family_bad,
            shard_index=shard_index,
            path=path,
            details={"expected_range": [lower, upper]},
            row_limit=row_limit,
        )

    for key in ("previous_actions", "sub_previous_actions"):
        tensor = data[key].long()
        if tensor.shape[0:1] != (n,):
            _append_contract_issue(
                issues,
                issue_counts,
                f"{key}_shape_mismatch",
                count=1,
                shard_index=shard_index,
                path=path,
                details={"expected_rows": n, "shape": list(tensor.shape)},
                row_limit=row_limit,
            )
            continue
        if n:
            row_bad = ((tensor < 0) | (tensor >= len(ACTION_NAMES))).reshape(n, -1).any(dim=1)
        else:
            row_bad = torch.zeros(0, dtype=torch.bool, device=tensor.device)
        _append_contract_issue(
            issues,
            issue_counts,
            f"bad_{key}_range",
            row_bad,
            shard_index=shard_index,
            path=path,
            details={f"{key}_minmax": _tensor_minmax(tensor)},
            row_limit=row_limit,
        )

    for key in ("visible_tiles", "game_features", "sub_visible_tiles", "sub_game_features", "hidden_tiles"):
        tensor = data[key].float()
        if tensor.shape[0:1] != (n,):
            _append_contract_issue(
                issues,
                issue_counts,
                f"{key}_shape_mismatch",
                count=1,
                shard_index=shard_index,
                path=path,
                details={"expected_rows": n, "shape": list(tensor.shape)},
                row_limit=row_limit,
            )
            continue
        if n:
            finite_rows = torch.isfinite(tensor.reshape(n, -1)).all(dim=1)
        else:
            finite_rows = torch.zeros(0, dtype=torch.bool, device=tensor.device)
        _append_contract_issue(
            issues,
            issue_counts,
            f"nonfinite_feature:{key}",
            ~finite_rows,
            shard_index=shard_index,
            path=path,
            row_limit=row_limit,
        )

    valid_action = ~bad_action
    for key, issue_name in (
        ("game_features", "label_outside_action_mask_any"),
        ("sub_game_features", "label_outside_sub_action_mask_any"),
    ):
        game = data[key].float()
        if game.ndim != 3 or game.shape[0] != n or game.shape[1] < 1 or game.shape[-1] < ACTION_MASK_OFFSET + len(ACTION_NAMES):
            _append_contract_issue(
                issues,
                issue_counts,
                f"{key}_action_mask_shape_mismatch",
                count=1,
                shard_index=shard_index,
                path=path,
                details={"shape": list(game.shape)},
                row_limit=row_limit,
            )
            continue
        action_mask = game[:, -1, ACTION_MASK_OFFSET : ACTION_MASK_OFFSET + len(ACTION_NAMES)] > 0
        safe_action = action.clamp(0, len(ACTION_NAMES) - 1)
        outside_mask = valid_action & ~action_mask.gather(1, safe_action[:, None]).squeeze(1)
        _append_contract_issue(
            issues,
            issue_counts,
            issue_name,
            outside_mask,
            shard_index=shard_index,
            path=path,
            row_limit=row_limit,
        )

    sub_visible = data["sub_visible_tiles"].float()
    if sub_visible.ndim != 4 or sub_visible.shape[0] != n or sub_visible.shape[1] < 1 or sub_visible.shape[2] <= HAND_SELF_VISIBLE_ROW or sub_visible.shape[3] < DISCARD_SIZE:
        _append_contract_issue(
            issues,
            issue_counts,
            "sub_visible_tiles_hand_shape_mismatch",
            count=1,
            shard_index=shard_index,
            path=path,
            details={"shape": list(sub_visible.shape)},
            row_limit=row_limit,
        )
    else:
        sub_hand = sub_visible[:, -1, HAND_SELF_VISIBLE_ROW, :DISCARD_SIZE]
        safe_discard = discard.clamp(0, DISCARD_SIZE - 1)
        discard_in_sub_hand = sub_hand.gather(1, safe_discard[:, None]).squeeze(1) > 0
        missing_discard = discard_mask & (bad_active_discard | ~discard_in_sub_hand)
        _append_contract_issue(
            issues,
            issue_counts,
            "active_discard_not_in_sub_hand",
            missing_discard,
            shard_index=shard_index,
            path=path,
            details={"discard_minmax": _tensor_minmax(discard)},
            row_limit=row_limit,
        )

    return issue_counts, issues


def validate_supervised_dataset_contract(
    payload: dict,
    *,
    row_limit: int = 16,
) -> dict[str, object]:
    """Validate the full hierarchical supervised-label contract before training."""

    summary: dict[str, object] = {
        "format": payload.get("format", "monolithic"),
        "examples": int(payload.get("examples", 0) or 0),
        "sharded": bool(payload.get("format") == SHARD_INDEX_FORMAT),
        "shards": 0,
        "issue_counts": {},
        "issues": [],
    }
    total_issue_counts: dict[str, int] = {}
    sample_issues: list[dict[str, object]] = []

    if payload.get("format") == SHARD_INDEX_FORMAT:
        index_path = Path(str(payload.get("_index_path") or ".")).resolve()
        dataset = ShardedTensorDataset(payload, index_path=index_path)
        summary["examples"] = len(dataset)
        summary["shards"] = dataset.shard_count
        for shard_index in range(dataset.shard_count):
            shard_path = dataset.shard_path(shard_index)
            shard_counts, shard_issues = _validate_supervised_payload_contract(
                dataset.load_shard_payload(shard_index),
                shard_index=shard_index,
                path=shard_path,
                row_limit=row_limit,
            )
            for key, value in shard_counts.items():
                total_issue_counts[key] = total_issue_counts.get(key, 0) + int(value)
            if len(sample_issues) < 64:
                sample_issues.extend(shard_issues[: max(0, 64 - len(sample_issues))])
    else:
        counts, issues = _validate_supervised_payload_contract(payload, row_limit=row_limit)
        total_issue_counts.update(counts)
        sample_issues.extend(issues)
        if "action_label" in payload:
            summary["examples"] = int(payload["action_label"].numel())

    summary["issue_counts"] = dict(sorted(total_issue_counts.items()))
    summary["issues"] = sample_issues
    summary["passed"] = not total_issue_counts
    if total_issue_counts:
        raise ValueError("supervised dataset contract failed: " + json.dumps(summary, sort_keys=True))
    return summary


def first_nonfinite_parameter_names(model: nn.Module, *, limit: int = 8) -> list[str]:
    named_parameters = model.module.named_parameters() if isinstance(model, nn.DataParallel) else model.named_parameters()
    bad: list[str] = []
    for name, parameter in named_parameters:
        if not torch.isfinite(parameter.detach()).all().item():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def unpack_batch(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    *,
    include_hidden_tiles: bool = True,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    (
        visible_tiles,
        game_features,
        rewards,
        previous_actions,
        sub_visible_tiles,
        sub_game_features,
        sub_rewards,
        sub_previous_actions,
        hidden_tiles,
        action_label,
        claim_label,
        discard_label,
    ) = batch
    visible_tiles = visible_tiles.to(device, non_blocking=True).float()
    game_features = game_features.to(device, non_blocking=True).float()
    rewards = rewards.to(device, non_blocking=True).float()
    previous_actions = previous_actions.to(device, non_blocking=True).long()
    sub_visible_tiles = sub_visible_tiles.to(device, non_blocking=True).float()
    sub_game_features = sub_game_features.to(device, non_blocking=True).float()
    sub_rewards = sub_rewards.to(device, non_blocking=True).float()
    sub_previous_actions = sub_previous_actions.to(device, non_blocking=True).long()
    if include_hidden_tiles:
        hidden_tiles = hidden_tiles.to(device, non_blocking=True).float()
    else:
        hidden_tiles = None
    action_label = action_label.to(device, non_blocking=True).long()
    claim_label = claim_label.to(device, non_blocking=True).long()
    discard_label = discard_label.to(device, non_blocking=True).long()
    inputs = {
        "visible_tiles": visible_tiles,
        "game_features": game_features,
        "rewards": rewards,
        "previous_actions": previous_actions,
        "sub_visible_tiles": sub_visible_tiles,
        "sub_game_features": sub_game_features,
        "sub_rewards": sub_rewards,
        "sub_previous_actions": sub_previous_actions,
    }
    if include_hidden_tiles:
        inputs["hidden_tiles"] = hidden_tiles
    return inputs, (action_label, claim_label, discard_label)


def decision_masks(action_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    claim_actions = torch.tensor(CLAIM_ACTION_INDICES, device=action_label.device)
    claim_mask = (action_label.unsqueeze(-1) == claim_actions).any(dim=-1)
    discard_mask = action_label == ACTION_TO_INDEX["DISCARD"]
    return claim_mask, discard_mask


def supervised_loss_components(
    outputs: dict[str, torch.Tensor],
    labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    action_label, claim_label, discard_label = labels
    validate_supervised_labels(labels)
    ce = nn.CrossEntropyLoss()
    action_loss = ce(outputs["action_logits"], action_label)
    total = action_loss
    claim_mask, discard_mask = decision_masks(action_label)
    claim_loss = None
    if claim_mask.any():
        claim_loss = ce(outputs["claim_logits"][claim_mask], claim_label[claim_mask])
        total = total + claim_loss
    discard_loss = None
    if discard_mask.any():
        discard_loss = ce(outputs["discard_logits"][discard_mask], discard_label[discard_mask])
        total = total + discard_loss
    return {
        "action": action_loss,
        "claim": claim_loss,
        "discard": discard_loss,
        "total": total,
    }


def supervised_loss(outputs: dict[str, torch.Tensor], labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    loss = supervised_loss_components(outputs, labels)["total"]
    assert loss is not None
    return loss


def nonfinite_supervised_context(
    outputs: dict[str, torch.Tensor],
    labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    loss_components: dict[str, torch.Tensor | None],
    model: nn.Module | None = None,
) -> dict[str, object]:
    component_context: dict[str, object] = {}
    for name, value in loss_components.items():
        if value is None:
            component_context[name] = None
        else:
            component_context[name] = tensor_finite_context(value)
            component_context[name]["value"] = float(value.detach().item()) if torch.isfinite(value).item() else str(value.detach().item())
    context: dict[str, object] = {
        "loss_components": component_context,
        "label_ranges": label_range_context(labels),
        "output_tensors": {
            name: tensor_finite_context(tensor)
            for name, tensor in outputs.items()
            if name in {"action_logits", "claim_logits", "discard_logits"}
        },
    }
    if model is not None:
        context["nonfinite_parameters"] = first_nonfinite_parameter_names(model)
    return context


def batch_metric_sums(
    outputs: dict[str, torch.Tensor],
    labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    action_label, claim_label, discard_label = labels
    claim_mask, discard_mask = decision_masks(action_label)
    action_pred = outputs["action_logits"].argmax(dim=-1)
    claim_pred = outputs["claim_logits"].argmax(dim=-1)
    discard_pred = outputs["discard_logits"].argmax(dim=-1)
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    metrics = {
        "action_loss_sum": float(ce_sum(outputs["action_logits"], action_label).item()),
        "action_count": float(action_label.numel()),
        "action_correct": float((action_pred == action_label).sum().item()),
        "claim_loss_sum": 0.0,
        "claim_count": float(claim_mask.sum().item()),
        "claim_correct": 0.0,
        "discard_loss_sum": 0.0,
        "discard_count": float(discard_mask.sum().item()),
        "discard_correct": 0.0,
    }
    for index, name in enumerate(ACTION_NAMES):
        label_mask = action_label == index
        metrics[f"action_breakdown:{name}:count"] = float(label_mask.sum().item())
        metrics[f"action_breakdown:{name}:correct"] = float((action_pred[label_mask] == index).sum().item())
        metrics[f"action_breakdown:{name}:predicted"] = float((action_pred == index).sum().item())
    if claim_mask.any():
        metrics["claim_loss_sum"] = float(ce_sum(outputs["claim_logits"][claim_mask], claim_label[claim_mask]).item())
        metrics["claim_correct"] = float((claim_pred[claim_mask] == claim_label[claim_mask]).sum().item())
    for name in CLAIM_ACTION_NAMES:
        label_mask = action_label == ACTION_TO_INDEX[name]
        metrics[f"claim_breakdown:{name}:count"] = float(label_mask.sum().item())
        metrics[f"claim_breakdown:{name}:correct"] = float((claim_pred[label_mask] == claim_label[label_mask]).sum().item())
    if discard_mask.any():
        metrics["discard_loss_sum"] = float(
            ce_sum(outputs["discard_logits"][discard_mask], discard_label[discard_mask]).item()
        )
        metrics["discard_correct"] = float((discard_pred[discard_mask] == discard_label[discard_mask]).sum().item())
    for index, name in enumerate(TILE_NAMES):
        label_mask = discard_mask & (discard_label == index)
        metrics[f"discard_tile_breakdown:{name}:count"] = float(label_mask.sum().item())
        metrics[f"discard_tile_breakdown:{name}:correct"] = float((discard_pred[label_mask] == index).sum().item())
        metrics[f"discard_tile_breakdown:{name}:predicted"] = float((discard_mask & (discard_pred == index)).sum().item())
    return metrics


def merge_metric_sums(total: dict[str, float], batch: dict[str, float]) -> None:
    for key, value in batch.items():
        total[key] = total.get(key, 0.0) + float(value)


def finalize_metric_sums(metric_sums: dict[str, float]) -> dict[str, float | None]:
    action_count = metric_sums.get("action_count", 0.0)
    claim_count = metric_sums.get("claim_count", 0.0)
    discard_count = metric_sums.get("discard_count", 0.0)
    decision_count = action_count + claim_count + discard_count
    decision_loss_sum = (
        metric_sums.get("action_loss_sum", 0.0)
        + metric_sums.get("claim_loss_sum", 0.0)
        + metric_sums.get("discard_loss_sum", 0.0)
    )
    metrics = {
        "decision_loss": decision_loss_sum / decision_count if decision_count else None,
        "action_loss": metric_sums.get("action_loss_sum", 0.0) / action_count if action_count else None,
        "action_accuracy": metric_sums.get("action_correct", 0.0) / action_count if action_count else None,
        "action_count": int(action_count),
        "claim_loss": metric_sums.get("claim_loss_sum", 0.0) / claim_count if claim_count else None,
        "claim_accuracy": metric_sums.get("claim_correct", 0.0) / claim_count if claim_count else None,
        "claim_count": int(claim_count),
        "discard_loss": metric_sums.get("discard_loss_sum", 0.0) / discard_count if discard_count else None,
        "discard_accuracy": metric_sums.get("discard_correct", 0.0) / discard_count if discard_count else None,
        "discard_count": int(discard_count),
    }
    metrics["action_breakdown"] = _named_breakdown(metric_sums, "action_breakdown", ACTION_NAMES)
    metrics["claim_breakdown"] = _named_breakdown(metric_sums, "claim_breakdown", CLAIM_ACTION_NAMES)
    metrics["discard_tile_breakdown"] = _named_breakdown(metric_sums, "discard_tile_breakdown", TILE_NAMES)
    return metrics


def _named_breakdown(metric_sums: dict[str, float], prefix: str, names: tuple[str, ...]) -> dict[str, dict[str, float | int | None]]:
    breakdown: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        count = metric_sums.get(f"{prefix}:{name}:count", 0.0)
        correct = metric_sums.get(f"{prefix}:{name}:correct", 0.0)
        predicted = metric_sums.get(f"{prefix}:{name}:predicted", 0.0)
        item: dict[str, float | int | None] = {
            "count": int(count),
            "correct": int(correct),
            "accuracy": correct / count if count else None,
        }
        if f"{prefix}:{name}:predicted" in metric_sums:
            item["predicted"] = int(predicted)
        breakdown[name] = item
    return breakdown


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, float | None]:
    model.eval()
    metric_sums: dict[str, float] = {}
    with torch.no_grad():
        for batch in iter_supervised_batches(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ):
            inputs, labels = unpack_batch(batch, device, include_hidden_tiles=False)
            outputs = model(**inputs)
            merge_metric_sums(metric_sums, batch_metric_sums(outputs, labels))
    return finalize_metric_sums(metric_sums)


def train(args: argparse.Namespace) -> dict:
    seed = int(getattr(args, "seed", 0) or 0)
    configure_seed(seed)
    configure_cuda_device(getattr(args, "device", None))
    grad_clip = float(getattr(args, "grad_clip", 0.0) or 0.0)
    fail_on_nonfinite = bool(getattr(args, "fail_on_nonfinite", False))
    force_math_sdp = bool(getattr(args, "force_math_sdp", False))
    max_steps = int(getattr(args, "max_steps", 0) or 0)
    sync_debug = bool(getattr(args, "cuda_sync_debug", False))
    cuda_attention = configure_cuda_attention(force_math_sdp=force_math_sdp)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = TjongConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    validate_paper_supervised_args(args, config)
    train_payload = load_tensor_payload(Path(args.train_pt), expected_encoding_version=args.require_encoding_version)
    preflight_summary: dict[str, object] | None = None
    if bool(getattr(args, "preflight_supervised_contract", False)):
        preflight_summary = validate_supervised_dataset_contract(train_payload)
        print("supervised_contract_preflight " + json.dumps(preflight_summary, sort_keys=True), flush=True)
    encoding_schema = train_payload.get("encoding_schema") or tensor_encoding_schema()
    checkpoint_encoding_version = encoding_schema.get("version") or args.require_encoding_version
    dataset = tensor_dataset_from_payload(train_payload)
    train_data_format = dataset_data_format(dataset)
    base_model = TjongNetwork(config)
    if bool(getattr(args, "data_parallel", False)) and device.type == "cuda" and torch.cuda.device_count() > 1:
        model: nn.Module = nn.DataParallel(base_model).to(device)
    else:
        model = base_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    start_epoch = 1
    global_batch = 0
    metrics = {
        "paper": "Tjong CIT2.12298",
        "optimizer": "Adam",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": seed,
        "shuffle_seed_mode": "seed_plus_epoch",
        "grad_clip": grad_clip,
        "fail_on_nonfinite": fail_on_nonfinite,
        "check_param_finiteness": bool(getattr(args, "check_param_finiteness", False)),
        "cuda_sync_debug": sync_debug,
        "max_steps": max_steps,
        "force_math_sdp": force_math_sdp,
        "cuda_attention": cuda_attention,
        "required_encoding_version": args.require_encoding_version,
        "paper_tensor_encoding_version": TENSOR_ENCODING_VERSION,
        "checkpoint_encoding_version": checkpoint_encoding_version,
        "encoding_schema": encoding_schema,
        "train_examples": len(dataset),
        "train_data_format": train_data_format,
        "train_shard_count": dataset.shard_count if isinstance(dataset, ShardedTensorDataset) else 0,
        "preflight_supervised_contract": preflight_summary,
        "model_parameters": base_model.parameter_count(),
        "data_parallel": isinstance(model, nn.DataParallel),
        "device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "epochs": [],
    }
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if resume_checkpoint:
        resume_payload = torch.load(resume_checkpoint, map_location="cpu")
        validate_resume_config(resume_payload, config)
        base_model.load_state_dict(resume_payload["model"])
        if resume_payload.get("optimizer"):
            optimizer.load_state_dict(resume_payload["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        metrics = resume_payload.get("metrics") or metrics
        metrics["resumed_from"] = str(resume_checkpoint)
        start_epoch = int(resume_payload.get("epoch", len(metrics.get("epochs", [])))) + 1
        if metrics.get("epochs"):
            global_batch = int(metrics["epochs"][-1].get("global_batch", 0) or 0)
    metrics["grad_clip"] = grad_clip
    metrics["fail_on_nonfinite"] = fail_on_nonfinite
    metrics["check_param_finiteness"] = bool(getattr(args, "check_param_finiteness", False))
    metrics["cuda_sync_debug"] = sync_debug
    metrics["max_steps"] = max_steps
    metrics["force_math_sdp"] = force_math_sdp
    metrics["cuda_attention"] = cuda_attention
    metrics["train_examples"] = len(dataset)
    metrics["train_data_format"] = train_data_format
    metrics["train_shard_count"] = dataset.shard_count if isinstance(dataset, ShardedTensorDataset) else 0
    if preflight_summary is not None:
        metrics["preflight_supervised_contract"] = preflight_summary

    checkpoint_every_batches = int(getattr(args, "checkpoint_every_batches", 0) or 0)
    checkpoint_at_global_batches = parse_batch_set(getattr(args, "checkpoint_at_global_batches", None))
    checkpoint_at_epoch_batches = parse_batch_set(getattr(args, "checkpoint_at_epoch_batches", None))
    log_every_batches = int(getattr(args, "log_every_batches", 0) or 0)
    batch_metrics_jsonl = getattr(args, "batch_metrics_jsonl", None)
    stop_after_epoch = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        batches = 0
        max_grad_norm_before_clip = 0.0
        metric_sums: dict[str, float] = {}
        loader = iter_supervised_batches(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=seed + epoch,
        )
        for batch in loader:
            batches += 1
            global_batch += 1
            inputs, labels = unpack_batch(batch, device, include_hidden_tiles=False)
            optimizer.zero_grad(set_to_none=True)
            if fail_on_nonfinite and bool(getattr(args, "check_param_finiteness", False)):
                bad_parameters = first_nonfinite_parameter_names(model)
                if bad_parameters:
                    raise FloatingPointError(
                        "non-finite supervised parameters before forward at "
                        f"epoch={epoch} batch={batches} global_batch={global_batch} "
                        f"parameters={bad_parameters}"
                    )
            outputs = model(**inputs)
            cuda_sync_debug(args, device)
            if fail_on_nonfinite:
                bad_outputs = {
                    name: tensor_finite_context(tensor)
                    for name, tensor in outputs.items()
                    if name in {"action_logits", "claim_logits", "discard_logits"} and not torch.isfinite(tensor).all().item()
                }
                if bad_outputs:
                    raise FloatingPointError(
                        "non-finite supervised logits before CE at "
                        f"epoch={epoch} batch={batches} global_batch={global_batch} "
                        f"action_counts={action_count_context(labels[0])} "
                        f"context={json.dumps(bad_outputs, sort_keys=True)}"
                    )
            loss_components = supervised_loss_components(outputs, labels)
            loss = loss_components["total"]
            assert loss is not None
            cuda_sync_debug(args, device)
            if fail_on_nonfinite and not torch.isfinite(loss).item():
                context = nonfinite_supervised_context(outputs, labels, loss_components, model)
                raise FloatingPointError(
                    "non-finite supervised loss at "
                    f"epoch={epoch} batch={batches} global_batch={global_batch} "
                    f"loss={loss.detach().item()} action_counts={action_count_context(labels[0])} "
                    f"context={json.dumps(context, sort_keys=True)}"
                )
            loss.backward()
            cuda_sync_debug(args, device)
            if grad_clip > 0.0:
                grad_norm = safe_clip_grad_norm_(trainable_parameters, grad_clip)
                if math.isfinite(grad_norm):
                    max_grad_norm_before_clip = max(max_grad_norm_before_clip, grad_norm)
                elif fail_on_nonfinite:
                    raise FloatingPointError(
                        "non-finite supervised gradients before clipping at "
                        f"epoch={epoch} batch={batches} global_batch={global_batch} "
                        f"grad_norm={grad_norm} "
                        f"action_counts={action_count_context(labels[0])}"
                    )
            optimizer.step()
            batch_size = int(labels[0].numel())
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            merge_metric_sums(metric_sums, batch_metric_sums(outputs, labels))
            if log_every_batches and batches % log_every_batches == 0:
                batch_metrics = {
                    "event": "batch",
                    "epoch": epoch,
                    "batch": batches,
                    "global_batch": global_batch,
                    "optimization_loss": total_loss / max(1, total),
                    **finalize_metric_sums(metric_sums),
                }
                if grad_clip > 0.0:
                    batch_metrics["max_grad_norm_before_clip"] = max_grad_norm_before_clip
                print(json.dumps(batch_metrics, sort_keys=True), flush=True)
                if batch_metrics_jsonl:
                    append_jsonl(Path(batch_metrics_jsonl), batch_metrics)
            if (
                (checkpoint_every_batches and global_batch % checkpoint_every_batches == 0)
                or global_batch in checkpoint_at_global_batches
                or batches in checkpoint_at_epoch_batches
            ):
                checkpoint_dir = checkpoint_directory(args)
                save_supervised_checkpoint(
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    metrics=metrics,
                    encoding_schema=encoding_schema,
                    checkpoint_encoding_version=checkpoint_encoding_version,
                    epoch=epoch,
                    path=checkpoint_dir / f"batch_{global_batch:08d}.pt",
                    include_training_state=True,
                )
            if max_steps and global_batch >= max_steps:
                stop_after_epoch = True
                break
        epoch_metrics = {
            "epoch": epoch,
            "batches": batches,
            "global_batch": global_batch,
            "optimization_loss": total_loss / max(1, total),
            **finalize_metric_sums(metric_sums),
        }
        if grad_clip > 0.0:
            epoch_metrics["max_grad_norm_before_clip"] = max_grad_norm_before_clip
        metrics["epochs"].append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
        metrics_jsonl = getattr(args, "metrics_jsonl", None)
        if metrics_jsonl:
            append_jsonl(Path(metrics_jsonl), epoch_metrics)
        if args.metrics_out:
            write_metrics_file(Path(args.metrics_out), metrics)
        checkpoint_every_epochs = int(getattr(args, "checkpoint_every_epochs", 0) or 0)
        if checkpoint_every_epochs and epoch % checkpoint_every_epochs == 0:
            checkpoint_dir = checkpoint_directory(args)
            save_supervised_checkpoint(
                model=model,
                config=config,
                optimizer=optimizer,
                metrics=metrics,
                encoding_schema=encoding_schema,
                checkpoint_encoding_version=checkpoint_encoding_version,
                epoch=epoch,
                path=checkpoint_dir / f"epoch_{epoch:04d}.pt",
                include_training_state=True,
            )
            save_supervised_checkpoint(
                model=model,
                config=config,
                optimizer=optimizer,
                metrics=metrics,
                encoding_schema=encoding_schema,
                checkpoint_encoding_version=checkpoint_encoding_version,
                epoch=epoch,
                path=checkpoint_dir / "latest.pt",
                include_training_state=True,
            )
        if stop_after_epoch:
            break
    if args.checkpoint_out:
        Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
        save_supervised_checkpoint(
            model=model,
            config=config,
            optimizer=optimizer,
            metrics=metrics,
            encoding_schema=encoding_schema,
            checkpoint_encoding_version=checkpoint_encoding_version,
            epoch=args.epochs,
            path=Path(args.checkpoint_out),
            include_training_state=False,
        )
    if args.metrics_out:
        write_metrics_file(Path(args.metrics_out), metrics)
    return metrics


def save_supervised_checkpoint(
    *,
    model: nn.Module,
    config: TjongConfig,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
    encoding_schema: dict,
    checkpoint_encoding_version: str | None,
    epoch: int,
    path: Path,
    include_training_state: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    payload = {
        "model": state_dict,
        "config": config.__dict__,
        "metrics": metrics,
        "encoding_schema": encoding_schema,
        "tensor_encoding_version": checkpoint_encoding_version,
    }
    if include_training_state:
        payload["optimizer"] = optimizer.state_dict()
        payload["epoch"] = int(epoch)
    torch.save(payload, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pt", required=True)
    parser.add_argument("--checkpoint-out", default="models/tjong_supervised.pt")
    parser.add_argument("--metrics-out", default="runs/tjong_supervised_metrics.json")
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--require-paper-config", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--batch-metrics-jsonl", default=None)
    parser.add_argument("--checkpoint-every-batches", type=int, default=0)
    parser.add_argument("--checkpoint-at-global-batches", default=None)
    parser.add_argument("--checkpoint-at-epoch-batches", default=None)
    parser.add_argument("--log-every-batches", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=SUPERVISED_GRAD_CLIP)
    parser.add_argument("--force-math-sdp", action="store_true")
    parser.add_argument("--fail-on-nonfinite", action="store_true")
    parser.add_argument("--preflight-supervised-contract", action="store_true")
    parser.add_argument("--check-param-finiteness", action="store_true")
    parser.add_argument("--cuda-sync-debug", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
