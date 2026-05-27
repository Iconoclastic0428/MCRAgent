#!/usr/bin/env python3
"""Train a rule-gated Transformer candidate-action scorer for MCR logs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


LAWLORENTZ_DIR = Path(__file__).resolve().parents[1] / "external" / "Chinese-Standard-Mahjong-DRL"
if str(LAWLORENTZ_DIR) not in sys.path:
    sys.path.insert(0, str(LAWLORENTZ_DIR))

from feature import FeatureAgent  # noqa: E402

from build_lawlorentz_dataset import (  # noqa: E402
    BotzoneFeatureRuntime,
    _is_trainable_response,
    _safe_apply_own_response,
    _valid_observation,
    actual_response,
    response_to_valid_action,
)
from lawlorentz_policy import LawlorentzEffectiveScorer  # noqa: E402


DEFAULT_HISTORY_LEN = 80
DEFAULT_MAX_CANDIDATES = 96
CANDIDATE_RULE_FEATURES = 7
ACTION_RESPONSE_AGENT = FeatureAgent(0)
ACTION_TYPES = {
    "Pass": 0,
    "Hu": 1,
    "Play": 2,
    "Chi": 3,
    "Peng": 4,
    "Gang": 5,
    "BuGang": 6,
}
BOTZONE_ACTION_TYPES = {
    "PASS": 0,
    "HU": 1,
    "PLAY": 2,
    "CHI": 3,
    "PENG": 4,
    "GANG": 5,
    "BUGANG": 6,
    "DRAW": 7,
    "INIT": 8,
    "DEAL": 9,
}
TILE_IDS = {tile: index for index, tile in enumerate(FeatureAgent.TILE_LIST)}


@dataclass
class TransformerExample:
    obs: np.ndarray
    action_mask: np.ndarray
    act: int
    player: int
    turn: int
    response: str
    history_tokens: np.ndarray
    value_target: float
    allow_hu: bool
    candidate_rule_features: np.ndarray
    teacher_action_distribution: np.ndarray | None = None
    teacher_accept_top3: bool = False
    teacher_candidate_norms: tuple[str, ...] = ()


@dataclass
class ReviewTarget:
    candidates: list
    accept_top3: bool = False


def action_response(action: int) -> str:
    return ACTION_RESPONSE_AGENT.action2response(int(action))


def hu_gated_candidate_mask(
    action_mask: np.ndarray,
    action_to_response: Callable[[int], str] = action_response,
    *,
    allow_hu: bool,
) -> np.ndarray:
    """Return legal candidate mask with Hu removed unless fan validity is known."""

    gated = np.asarray(action_mask, dtype=np.int8).copy()
    for action in np.flatnonzero(gated > 0):
        if action_to_response(int(action)).split()[0] == "Hu" and not allow_hu:
            gated[int(action)] = 0
    return gated


def rule_gated_hu_allowed(
    action_mask: np.ndarray,
    action_to_response: Callable[[int], str] = action_response,
) -> bool:
    """Return whether the current legal-action mask contains a Hu action."""

    for action in np.flatnonzero(np.asarray(action_mask) > 0):
        if action_to_response(int(action)).split()[0] == "Hu":
            return True
    return False


def chaga_candidates_to_action_distribution(
    candidates: list,
    action_mask: np.ndarray,
    action_to_response: Callable[[int], str] = action_response,
    *,
    allow_hu: bool,
    temperature: float = 1.0,
) -> np.ndarray | None:
    """Convert weighted CHAGA candidates to a soft target over legal action IDs."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    gated_mask = hu_gated_candidate_mask(action_mask, action_to_response, allow_hu=allow_hu)
    legal_by_name: dict[str, list[int]] = {}
    for action in np.flatnonzero(gated_mask > 0):
        action = int(action)
        normalized = normalize_teacher_action(action_to_response(action))
        legal_by_name.setdefault(normalized, []).append(action)

    matched: list[tuple[int, float]] = []
    for item in candidates:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            score = float(item[0])
        except (TypeError, ValueError):
            continue
        normalized = normalize_teacher_action(str(item[1]))
        actions = legal_by_name.get(normalized)
        if not actions:
            continue
        share = score
        for action in actions:
            matched.append((action, share))
    if not matched:
        return None

    raw_scores = np.asarray([score for _, score in matched], dtype=np.float64)
    scaled = (raw_scores - float(np.max(raw_scores))) / float(temperature)
    weights = np.exp(scaled)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return None
    dist = np.zeros((len(action_mask),), dtype=np.float32)
    for (action, _), weight in zip(matched, weights / total):
        dist[action] += float(weight)
    if float(dist.sum()) <= 0.0:
        return None
    dist /= float(dist.sum())
    return dist


def normalize_teacher_action(action: str | None) -> str:
    if not action:
        return ""
    parts = str(action).strip().split()
    if not parts:
        return ""
    head = parts[0].upper()
    if head == "PLAY" and len(parts) >= 2:
        return f"PLAY {parts[1].upper()}"
    if head == "CHI" and len(parts) >= 2:
        return f"CHI {parts[1].upper()}"
    if head == "PENG":
        return "PENG"
    if head in {"GANG", "BUGANG"}:
        return head
    if head == "HU":
        return "HU"
    if head == "PASS":
        return "PASS"
    if head == "ABANDON":
        return "ABANDON"
    return " ".join(part.upper() for part in parts)


def normalize_teacher_candidates(candidates: list, *, limit: int | None = None) -> tuple[str, ...]:
    normalized: list[str] = []
    items = candidates[:limit] if limit is not None else candidates
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        action = normalize_teacher_action(str(item[1]))
        if action:
            normalized.append(action)
    return tuple(normalized)


def build_transformer_examples_from_record(
    record: dict,
    *,
    history_len: int = DEFAULT_HISTORY_LEN,
    teacher_lookup: Callable[..., list | None] | None = None,
    teacher_temperature: float = 1.0,
) -> tuple[list[TransformerExample], Counter[str]]:
    runtimes = [BotzoneFeatureRuntime() for _ in range(4)]
    skip_requests: list[Counter[str]] = [Counter() for _ in range(4)]
    examples: list[TransformerExample] = []
    stats: Counter[str] = Counter()
    history: list[int] = []
    logs = record.get("logs") or []
    train_players_raw = record.get("train_players")
    train_players = (
        {str(player) for player in train_players_raw}
        if train_players_raw is not None
        else None
    )
    record_id = str(record.get("source_record_id") or record.get("match_id") or record.get("id") or "")

    for turn in range(0, len(logs) - 1, 2):
        output = logs[turn].get("output") or {}
        requests = output.get("content") or {}
        if not isinstance(requests, dict):
            continue
        response_log = logs[turn + 1]
        if not isinstance(response_log, dict):
            continue

        history_after_turn: list[int] = []
        for player_text, request in requests.items():
            try:
                player = int(player_text)
            except (TypeError, ValueError):
                stats["bad_player"] += 1
                continue
            if player < 0 or player >= 4:
                stats["bad_player"] += 1
                continue
            request = str(request)
            if skip_requests[player][request] > 0:
                skip_requests[player][request] -= 1
                stats["skipped_duplicate_self_event"] += 1
                continue
            response = actual_response(response_log, str(player))
            if response is not None:
                history_after_turn.append(encode_history_event(player, request, response))
            try:
                obs = runtimes[player].observe(request)
            except Exception as exc:
                stats[f"observe_error:{type(exc).__name__}"] += 1
                continue
            if obs is None or response is None:
                continue
            if not _is_trainable_response(request, response):
                stats["untrainable_response"] += 1
                continue

            action = response_to_valid_action(runtimes[player].agent, obs, request, response)
            if action is None:
                stats[f"unmapped:{response.split()[0].upper()}"] += 1
                continue
            if train_players is not None and str(player) not in train_players:
                stats["filtered_train_player_examples"] += 1
                _advance_runtime_after_response(runtimes[player], request, response, action, skip_requests[player], stats)
                continue
            if not _valid_observation(obs):
                stats["invalid_observation"] += 1
                _advance_runtime_after_response(runtimes[player], request, response, action, skip_requests[player], stats)
                continue
            if int(np.sum(obs["action_mask"])) == 1:
                stats["single_action_mask"] += 1
                _advance_runtime_after_response(runtimes[player], request, response, action, skip_requests[player], stats)
                continue
            allow_hu = rule_gated_hu_allowed(obs["action_mask"], runtimes[player].agent.action2response)
            teacher_distribution = None
            teacher_accept_top3 = False
            teacher_candidate_norms: tuple[str, ...] = ()
            if teacher_lookup is not None:
                try:
                    teacher_result = teacher_lookup(
                        record=record,
                        record_id=record_id,
                        player=player,
                        turn=turn // 2,
                        request=request,
                        response=response,
                        action=action,
                        obs=obs,
                    )
                except Exception as exc:
                    stats[f"teacher_lookup_error:{type(exc).__name__}"] += 1
                    teacher_result = None
                if isinstance(teacher_result, ReviewTarget):
                    teacher_candidates = teacher_result.candidates
                    teacher_accept_top3 = bool(teacher_result.accept_top3)
                else:
                    teacher_candidates = teacher_result
                if teacher_candidates:
                    teacher_candidate_norms = normalize_teacher_candidates(teacher_candidates)
                    if teacher_candidate_norms:
                        stats["teacher_original_targets"] += 1
                    teacher_distribution = chaga_candidates_to_action_distribution(
                        teacher_candidates,
                        obs["action_mask"],
                        runtimes[player].agent.action2response,
                        allow_hu=allow_hu,
                        temperature=teacher_temperature,
                    )
                    if teacher_distribution is None:
                        stats["teacher_target_unmapped"] += 1
                    else:
                        stats["teacher_targets"] += 1

            examples.append(
                TransformerExample(
                    obs=obs["observation"].astype(np.float32),
                    action_mask=obs["action_mask"].astype(np.int8),
                    act=int(action),
                    player=player,
                    turn=turn // 2,
                    response=action_response(int(action)),
                    history_tokens=_history_window(history, history_len),
                    value_target=_value_target(record, player),
                    allow_hu=allow_hu,
                    candidate_rule_features=build_candidate_rule_features(
                        runtimes[player].agent,
                        obs,
                        allow_hu=allow_hu,
                    ),
                    teacher_action_distribution=teacher_distribution,
                    teacher_accept_top3=teacher_accept_top3,
                    teacher_candidate_norms=teacher_candidate_norms,
                )
            )
            stats["examples"] += 1
            stats[f"action:{examples[-1].response.split()[0]}"] += 1
            _advance_runtime_after_response(runtimes[player], request, response, action, skip_requests[player], stats)
        history.extend(history_after_turn)

    return examples, stats


def _advance_runtime_after_response(
    runtime: BotzoneFeatureRuntime,
    request: str,
    response: str,
    action: int,
    skip_requests: Counter[str],
    stats: Counter[str],
) -> None:
    runtime.remember_response(request, action)
    for skipped in _safe_apply_own_response(runtime, request, response, stats):
        skip_requests[skipped] += 1


def _history_window(history: list[int], history_len: int) -> np.ndarray:
    out = np.zeros((history_len,), dtype=np.int64)
    values = history[-history_len:]
    if values:
        out[-len(values) :] = np.asarray(values, dtype=np.int64)
    return out


def encode_history_event(player: int, request: str, response: str) -> int:
    response_tokens = response.strip().split()
    request_tokens = request.strip().split()
    head = response_tokens[0].upper() if response_tokens else "PASS"
    if head == "PASS" and len(request_tokens) >= 3:
        head = request_tokens[2].upper()
    action_type = BOTZONE_ACTION_TYPES.get(head, 0)
    tile = _first_tile(response_tokens[1:]) or _first_tile(request_tokens[1:]) or ""
    tile_mod = TILE_IDS.get(tile, 0) % 12
    return 1 + int(player) * 96 + action_type * 12 + tile_mod


def _first_tile(tokens: Iterable[str]) -> str | None:
    for token in tokens:
        if token in TILE_IDS:
            return token
    return None


def _value_target(record: dict, player: int) -> float:
    scores = record.get("scores") or {}
    value = scores.get(str(player), scores.get(player, 0))
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return float(math.tanh(score / 64.0))


def build_candidate_rule_features(
    agent: FeatureAgent | None,
    obs: dict,
    *,
    allow_hu: bool,
    effective_levels: int = 1,
) -> np.ndarray:
    features = np.zeros((FeatureAgent.ACT_SIZE, CANDIDATE_RULE_FEATURES), dtype=np.float32)
    if agent is None:
        return features
    try:
        scorer = LawlorentzEffectiveScorer(
            packs=tuple(agent.packs[0]),
            shown_tiles=Counter(agent.shownTiles),
            seat_wind=int(agent.seatWind),
            prevalent_wind=0,
            levels=effective_levels,
        )
    except Exception:
        scorer = None

    for action in np.flatnonzero(obs["action_mask"] > 0):
        action = int(action)
        try:
            response = agent.action2response(action)
        except Exception:
            continue
        parts = response.split()
        head = parts[0] if parts else "Pass"
        features[action, 0] = 1.0 if head == "Hu" and allow_hu else 0.0
        features[action, 1] = 1.0 if head == "Play" else 0.0
        features[action, 2] = 1.0 if head in {"Chi", "Peng", "Gang", "BuGang"} else 0.0
        if head == "Play" and len(parts) >= 2 and scorer is not None:
            tile = parts[1]
            after = list(agent.hand)
            try:
                after.remove(tile)
            except ValueError:
                continue
            shown = Counter(agent.shownTiles)
            shown[tile] += 1
            try:
                profile = scorer.profile(tuple(after), shown)
            except Exception:
                continue
            features[action, 3] = min(1.0, profile.fan8_wait_tiles / 32.0)
            features[action, 4] = min(1.0, profile.fan8_wait_types / 16.0)
            features[action, 5] = min(1.0, profile.first_effective_tiles / 32.0)
            features[action, 6] = max(-1.0, min(1.0, -profile.min_shanten / 8.0))
    return features


def collate_transformer_examples(
    examples: list[TransformerExample],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("empty Transformer batch")

    obs = torch.from_numpy(np.stack([example.obs for example in examples]).astype(np.float32))
    history = torch.from_numpy(np.stack([example.history_tokens for example in examples]).astype(np.int64))
    players = torch.tensor([example.player / 3.0 for example in examples], dtype=torch.float32)
    values = torch.tensor([example.value_target for example in examples], dtype=torch.float32)
    allow_hu = torch.tensor([1.0 if example.allow_hu else 0.0 for example in examples], dtype=torch.float32)

    candidate_actions = torch.zeros((len(examples), max_candidates), dtype=torch.long)
    candidate_mask = torch.zeros((len(examples), max_candidates), dtype=torch.bool)
    target_index = torch.zeros((len(examples),), dtype=torch.long)
    candidate_count = torch.zeros((len(examples),), dtype=torch.float32)
    candidate_rule_features = torch.zeros(
        (len(examples), max_candidates, CANDIDATE_RULE_FEATURES),
        dtype=torch.float32,
    )
    teacher_target_dist = torch.zeros((len(examples), max_candidates), dtype=torch.float32)
    has_teacher_target = torch.zeros((len(examples),), dtype=torch.bool)
    teacher_accept_top3 = torch.zeros((len(examples),), dtype=torch.bool)

    for row, example in enumerate(examples):
        gated_mask = hu_gated_candidate_mask(example.action_mask, allow_hu=example.allow_hu)
        candidates = [int(action) for action in np.flatnonzero(gated_mask > 0)]
        if example.act not in candidates:
            raise ValueError(f"target action {example.act} was removed from candidates")
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
            if example.act not in candidates:
                candidates[-1] = int(example.act)
        candidate_count[row] = len(candidates) / float(FeatureAgent.ACT_SIZE)
        candidate_actions[row, : len(candidates)] = torch.tensor(candidates, dtype=torch.long)
        candidate_mask[row, : len(candidates)] = True
        candidate_rule_features[row, : len(candidates), :] = torch.from_numpy(
            example.candidate_rule_features[candidates].astype(np.float32)
        )
        if example.teacher_action_distribution is not None:
            teacher_values = np.asarray(example.teacher_action_distribution[candidates], dtype=np.float32)
            teacher_total = float(np.sum(teacher_values))
            if teacher_total > 0.0:
                teacher_target_dist[row, : len(candidates)] = torch.from_numpy(teacher_values / teacher_total)
                has_teacher_target[row] = True
                teacher_accept_top3[row] = bool(example.teacher_accept_top3)
        target_index[row] = candidates.index(int(example.act))

    scalar_features = torch.stack([players, candidate_count, allow_hu], dim=1)
    return {
        "observation": obs,
        "history_tokens": history,
        "candidate_actions": candidate_actions,
        "candidate_mask": candidate_mask,
        "candidate_rule_features": candidate_rule_features,
        "target_index": target_index,
        "teacher_target_dist": teacher_target_dist,
        "has_teacher_target": has_teacher_target,
        "teacher_accept_top3": teacher_accept_top3,
        "value_target": values,
        "scalar_features": scalar_features,
    }


class TransformerCandidateModel(nn.Module):
    def __init__(
        self,
        *,
        act_size: int = FeatureAgent.ACT_SIZE,
        history_vocab_size: int = 1024,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.act_size = int(act_size)
        self.history_vocab_size = int(history_vocab_size)
        self.obs_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FeatureAgent.OBS_SIZE * 4 * 9, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.history_embedding = nn.Embedding(history_vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(DEFAULT_HISTORY_LEN, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.scalar_encoder = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.action_embedding = nn.Embedding(act_size, d_model)
        self.action_feature_encoder = nn.Sequential(
            nn.Linear(4 + CANDIDATE_RULE_FEATURES, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.fuse = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

        action_type_ids, is_hu, is_play = _build_action_type_buffers(act_size)
        self.register_buffer("action_type_ids", action_type_ids, persistent=False)
        self.register_buffer("is_hu_action", is_hu, persistent=False)
        self.register_buffer("is_play_action", is_play, persistent=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        obs = batch["observation"].float()
        history_tokens = batch["history_tokens"].long().clamp(0, self.history_vocab_size - 1)
        candidate_actions = batch["candidate_actions"].long().clamp(0, self.act_size - 1)
        candidate_mask = batch["candidate_mask"].bool()
        scalar_features = batch["scalar_features"].float()

        state = self.obs_encoder(obs)
        seq_len = history_tokens.shape[1]
        positions = torch.arange(seq_len, device=history_tokens.device).clamp(max=DEFAULT_HISTORY_LEN - 1)
        history_emb = self.history_embedding(history_tokens) + self.position_embedding(positions)[None, :, :]
        history_encoded = self.history_encoder(history_emb)
        non_padding = history_tokens.ne(0).float().unsqueeze(-1)
        denom = non_padding.sum(dim=1).clamp_min(1.0)
        history_state = (history_encoded * non_padding).sum(dim=1) / denom
        scalar_state = self.scalar_encoder(scalar_features)
        fused = self.fuse(torch.cat([state, history_state, scalar_state], dim=1))

        action_emb = self.action_embedding(candidate_actions)
        action_features = torch.cat(
            [self._candidate_features(candidate_actions), batch["candidate_rule_features"].float()],
            dim=-1,
        )
        action_feature_state = self.action_feature_encoder(action_features)
        fused_candidates = fused[:, None, :].expand_as(action_emb)
        logits = self.scorer(torch.cat([fused_candidates, action_emb, action_feature_state], dim=2)).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, float("-inf"))
        values = self.value_head(fused).squeeze(-1)
        return logits, values

    def _candidate_features(self, actions: torch.Tensor) -> torch.Tensor:
        type_ids = self.action_type_ids[actions].float() / 6.0
        action_norm = actions.float() / max(1.0, float(self.act_size - 1))
        return torch.stack(
            [
                action_norm,
                type_ids,
                self.is_hu_action[actions].float(),
                self.is_play_action[actions].float(),
            ],
            dim=-1,
        )


def _build_action_type_buffers(act_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    type_ids = torch.zeros((act_size,), dtype=torch.long)
    is_hu = torch.zeros((act_size,), dtype=torch.bool)
    is_play = torch.zeros((act_size,), dtype=torch.bool)
    for action in range(act_size):
        try:
            head = action_response(action).split()[0]
        except Exception:
            head = "Pass"
        type_ids[action] = ACTION_TYPES.get(head, 0)
        is_hu[action] = head == "Hu"
        is_play[action] = head == "Play"
    return type_ids, is_hu, is_play


def count_model_parameters(model: nn.Module) -> dict[str, int]:
    parameters = 0
    trainable_parameters = 0
    state_tensor_bytes = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        parameters += count
        if parameter.requires_grad:
            trainable_parameters += count
    for tensor in model.state_dict().values():
        state_tensor_bytes += int(tensor.numel() * tensor.element_size())
    return {
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "state_tensor_bytes": state_tensor_bytes,
    }


class TransformerRawDataset(Dataset):
    def __init__(self, examples: list[TransformerExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TransformerExample:
        return self.examples[index]


def load_examples(
    paths: list[Path],
    *,
    history_len: int,
    max_records_per_source: int | None = None,
    max_examples: int | None = None,
    teacher_lookup: Callable[..., list | None] | None = None,
    teacher_temperature: float = 1.0,
) -> tuple[list[TransformerExample], dict]:
    examples: list[TransformerExample] = []
    source_summaries: list[dict] = []
    totals: Counter[str] = Counter()
    for path in paths:
        stats: Counter[str] = Counter()
        records = 0
        with path.open("r", encoding="utf-8-sig") as src:
            for line in src:
                if not line.strip():
                    continue
                if max_records_per_source is not None and records >= max_records_per_source:
                    break
                record = json.loads(line)
                records += 1
                try:
                    record_examples, record_stats = build_transformer_examples_from_record(
                        record,
                        history_len=history_len,
                        teacher_lookup=teacher_lookup,
                        teacher_temperature=teacher_temperature,
                    )
                except Exception as exc:
                    stats[f"record_error:{type(exc).__name__}"] += 1
                    continue
                stats.update(record_stats)
                for example in record_examples:
                    if max_examples is not None and len(examples) >= max_examples:
                        break
                    examples.append(example)
                if max_examples is not None and len(examples) >= max_examples:
                    break
        stats["records"] = records
        stats["path"] = str(path)
        totals.update(stats)
        source_summaries.append(dict(stats))
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples, {"examples": len(examples), "source_summaries": source_summaries, "totals": dict(totals)}


def is_reviewed_example(example: TransformerExample) -> bool:
    return bool(getattr(example, "teacher_candidate_norms", ()))


def has_mapped_teacher_distribution(example: TransformerExample) -> bool:
    return example.teacher_action_distribution is not None


def filter_reviewed_examples(
    examples: list[TransformerExample],
    *,
    require_distribution: bool,
) -> tuple[list[TransformerExample], dict[str, int]]:
    reviewed = [example for example in examples if is_reviewed_example(example)]
    without_distribution = [
        example for example in reviewed if not has_mapped_teacher_distribution(example)
    ]
    filtered = [
        example
        for example in reviewed
        if not require_distribution or has_mapped_teacher_distribution(example)
    ]
    return filtered, {
        "reviewed_examples_before_filter": len(reviewed),
        "reviewed_without_teacher_distribution": len(without_distribution),
        "reviewed_examples_after_filter": len(filtered),
    }


def reviewed_sampling_weights(
    examples: list[TransformerExample],
    *,
    reviewed_batch_fraction: float | None,
) -> torch.Tensor | None:
    if reviewed_batch_fraction is None or reviewed_batch_fraction <= 0.0:
        return None
    if reviewed_batch_fraction >= 1.0:
        raise ValueError("reviewed_batch_fraction must be less than 1.0")
    reviewed_mask = [is_reviewed_example(example) for example in examples]
    reviewed_count = sum(1 for value in reviewed_mask if value)
    unreviewed_count = len(examples) - reviewed_count
    if reviewed_count == 0 or unreviewed_count == 0:
        return None
    reviewed_weight = float(reviewed_batch_fraction) / float(reviewed_count)
    unreviewed_weight = (1.0 - float(reviewed_batch_fraction)) / float(unreviewed_count)
    return torch.tensor(
        [reviewed_weight if reviewed else unreviewed_weight for reviewed in reviewed_mask],
        dtype=torch.double,
    )


def validate_reviewed_training_args(args: argparse.Namespace) -> None:
    if (args.train_reviewed_only or args.val_reviewed_only) and args.max_candidates < FeatureAgent.ACT_SIZE:
        raise ValueError(f"reviewed-only CHAGA training requires --max-candidates {FeatureAgent.ACT_SIZE}")
    if args.reviewed_batch_fraction is not None and not 0.0 <= args.reviewed_batch_fraction < 1.0:
        raise ValueError("--reviewed-batch-fraction must be in [0, 1)")


class ReviewTargetLookup:
    def __init__(self, entries_by_key: dict[tuple, Iterable[ReviewTarget]]) -> None:
        self.entries_by_key = {tuple(key): deque(value) for key, value in entries_by_key.items()}

    def __call__(self, **kwargs) -> ReviewTarget | None:
        turn = str(kwargs.get("turn") if kwargs.get("turn") is not None else "")
        base_key = (
            str(kwargs.get("record_id") or ""),
            str(kwargs.get("player")),
            str(kwargs.get("request") or ""),
            normalize_teacher_action(str(kwargs.get("response") or "")),
        )
        key = (*base_key, turn)
        queue = self.entries_by_key.get(key)
        if not queue:
            queue = self.entries_by_key.get((*base_key, ""))
        if not queue:
            queue = self.entries_by_key.get(base_key)
        if not queue:
            return None
        return queue.popleft()


def load_review_target_lookup(path: Path) -> ReviewTargetLookup:
    entries: dict[tuple[str, str, str, str, str], deque[ReviewTarget]] = {}
    with path.open("r", encoding="utf-8-sig") as src:
        for line in src:
            if not line.strip():
                continue
            entry = json.loads(line)
            checks = entry.get("checks") or {}
            if not _review_checks_pass(checks):
                continue
            state_context = entry.get("state_context") or {}
            candidates = entry.get("chaga_top5_candidates") or []
            if not candidates:
                continue
            turn_value = state_context.get("turn", entry.get("state_turn", entry.get("turn", "")))
            key = (
                str(entry.get("record_id") or ""),
                str(entry.get("seat")),
                str(state_context.get("request") or ""),
                normalize_teacher_action(str(state_context.get("state_actual_response") or entry.get("state_actual_response") or entry.get("human_action") or "")),
                str(turn_value if turn_value is not None else ""),
            )
            accept_top3 = _is_first_six_discard_review_entry(entry)
            entries.setdefault(key, deque()).append(ReviewTarget(candidates=candidates, accept_top3=accept_top3))
    return ReviewTargetLookup(entries)


def make_review_lookup(path: str | None) -> ReviewTargetLookup | None:
    return load_review_target_lookup(Path(path)) if path else None


def _review_checks_pass(checks: dict) -> bool:
    required = [
        "offered_tile_matches",
        "drawn_tile_matches",
        "current_actor_matches",
        "window_matches",
        "hand_size_mod_ok",
        "top1_in_legal_mask",
    ]
    return all(bool(checks.get(key)) for key in required)


def _is_first_six_discard_review_entry(entry: dict) -> bool:
    action = normalize_teacher_action(str(entry.get("human_action") or entry.get("state_actual_response") or ""))
    if not action.startswith("PLAY "):
        return False
    try:
        ordinal = int(entry.get("play_ordinal") or 0)
    except (TypeError, ValueError):
        return False
    return 1 <= ordinal <= 6


def split_examples(
    examples: list[TransformerExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[TransformerExample], list[TransformerExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_count = int(len(shuffled) * val_ratio)
    return shuffled[val_count:], shuffled[:val_count]


def evaluate_model(
    model: TransformerCandidateModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int | None]:
    model.eval()
    total = correct = play_total = play_correct = 0
    teacher_total = teacher_top1_correct = teacher_top3_correct = teacher_relaxed_correct = 0
    teacher_play_total = teacher_play_top1_correct = teacher_play_top3_correct = teacher_play_relaxed_correct = 0
    loss_total = value_loss_total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits, values = model(batch)
            policy_loss = F.cross_entropy(logits, batch["target_index"])
            value_loss = F.mse_loss(values, batch["value_target"])
            pred_index = torch.argmax(logits, dim=1)
            pred_actions = batch["candidate_actions"].gather(1, pred_index[:, None]).squeeze(1)
            target_actions = batch["candidate_actions"].gather(1, batch["target_index"][:, None]).squeeze(1)
            has_teacher = batch.get("has_teacher_target")
            if has_teacher is not None and bool(has_teacher.any().item()):
                teacher_dist = batch["teacher_target_dist"]
                accept_top3_flags = batch.get("teacher_accept_top3")
                for row in torch.nonzero(has_teacher, as_tuple=False).flatten().detach().cpu().tolist():
                    row_dist = teacher_dist[row]
                    positive = torch.nonzero(row_dist > 0, as_tuple=False).flatten()
                    if int(positive.numel()) == 0:
                        continue
                    teacher_total += 1
                    top1_index = int(torch.argmax(row_dist).item())
                    topk = _teacher_topk_indices(row_dist, batch["candidate_mask"][row], limit=3)
                    pred_slot = int(pred_index[row].item())
                    top1_action = int(batch["candidate_actions"][row, top1_index].item())
                    accept_top3 = bool(accept_top3_flags[row].item()) if accept_top3_flags is not None else False
                    if pred_slot == top1_index:
                        teacher_top1_correct += 1
                    top3_match = bool((topk == pred_slot).any().item())
                    if top3_match:
                        teacher_top3_correct += 1
                    if pred_slot == top1_index or (accept_top3 and top3_match):
                        teacher_relaxed_correct += 1
                    if action_response(top1_action).startswith("Play "):
                        teacher_play_total += 1
                        if pred_slot == top1_index:
                            teacher_play_top1_correct += 1
                        if top3_match:
                            teacher_play_top3_correct += 1
                        if pred_slot == top1_index or (accept_top3 and top3_match):
                            teacher_play_relaxed_correct += 1
            batch_size = int(target_actions.numel())
            total += batch_size
            correct += int((pred_actions == target_actions).sum().item())
            for predicted, target in zip(pred_actions.detach().cpu().tolist(), target_actions.detach().cpu().tolist()):
                if action_response(int(target)).startswith("Play "):
                    play_total += 1
                    if int(predicted) == int(target):
                        play_correct += 1
            loss_total += float(policy_loss.item()) * batch_size
            value_loss_total += float(value_loss.item()) * batch_size
    return {
        "loss": loss_total / total if total else None,
        "value_loss": value_loss_total / total if total else None,
        "accuracy": correct / total if total else None,
        "play_exact_accuracy": play_correct / play_total if play_total else None,
        "teacher_top1_accuracy": teacher_top1_correct / teacher_total if teacher_total else None,
        "teacher_top3_inclusion": teacher_top3_correct / teacher_total if teacher_total else None,
        "teacher_relaxed_accuracy": teacher_relaxed_correct / teacher_total if teacher_total else None,
        "teacher_samples": teacher_total,
        "teacher_play_top1_accuracy": teacher_play_top1_correct / teacher_play_total if teacher_play_total else None,
        "teacher_play_top3_inclusion": teacher_play_top3_correct / teacher_play_total if teacher_play_total else None,
        "teacher_play_relaxed_accuracy": teacher_play_relaxed_correct / teacher_play_total if teacher_play_total else None,
        "teacher_play_samples": teacher_play_total,
        "samples": total,
        "play_samples": play_total,
    }


def _teacher_topk_indices(row_dist: torch.Tensor, candidate_mask: torch.Tensor, *, limit: int) -> torch.Tensor:
    positive = (row_dist > 0) & candidate_mask.bool()
    if not bool(positive.any().item()):
        return torch.empty((0,), dtype=torch.long, device=row_dist.device)
    scores = row_dist.masked_fill(~positive, float("-inf"))
    k = min(int(limit), int(positive.sum().item()))
    return torch.topk(scores, k=k).indices


def policy_loss_with_optional_teacher(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    hard_loss_weight: float = 1.0,
    teacher_loss_weight: float = 0.0,
    reviewed_hard_loss_weight: float | None = None,
    reviewed_teacher_loss_weight: float | None = None,
    unreviewed_hard_loss_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hard_losses = F.cross_entropy(logits, batch["target_index"], reduction="none")
    hard_loss = hard_losses.mean()
    has_teacher = batch.get("has_teacher_target")
    if has_teacher is None:
        has_teacher = torch.zeros_like(batch["target_index"], dtype=torch.bool)
    else:
        has_teacher = has_teacher.bool()
    teacher_loss = logits.new_tensor(0.0)
    reviewed_teacher_loss = logits.new_tensor(0.0)
    if bool(has_teacher.any().item()):
        log_probs = F.log_softmax(logits[has_teacher], dim=1)
        teacher_dist = batch["teacher_target_dist"][has_teacher].to(log_probs.device)
        safe_log_probs = torch.where(teacher_dist > 0.0, log_probs, torch.zeros_like(log_probs))
        reviewed_teacher_loss = -(teacher_dist * safe_log_probs).sum(dim=1).mean()
        teacher_loss = reviewed_teacher_loss
    reviewed_hard_loss = hard_losses[has_teacher].mean() if bool(has_teacher.any().item()) else logits.new_tensor(0.0)
    unreviewed_mask = ~has_teacher
    unreviewed_hard_loss = (
        hard_losses[unreviewed_mask].mean() if bool(unreviewed_mask.any().item()) else logits.new_tensor(0.0)
    )
    if (
        reviewed_hard_loss_weight is not None
        or reviewed_teacher_loss_weight is not None
        or unreviewed_hard_loss_weight is not None
    ):
        loss = (
            float(reviewed_hard_loss_weight if reviewed_hard_loss_weight is not None else hard_loss_weight)
            * reviewed_hard_loss
            + float(reviewed_teacher_loss_weight if reviewed_teacher_loss_weight is not None else teacher_loss_weight)
            * reviewed_teacher_loss
            + float(unreviewed_hard_loss_weight if unreviewed_hard_loss_weight is not None else hard_loss_weight)
            * unreviewed_hard_loss
        )
    else:
        loss = hard_loss_weight * hard_loss + teacher_loss_weight * teacher_loss
    return loss, {
        "hard_policy_loss": hard_loss.detach(),
        "teacher_policy_loss": teacher_loss.detach(),
        "reviewed_hard_policy_loss": reviewed_hard_loss.detach(),
        "reviewed_teacher_policy_loss": reviewed_teacher_loss.detach(),
        "unreviewed_hard_policy_loss": unreviewed_hard_loss.detach(),
    }


def is_better_checkpoint_metric(
    candidate: dict[str, float | int | None],
    incumbent: dict[str, float | int | None] | None,
    *,
    monitor_metric: str = "val_play_exact_accuracy",
) -> bool:
    """Rank checkpoints by CHAGA PLAY exact first, then overall exact and loss."""

    if incumbent is None:
        return True
    return _checkpoint_metric_tuple(candidate, monitor_metric) > _checkpoint_metric_tuple(incumbent, monitor_metric)


def _checkpoint_metric_tuple(metrics: dict[str, float | int | None], monitor_metric: str) -> tuple[float, float, float]:
    primary = _metric_value(metrics.get(monitor_metric), default=float("-inf"))
    val_accuracy = _metric_value(metrics.get("val_accuracy"), default=float("-inf"))
    val_loss = _metric_value(metrics.get("val_loss"), default=float("inf"))
    return (primary, val_accuracy, -val_loss)


def _metric_value(value: float | int | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def train(args: argparse.Namespace) -> dict:
    validate_reviewed_training_args(args)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_teacher_lookup = make_review_lookup(args.review_audit_jsonl)

    train_examples, train_load_summary = load_examples(
        [Path(path) for path in args.raw],
        history_len=args.history_len,
        max_records_per_source=args.max_records_per_source,
        max_examples=args.max_train_examples,
        teacher_lookup=train_teacher_lookup,
        teacher_temperature=args.teacher_temperature,
    )
    if args.eval_raw:
        val_teacher_lookup = make_review_lookup(args.review_audit_jsonl)
        val_examples, val_load_summary = load_examples(
            [Path(path) for path in args.eval_raw],
            history_len=args.history_len,
            max_records_per_source=args.max_eval_records_per_source,
            max_examples=args.max_val_examples,
            teacher_lookup=val_teacher_lookup,
            teacher_temperature=args.teacher_temperature,
        )
    else:
        train_examples, val_examples = split_examples(train_examples, val_ratio=args.val_ratio, seed=seed)
        val_load_summary = {"examples": len(val_examples), "source_summaries": [], "totals": {}}
    train_review_filter_summary: dict[str, int] | None = None
    val_review_filter_summary: dict[str, int] | None = None
    if args.train_reviewed_only:
        train_examples, train_review_filter_summary = filter_reviewed_examples(
            train_examples,
            require_distribution=args.require_teacher_distribution,
        )
    if args.val_reviewed_only:
        val_examples, val_review_filter_summary = filter_reviewed_examples(
            val_examples,
            require_distribution=args.require_teacher_distribution,
        )
    if not train_examples:
        raise ValueError("no training examples built")
    if not val_examples:
        raise ValueError("no validation examples built")

    collate = lambda items: collate_transformer_examples(items, max_candidates=args.max_candidates)
    train_sampling_weights = reviewed_sampling_weights(
        train_examples,
        reviewed_batch_fraction=args.reviewed_batch_fraction,
    )
    train_sampler = (
        WeightedRandomSampler(train_sampling_weights, num_samples=len(train_examples), replacement=True)
        if train_sampling_weights is not None
        else None
    )
    train_loader = DataLoader(
        TransformerRawDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        TransformerRawDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TransformerCandidateModel(
        history_vocab_size=args.history_vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    model_size = count_model_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics = {
        "format": "mcr_transformer_candidate_v1",
        "raw": args.raw,
        "eval_raw": args.eval_raw,
        "train_load_summary": train_load_summary,
        "val_load_summary": val_load_summary,
        "train_review_filter_summary": train_review_filter_summary,
        "val_review_filter_summary": val_review_filter_summary,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "history_len": args.history_len,
        "max_candidates": args.max_candidates,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hard_loss_weight": args.hard_loss_weight,
        "teacher_loss_weight": args.teacher_loss_weight,
        "reviewed_batch_fraction": args.reviewed_batch_fraction,
        "reviewed_hard_loss_weight": args.reviewed_hard_loss_weight,
        "reviewed_teacher_loss_weight": args.reviewed_teacher_loss_weight,
        "unreviewed_hard_loss_weight": args.unreviewed_hard_loss_weight,
        "train_sampling": {
            "weighted": train_sampler is not None,
            "reviewed_examples": sum(1 for example in train_examples if is_reviewed_example(example)),
            "unreviewed_examples": sum(1 for example in train_examples if not is_reviewed_example(example)),
        },
        "teacher_temperature": args.teacher_temperature,
        "review_audit_jsonl": args.review_audit_jsonl,
        "monitor_metric": args.monitor_metric,
        "model_size": model_size,
        "device": str(device),
        "started_at": int(time.time()),
        "history": [],
    }

    best_epoch_metrics: dict[str, float | int | None] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        model.train()
        total = correct = 0
        loss_total = policy_loss_total = value_loss_total = 0.0
        for batch in train_loader:
            batch = _to_device(batch, device)
            logits, values = model(batch)
            policy_loss, policy_parts = policy_loss_with_optional_teacher(
                logits,
                batch,
                hard_loss_weight=args.hard_loss_weight,
                teacher_loss_weight=args.teacher_loss_weight,
                reviewed_hard_loss_weight=args.reviewed_hard_loss_weight,
                reviewed_teacher_loss_weight=args.reviewed_teacher_loss_weight,
                unreviewed_hard_loss_weight=args.unreviewed_hard_loss_weight,
            )
            value_loss = F.mse_loss(values, batch["value_target"])
            loss = policy_loss + args.value_loss_weight * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            pred_index = torch.argmax(logits.detach(), dim=1)
            pred_actions = batch["candidate_actions"].gather(1, pred_index[:, None]).squeeze(1)
            target_actions = batch["candidate_actions"].gather(1, batch["target_index"][:, None]).squeeze(1)
            batch_size = int(target_actions.numel())
            total += batch_size
            correct += int((pred_actions == target_actions).sum().item())
            loss_total += float(loss.item()) * batch_size
            policy_loss_total += float(policy_loss.item()) * batch_size
            value_loss_total += float(value_loss.item()) * batch_size
        val_metrics = evaluate_model(model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": loss_total / total if total else None,
            "train_policy_loss": policy_loss_total / total if total else None,
            "train_value_loss": value_loss_total / total if total else None,
            "train_accuracy": correct / total if total else None,
            "train_samples": total,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        metrics["history"].append(epoch_metrics)
        if is_better_checkpoint_metric(epoch_metrics, best_epoch_metrics, monitor_metric=args.monitor_metric):
            best_epoch_metrics = dict(epoch_metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(json.dumps(epoch_metrics, ensure_ascii=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    checkpoint = {
        "model_state": best_state,
        "config": {
            "history_vocab_size": args.history_vocab_size,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "history_len": args.history_len,
            "max_candidates": args.max_candidates,
        },
        "selection": {
            "monitor_metric": args.monitor_metric,
            "best_epoch": best_epoch_metrics,
        },
    }
    torch.save(checkpoint, out_path)
    metrics["checkpoint"] = str(out_path)
    metrics["best_epoch"] = best_epoch_metrics
    metrics["checkpoint_epoch"] = best_epoch_metrics.get("epoch") if best_epoch_metrics else None
    metrics["finished_at"] = int(time.time())
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="append", required=True, help="Training raw/converted JSONL path; repeatable")
    parser.add_argument("--eval-raw", action="append", default=None, help="Held-out raw/converted JSONL path; repeatable")
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--value-loss-weight", type=float, default=0.2)
    parser.add_argument("--hard-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0)
    parser.add_argument("--reviewed-batch-fraction", type=float, default=None)
    parser.add_argument("--reviewed-hard-loss-weight", type=float, default=None)
    parser.add_argument("--reviewed-teacher-loss-weight", type=float, default=None)
    parser.add_argument("--unreviewed-hard-loss-weight", type=float, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--review-audit-jsonl", default=None)
    parser.add_argument("--train-reviewed-only", action="store_true")
    parser.add_argument("--val-reviewed-only", action="store_true")
    parser.add_argument("--require-teacher-distribution", action="store_true")
    parser.add_argument("--monitor-metric", default="val_play_exact_accuracy")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--max-eval-records-per-source", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--history-len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--history-vocab-size", type=int, default=1024)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    metrics = train(args)
    printable = {key: value for key, value in metrics.items() if key != "history"}
    printable["last_epoch"] = metrics["history"][-1] if metrics["history"] else None
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
