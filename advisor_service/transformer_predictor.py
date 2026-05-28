"""Live inference wrapper for Transformer candidate checkpoints."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
LAWLORENTZ_DIR = WORKSPACE_ROOT / "external" / "Chinese-Standard-Mahjong-DRL"
for path in (SCRIPTS_DIR, LAWLORENTZ_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lawlorentz_policy import LawlorentzEffectiveScorer  # noqa: E402


class TransformerCheckpointPredictor:
    """Rank already fan-gated live advisor candidates with a trained checkpoint.

    The live extension currently observes a normalized Tziakcha snapshot, not a
    full Botzone replay stream. This wrapper therefore supplies a conservative
    hand-only observation and lets the existing rule layer remain authoritative
    for legality and Hu fan checks.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        model: Any | None = None,
        config: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        (
            self.torch,
            self.FeatureAgent,
            self.TransformerCandidateModel,
            self.CANDIDATE_RULE_FEATURES,
        ) = _training_symbols()
        self.device = self.torch.device(device or ("cuda" if self.torch.cuda.is_available() else "cpu"))
        if model is None:
            if self.model_path is None:
                raise ValueError("model_path is required when model is not supplied")
            model, loaded_config = self._load_checkpoint(self.model_path)
            self.config = loaded_config
        else:
            self.config = dict(config or {})
        self.model = model.to(self.device) if hasattr(model, "to") else model
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.history_len = int(self.config.get("history_len", 80))
        self.max_candidates = int(self.config.get("max_candidates", 96))
        self.act_size = int(self.FeatureAgent.ACT_SIZE)
        self.obs_shape = (int(self.FeatureAgent.OBS_SIZE), 4, 9)

    def _load_checkpoint(self, path: Path) -> tuple[Any, dict[str, Any]]:
        checkpoint = self.torch.load(path, map_location=self.device)
        config = dict(checkpoint.get("config") or {})
        model = self.TransformerCandidateModel(
            history_vocab_size=config.get("history_vocab_size", 2048),
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 4),
            dim_feedforward=config.get("dim_feedforward", 512),
            dropout=config.get("dropout", 0.1),
        )
        model.load_state_dict(checkpoint["model_state"])
        return model, config

    def info(self) -> dict[str, Any]:
        return {
            "type": "transformer-checkpoint",
            "path": str(self.model_path) if self.model_path else None,
            "device": str(self.device),
            "config": dict(self.config),
        }

    def predict_legal_response(
        self,
        input_text: str,
        hand: Counter[str],
        player_id: int,
        request: str,
        candidates: list[str],
    ) -> str:
        del input_text
        mappings = self._map_candidates(request, candidates)
        if not mappings:
            return candidates[0] if candidates else "PASS"

        ordered_actions: list[int] = []
        responses_by_action: dict[int, list[str]] = {}
        for response, action in mappings:
            if action not in responses_by_action:
                ordered_actions.append(action)
                responses_by_action[action] = []
            responses_by_action[action].append(response)
        ordered_actions = ordered_actions[: self.max_candidates]
        if not ordered_actions:
            return candidates[0] if candidates else "PASS"

        batch = self._batch_for_live_state(hand, player_id, ordered_actions)
        with self.torch.no_grad():
            logits, _ = self.model(batch)
        pred_slot = int(self.torch.argmax(logits[0]).item())
        selected_action = int(ordered_actions[pred_slot])
        selected_responses = responses_by_action.get(selected_action) or [candidates[0]]
        return self._choose_response_for_action(selected_responses, hand, request)

    def _batch_for_live_state(
        self,
        hand: Counter[str],
        player_id: int,
        candidate_actions: list[int],
    ) -> dict[str, Any]:
        torch = self.torch
        observation = self._hand_observation(hand, player_id)
        padded_actions = np.zeros((1, self.max_candidates), dtype=np.int64)
        candidate_mask = np.zeros((1, self.max_candidates), dtype=np.bool_)
        count = min(len(candidate_actions), self.max_candidates)
        padded_actions[0, :count] = np.asarray(candidate_actions[:count], dtype=np.int64)
        candidate_mask[0, :count] = True
        rule_features = np.zeros(
            (1, self.max_candidates, self.CANDIDATE_RULE_FEATURES),
            dtype=np.float32,
        )
        scalar_features = np.asarray(
            [[float(player_id) / 3.0, float(count) / max(1.0, float(self.act_size)), 1.0 if 1 in candidate_actions else 0.0]],
            dtype=np.float32,
        )
        return {
            "observation": torch.from_numpy(observation[None, :, :, :]).to(self.device),
            "history_tokens": torch.zeros((1, self.history_len), dtype=torch.long, device=self.device),
            "candidate_actions": torch.from_numpy(padded_actions).to(self.device),
            "candidate_mask": torch.from_numpy(candidate_mask).to(self.device),
            "candidate_rule_features": torch.from_numpy(rule_features).to(self.device),
            "scalar_features": torch.from_numpy(scalar_features).to(self.device),
        }

    def _hand_observation(self, hand: Counter[str], player_id: int) -> np.ndarray:
        obs = np.zeros(self.obs_shape, dtype=np.float32)
        try:
            wind = f"F{int(player_id) + 1}"
            channel, column = divmod(self.FeatureAgent.OFFSET_TILE[wind], 9)
            obs[self.FeatureAgent.OFFSET_OBS["SEAT_WIND"], channel, column] = 1.0
        except Exception:
            pass
        hand_offset = int(self.FeatureAgent.OFFSET_OBS["HAND"])
        for tile, count in hand.items():
            tile_index = self.FeatureAgent.OFFSET_TILE.get(tile)
            if tile_index is None:
                continue
            channel, column = divmod(tile_index, 9)
            for copy_index in range(min(4, int(count))):
                obs[hand_offset + copy_index, channel, column] = 1.0
        return obs

    def _map_candidates(self, request: str, candidates: list[str]) -> list[tuple[str, int]]:
        mappings: list[tuple[str, int]] = []
        for response in candidates:
            action = self._response_to_action(request, response)
            if action is not None and 0 <= action < self.act_size:
                mappings.append((response, int(action)))
        return mappings

    def _response_to_action(self, request: str, response: str) -> int | None:
        parts = response.strip().split()
        if not parts:
            return None
        head = parts[0].upper()
        if head == "PASS":
            return int(self.FeatureAgent.OFFSET_ACT["Pass"])
        if head == "HU":
            return int(self.FeatureAgent.OFFSET_ACT["Hu"])
        if head == "PLAY" and len(parts) >= 2:
            return self._tile_action("Play", parts[1])
        if head == "PENG":
            event_tile = _event_tile(request)
            return self._tile_action("Peng", event_tile) if event_tile else self._first_action_with_head("Peng")
        if head == "CHI" and len(parts) >= 2:
            event_tile = _event_tile(request)
            if event_tile:
                try:
                    return int(self.FeatureAgent(0).response2action(f"Chi {event_tile} {parts[1]}"))
                except Exception:
                    pass
            return self._first_matching_action(f"Chi {parts[1]}")
        if head == "GANG":
            tile = parts[1] if len(parts) >= 2 else _event_tile(request)
            if tile:
                action = self._first_matching_action(f"Gang {tile}")
                if action is not None:
                    return action
                return self._tile_action("Gang", tile)
            return self._first_action_with_head("Gang")
        if head == "BUGANG" and len(parts) >= 2:
            return self._tile_action("BuGang", parts[1])
        return None

    def _tile_action(self, action: str, tile: str | None) -> int | None:
        if not tile:
            return None
        tile_index = self.FeatureAgent.OFFSET_TILE.get(tile)
        if tile_index is None:
            return None
        return int(self.FeatureAgent.OFFSET_ACT[action] + tile_index)

    def _first_matching_action(self, response: str) -> int | None:
        wanted = response.strip()
        for action in range(self.act_size):
            try:
                if self.FeatureAgent(0).action2response(action) == wanted:
                    return int(action)
            except Exception:
                continue
        return None

    def _first_action_with_head(self, head: str) -> int | None:
        for action in range(self.act_size):
            try:
                if self.FeatureAgent(0).action2response(action).split()[0] == head:
                    return int(action)
            except Exception:
                continue
        return None

    def _choose_response_for_action(self, responses: list[str], hand: Counter[str], request: str) -> str:
        if len(responses) == 1:
            return responses[0]
        claim_discards = [response for response in responses if _response_discard(response)]
        if not claim_discards:
            return responses[0]
        try:
            scorer = LawlorentzEffectiveScorer(packs=(), shown_tiles={}, seat_wind=0, prevalent_wind=0, levels=1)
            return max(
                claim_discards,
                key=lambda response: scorer.discard_key(
                    list(_after_claim_hand(hand, request, response).elements()),
                    _response_discard(response) or "",
                ),
            )
        except Exception:
            return claim_discards[0]


def _training_symbols():
    import torch
    from feature import FeatureAgent
    from train_transformer_candidate import CANDIDATE_RULE_FEATURES, TransformerCandidateModel

    return torch, FeatureAgent, TransformerCandidateModel, CANDIDATE_RULE_FEATURES


def _event_tile(request: str) -> str | None:
    tokens = request.strip().split()
    if len(tokens) >= 4 and tokens[0] == "3":
        return tokens[3] if tokens[2].upper() == "BUGANG" else tokens[-1]
    return None


def _response_discard(response: str) -> str | None:
    parts = response.strip().split()
    if len(parts) == 2 and parts[0].upper() == "PENG":
        return parts[1]
    if len(parts) == 3 and parts[0].upper() == "CHI":
        return parts[2]
    return None


def _after_claim_hand(hand: Counter[str], request: str, response: str) -> Counter[str]:
    after = Counter(hand)
    event_tile = _event_tile(request)
    parts = response.strip().split()
    if not parts or not event_tile:
        return after
    head = parts[0].upper()
    if head == "PENG":
        after[event_tile] -= 2
    elif head == "CHI" and len(parts) >= 2:
        middle = parts[1]
        try:
            rank = int(middle[1])
            sequence = [f"{middle[0]}{rank - 1}", middle, f"{middle[0]}{rank + 1}"]
        except (IndexError, ValueError):
            sequence = []
        needed = Counter(sequence)
        needed[event_tile] -= 1
        for tile, count in needed.items():
            if count > 0:
                after[tile] -= count
    for tile in list(after):
        if after[tile] <= 0:
            del after[tile]
    return after
