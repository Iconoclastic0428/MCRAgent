"""FeatureAgent-backed Transformer policy for exact official-judge self-play."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from advisor_service.transformer_predictor import TransformerCheckpointPredictor
from build_lawlorentz_dataset import (
    BotzoneFeatureRuntime,
    _is_trainable_response,
    _safe_apply_own_response,
    response_to_valid_action,
)
from legal_actions import apply_response
from policy_bot import BotzonePolicy
from train_transformer_candidate import (
    build_candidate_rule_features,
    encode_history_event,
    valid_transformer_observation,
)


class FeatureTransformerBotzonePolicy(BotzonePolicy):
    """Official-judge policy that scores candidates from full FeatureAgent state.

    ``BotzonePolicy`` remains the source of official response candidates and fan
    gating. This class only replaces the ranking input so the Transformer sees
    the same observation/rule-feature shape used during training.
    """

    def __init__(
        self,
        predictor: TransformerCheckpointPredictor,
        *,
        fan_checker=None,
        history_len: int | None = None,
    ) -> None:
        super().__init__(predictor=None, fan_checker=fan_checker)
        self.feature_predictor = predictor
        self.feature_runtime = BotzoneFeatureRuntime()
        self.feature_history: list[int] = []
        self.feature_skip_requests: Counter[str] = Counter()
        self.feature_history_len = int(history_len or getattr(predictor, "history_len", 80))
        self._feature_obs: dict[str, Any] | None = None

    def respond(self, request: str) -> str:
        self._feature_obs = self._observe_feature_state(request)
        response = super().respond(request)
        self._advance_feature_state(request, response)
        self._feature_obs = None
        return response

    def _observe_feature_state(self, request: str) -> dict[str, Any] | None:
        if self.feature_skip_requests[request] > 0:
            self.feature_skip_requests[request] -= 1
            self.stats["feature_skipped_self_events"] += 1
            return None
        try:
            obs = self.feature_runtime.observe(request)
        except Exception:
            self.stats["feature_observe_errors"] += 1
            return None
        return obs if isinstance(obs, dict) else None

    def _advance_feature_state(self, request: str, response: str) -> None:
        obs = self._feature_obs
        if obs is None or not _is_trainable_response(request, response):
            return
        action = response_to_valid_action(self.feature_runtime.agent, obs, request, response)
        if action is None:
            self.stats["feature_unmapped_responses"] += 1
            return
        self.feature_runtime.remember_response(request, action)
        for skipped in _safe_apply_own_response(
            self.feature_runtime,
            request,
            response,
            self.stats,
        ):
            self.feature_skip_requests[skipped] += 1
        player = int(self.player_id or 0)
        self.feature_history.append(encode_history_event(player, request, response))
        if len(self.feature_history) > self.feature_history_len * 4:
            self.feature_history = self.feature_history[-self.feature_history_len * 2 :]

    def _choose_draw_response(self, request: str, drawn_tile: str) -> str:
        self.stats["draw_turns"] += 1
        candidates = self._legal_responses(request)
        predicted = self._feature_prediction(request, candidates)
        if predicted in candidates:
            if predicted == "HU":
                self.stats["legal_hu_seen"] += 1
                self.stats["hu_taken"] += 1
            self._apply_draw_response(predicted)
            return predicted

        self.last_illegal_prediction = True
        self.stats["feature_prediction_fallbacks"] += 1
        fallback = f"PLAY {drawn_tile if self.hand[drawn_tile] else self._first_tile()}"
        self.last_fallback_used = True
        self.stats["fallbacks"] += 1
        self._apply_draw_response(fallback)
        return fallback

    def _choose_reaction_response(self, request: str) -> str:
        self.stats["reaction_turns"] += 1
        candidates = self._legal_responses(request)
        predicted = self._feature_prediction(request, candidates)
        if predicted in candidates:
            if predicted == "HU":
                self.stats["legal_hu_seen"] += 1
                self.stats["hu_taken"] += 1
            self._record_own_reaction_pack(request, predicted)
            apply_response(self.hand, request, predicted)
            return predicted

        self.last_illegal_prediction = True
        self.stats["feature_prediction_fallbacks"] += 1
        self.last_fallback_used = True
        self.stats["fallbacks"] += 1
        return "PASS"

    def _feature_prediction(self, request: str, candidates: list[str]) -> str:
        if not candidates:
            return "PASS"
        obs = self._feature_obs
        if obs is None or not valid_transformer_observation(obs, require_shanten=True):
            self.stats["feature_invalid_observations"] += 1
            return "PASS"
        candidate_actions: list[int] = []
        responses_by_action: dict[int, list[str]] = {}
        for response in candidates:
            action = response_to_valid_action(self.feature_runtime.agent, obs, request, response)
            if action is None:
                self.stats["feature_unmapped_candidates"] += 1
                continue
            action = int(action)
            if action not in responses_by_action:
                candidate_actions.append(action)
                responses_by_action[action] = []
            responses_by_action[action].append(response)
        if not candidate_actions:
            self.stats["feature_empty_candidate_actions"] += 1
            return "PASS"
        allow_hu = any(response.split()[0].upper() == "HU" for response in candidates)
        rule_features = build_candidate_rule_features(
            self.feature_runtime.agent,
            obs,
            allow_hu=allow_hu,
        )
        history = _history_window(self.feature_history, self.feature_history_len)
        try:
            prediction = self.feature_predictor.predict_feature_response(
                observation=np.asarray(obs["observation"], dtype=np.float32),
                history_tokens=history,
                player_id=int(self.player_id or 0),
                candidate_actions=candidate_actions,
                candidate_rule_features=rule_features,
                responses_by_action=responses_by_action,
            ).strip()
        except Exception:
            self.stats["feature_prediction_errors"] += 1
            return "PASS"
        self.last_model_response = prediction
        self.stats["feature_predictions"] += 1
        return prediction

    def diagnostics(self) -> dict[str, int | str | bool | None]:
        data = super().diagnostics()
        data.update(
            {
                "kind": "feature_transformer_policy",
                "feature_history_events": len(self.feature_history),
                "feature_pending_skip_requests": sum(self.feature_skip_requests.values()),
            }
        )
        return data


def _history_window(history: list[int], history_len: int) -> np.ndarray:
    out = np.zeros((history_len,), dtype=np.int64)
    values = history[-history_len:]
    if values:
        out[-len(values) :] = np.asarray(values, dtype=np.int64)
    return out

