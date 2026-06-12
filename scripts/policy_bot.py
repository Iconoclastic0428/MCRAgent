#!/usr/bin/env python3
"""Botzone-style policy wrapper for trained MCR behavior-cloning models."""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Protocol

from hand_features import candidate_feature_text, min_shanten, regular_shanten, remove_one_tile
from feature_ranker import featurize_legal_responses
from fan_feature_ranker import featurize_fan_legal_responses
from effective_tiles import evaluate_discard_candidates
from legal_actions import (
    apply_response,
    generate_legal_responses,
    request_event_tile,
    response_candidate_text,
)


class Predictor(Protocol):
    def predict_response(self, input_text: str) -> str:
        ...


class SklearnPredictor:
    def __init__(self, model_path: Path):
        with model_path.open("rb") as src:
            payload = pickle.load(src)
        self.payload = payload
        self.pipeline = payload.get("pipeline")
        self.model = payload.get("model")
        self.kind = payload.get("kind", "response_classifier")
        self.suppress_hu = bool(payload.get("suppress_hu", False))
        self.prefer_hu = bool(payload.get("prefer_hu", False))
        self.suppress_actions = {
            str(action).upper() for action in payload.get("suppress_actions", [])
        }
        self.effective_tile_overlay = payload.get("effective_tile_overlay")
        if self.kind in {"composite_policy", "draw_ensemble_composite_policy"}:
            self.draw_pipeline = payload.get("draw_payload", {}).get("pipeline")
            self.draw_kind = payload.get("draw_payload", {}).get("kind", "discard_ranker")
            self.draw_payloads = payload.get("draw_payloads") or []
            self.draw_weights = payload.get("draw_weights") or [1.0] * len(self.draw_payloads)
            self.reaction_pipeline = payload["reaction_payload"].get("pipeline")
            self.reaction_kind = payload["reaction_payload"].get("kind", "legal_action_ranker")
            self.reaction_thresholds = payload.get("reaction_thresholds") or {}
            self.suppress_hu = self.suppress_hu or bool(
                payload.get("draw_payload", {}).get("suppress_hu", False)
            ) or bool(payload["reaction_payload"].get("suppress_hu", False))
            self.suppress_actions.update(
                str(action).upper()
                for action in payload.get("draw_payload", {}).get("suppress_actions", [])
            )
            for draw_payload in self.draw_payloads:
                self.suppress_actions.update(
                    str(action).upper() for action in draw_payload.get("suppress_actions", [])
                )
            self.suppress_actions.update(
                str(action).upper()
                for action in payload["reaction_payload"].get("suppress_actions", [])
            )

    def predict_response(self, input_text: str) -> str:
        if self.pipeline is None:
            return "PASS"
        return str(self.pipeline.predict([input_text])[0])

    def predict_draw_response(self, input_text: str, hand: Counter[str]) -> str:
        if self.kind == "composite_policy":
            if self.draw_kind == "feature_action_ranker":
                return self._predict_feature_payload_response(
                    input_text, hand, None, "2 _", self.payload["draw_payload"]
                )
            return self._predict_discard_ranker_response(input_text, hand, self.draw_pipeline)
        if self.kind == "draw_ensemble_composite_policy":
            return self._predict_feature_ensemble_response(input_text, hand, None, "2 _")
        if self.kind == "feature_action_ranker":
            return self._predict_feature_payload_response(input_text, hand, None, "2 _", self.payload)
        if self.kind != "discard_ranker":
            return self.predict_response(input_text)
        return self._predict_discard_ranker_response(input_text, hand, self.pipeline)

    def _predict_discard_ranker_response(self, input_text: str, hand: Counter[str], pipeline) -> str:
        candidates = sorted(tile for tile, count in hand.items() if count > 0)
        if not candidates:
            return "PLAY W1"
        hand_tiles = list(hand.elements())
        drawn_tile = ShantenHeuristicPredictor()._drawn_tile(input_text)
        texts = [
            f"{input_text}\n{candidate_feature_text(hand_tiles, tile, drawn_tile=drawn_tile)}"
            for tile in candidates
        ]
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(texts)
            classes = list(pipeline.classes_)
            positive_index = classes.index(1) if 1 in classes else classes.index("1")
            scores = [row[positive_index] for row in proba]
        else:
            scores = pipeline.decision_function(texts)
        best_tile = candidates[max(range(len(candidates)), key=lambda index: scores[index])]
        return f"PLAY {best_tile}"

    def predict_legal_response(
        self,
        input_text: str,
        hand: Counter[str],
        player_id: int | None,
        request: str,
        candidates: list[str] | None = None,
    ) -> str:
        candidates = self._filter_candidates(
            candidates or generate_legal_responses(player_id, request, hand)
        )
        if not candidates:
            return "PASS"
        if self.kind == "composite_policy" and request.startswith("2 "):
            if self.draw_kind == "feature_action_ranker":
                return self._predict_feature_payload_response(
                    input_text, hand, player_id, request, self.payload["draw_payload"], candidates
                )
            return self.predict_draw_response(input_text, hand)
        if self.kind == "draw_ensemble_composite_policy" and request.startswith("2 "):
            return self._predict_feature_ensemble_response(
                input_text, hand, player_id, request, candidates
            )
        if self.kind == "feature_action_ranker":
            return self._predict_feature_payload_response(
                input_text, hand, player_id, request, self.payload, candidates
            )
        if (
            self.kind in {"composite_policy", "draw_ensemble_composite_policy"}
            and self.reaction_kind == "feature_action_ranker"
        ):
            return self._predict_feature_payload_response(
                input_text, hand, player_id, request, self.payload["reaction_payload"], candidates
            )
        pipeline = (
            self.reaction_pipeline
            if self.kind in {"composite_policy", "draw_ensemble_composite_policy"}
            else self.pipeline
        )
        texts = [
            response_candidate_text(input_text, response, hand, request)
            for response in candidates
        ]
        scores = self._score_text_candidates(pipeline, texts)
        return self._choose_thresholded_response(candidates, scores, request)

    def score_legal_response_candidates(
        self,
        input_text: str,
        hand: Counter[str],
        player_id: int | None,
        request: str,
        candidates: list[str],
    ) -> list[float]:
        if self.kind == "draw_ensemble_composite_policy" and request.startswith("2 "):
            return self._score_feature_ensemble_responses(input_text, hand, request, candidates)
        if self.kind == "composite_policy" and request.startswith("2 "):
            if self.draw_kind == "feature_action_ranker":
                return self._score_feature_payload_response(
                    input_text, hand, request, self.payload["draw_payload"], candidates
                )
            return _one_hot_response_scores(
                candidates, self._predict_discard_ranker_response(input_text, hand, self.draw_pipeline)
            )
        if self.kind == "feature_action_ranker":
            return self._score_feature_payload_response(
                input_text, hand, request, self.payload, candidates
            )
        predicted = self.predict_legal_response(input_text, hand, player_id, request, candidates)
        return _one_hot_response_scores(candidates, predicted)

    def _score_text_candidates(self, pipeline, texts: list[str]) -> list[float]:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(texts)
            classes = list(pipeline.classes_)
            positive_index = classes.index(1) if 1 in classes else classes.index("1")
            return [float(row[positive_index]) for row in proba]
        return [float(score) for score in pipeline.decision_function(texts)]

    def _filter_candidates(self, candidates: list[str]) -> list[str]:
        if not self.suppress_hu and not self.suppress_actions:
            return candidates
        filtered = []
        for candidate in candidates:
            action = candidate.split()[0].upper() if candidate else "PASS"
            if self.suppress_hu and action == "HU":
                continue
            if action in self.suppress_actions:
                continue
            filtered.append(candidate)
        return filtered or ["PASS"]

    def _choose_thresholded_response(
        self, candidates: list[str], scores: list[float], request: str
    ) -> str:
        best_index = max(range(len(candidates)), key=lambda index: scores[index])
        best_response = candidates[best_index]
        thresholds = getattr(self, "reaction_thresholds", {}) if request.startswith("3 ") else {}
        if not thresholds or best_response == "PASS":
            return best_response

        pass_score = scores[candidates.index("PASS")] if "PASS" in candidates else 0.0
        min_margin = float(thresholds.get("min_margin", 0.0))
        min_score = thresholds.get("min_score")
        if scores[best_index] - pass_score < min_margin:
            return "PASS"
        if min_score is not None and scores[best_index] < float(min_score):
            return "PASS"
        return best_response

    def _predict_feature_payload_response(
        self,
        input_text: str,
        hand: Counter[str],
        player_id: int | None,
        request: str,
        payload: dict,
        candidates: list[str] | None = None,
    ) -> str:
        responses = candidates or generate_legal_responses(player_id, request, hand)
        if not responses:
            return "PASS"
        model = payload["model"]
        features = self._featurize_payload_responses(input_text, request, hand, responses, payload)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            scores = [row[1] for row in proba]
        else:
            scores = model.decision_function(features)
        return responses[max(range(len(responses)), key=lambda index: scores[index])]

    def _score_feature_payload_response(
        self,
        input_text: str,
        hand: Counter[str],
        request: str,
        payload: dict,
        responses: list[str],
    ) -> list[float]:
        if payload.get("kind") != "feature_action_ranker":
            raise ValueError("draw ensemble currently supports feature_action_ranker payloads only")
        model = payload["model"]
        features = self._featurize_payload_responses(input_text, request, hand, responses, payload)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            return [float(row[1]) for row in proba]
        return [float(score) for score in model.decision_function(features)]

    def _featurize_payload_responses(
        self,
        input_text: str,
        request: str,
        hand: Counter[str],
        responses: list[str],
        payload: dict,
    ):
        if payload.get("feature_mode") == "numeric_fan_v1":
            return featurize_fan_legal_responses(input_text, request, hand, responses)
        return featurize_legal_responses(input_text, request, hand, responses)

    def _predict_feature_ensemble_response(
        self,
        input_text: str,
        hand: Counter[str],
        player_id: int | None,
        request: str,
        candidates: list[str] | None = None,
    ) -> str:
        responses = candidates or generate_legal_responses(player_id, request, hand)
        if not responses:
            return "PASS"
        scores = self._score_feature_ensemble_responses(input_text, hand, request, responses)
        return responses[max(range(len(responses)), key=lambda index: scores[index])]

    def _score_feature_ensemble_responses(
        self,
        input_text: str,
        hand: Counter[str],
        request: str,
        responses: list[str],
    ) -> list[float]:
        weights = [float(weight) for weight in self.draw_weights]
        if len(weights) != len(self.draw_payloads):
            weights = [1.0] * len(self.draw_payloads)
        total_weight = sum(weights) or 1.0
        scores = [0.0] * len(responses)
        for payload, weight in zip(self.draw_payloads, weights):
            payload_scores = self._score_feature_payload_response(
                input_text, hand, request, payload, responses
            )
            for index, score in enumerate(payload_scores):
                scores[index] += (weight / total_weight) * score
        return scores


def _one_hot_response_scores(candidates: list[str], predicted: str) -> list[float]:
    return [1.0 if candidate == predicted else 0.0 for candidate in candidates]


def dedupe_responses(responses: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for response in responses:
        if response not in seen:
            result.append(response)
            seen.add(response)
    return result


def hand_elements(hand: Counter[str] | dict[str, int]) -> list[str]:
    tiles: list[str] = []
    for tile, count in hand.items():
        tiles.extend([tile] * int(count))
    return tiles


class ShantenHeuristicPredictor:
    def predict_response(self, input_text: str) -> str:
        return "PASS"

    def predict_draw_response(self, input_text: str, hand: Counter[str] | dict[str, int]) -> str:
        candidates = sorted(tile for tile, count in hand.items() if count > 0)
        if not candidates:
            return "PLAY W1"
        drawn_tile = self._drawn_tile(input_text)
        hand_tiles = hand_elements(hand)

        def score(tile: str) -> tuple[int, int, int, str]:
            after = remove_one_tile(hand_tiles, tile)
            drawn_penalty = 0 if tile == drawn_tile else 1
            return (min_shanten(after), regular_shanten(after), drawn_penalty, tile)

        return f"PLAY {min(candidates, key=score)}"

    def _drawn_tile(self, input_text: str) -> str | None:
        for line in reversed(input_text.splitlines()):
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "REQ" and parts[1] == "2":
                return parts[2]
        return None


def split_tiles(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token and token[0] in {"W", "B", "T", "F", "J", "H"}]


class BotzonePolicy:
    def __init__(self, predictor: Predictor | None = None, fan_checker=None):
        self.predictor = predictor
        self.fan_checker = fan_checker if fan_checker is not None else self._default_fan_checker()
        self.history: list[str] = []
        self.hand: Counter[str] = Counter()
        self.player_id: int | None = None
        self.quan = 0
        self.packs: list[dict] = []
        self.flower_counts = [0, 0, 0, 0]
        self.wall_counts = [23, 23, 23, 23]
        self.next_draw_about_kong = False
        self.last_model_response: str | None = None
        self.last_overlay_response: str | None = None
        self.last_fallback_used = False
        self.last_illegal_prediction = False
        self.visible_counts: Counter[str] = Counter()
        self.stats: Counter[str] = Counter()

    def _default_fan_checker(self):
        try:
            from official_fan import OfficialFanChecker
        except ImportError:
            return None
        return OfficialFanChecker.default()

    def respond(self, request: str) -> str:
        self.last_model_response = None
        self.last_overlay_response = None
        self.last_fallback_used = False
        self.last_illegal_prediction = False

        tokens = request.strip().split()
        if not tokens:
            response = "PASS"
        elif tokens[0] == "0" and len(tokens) >= 2:
            self.player_id = int(tokens[1])
            if len(tokens) >= 3:
                self.quan = int(tokens[2])
            response = "PASS"
        elif tokens[0] == "1":
            self._load_initial_hand(tokens)
            response = "PASS"
        elif tokens[0] == "2" and len(tokens) >= 2:
            if self.player_id is not None:
                self._record_wall_draw(self.player_id)
            self.hand[tokens[1]] += 1
            response = self._choose_draw_response(request, tokens[1])
        elif tokens[0] == "3":
            self._record_public_event(tokens)
            response = self._choose_reaction_response(request)
        else:
            response = "PASS"

        self.history.append(f"REQ {request}")
        self.history.append(f"RES {response}")
        return response

    def _load_initial_hand(self, tokens: list[str]) -> None:
        self.hand.clear()
        self.visible_counts.clear()
        # Format: 1 hua0 hua1 hua2 hua3 Card1 ... Card13 ...
        if len(tokens) >= 5:
            self.flower_counts = [int(value) for value in tokens[1:5]]
            self.wall_counts = [21 - count for count in self.flower_counts]
        for tile in split_tiles(tokens[5:18]):
            self.hand[tile] += 1

    def _record_wall_draw(self, player: int) -> None:
        if 0 <= player < 4:
            self.wall_counts[player] -= 1

    def _record_public_event(self, tokens: list[str]) -> None:
        if len(tokens) < 3:
            return
        try:
            actor = int(tokens[1])
        except ValueError:
            return
        if tokens[2] == "DRAW":
            self._record_wall_draw(actor)
        elif tokens[2] == "BUHUA":
            self._record_wall_draw(actor)
            if 0 <= actor < 4:
                self.flower_counts[actor] += 1
        elif tokens[2] == "PLAY":
            tile = request_event_tile(tokens)
            if tile:
                self.visible_counts[tile] += 1
        elif tokens[2] == "BUGANG" and len(tokens) >= 4:
            self.visible_counts[tokens[3]] += 1
        elif tokens[2] == "GANG" and len(tokens) >= 4:
            self.visible_counts[tokens[3]] = max(self.visible_counts[tokens[3]], 4)
        elif tokens[2] == "PENG":
            tile = request_event_tile(tokens)
            if tile:
                self.visible_counts[tile] += 2

    def _input_text(self, request: str) -> str:
        return "\n".join([*self.history, f"REQ {request}"])

    def _choose_draw_response(self, request: str, drawn_tile: str) -> str:
        self.stats["draw_turns"] += 1
        candidates = self._legal_responses(request)
        if getattr(self.predictor, "prefer_hu", False) and "HU" in candidates:
            self.stats["legal_hu_seen"] += 1
            self.stats["hu_taken"] += 1
            return "HU"
        if self.predictor is not None:
            input_text = self._input_text(request)
            if (
                hasattr(self.predictor, "predict_legal_response")
                and getattr(self.predictor, "kind", None)
                in {"legal_action_ranker", "composite_policy", "feature_action_ranker"}
            ):
                predicted = self.predictor.predict_legal_response(
                    input_text, self.hand, self.player_id, request, candidates
                ).strip()
            elif hasattr(self.predictor, "predict_draw_response"):
                predicted = self.predictor.predict_draw_response(input_text, self.hand).strip()
            else:
                predicted = self.predictor.predict_response(input_text).strip()
            self.last_model_response = predicted
            self.stats["draw_model_predictions"] += 1
            overlay = self._effective_overlay_response(input_text, request, candidates, predicted)
            if overlay is not None:
                predicted = overlay
            if predicted in candidates:
                if predicted == "HU":
                    self.stats["legal_hu_seen"] += 1
                    self.stats["hu_taken"] += 1
                self._apply_draw_response(predicted)
                return predicted
            self.last_illegal_prediction = True
            self.stats["illegal_predictions"] += 1

        fallback = f"PLAY {drawn_tile if self.hand[drawn_tile] else self._first_tile()}"
        self.last_fallback_used = True
        self.stats["fallbacks"] += 1
        self._apply_draw_response(fallback)
        return fallback

    def _effective_overlay_response(
        self,
        input_text: str,
        request: str,
        candidates: list[str],
        predicted: str,
    ) -> str | None:
        config = getattr(self.predictor, "effective_tile_overlay", None)
        if not config or self.fan_checker is None or self.player_id is None:
            return None
        play_candidates = [candidate for candidate in candidates if candidate.startswith("PLAY ")]
        if not play_candidates:
            return None
        if hasattr(self.predictor, "score_legal_response_candidates"):
            all_scores = self.predictor.score_legal_response_candidates(
                input_text, self.hand, self.player_id, request, candidates
            )
            score_by_candidate = dict(zip(candidates, all_scores))
            base_scores = [score_by_candidate.get(candidate, 0.0) for candidate in play_candidates]
        else:
            base_scores = _one_hot_response_scores(play_candidates, predicted)
        scored = evaluate_discard_candidates(
            hand=self.hand,
            responses=play_candidates,
            fan_checker=self.fan_checker,
            base_scores=base_scores,
            packs=self.packs,
            visible_counts=self.visible_counts,
            flower_count=self.flower_counts[self.player_id],
            seat_wind=self.player_id,
            prevalent_wind=self.quan,
            player=self.player_id,
            levels=int(config.get("levels", 1)) if isinstance(config, dict) else 1,
            base_score_weight=float(config.get("base_score_weight", 0.05))
            if isinstance(config, dict)
            else 0.05,
            guideline_weight=float(config.get("guideline_weight", 1.0))
            if isinstance(config, dict)
            else 1.0,
        )
        if not scored:
            return None
        if isinstance(config, dict) and config.get("require_positive_fan8"):
            if max(item.profile.fan8_wait_tiles for item in scored) <= 0:
                return None
        response = scored[0].response
        self.last_overlay_response = response
        self.stats["draw_overlay_choices"] += 1
        if response != predicted:
            self.stats["draw_overlay_changed_tile"] += 1
        return response

    def _choose_reaction_response(self, request: str) -> str:
        self.stats["reaction_turns"] += 1
        candidates = self._legal_responses(request)
        if getattr(self.predictor, "prefer_hu", False) and "HU" in candidates:
            self.stats["legal_hu_seen"] += 1
            self.stats["hu_taken"] += 1
            return "HU"
        if self.predictor is not None:
            input_text = self._input_text(request)
            if hasattr(self.predictor, "predict_legal_response"):
                predicted = self.predictor.predict_legal_response(
                    input_text, self.hand, self.player_id, request, candidates
                ).strip()
            else:
                predicted = self.predictor.predict_response(input_text).strip()
            self.last_model_response = predicted
            if predicted in candidates:
                if predicted == "HU":
                    self.stats["legal_hu_seen"] += 1
                    self.stats["hu_taken"] += 1
                self._record_own_reaction_pack(request, predicted)
                apply_response(self.hand, request, predicted)
                return predicted
            self.last_illegal_prediction = True
            self.stats["illegal_predictions"] += 1
        self.last_fallback_used = True
        self.stats["fallbacks"] += 1
        return "PASS"

    def _first_tile(self) -> str:
        for tile in sorted(self.hand):
            if self.hand[tile] > 0:
                return tile
        return "W1"

    def _is_legal_draw_response(self, response: str) -> bool:
        return response in self._legal_responses("2 _")

    def _apply_draw_response(self, response: str) -> None:
        parts = response.split()
        if len(parts) == 2 and parts[0].upper() in {"PLAY", "GANG", "BUGANG"}:
            action = parts[0].upper()
            tile = parts[1]
            if action == "GANG":
                self.packs.append({"type": "GANG", "tile": tile, "offer": self.player_id or 0})
                self.next_draw_about_kong = True
                self.visible_counts[tile] = max(self.visible_counts[tile], 4)
            elif action == "BUGANG":
                for pack in self.packs:
                    if pack["type"] == "PENG" and pack["tile"] == tile:
                        pack["type"] = "GANG"
                        break
                self.next_draw_about_kong = True
                self.visible_counts[tile] += 1
            elif action == "PLAY":
                self.visible_counts[tile] += 1
            remove_count = 4 if action == "GANG" else 1
            self.hand[tile] -= remove_count
            if self.hand[tile] <= 0:
                del self.hand[tile]

    def _legal_responses(self, request: str) -> list[str]:
        responses = generate_legal_responses(
            self.player_id,
            request,
            self.hand,
            meld_count=len(self.packs),
        )
        responses.extend(self._bugang_responses(request))
        return [
            response
            for response in dedupe_responses(responses)
            if self._passes_wall_legality(request, response)
            and self._passes_hu_legality(request, response)
        ]

    def _bugang_responses(self, request: str) -> list[str]:
        tokens = request.strip().split()
        if not tokens or tokens[0] != "2":
            return []
        candidates = []
        for pack in self.packs:
            tile = str(pack.get("tile", ""))
            if pack.get("type") == "PENG" and self.hand[tile] > 0:
                candidates.append(f"BUGANG {tile}")
        return sorted(candidates)

    def _passes_wall_legality(self, request: str, response: str) -> bool:
        action = response.split()[0].upper() if response else "PASS"
        if action not in {"GANG", "BUGANG", "PENG", "CHI"}:
            return True
        tokens = request.strip().split()
        if not tokens:
            return True
        if tokens[0] == "2":
            if action not in {"GANG", "BUGANG"} or self.player_id is None:
                return True
            return (
                self.wall_counts[self.player_id] > 0
                and self.wall_counts[(self.player_id + 1) % 4] > 0
            )
        if tokens[0] == "3" and len(tokens) >= 2:
            try:
                actor = int(tokens[1])
            except ValueError:
                return True
            return self.wall_counts[(actor + 1) % 4] > 0
        return True

    def _passes_hu_legality(self, request: str, response: str) -> bool:
        action = response.split()[0].upper() if response else "PASS"
        if action != "HU":
            return True
        if self.fan_checker is None or self.player_id is None:
            self.stats["fan_check_missing"] += 1
            return False
        try:
            self.stats["fan_check_calls"] += 1
            accepted = bool(self.fan_checker.can_hu(**self._hu_context(request)))
            if accepted:
                self.stats["fan_check_accepts"] += 1
            else:
                self.stats["fan_check_rejects"] += 1
            return accepted
        except Exception:
            self.stats["fan_check_errors"] += 1
            return False

    def _hu_context(self, request: str) -> dict:
        tokens = request.strip().split()
        win_tile = tokens[1] if tokens and tokens[0] == "2" and len(tokens) >= 2 else None
        actor = self.player_id
        is_self_draw = bool(tokens and tokens[0] == "2")
        if tokens and tokens[0] == "3":
            try:
                actor = int(tokens[1])
            except (IndexError, ValueError):
                actor = self.player_id
            win_tile = request_event_tile(tokens)
        standing_tiles = list(self.hand.elements())
        if is_self_draw and win_tile in standing_tiles:
            standing_tiles.remove(win_tile)
        return {
            "packs": list(self.packs),
            "hand": standing_tiles,
            "win_tile": win_tile,
            "flower_count": self.flower_counts[self.player_id],
            "is_self_draw": is_self_draw,
            "is_4th_tile": False,
            "is_about_kong": self.next_draw_about_kong or bool(
                tokens and len(tokens) >= 3 and tokens[2] == "BUGANG"
            ),
            "is_last": False,
            "seat_wind": self.player_id,
            "prevalent_wind": self.quan,
            "player": self.player_id,
        }

    def _record_own_reaction_pack(self, request: str, response: str) -> None:
        parts = response.split()
        tokens = request.strip().split()
        if not parts or len(tokens) < 4 or self.player_id is None:
            return
        action = parts[0].upper()
        try:
            actor = int(tokens[1])
        except ValueError:
            return
        event_tile = request_event_tile(tokens)
        if action == "PENG" and event_tile:
            self.packs.append({"type": "PENG", "tile": event_tile, "offer": actor})
            self.visible_counts[event_tile] += 2
            if len(parts) >= 2:
                self.visible_counts[parts[1]] += 1
        elif action == "GANG" and event_tile:
            self.packs.append({"type": "GANG", "tile": event_tile, "offer": actor})
            self.next_draw_about_kong = True
            self.visible_counts[event_tile] = max(self.visible_counts[event_tile], 4)
        elif action == "CHI" and event_tile and len(parts) >= 3:
            middle = parts[1]
            offer = int(event_tile[1]) - int(middle[1]) + 1
            self.packs.append({"type": "CHI", "tile": middle, "offer": offer})
            sequence = [f"{middle[0]}{int(middle[1]) - 1}", middle, f"{middle[0]}{int(middle[1]) + 1}"]
            needed = Counter(sequence)
            needed[event_tile] -= 1
            for tile, count in needed.items():
                if count > 0:
                    self.visible_counts[tile] += count
            self.visible_counts[parts[2]] += 1

    def diagnostics(self) -> dict[str, int | str | bool | None]:
        data: dict[str, int | str | bool | None] = {
            key: int(value) for key, value in sorted(self.stats.items())
        }
        data.update(
            {
                "kind": "botzone_policy",
                "last_model_response": self.last_model_response,
                "last_overlay_response": self.last_overlay_response,
                "last_fallback_used": self.last_fallback_used,
                "last_illegal_prediction": self.last_illegal_prediction,
            }
        )
        return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    predictor = SklearnPredictor(Path(args.model)) if args.model else None
    policy = BotzonePolicy(predictor)
    for line in sys.stdin:
        request = line.strip()
        if request:
            print(policy.respond(request), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
