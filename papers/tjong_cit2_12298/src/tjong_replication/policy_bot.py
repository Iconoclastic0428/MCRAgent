"""Botzone policy wrapper for a trained Tjong checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .actions import ACTION_NAMES, ACTION_TO_INDEX, flatten_claim
from .encoding import build_game_features, stack_hidden_tile_features, stack_visible_tile_features
from .evaluate_supervised import load_model
from .tensorize_botzone import (
    HIDDEN_TILE_ROW_NAMES,
    MemoryFrame,
    ReplayState,
    chow_claim_index,
    chow_needed_from_hand,
    request_event_tile,
    selected_action_mask,
    zero_memory_frame,
)
from .tiles import TILE_NAMES, tile_id

PASS_ACTION = ACTION_TO_INDEX["PASS"]
DISCARD_ACTION = ACTION_TO_INDEX["DISCARD"]
HU_ACTION = ACTION_TO_INDEX["HU"]


class TjongCheckpointPredictor:
    """Score legal Botzone responses with the paper hierarchical heads."""

    kind = "legal_action_ranker"
    requires_botzone_history = True

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str | None = None,
        require_encoding_version: str | None = None,
        require_paper_config: bool = False,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(
            Path(checkpoint),
            self.device,
            expected_encoding_version=require_encoding_version,
            require_paper_config=require_paper_config,
        )
        self.model.eval()

    def predict_legal_response(
        self,
        input_text: str,
        hand: Counter[str],
        player_id: int | None,
        request: str,
        candidates: list[str] | None = None,
    ) -> str:
        candidates = candidates or ["PASS"]
        return self.predict_hierarchical_response(input_text, hand, player_id, request, candidates)

    def predict_hierarchical_response(
        self,
        input_text: str,
        hand: Counter[str],
        player_id: int | None,
        request: str,
        candidates: list[str],
    ) -> str:
        if not candidates:
            return "PASS"
        labels = [response_to_labels(request, response) for response in candidates]
        action_mask = action_mask_from_labels(labels)
        if not any(action_mask):
            return "PASS" if "PASS" in candidates else candidates[0]

        context = build_runtime_context(
            input_text=input_text,
            hand=hand,
            player_id=player_id or 0,
            current_action_mask=action_mask,
            memory_len=self.model.config.memory_len,
        )
        primary_memory = context.memory_for_mask(action_mask, previous_action=context.last_action)
        with torch.no_grad():
            outputs = self.model(
                visible_tiles=primary_memory.visible_tiles.unsqueeze(0).to(self.device),
                game_features=primary_memory.game_features.unsqueeze(0).to(self.device),
                rewards=primary_memory.rewards.unsqueeze(0).to(self.device),
                previous_actions=primary_memory.previous_actions.unsqueeze(0).to(self.device),
                hidden_tiles=primary_memory.hidden_tiles.unsqueeze(0).to(self.device),
            )
        masked_action_logits = outputs["action_logits"].masked_fill(
            torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0) <= 0,
            torch.finfo(outputs["action_logits"].dtype).min,
        )
        action_order = torch.argsort(masked_action_logits[0], descending=True).tolist()
        for action in action_order:
            if action_mask[int(action)] <= 0:
                continue
            response = self._response_for_selected_action(context, request, candidates, labels, int(action), action_mask)
            if response is not None:
                return response
        return "PASS" if "PASS" in candidates else candidates[0]

    def _response_for_selected_action(
        self,
        context: "RuntimeFeatureContext",
        request: str,
        candidates: list[str],
        labels: list[tuple[int, int, int] | None],
        action: int,
        action_mask: list[float],
    ) -> str | None:
        matching = [index for index, label in enumerate(labels) if label is not None and label[0] == action]
        if not matching:
            return None
        if action in {PASS_ACTION, HU_ACTION}:
            return candidates[matching[0]]
        if action == DISCARD_ACTION:
            return self._choose_discard_candidate(context, candidates, labels, matching, action_mask)
        if action in CLAIM_ACTIONS:
            claim_index = self._choose_claim_index(context, labels, matching, action, action_mask)
            if claim_index is None:
                return None
            claim_matching = [index for index in matching if labels[index] is not None and labels[index][1] == claim_index]
            if not claim_matching:
                return None
            if action in {ACTION_TO_INDEX["CHOW"], ACTION_TO_INDEX["PONG"]}:
                return self._choose_post_claim_discard_candidate(
                    context,
                    request,
                    candidates,
                    labels,
                    claim_matching,
                    action,
                )
            return candidates[claim_matching[0]]
        return None

    def _choose_claim_index(
        self,
        context: "RuntimeFeatureContext",
        labels: list[tuple[int, int, int] | None],
        matching: list[int],
        action: int,
        action_mask: list[float],
    ) -> int | None:
        sub_memory = context.memory_for_mask(selected_action_mask(action), previous_action=action)
        primary_memory = context.memory_for_mask(action_mask, previous_action=context.last_action)
        with torch.no_grad():
            outputs = self.model(
                visible_tiles=primary_memory.visible_tiles.unsqueeze(0).to(self.device),
                game_features=primary_memory.game_features.unsqueeze(0).to(self.device),
                rewards=primary_memory.rewards.unsqueeze(0).to(self.device),
                previous_actions=primary_memory.previous_actions.unsqueeze(0).to(self.device),
                sub_visible_tiles=sub_memory.visible_tiles.unsqueeze(0).to(self.device),
                sub_game_features=sub_memory.game_features.unsqueeze(0).to(self.device),
                sub_rewards=sub_memory.rewards.unsqueeze(0).to(self.device),
                sub_previous_actions=sub_memory.previous_actions.unsqueeze(0).to(self.device),
                hidden_tiles=primary_memory.hidden_tiles.unsqueeze(0).to(self.device),
            )
        claim_logits = outputs["claim_logits"][0]
        allowed_claims = sorted({int(labels[index][1]) for index in matching if labels[index] is not None})
        if not allowed_claims:
            return None
        return max(allowed_claims, key=lambda claim: float(claim_logits[claim].item()))

    def _choose_discard_candidate(
        self,
        context: "RuntimeFeatureContext",
        candidates: list[str],
        labels: list[tuple[int, int, int] | None],
        matching: list[int],
        action_mask: list[float],
    ) -> str | None:
        primary_memory = context.memory_for_mask(action_mask, previous_action=context.last_action)
        sub_memory = context.memory_for_mask(selected_action_mask(DISCARD_ACTION), previous_action=DISCARD_ACTION)
        with torch.no_grad():
            outputs = self.model(
                visible_tiles=primary_memory.visible_tiles.unsqueeze(0).to(self.device),
                game_features=primary_memory.game_features.unsqueeze(0).to(self.device),
                rewards=primary_memory.rewards.unsqueeze(0).to(self.device),
                previous_actions=primary_memory.previous_actions.unsqueeze(0).to(self.device),
                sub_visible_tiles=sub_memory.visible_tiles.unsqueeze(0).to(self.device),
                sub_game_features=sub_memory.game_features.unsqueeze(0).to(self.device),
                sub_rewards=sub_memory.rewards.unsqueeze(0).to(self.device),
                sub_previous_actions=sub_memory.previous_actions.unsqueeze(0).to(self.device),
                hidden_tiles=primary_memory.hidden_tiles.unsqueeze(0).to(self.device),
            )
        discard_logits = outputs["discard_logits"][0]
        return candidates[
            max(
                matching,
                key=lambda index: float(discard_logits[int(labels[index][2])].item()) if labels[index] is not None else float("-inf"),
            )
        ]

    def _choose_post_claim_discard_candidate(
        self,
        context: "RuntimeFeatureContext",
        request: str,
        candidates: list[str],
        labels: list[tuple[int, int, int] | None],
        matching: list[int],
        claim_action: int,
    ) -> str | None:
        scored: list[tuple[float, int]] = []
        for index in matching:
            candidate = candidates[index]
            discard = response_discard_label(candidate)
            post_context = context.post_claim_context(request, candidate)
            if discard is None or post_context is None:
                scored.append((0.0, index))
                continue
            primary_memory = post_context.memory_for_mask(
                selected_action_mask(DISCARD_ACTION),
                previous_action=claim_action,
            )
            sub_memory = post_context.memory_for_mask(
                selected_action_mask(DISCARD_ACTION),
                previous_action=DISCARD_ACTION,
            )
            with torch.no_grad():
                outputs = self.model(
                    visible_tiles=primary_memory.visible_tiles.unsqueeze(0).to(self.device),
                    game_features=primary_memory.game_features.unsqueeze(0).to(self.device),
                    rewards=primary_memory.rewards.unsqueeze(0).to(self.device),
                    previous_actions=primary_memory.previous_actions.unsqueeze(0).to(self.device),
                    sub_visible_tiles=sub_memory.visible_tiles.unsqueeze(0).to(self.device),
                    sub_game_features=sub_memory.game_features.unsqueeze(0).to(self.device),
                    sub_rewards=sub_memory.rewards.unsqueeze(0).to(self.device),
                    sub_previous_actions=sub_memory.previous_actions.unsqueeze(0).to(self.device),
                    hidden_tiles=primary_memory.hidden_tiles.unsqueeze(0).to(self.device),
                )
            scored.append((float(outputs["discard_logits"][0, discard].item()), index))
        if not scored:
            return None
        return candidates[max(scored, key=lambda item: item[0])[1]]

    def score_legal_response_candidates(
        self,
        input_text: str,
        hand: Counter[str],
        player_id: int | None,
        request: str,
        candidates: list[str],
    ) -> list[float]:
        if not candidates:
            return []
        labels = [response_to_labels(request, response) for response in candidates]
        action_mask = [0.0] * 8
        for label in labels:
            if label is not None:
                action_mask[label[0]] = 1.0
        if not any(action_mask):
            return [0.0 if candidate == "PASS" else float("-inf") for candidate in candidates]

        context = build_runtime_context(
            input_text=input_text,
            hand=hand,
            player_id=player_id or 0,
            current_action_mask=action_mask,
            memory_len=self.model.config.memory_len,
        )
        primary_memory = context.memory_for_mask(action_mask, previous_action=context.last_action)
        sub_memories = [
            context.memory_for_mask(
                selected_action_mask(label[0] if label is not None else PASS_ACTION),
                previous_action=label[0] if label is not None else PASS_ACTION,
            )
            for label in labels
        ]
        forced_discard_indices: list[int] = []
        forced_discard_memories: list[tuple[PolicyMemory, PolicyMemory]] = []
        discard_only_mask = selected_action_mask(ACTION_TO_INDEX["DISCARD"])
        for index, (candidate, label) in enumerate(zip(candidates, labels)):
            if label is None or label[0] not in CLAIM_ACTIONS or response_discard_label(candidate) is None:
                continue
            post_context = context.post_claim_context(request, candidate)
            if post_context is None:
                continue
            post_primary_memory = post_context.memory_for_mask(discard_only_mask, previous_action=label[0])
            post_sub_memory = post_context.memory_for_mask(discard_only_mask, previous_action=DISCARD_ACTION)
            forced_discard_indices.append(index)
            forced_discard_memories.append((post_primary_memory, post_sub_memory))
        visible_batch = primary_memory.visible_tiles.unsqueeze(0).repeat(len(candidates), 1, 1, 1).to(self.device)
        game_batch = primary_memory.game_features.unsqueeze(0).repeat(len(candidates), 1, 1).to(self.device)
        rewards_batch = primary_memory.rewards.unsqueeze(0).repeat(len(candidates), 1).to(self.device)
        previous_actions_batch = primary_memory.previous_actions.unsqueeze(0).repeat(len(candidates), 1).to(self.device)
        sub_visible_batch = torch.stack([item.visible_tiles for item in sub_memories], dim=0).to(self.device)
        sub_game_batch = torch.stack([item.game_features for item in sub_memories], dim=0).to(self.device)
        sub_rewards_batch = torch.stack([item.rewards for item in sub_memories], dim=0).to(self.device)
        sub_previous_actions_batch = torch.stack([item.previous_actions for item in sub_memories], dim=0).to(self.device)
        hidden_batch = primary_memory.hidden_tiles.unsqueeze(0).repeat(len(candidates), 1, 1).to(self.device)
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                visible_tiles=visible_batch,
                game_features=game_batch,
                rewards=rewards_batch,
                previous_actions=previous_actions_batch,
                sub_visible_tiles=sub_visible_batch,
                sub_game_features=sub_game_batch,
                sub_rewards=sub_rewards_batch,
                sub_previous_actions=sub_previous_actions_batch,
                hidden_tiles=hidden_batch,
            )
            forced_discard_logp: dict[int, torch.Tensor] = {}
            if forced_discard_memories:
                forced_visible_batch = torch.stack([item[0].visible_tiles for item in forced_discard_memories], dim=0).to(self.device)
                forced_game_batch = torch.stack([item[0].game_features for item in forced_discard_memories], dim=0).to(self.device)
                forced_rewards_batch = torch.stack([item[0].rewards for item in forced_discard_memories], dim=0).to(self.device)
                forced_previous_actions_batch = torch.stack([item[0].previous_actions for item in forced_discard_memories], dim=0).to(self.device)
                forced_sub_visible_batch = torch.stack([item[1].visible_tiles for item in forced_discard_memories], dim=0).to(self.device)
                forced_sub_game_batch = torch.stack([item[1].game_features for item in forced_discard_memories], dim=0).to(self.device)
                forced_sub_rewards_batch = torch.stack([item[1].rewards for item in forced_discard_memories], dim=0).to(self.device)
                forced_sub_previous_actions_batch = torch.stack([item[1].previous_actions for item in forced_discard_memories], dim=0).to(self.device)
                forced_hidden_batch = torch.stack([item[0].hidden_tiles for item in forced_discard_memories], dim=0).to(self.device)
                forced_outputs = self.model(
                    visible_tiles=forced_visible_batch,
                    game_features=forced_game_batch,
                    rewards=forced_rewards_batch,
                    previous_actions=forced_previous_actions_batch,
                    sub_visible_tiles=forced_sub_visible_batch,
                    sub_game_features=forced_sub_game_batch,
                    sub_rewards=forced_sub_rewards_batch,
                    sub_previous_actions=forced_sub_previous_actions_batch,
                    hidden_tiles=forced_hidden_batch,
                )
                forced_logp_batch = torch.log_softmax(forced_outputs["discard_logits"], dim=-1)
                forced_discard_logp = {
                    candidate_index: forced_logp_batch[row_index]
                    for row_index, candidate_index in enumerate(forced_discard_indices)
                }
        masked_action_logits = outputs["action_logits"].masked_fill(
            action_mask_tensor <= 0,
            torch.finfo(outputs["action_logits"].dtype).min,
        )
        action_logp = torch.log_softmax(masked_action_logits, dim=-1)
        claim_logp = torch.log_softmax(outputs["claim_logits"], dim=-1)
        discard_logp = torch.log_softmax(outputs["discard_logits"], dim=-1)
        scores: list[float] = []
        for index, (candidate, label) in enumerate(zip(candidates, labels)):
            if label is None:
                scores.append(float("-inf"))
                continue
            action, claim, discard = label
            score = action_logp[index, action]
            if action in CLAIM_ACTIONS:
                score = score + claim_logp[index, claim]
            forced_discard = response_discard_label(candidate)
            if action == ACTION_TO_INDEX["DISCARD"] or forced_discard is not None:
                if forced_discard is not None and index in forced_discard_logp:
                    score = score + forced_discard_logp[index][forced_discard]
                else:
                    score = score + discard_logp[index, discard if forced_discard is None else forced_discard]
            scores.append(float(score.item()))
        return scores


CLAIM_ACTIONS = {
    ACTION_TO_INDEX["CHOW"],
    ACTION_TO_INDEX["PONG"],
    ACTION_TO_INDEX["MINGKONG"],
    ACTION_TO_INDEX["BUKONG"],
    ACTION_TO_INDEX["ANKONG"],
}


@dataclass
class PolicyMemory:
    visible_tiles: torch.Tensor
    game_features: torch.Tensor
    rewards: torch.Tensor
    previous_actions: torch.Tensor
    hidden_tiles: torch.Tensor


@dataclass
class LivePolicyState:
    state: ReplayState
    hand_counts: list[int]
    wall_counts: list[int]
    player_id: int


@dataclass
class RuntimeFeatureContext:
    live: LivePolicyState
    history: list[MemoryFrame]
    last_action: int
    player_id: int
    current_request: str
    memory_len: int

    def memory_for_mask(
        self,
        action_mask: Iterable[float],
        *,
        previous_action: int,
        live: LivePolicyState | None = None,
    ) -> PolicyMemory:
        live_state = self.live if live is None else live
        visible, game, hidden = encode_live_policy_state(live_state, self.player_id, action_mask)
        frames = list(self.history)[-(self.memory_len - 1) :] + [
            MemoryFrame(visible, game, previous_action=int(previous_action), reward=0.0)
        ]
        while len(frames) < self.memory_len:
            frames.insert(0, zero_memory_frame(previous_action=PASS_ACTION))
        return PolicyMemory(
            visible_tiles=torch.stack([frame.visible_tiles for frame in frames], dim=0),
            game_features=torch.stack([frame.game_features for frame in frames], dim=0),
            rewards=torch.tensor([frame.reward for frame in frames], dtype=torch.float32),
            previous_actions=torch.tensor([frame.previous_action for frame in frames], dtype=torch.long),
            hidden_tiles=hidden,
        )

    def post_claim_context(self, request: str, response: str) -> "RuntimeFeatureContext | None":
        post_replay = self.live.state.after_claim(self.player_id, request, response)
        if post_replay is None:
            return None
        post_live = copy.deepcopy(self.live)
        post_live.state = post_replay
        head = response.split()[0].upper() if response.split() else ""
        if head in {"CHI", "PENG"}:
            post_live.hand_counts[self.player_id] = max(0, post_live.hand_counts[self.player_id] - 2)
        return RuntimeFeatureContext(
            live=post_live,
            history=list(self.history),
            last_action=self.last_action,
            player_id=self.player_id,
            current_request=request,
            memory_len=self.memory_len,
        )


def action_mask_from_labels(labels: Iterable[tuple[int, int, int] | None]) -> list[float]:
    mask = [0.0] * len(ACTION_NAMES)
    for label in labels:
        if label is not None:
            mask[int(label[0])] = 1.0
    return mask


def build_runtime_context(
    *,
    input_text: str,
    hand: Counter[str],
    player_id: int,
    current_action_mask: Iterable[float],
    memory_len: int,
) -> RuntimeFeatureContext:
    live = LivePolicyState(
        state=ReplayState(),
        hand_counts=[13, 13, 13, 13],
        wall_counts=[21, 21, 21, 21],
        player_id=int(player_id),
    )
    history: deque[MemoryFrame] = deque(maxlen=max(0, memory_len - 1))
    last_action = PASS_ACTION
    current_request = ""
    entries = list(iter_policy_history(input_text))
    for request, response in entries:
        current_request = request
        apply_request_to_live_state(live, request)
        tokens = request.strip().split()
        if response is None:
            break
        label = response_to_labels(request, response)
        if tokens and tokens[0] in {"2", "3"} and label is not None:
            action_mask = runtime_action_type_mask(live, request, response)
            if action_mask[label[0]] <= 0.0:
                action_mask[label[0]] = 1.0
            if sum(1 for value in action_mask if value > 0.0) <= 1 and label[0] != DISCARD_ACTION:
                continue
            visible, game, _ = encode_live_policy_state(live, live.player_id, action_mask)
            history.append(MemoryFrame(visible, game, previous_action=int(label[0]), reward=0.0))
            last_action = int(label[0])

    # The generic live replay is intentionally public-information only. The
    # policy wrapper already owns the exact current concealed hand, so use it
    # for the current decision frame even if the textual replay was truncated.
    live.player_id = int(player_id)
    live.state.hands[int(player_id)] = Counter(hand)
    live.hand_counts[int(player_id)] = int(sum(hand.values()))
    return RuntimeFeatureContext(
        live=live,
        history=list(history),
        last_action=last_action,
        player_id=int(player_id),
        current_request=current_request,
        memory_len=int(memory_len),
    )


def iter_policy_history(input_text: str) -> Iterable[tuple[str, str | None]]:
    pending_request: str | None = None
    for raw_line in input_text.splitlines():
        line = raw_line.strip()
        if line.startswith("REQ "):
            if pending_request is not None:
                yield pending_request, None
            pending_request = line[4:].strip()
        elif line.startswith("RES ") and pending_request is not None:
            yield pending_request, line[4:].strip()
            pending_request = None
    if pending_request is not None:
        yield pending_request, None


def runtime_action_type_mask(live: LivePolicyState, request: str, response: str | None = None) -> list[float]:
    mask = [0.0] * len(ACTION_NAMES)
    tokens = request.strip().split()
    if not tokens:
        mask[PASS_ACTION] = 1.0
        return mask
    response_head = response.split()[0].upper() if response and response.split() else ""
    player = live.player_id
    hand = live.state.hands[player]
    if tokens[0] == "2":
        if response_head == "HU":
            mask[HU_ACTION] = 1.0
        if hand:
            mask[DISCARD_ACTION] = 1.0
        if any(count >= 4 for count in hand.values()):
            mask[ACTION_TO_INDEX["ANKONG"]] = 1.0
        if any(live.state.pengs[player][tile] > 0 and hand[tile] > 0 for tile in live.state.pengs[player]):
            mask[ACTION_TO_INDEX["BUKONG"]] = 1.0
        return mask

    mask[PASS_ACTION] = 1.0
    if response_head == "HU":
        mask[HU_ACTION] = 1.0
    if tokens[0] != "3" or len(tokens) < 3:
        return mask
    try:
        actor = int(tokens[1])
    except ValueError:
        return mask
    event_tile = request_event_tile(tokens)
    if actor == player or event_tile not in TILE_NAMES or tokens[2].upper() not in {"PLAY", "PENG", "CHI"}:
        return mask
    if hand[event_tile] >= 2:
        mask[ACTION_TO_INDEX["PONG"]] = 1.0
    if hand[event_tile] >= 3:
        mask[ACTION_TO_INDEX["MINGKONG"]] = 1.0
    if player == (actor + 1) % 4 and runtime_any_chow(hand, event_tile):
        mask[ACTION_TO_INDEX["CHOW"]] = 1.0
    return mask


def apply_request_to_live_state(live: LivePolicyState, request: str) -> None:
    tokens = request.strip().split()
    if not tokens:
        return
    if tokens[0] == "0" and len(tokens) >= 2:
        try:
            live.player_id = int(tokens[1])
            if len(tokens) >= 3:
                live.state.prevailing_wind = int(tokens[2])
        except ValueError:
            return
        return
    if tokens[0] == "1":
        apply_initial_hand_request(live, tokens)
        return
    if tokens[0] == "2" and len(tokens) >= 2:
        tile = tokens[1]
        player = live.player_id
        if tile in TILE_NAMES:
            live.state.hands[player][tile] += 1
            live.hand_counts[player] += 1
        decrement_wall(live, player)
        return
    if tokens[0] != "3" or len(tokens) < 3:
        return
    try:
        actor = int(tokens[1])
    except ValueError:
        return
    action = tokens[2].upper()
    if action == "DRAW":
        live.hand_counts[actor] += 1
        decrement_wall(live, actor)
    elif action == "BUHUA":
        decrement_wall(live, actor)
    elif action == "PLAY":
        apply_public_play(live, actor, tokens)
    elif action == "PENG":
        apply_public_peng(live, actor, tokens)
    elif action == "CHI":
        apply_public_chow(live, actor, tokens)
    elif action == "GANG":
        apply_public_kong(live, actor, tokens)
    elif action == "BUGANG":
        apply_public_bukong(live, actor, tokens)
    live.state.tile_counts = [max(0, int(value)) for value in live.wall_counts]


def apply_initial_hand_request(live: LivePolicyState, tokens: list[str]) -> None:
    player = live.player_id
    if len(tokens) >= 5:
        try:
            flowers = [int(value) for value in tokens[1:5]]
            live.wall_counts = [max(0, 21 - value) for value in flowers]
        except ValueError:
            pass
    tiles = [tile for tile in tokens[5:] if tile in TILE_NAMES]
    live.state.hands[player] = Counter(tiles)
    own_count = len(tiles) if tiles else 13
    live.hand_counts = [13, 13, 13, 13]
    live.hand_counts[player] = own_count
    live.state.tile_counts = [max(0, int(value)) for value in live.wall_counts]
    live.state.claimable_discard = None


def apply_public_play(live: LivePolicyState, actor: int, tokens: list[str]) -> None:
    tile = request_event_tile(tokens)
    if tile not in TILE_NAMES:
        return
    live.state._remove_from_hand(actor, tile, 1)
    live.state.discards[actor][tile] += 1
    live.state.claimable_discard = (actor, tile)
    live.hand_counts[actor] = max(0, live.hand_counts[actor] - 1)


def apply_public_peng(live: LivePolicyState, actor: int, tokens: list[str]) -> None:
    event_tile = live.state.claimable_discard[1] if live.state.claimable_discard else None
    discard = tokens[3] if len(tokens) >= 4 else None
    if event_tile not in TILE_NAMES:
        live.state.claimable_discard = None
        return
    live.state._consume_claimable_discard(event_tile)
    live.state.claimable_discard = None
    live.state._remove_from_hand(actor, event_tile, 2)
    live.state.pengs[actor][event_tile] += 1
    live.hand_counts[actor] = max(0, live.hand_counts[actor] - 2)
    if discard in TILE_NAMES:
        live.state._remove_from_hand(actor, discard, 1)
        live.state.discards[actor][discard] += 1
        live.state.claimable_discard = (actor, discard)
        live.hand_counts[actor] = max(0, live.hand_counts[actor] - 1)


def apply_public_chow(live: LivePolicyState, actor: int, tokens: list[str]) -> None:
    event_tile = live.state.claimable_discard[1] if live.state.claimable_discard else None
    middle = tokens[3] if len(tokens) >= 4 else None
    discard = tokens[4] if len(tokens) >= 5 else None
    if event_tile not in TILE_NAMES or middle not in TILE_NAMES:
        live.state.claimable_discard = None
        return
    live.state._consume_claimable_discard(event_tile)
    live.state.claimable_discard = None
    needed = chow_needed_from_hand(middle, event_tile)
    for tile, count in needed.items():
        live.state._remove_from_hand(actor, tile, count)
    for tile in safe_chow_sequence(middle):
        live.state.chows[actor][tile] += 1
    live.hand_counts[actor] = max(0, live.hand_counts[actor] - sum(needed.values()))
    if discard in TILE_NAMES:
        live.state._remove_from_hand(actor, discard, 1)
        live.state.discards[actor][discard] += 1
        live.state.claimable_discard = (actor, discard)
        live.hand_counts[actor] = max(0, live.hand_counts[actor] - 1)


def apply_public_kong(live: LivePolicyState, actor: int, tokens: list[str]) -> None:
    tile = tokens[3] if len(tokens) >= 4 else None
    if tile not in TILE_NAMES:
        live.state.claimable_discard = None
        return
    if live.state.claimable_discard and live.state.claimable_discard[1] == tile:
        live.state._consume_claimable_discard(tile)
        live.state._remove_from_hand(actor, tile, 3)
        live.hand_counts[actor] = max(0, live.hand_counts[actor] - 3)
    else:
        live.state._remove_from_hand(actor, tile, 4)
        live.state.concealed_kongs[actor] += 1
        live.state.concealed_kong_tiles[actor][tile] += 4
        live.hand_counts[actor] = max(0, live.hand_counts[actor] - 4)
    live.state.kongs[actor][tile] += 1
    live.state.claimable_discard = None


def apply_public_bukong(live: LivePolicyState, actor: int, tokens: list[str]) -> None:
    tile = tokens[3] if len(tokens) >= 4 else None
    live.state.claimable_discard = None
    if tile not in TILE_NAMES:
        return
    live.state._remove_from_hand(actor, tile, 1)
    if live.state.pengs[actor][tile] > 0:
        live.state.pengs[actor][tile] -= 1
    live.state.kongs[actor][tile] += 1
    live.hand_counts[actor] = max(0, live.hand_counts[actor] - 1)


def decrement_wall(live: LivePolicyState, player: int) -> None:
    if 0 <= player < 4:
        live.wall_counts[player] = max(0, live.wall_counts[player] - 1)
        live.state.tile_counts = [max(0, int(value)) for value in live.wall_counts]


def encode_live_policy_state(
    live: LivePolicyState,
    player: int,
    action_mask: Iterable[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = live.state
    rows: dict[str, list[int] | torch.Tensor] = {
        "hand_self": counter_to_ids(state.hands[player]),
    }
    live_counts = state.live_counts_for(player)
    rows["available_tile_mask"] = (live_counts > 0).float()
    for rel in range(4):
        absolute = (player + rel) % 4
        rows[f"discard_p{rel}"] = counter_to_ids(state.discards[absolute])
        rows[f"peng_p{rel}"] = counter_to_ids(expand_counter(state.pengs[absolute], 3))
        rows[f"chow_p{rel}"] = counter_to_ids(state.chows[absolute])
        rows[f"kong_p{rel}"] = counter_to_ids(expand_counter(state.kongs[absolute], 4))
        rows[f"remaining_tiles_p{rel}"] = live_counts
    opponents = [(player + rel) % 4 for rel in range(1, 4)]
    visible = stack_visible_tile_features(rows)
    game = build_game_features(
        prevailing_wind=state.prevailing_wind,
        seat_wind=player,
        opponent_concealed_kongs=[float(state.concealed_kongs[other]) for other in opponents],
        remaining_tile_counts=[float(live.wall_counts[(player + rel) % 4]) for rel in range(4)],
        hand_tile_counts=[float(live.hand_counts[(player + rel) % 4]) for rel in range(4)],
        action_mask=list(action_mask),
    )
    hidden = stack_hidden_tile_features(
        [
            torch.zeros(34),
            torch.zeros(34),
            torch.zeros(34),
            counter_to_tensor(sum((state.concealed_kong_tiles[index] for index in range(4)), Counter())),
            live_counts,
        ]
    )
    return visible, game, hidden


def runtime_any_chow(hand: Counter[str], event_tile: str) -> bool:
    if len(event_tile) != 2 or event_tile[0] not in {"W", "T", "B"}:
        return False
    try:
        rank = int(event_tile[1])
    except ValueError:
        return False
    for middle_rank in range(rank - 1, rank + 2):
        middle = f"{event_tile[0]}{middle_rank}"
        if middle not in TILE_NAMES:
            continue
        try:
            needed = chow_needed_from_hand(middle, event_tile)
        except ValueError:
            continue
        if all(hand[tile] >= count for tile, count in needed.items()):
            return True
    return False


def current_memory(visible: torch.Tensor, game: torch.Tensor, memory_len: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    return visible.unsqueeze(0).repeat(memory_len, 1, 1), game.unsqueeze(0).repeat(memory_len, 1)


def encode_runtime_state(
    *,
    input_text: str,
    hand: Counter[str],
    player_id: int,
    action_mask: Iterable[float],
    post_claim_request: str | None = None,
    post_claim_response: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    discards = [Counter() for _ in range(4)]
    pengs = [Counter() for _ in range(4)]
    chows = [Counter() for _ in range(4)]
    kongs = [Counter() for _ in range(4)]
    concealed_kong_tiles = Counter()
    tile_counts = [21, 21, 21, 21]
    prevailing_wind = 0
    previous_request_action: str | None = None
    previous_request_actor: int | None = None
    for line in input_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 2 or parts[0] != "REQ":
            continue
        tokens = parts[1:]
        if not tokens:
            continue
        if tokens[0] == "0" and len(tokens) >= 3:
            prevailing_wind = int(tokens[2])
        elif tokens[0] == "2":
            previous_request_action = "DRAW"
            previous_request_actor = player_id
        elif tokens[0] == "3" and len(tokens) >= 3:
            try:
                actor = int(tokens[1])
            except ValueError:
                continue
            action = tokens[2].upper()
            if action == "PLAY" and len(tokens) >= 4 and tokens[3] in TILE_NAMES:
                discards[actor][tokens[3]] += 1
            elif action == "PENG" and len(tokens) >= 4 and tokens[3] in TILE_NAMES:
                pengs[actor][tokens[3]] += 1
            elif action == "GANG" and len(tokens) >= 4 and tokens[3] in TILE_NAMES:
                kongs[actor][tokens[3]] += 1
                if previous_request_action == "DRAW" and previous_request_actor == actor:
                    concealed_kong_tiles.update([tokens[3]] * 4)
            elif action == "BUGANG" and len(tokens) >= 4 and tokens[3] in TILE_NAMES:
                kongs[actor][tokens[3]] += 1
            elif action == "CHI" and len(tokens) >= 5 and tokens[3] in TILE_NAMES:
                for tile in safe_chow_sequence(tokens[3]):
                    chows[actor][tile] += 1
            previous_request_action = action
            previous_request_actor = actor

    runtime_hand = Counter(hand)
    if post_claim_request and post_claim_response:
        apply_runtime_post_claim_state(
            hand=runtime_hand,
            player_id=player_id,
            request=post_claim_request,
            response=post_claim_response,
            discards=discards,
            pengs=pengs,
            chows=chows,
            kongs=kongs,
        )

    visible_counts = Counter(runtime_hand)
    for counter in discards:
        visible_counts.update(counter)
    for counter in pengs:
        visible_counts.update(expand_counter(counter, 3))
    for counter in chows:
        visible_counts.update(counter)
    for counter in kongs:
        visible_counts.update(expand_counter(counter, 4))
    live_counts = torch.zeros(34, dtype=torch.float32)
    for tile in TILE_NAMES:
        live_counts[tile_id(tile)] = max(0.0, 4.0 - float(visible_counts[tile]))
    rows = {"hand_self": counter_to_ids(runtime_hand), "available_tile_mask": (live_counts > 0).float()}
    for rel in range(4):
        absolute = (player_id + rel) % 4
        rows[f"discard_p{rel}"] = counter_to_ids(discards[absolute])
        rows[f"peng_p{rel}"] = counter_to_ids(expand_counter(pengs[absolute], 3))
        rows[f"chow_p{rel}"] = counter_to_ids(chows[absolute])
        rows[f"kong_p{rel}"] = counter_to_ids(expand_counter(kongs[absolute], 4))
        rows[f"remaining_tiles_p{rel}"] = live_counts
    visible = stack_visible_tile_features(rows)
    game = build_game_features(
        prevailing_wind=prevailing_wind,
        seat_wind=player_id,
        opponent_concealed_kongs=[0.0, 0.0, 0.0],
        remaining_tile_counts=tile_counts,
        hand_tile_counts=[float(sum(runtime_hand.values())), 0.0, 0.0, 0.0],
        action_mask=list(action_mask),
    )
    hidden = stack_hidden_tile_features(
        [
            torch.zeros(34),
            torch.zeros(34),
            torch.zeros(34),
            counter_to_tensor(concealed_kong_tiles),
            live_counts,
        ]
    )
    return visible, game, hidden


def apply_runtime_post_claim_state(
    *,
    hand: Counter[str],
    player_id: int,
    request: str,
    response: str,
    discards: list[Counter[str]],
    pengs: list[Counter[str]],
    chows: list[Counter[str]],
    kongs: list[Counter[str]],
) -> None:
    """Mutate counters to the state after a claim and before its forced discard."""

    parts = response.strip().split()
    request_tokens = request.strip().split()
    if not parts or not request_tokens or request_tokens[0] != "3":
        return
    head = parts[0].upper()
    event_tile = request_event_tile(request_tokens)
    if event_tile not in TILE_NAMES:
        return
    try:
        actor = int(request_tokens[1])
    except (IndexError, ValueError):
        actor = None

    def consume_event_discard() -> None:
        if actor is not None and 0 <= actor < 4 and discards[actor][event_tile] > 0:
            discards[actor][event_tile] -= 1
            if discards[actor][event_tile] <= 0:
                del discards[actor][event_tile]

    if head == "PENG":
        consume_event_discard()
        remove_tiles_from_counter(hand, Counter({event_tile: 2}))
        pengs[player_id][event_tile] += 1
    elif head == "CHI" and len(parts) >= 2:
        middle = parts[1]
        sequence = safe_chow_sequence(middle)
        if event_tile not in sequence:
            return
        consume_event_discard()
        needed = Counter(sequence)
        needed[event_tile] -= 1
        remove_tiles_from_counter(hand, needed)
        for tile in sequence:
            chows[player_id][tile] += 1
    elif head == "GANG":
        consume_event_discard()
        remove_tiles_from_counter(hand, Counter({event_tile: 3}))
        kongs[player_id][event_tile] += 1


def runtime_hidden_schema_rows() -> tuple[str, ...]:
    return HIDDEN_TILE_ROW_NAMES


def response_to_labels(request: str, response: str) -> tuple[int, int, int] | None:
    parts = response.strip().split()
    request_tokens = request.strip().split()
    if not parts:
        return None
    head = parts[0].upper()
    event_tile = request_event_tile(request_tokens)
    if head == "PASS":
        return (ACTION_TO_INDEX["PASS"], 0, 0)
    if head == "HU":
        return (ACTION_TO_INDEX["HU"], 0, 0)
    if head == "PLAY" and len(parts) >= 2 and parts[1] in TILE_NAMES:
        return (ACTION_TO_INDEX["DISCARD"], 0, tile_id(parts[1]))
    if head == "PENG" and event_tile in TILE_NAMES:
        discard = tile_id(parts[1]) if len(parts) >= 2 and parts[1] in TILE_NAMES else 0
        return (ACTION_TO_INDEX["PONG"], flatten_claim("PONG", tile_id(event_tile)), discard)
    if head == "GANG":
        if request_tokens and request_tokens[0] == "2" and len(parts) >= 2 and parts[1] in TILE_NAMES:
            return (ACTION_TO_INDEX["ANKONG"], flatten_claim("ANKONG", tile_id(parts[1])), 0)
        if event_tile in TILE_NAMES:
            return (ACTION_TO_INDEX["MINGKONG"], flatten_claim("MINGKONG", tile_id(event_tile)), 0)
    if head == "BUGANG" and len(parts) >= 2 and parts[1] in TILE_NAMES:
        return (ACTION_TO_INDEX["BUKONG"], flatten_claim("BUKONG", tile_id(parts[1])), 0)
    if head == "CHI" and len(parts) >= 2 and event_tile in TILE_NAMES:
        try:
            discard = tile_id(parts[2]) if len(parts) >= 3 and parts[2] in TILE_NAMES else 0
            return (ACTION_TO_INDEX["CHOW"], flatten_claim("CHOW", chow_claim_index(parts[1], event_tile)), discard)
        except ValueError:
            return None
    return None


def response_discard_label(response: str) -> int | None:
    parts = response.strip().split()
    if len(parts) >= 2 and parts[0].upper() in {"PLAY", "PENG"} and parts[1] in TILE_NAMES:
        return tile_id(parts[1])
    if len(parts) >= 3 and parts[0].upper() == "CHI" and parts[2] in TILE_NAMES:
        return tile_id(parts[2])
    return None


def first_candidate_for_action(candidates: list[str], labels: list[tuple[int, int, int] | None], action: int) -> str | None:
    for candidate, label in zip(candidates, labels):
        if label is not None and int(label[0]) == int(action):
            return candidate
    return None


def counter_to_ids(counter: Counter[str]) -> list[int]:
    ids: list[int] = []
    for tile, count in counter.items():
        if tile in TILE_NAMES and count > 0:
            ids.extend([tile_id(tile)] * int(count))
    return ids


def counter_to_tensor(counter: Counter[str]) -> torch.Tensor:
    tensor = torch.zeros(34, dtype=torch.float32)
    for tile, count in counter.items():
        if tile in TILE_NAMES and count > 0:
            tensor[tile_id(tile)] += float(count)
    return tensor


def expand_counter(counter: Counter[str], width: int) -> Counter[str]:
    expanded = Counter()
    for tile, count in counter.items():
        expanded[tile] += int(count) * int(width)
    return expanded


def remove_tiles_from_counter(counter: Counter[str], tiles: Counter[str]) -> None:
    for tile, count in tiles.items():
        if tile in TILE_NAMES and count > 0:
            counter[tile] -= int(count)
            if counter[tile] <= 0:
                del counter[tile]


def safe_chow_sequence(middle: str) -> list[str]:
    try:
        rank = int(middle[1])
    except (IndexError, ValueError):
        return []
    if len(middle) != 2 or middle[0] not in {"W", "T", "B"} or rank < 2 or rank > 8:
        return []
    return [f"{middle[0]}{rank - 1}", middle, f"{middle[0]}{rank + 1}"]


def respond_json(
    payload: dict,
    checkpoint: str,
    *,
    device: str | None = None,
    require_encoding_version: str | None = None,
    require_paper_config: bool = False,
) -> str:
    predictor = TjongCheckpointPredictor(
        checkpoint,
        device=device,
        require_encoding_version=require_encoding_version,
        require_paper_config=require_paper_config,
    )
    return respond_json_with_predictor(payload, predictor)


class _ForcedReplayPredictor:
    kind = "legal_action_ranker"

    def __init__(self, response: str = "PASS"):
        self.response = response

    def predict_legal_response(
        self,
        input_text: str,
        hand: Counter[str],
        player_id: int | None,
        request: str,
        candidates: list[str] | None = None,
    ) -> str:
        return self.response

    def predict_response(self, input_text: str) -> str:
        return self.response


def respond_json_with_predictor(payload: dict, predictor, *, fan_checker=None) -> str:
    # Import here so this module can be used in tests without adding scripts/ to sys.path.
    repo_root = Path(__file__).resolve().parents[4]
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from policy_bot import BotzonePolicy  # noqa: PLC0415

    requests = [str(item) for item in payload.get("requests", [])]
    responses = [str(item) for item in payload.get("responses", [])]
    if not requests:
        return "PASS"

    replay_predictor = _ForcedReplayPredictor()
    policy = BotzonePolicy(replay_predictor, fan_checker=fan_checker)
    for request, expected_response in zip(requests[:-1], responses):
        replay_predictor.response = expected_response
        actual_response = policy.respond(request)
        if actual_response != expected_response:
            raise ValueError(
                "Botzone replay diverged before current decision: "
                f"request={request!r} expected={expected_response!r} actual={actual_response!r}"
            )

    policy.predictor = predictor
    return policy.respond(requests[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-encoding-version", default=None)
    parser.add_argument("--require-paper-config", action="store_true")
    args = parser.parse_args()
    payload = json.loads(sys.stdin.read() or "{}")
    print(
        json.dumps(
            {
                "response": respond_json(
                    payload,
                    args.checkpoint,
                    device=args.device,
                    require_encoding_version=args.require_encoding_version,
                    require_paper_config=args.require_paper_config,
                )
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
