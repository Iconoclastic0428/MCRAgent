"""Botzone policy wrapper for a trained Tjong checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

from .actions import ACTION_TO_INDEX, flatten_claim
from .encoding import build_game_features, stack_hidden_tile_features, stack_visible_tile_features
from .evaluate_supervised import load_model
from .tensorize_botzone import HIDDEN_TILE_ROW_NAMES, chow_claim_index, request_event_tile, selected_action_mask
from .tiles import TILE_NAMES, tile_id


class TjongCheckpointPredictor:
    """Score legal Botzone responses with the paper hierarchical heads."""

    kind = "legal_action_ranker"

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
        scores = self.score_legal_response_candidates(input_text, hand, player_id, request, candidates)
        if not scores:
            return "PASS" if "PASS" in candidates else candidates[0]
        best_index = max(range(len(candidates)), key=lambda index: scores[index])
        if scores[best_index] == float("-inf"):
            return "PASS" if "PASS" in candidates else candidates[0]
        return candidates[best_index]

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

        visible, game, hidden = encode_runtime_state(
            input_text=input_text,
            hand=hand,
            player_id=player_id or 0,
            action_mask=action_mask,
        )
        visible_memory, game_memory = current_memory(visible, game)
        sub_memories = [
            current_memory(
                *encode_runtime_state(
                    input_text=input_text,
                    hand=hand,
                    player_id=player_id or 0,
                    action_mask=selected_action_mask(label[0] if label is not None else ACTION_TO_INDEX["PASS"]),
                )[:2]
            )
            for label in labels
        ]
        forced_discard_indices: list[int] = []
        forced_discard_memories: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        discard_only_mask = selected_action_mask(ACTION_TO_INDEX["DISCARD"])
        for index, (candidate, label) in enumerate(zip(candidates, labels)):
            if label is None or label[0] not in CLAIM_ACTIONS or response_discard_label(candidate) is None:
                continue
            post_visible, post_game, post_hidden = encode_runtime_state(
                input_text=input_text,
                hand=hand,
                player_id=player_id or 0,
                action_mask=discard_only_mask,
                post_claim_request=request,
                post_claim_response=candidate,
            )
            post_visible_memory, post_game_memory = current_memory(post_visible, post_game)
            forced_discard_indices.append(index)
            forced_discard_memories.append((post_visible_memory, post_game_memory, post_hidden))
        visible_batch = visible_memory.unsqueeze(0).repeat(len(candidates), 1, 1, 1).to(self.device)
        game_batch = game_memory.unsqueeze(0).repeat(len(candidates), 1, 1).to(self.device)
        sub_visible_batch = torch.stack([item[0] for item in sub_memories], dim=0).to(self.device)
        sub_game_batch = torch.stack([item[1] for item in sub_memories], dim=0).to(self.device)
        hidden_batch = hidden.unsqueeze(0).repeat(len(candidates), 1, 1).to(self.device)
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                visible_tiles=visible_batch,
                game_features=game_batch,
                sub_visible_tiles=sub_visible_batch,
                sub_game_features=sub_game_batch,
                hidden_tiles=hidden_batch,
            )
            forced_discard_logp: dict[int, torch.Tensor] = {}
            if forced_discard_memories:
                forced_visible_batch = torch.stack([item[0] for item in forced_discard_memories], dim=0).to(self.device)
                forced_game_batch = torch.stack([item[1] for item in forced_discard_memories], dim=0).to(self.device)
                forced_hidden_batch = torch.stack([item[2] for item in forced_discard_memories], dim=0).to(self.device)
                forced_outputs = self.model(
                    visible_tiles=forced_visible_batch,
                    game_features=forced_game_batch,
                    sub_visible_tiles=forced_visible_batch,
                    sub_game_features=forced_game_batch,
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
    # Import here so this module can be used in tests without adding scripts/ to sys.path.
    repo_root = Path(__file__).resolve().parents[4]
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from policy_bot import BotzonePolicy  # noqa: PLC0415

    predictor = TjongCheckpointPredictor(
        checkpoint,
        device=device,
        require_encoding_version=require_encoding_version,
        require_paper_config=require_paper_config,
    )
    policy = BotzonePolicy(predictor)
    requests = [str(item) for item in payload.get("requests", [])]
    responses = [str(item) for item in payload.get("responses", [])]
    for request, expected_response in zip(requests[:-1], responses):
        actual_response = policy.respond(request)
        if actual_response != expected_response:
            continue
    return policy.respond(requests[-1]) if requests else "PASS"


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
