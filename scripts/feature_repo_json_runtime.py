#!/usr/bin/env python3
"""Replay official-judge Botzone payloads through feature-style bots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class ReplayFeatureJsonBot:
    def __init__(self, model: Any, *, obs_mode: str) -> None:
        self.model = model
        self.obs_mode = obs_mode
        self.agent: Any | None = None
        self.seat_wind: int | None = None
        self.prevalent_wind = 0
        self.pending_angang_tile: str | None = None
        self.after_draw = False

    def replay_payload(self, payload: dict[str, Any]) -> str:
        requests = [str(item) for item in payload.get("requests", [])]
        responses = [str(item) for item in payload.get("responses", [])]
        if not requests:
            return "PASS"

        for request, response in zip(requests[:-1], responses):
            self._observe_request(request)
            self._remember_response(request, response)
        return self._respond_to_request(requests[-1])

    def respond(self, request: str) -> str:
        response = self._respond_to_request(request)
        self._remember_response(request, response)
        return response

    def _predict_response(self, obs: dict[str, Any]) -> str:
        obs_tensor = torch.from_numpy(
            np.expand_dims(np.asarray(obs["observation"], dtype=np.float32), 0)
        )
        mask_tensor = torch.from_numpy(
            np.expand_dims(np.asarray(obs["action_mask"], dtype=np.float32), 0)
        )
        input_obs = {"observation": obs_tensor, "action_mask": mask_tensor}
        if self.obs_mode == "vec":
            input_obs["vec"] = torch.from_numpy(
                np.expand_dims(np.asarray(obs["vec"], dtype=np.float32), 0)
            )
        elif self.obs_mode == "global":
            glob = obs.get("global")
            if glob is None:
                glob = np.zeros(10, dtype=np.float32)
            input_obs["global"] = torch.from_numpy(
                np.expand_dims(np.asarray(glob, dtype=np.float32), 0)
            )
        with torch.no_grad():
            logits = self.model({"is_training": False, "obs": input_obs})
        action = int(logits.detach().cpu().numpy().reshape(-1).argmax())
        assert self.agent is not None
        return str(self.agent.action2response(action))

    def _remember_response(self, request: str, response: str) -> None:
        request_tokens = request.strip().split()
        response_tokens = response.strip().split()
        if not request_tokens or not response_tokens:
            return
        if request_tokens[0] == "2" and response_tokens[0].upper() == "GANG" and len(response_tokens) >= 2:
            self.pending_angang_tile = response_tokens[1]
        elif response_tokens[0].upper() != "GANG":
            self.pending_angang_tile = None

    def _respond_to_request(self, request: str) -> str:
        tokens, obs = self._observe_request(request)
        if not tokens:
            return "PASS"
        if tokens[0] == "2":
            if obs is None:
                return "PASS"
            return self._format_draw_response(self._predict_response(obs))
        if tokens[0] != "3" or len(tokens) < 3:
            return "PASS"
        action = tokens[2].upper()
        actor = int(tokens[1])
        if action == "BUGANG":
            if actor == self.seat_wind or obs is None:
                return "PASS"
            return "HU" if self._predict_response(obs) == "Hu" else "PASS"
        if action in {"PLAY", "PENG", "CHI"}:
            if actor == self.seat_wind or obs is None:
                return "PASS"
            return self._format_reaction_response(self._predict_response(obs))
        return "PASS"

    def _format_draw_response(self, response: str) -> str:
        parts = response.split()
        if not parts:
            return "PASS"
        head = parts[0]
        if head == "Hu":
            return "HU"
        if head == "Play" and len(parts) >= 2:
            return f"PLAY {parts[1]}"
        if head in {"Gang", "AnGang"} and len(parts) >= 2:
            return f"GANG {parts[1]}"
        if head == "BuGang" and len(parts) >= 2:
            return f"BUGANG {parts[1]}"
        return "PASS"

    def _format_reaction_response(self, response: str) -> str:
        parts = response.split()
        if not parts:
            return "PASS"
        head = parts[0]
        if head == "Hu":
            return "HU"
        if head == "Pass":
            return "PASS"
        if head == "Gang":
            self.pending_angang_tile = None
            return "GANG"
        if head not in {"Peng", "Chi"}:
            return "PASS"
        assert self.agent is not None
        assert self.seat_wind is not None
        post_obs = self.agent.request2obs(f"Player {self.seat_wind} {response}")
        discard_parts = self._predict_response(post_obs).split()
        discard = discard_parts[-1] if discard_parts else ""
        self.agent.request2obs(f"Player {self.seat_wind} Un{response}")
        formatted = [head.upper(), *parts[1:]]
        if discard:
            formatted.append(discard)
        return " ".join(formatted).strip() or "PASS"

    def _observe_request(self, request: str) -> tuple[list[str], dict[str, Any] | None]:
        tokens = request.strip().split()
        if not tokens:
            return [], None
        if tokens[0] == "0":
            self._handle_init(tokens)
            return tokens, None
        if self.agent is None:
            return tokens, None
        if tokens[0] == "1":
            hand_tiles = [tile for tile in tokens[5:] if not tile.startswith("H")]
            self.agent.request2obs(" ".join(["Deal", *hand_tiles[:13]]))
            return tokens, None
        if tokens[0] == "2":
            tile = tokens[1] if len(tokens) >= 2 else ""
            self.after_draw = False
            if tile.startswith("H"):
                return tokens, None
            return tokens, self.agent.request2obs(f"Draw {tile}")
        if tokens[0] != "3" or len(tokens) < 3:
            return tokens, None

        actor = int(tokens[1])
        event = tokens[2].upper()
        if event == "DRAW":
            self.agent.request2obs(f"Player {actor} Draw")
            self.after_draw = True
            return tokens, None
        if event == "BUHUA":
            self.agent.request2obs(f"Player {actor} Draw")
            self.after_draw = True
            return tokens, None
        if event == "GANG":
            if actor == self.seat_wind and self.pending_angang_tile:
                self.agent.request2obs(f"Player {actor} AnGang {self.pending_angang_tile}")
                self.pending_angang_tile = None
            elif self.after_draw:
                self.agent.request2obs(f"Player {actor} AnGang")
            else:
                self.agent.request2obs(f"Player {actor} Gang")
            self.after_draw = False
            return tokens, None
        if event == "BUGANG" and len(tokens) >= 4:
            self.after_draw = False
            return tokens, self.agent.request2obs(f"Player {actor} BuGang {tokens[3]}")
        if event == "PLAY" and len(tokens) >= 4:
            self.after_draw = False
            return tokens, self.agent.request2obs(f"Player {actor} Play {tokens[3]}")
        if event == "PENG" and len(tokens) >= 4:
            self.after_draw = False
            self.agent.request2obs(f"Player {actor} Peng")
            return tokens, self.agent.request2obs(f"Player {actor} Play {tokens[3]}")
        if event == "CHI" and len(tokens) >= 5:
            self.after_draw = False
            self.agent.request2obs(f"Player {actor} Chi {tokens[3]}")
            return tokens, self.agent.request2obs(f"Player {actor} Play {tokens[4]}")
        return tokens, None

    def _handle_init(self, tokens: list[str]) -> None:
        if len(tokens) < 3:
            return
        self.seat_wind = int(tokens[1])
        self.prevalent_wind = int(tokens[2])
        self.agent = self._new_agent(self.seat_wind)
        self.agent.request2obs(f"Wind {self.prevalent_wind}")
        self.pending_angang_tile = None
        self.after_draw = False

    def _new_agent(self, seat_wind: int) -> Any:
        raise NotImplementedError


def main_from_factory(factory) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("json", "text"), default="json")
    args = parser.parse_args()

    bot = factory()
    if args.protocol == "text":
        try:
            input()
            while True:
                request = input()
                while not request.strip():
                    request = input()
                print(bot.respond(request), flush=True)
                print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<", flush=True)
        except EOFError:
            return 0
        return 0

    payload = json.loads(sys.stdin.read() or "{}")
    print(json.dumps({"response": bot.replay_payload(payload)}, ensure_ascii=False), flush=True)
    return 0
