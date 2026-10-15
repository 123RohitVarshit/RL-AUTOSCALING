"""Canonical policy architecture + observation normalization.

Training and inference both import from this file so they cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from agents.base import AutoscalerAgent

# Must match MathClusterEnv defaults and _obs().
MAX_SERVERS = 8
MAX_QUEUE = 50
MAX_ARRIVAL = 25.0
SCALE_WINDOW = 50.0

OBS_SCALE = np.array(
    [MAX_SERVERS, MAX_SERVERS, MAX_QUEUE, MAX_ARRIVAL, SCALE_WINDOW],
    dtype=np.float32,
)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_size: int = 5, act_size: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor = nn.Linear(64, act_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


def state_to_obs(state: dict) -> np.ndarray:
    """Raw grader state -> normalized network input. Single source of truth."""
    raw = np.array([
        state.get("active_servers", 2),
        state.get("booting_servers", 0),
        state.get("queue", 0.0),
        state.get("arrival_rate", 5.0),
        min(state.get("time_since_scale", 50), SCALE_WINDOW),
    ], dtype=np.float32)
    return raw / OBS_SCALE


class RLAutoscaler(AutoscalerAgent):
    def __init__(self, model_path=None):
        self.policy = PolicyNetwork()
        if model_path:
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
            self.policy.load_state_dict(sd)
        self.policy.eval()

    def reset(self):
        pass

    def decide(self, state: dict) -> int:
        obs = torch.from_numpy(state_to_obs(state)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.policy(obs)
        return int(torch.argmax(logits, dim=-1).item())
