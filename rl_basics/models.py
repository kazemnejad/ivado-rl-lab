"""MLP policy and value networks for tabular RL on FourRoomsTL.

See SPEC §5.2: one-hot state encoding -> 1 hidden layer -> output head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """One-hot state -> hidden -> n_actions logits."""

    def __init__(self, n_states: int = 289, n_actions: int = 4, hidden: int = 64):
        super().__init__()
        self.n_states = n_states
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, n_actions)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: (B,) long; returns (B, n_actions) float logits.
        x = F.one_hot(states, self.n_states).float()
        return self.fc2(F.relu(self.fc1(x)))


class ValueNetwork(nn.Module):
    """One-hot state -> hidden -> scalar V."""

    def __init__(self, n_states: int = 289, hidden: int = 64):
        super().__init__()
        self.n_states = n_states
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: (B,) long; returns (B,) float (squeezed last dim).
        x = F.one_hot(states, self.n_states).float()
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)
