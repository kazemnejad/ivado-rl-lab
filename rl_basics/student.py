"""Student-facing fill-ins — SPEC §5.4.

This module contains the public reference implementations of the small
primitives the student fills in during the lab walkthrough.

Dual-location pattern: each function defined here also has a private twin
``_<name>_ref`` in :mod:`rl_basics.utils`. The twin is used as an independent
oracle in the test suite (see ``tests/test_student.py``) and in the Task 12
student stress suite. When the lab notebook ships, a separate build script
will stub the bodies in this module while leaving the ``utils.py`` reference
intact — so the tests still have something to compare against.
"""

from __future__ import annotations

import torch


def compute_returns_to_go(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Returns G_t = Σ_{k≥t} γ^(k-t) · r_k.

    rewards : (B, T) float
    mask    : (B, T) bool — True where alive
    gamma   : float in (0, 1]
    returns : (B, T) float

    Hint: do this BACKWARD. Initialize a running sum, sweep t from T-1 to 0.
    """
    B, T = rewards.shape
    assert mask.shape == (B, T), f"mask shape mismatch: {mask.shape}"
    # === STUDENT TODO === #
    G = torch.zeros_like(rewards)
    running = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    for t in range(T - 1, -1, -1):
        running = rewards[:, t] + gamma * running
        G[:, t] = running
    return G


def compute_advantage_vanilla(
    returns: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """A_t = G_t.  (B, T) → (B, T).  One liner.  Mask is unused here."""
    # === STUDENT TODO === #
    return returns


def compute_advantage_with_value_baseline(
    returns: torch.Tensor,
    value_pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """A_t = G_t − V_φ(s_t).  All (B, T).  value_pred should be detached
    upstream (we don't backprop through value_net for the policy gradient).

    One-liner.
    """
    # === STUDENT TODO === #
    return returns - value_pred


def compute_advantage_with_batch_baseline(
    returns: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """A_t = G_t − μ_t   where μ_t = mean over the batch axis at timestep t.

    Mask-aware: only average G_t over alive trajectories at time t.

    Shape: (B, T) → (B, T).
    """
    # === STUDENT TODO === #
    m = mask.float()
    denom = m.sum(dim=0).clamp(min=1.0)                 # (T,)
    mean_t = (returns * m).sum(dim=0) / denom           # (T,)
    return returns - mean_t.unsqueeze(0)                # (B, T)


def policy_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """The REINFORCE policy-gradient loss.

    All inputs (B, T).  Returns a scalar.

    Formula:  L = − (1/Σ mask) · Σ_{(b,t) alive} A_{b,t} · log π(a|s)

    This is the EXACT same expression as in the LLM Lab 2 — keep it in mind.
    """
    # === STUDENT TODO === #
    m = mask.float()
    return -(advantages * log_probs * m).sum() / m.sum().clamp(min=1)
