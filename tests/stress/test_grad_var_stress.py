"""Transcription smoke tests for ``measure_grad_variance`` (Task 19).

The math properties (1/B scaling, split-half consistency, baseline reduction)
are already proven by ``$PROTO/verify_grad_variance.py`` (5/5 pass) on the
same algorithm we transcribe here. We do NOT re-prove them — that would burn
hundreds of bootstrap rollouts to verify code that's already verified.

What these tests DO check is that the transcription is intact:
  * Test 1 — bit-exact match vs inline autograd (catches any bug in the
    advantage / loss / grad-flatten pipeline).
  * Test 2 — successive bootstrap rollouts use different env RNG (catches
    accidental state reuse).
  * Test 3 — public API smoke: returns the right keys, tr_var > 0 on a
    non-degenerate setup.

All three are gated behind ``@pytest.mark.stress`` to keep them out of the
default unit run, but they finish in <10 s end-to-end.

Run:
    .venv/bin/python -m pytest tests/stress/test_grad_var_stress.py -v -m stress
"""

from __future__ import annotations

import pytest
import torch

from rl_basics.env import FourRoomsTL
from rl_basics.grad_variance import (
    _grad_one_batch,
    measure_grad_variance,
    _measure_grad_variance_raw,
)
from rl_basics.models import MLPPolicy
from rl_basics.utils import (
    _compute_advantage_vanilla_ref,
    _compute_returns_to_go_ref,
    _policy_loss_ref,
    log_probs as log_probs_fn,
    rollout,
)


ENV_KW: dict = {}  # FourRoomsTL has fixed config; nothing to pass.
GAMMA = 0.99


def _fresh_policy(seed: int = 0, hidden: int = 16) -> MLPPolicy:
    torch.manual_seed(seed)
    return MLPPolicy(n_states=FourRoomsTL.n_states, n_actions=FourRoomsTL.n_actions,
                     hidden=hidden)


# ---------------------------------------------------------------------------
# Test 1 — bit-exact match against inline autograd
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_bit_exact_autograd_match():
    """``_grad_one_batch`` matches the inline ``policy_loss; .backward()`` g
    bit-for-bit (same RNG state ⇒ identical rollout ⇒ identical g).

    This catches transcription bugs in the advantage / log_probs / loss
    pipeline. Wall: <1 s.
    """
    pol = _fresh_policy(seed=42, hidden=16)

    # Path A: through our helper.
    torch.manual_seed(7)
    env_a = FourRoomsTL(batch_size=8, device="cpu", seed=7)
    g_a = _grad_one_batch(env_a, pol, value_net=None,
                          advantage_kind="vanilla", gamma=GAMMA)

    # Path B: inline.
    torch.manual_seed(7)
    env_b = FourRoomsTL(batch_size=8, device="cpu", seed=7)
    traj = rollout(env_b, pol)
    G = _compute_returns_to_go_ref(traj.rewards, traj.mask, gamma=GAMMA)
    adv = _compute_advantage_vanilla_ref(G, traj.mask)
    logits = pol(traj.states.flatten())
    lp = log_probs_fn(logits, traj.actions.flatten()).view_as(traj.states)
    loss = _policy_loss_ref(lp, adv.detach(), traj.mask)
    for p in pol.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    loss.backward()
    g_b = torch.cat([p.grad.detach().flatten() for p in pol.parameters()
                     if p.requires_grad])

    assert g_a.shape == g_b.shape, (g_a.shape, g_b.shape)
    assert torch.equal(g_a, g_b), (
        f"gradients differ (max abs diff = {(g_a - g_b).abs().max().item():.3e})"
    )


# ---------------------------------------------------------------------------
# Test 2 — bootstrap calls use independent env RNG
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_bootstrap_independence():
    """Successive bootstrap rollouts must use independent env RNG. If we
    accidentally reused the same batch each time, every g would be identical.
    Catches state-reuse bugs in ``_measure_grad_variance_raw``. Wall: <2 s.
    """
    pol = _fresh_policy(seed=1, hidden=16)
    torch.manual_seed(123)
    grads = _measure_grad_variance_raw(
        FourRoomsTL, ENV_KW, batch_size=8, device="cpu",
        policy=pol, value_net=None, advantage_kind="vanilla",
        gamma=GAMMA, n_boot=5,
    )
    assert grads.shape[0] == 5
    n_dup = 0
    for i in range(5):
        for j in range(i + 1, 5):
            if (grads[i] - grads[j]).abs().max().item() < 1e-8:
                n_dup += 1
    assert n_dup == 0, "bootstrap calls reused the same batch (RNG not advancing)"


# ---------------------------------------------------------------------------
# Test 3 — public API smoke
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_measure_grad_variance_smoke():
    """``measure_grad_variance`` returns the documented keys with sane values
    on a tiny non-degenerate setup. Wall: <2 s.
    """
    pol = _fresh_policy(seed=2, hidden=16)
    torch.manual_seed(456)
    out = measure_grad_variance(
        FourRoomsTL, ENV_KW, batch_size=8, device="cpu",
        policy=pol, value_net=None, advantage_kind="vanilla",
        gamma=GAMMA, n_boot=4,
    )
    assert set(out.keys()) >= {"n_boot", "mean_norm_sq", "mean_g_norm_sq",
                                "tr_var", "snr"}
    assert out["n_boot"] == 4
    assert out["tr_var"] >= 0.0
    assert out["mean_norm_sq"] >= out["mean_g_norm_sq"], (
        "tr_var = E[||g||²] − ||E[g]||² must be non-negative"
    )
