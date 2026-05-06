"""Student fill-in stress suite — SPEC §7.5.

For each of the 5 student fill-ins, sweep ``B ∈ {1, 8, 16, 64} × T ∈ {1, 10, 400}``
× ``seed ∈ {0, 1, 2}`` and compare against the shipped reference in
:mod:`rl_basics.utils` within fp tolerance.

Plus mask edge cases (all-False, all-True, half-half) for advantages + policy_loss
and differentiability checks (grad flow through log_probs / advantages /
value_pred).

All tests carry ``@pytest.mark.stress`` and are deselected by default via the
``addopts`` setting in ``pyproject.toml``. Opt in with ``-m stress``.
"""

from __future__ import annotations

import pytest
import torch

from rl_basics.student import (
    compute_advantage_vanilla,
    compute_advantage_with_batch_baseline,
    compute_advantage_with_value_baseline,
    compute_returns_to_go,
    policy_loss,
)
from rl_basics.utils import (
    _compute_advantage_vanilla_ref,
    _compute_advantage_with_batch_baseline_ref,
    _compute_advantage_with_value_baseline_ref,
    _compute_returns_to_go_ref,
    _policy_loss_ref,
)

# ---------------------------------------------------------------------------
# Reference-vs-impl matrix tests
# ---------------------------------------------------------------------------

_BS = [1, 8, 16, 64]
_TS = [1, 10, 400]
_SEEDS = [0, 1, 2]


def _make_mask(B: int, T: int, p: float = 0.7) -> torch.Tensor:
    return torch.bernoulli(torch.full((B, T), p)).bool()


@pytest.mark.stress
@pytest.mark.parametrize("B", _BS)
@pytest.mark.parametrize("T", _TS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_returns_to_go_matches_reference(B: int, T: int, seed: int) -> None:
    torch.manual_seed(seed)
    rewards = torch.randn(B, T)
    mask = _make_mask(B, T)
    for gamma in (0.5, 0.99, 1.0):
        ref = _compute_returns_to_go_ref(rewards, mask, gamma=gamma)
        got = compute_returns_to_go(rewards, mask, gamma=gamma)
        assert got.shape == ref.shape == (B, T)
        assert got.dtype == ref.dtype
        assert torch.allclose(got, ref, atol=1e-6), (
            f"returns_to_go mismatch B={B} T={T} seed={seed} gamma={gamma}"
        )


@pytest.mark.stress
@pytest.mark.parametrize("B", _BS)
@pytest.mark.parametrize("T", _TS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_advantage_vanilla_matches_reference(B: int, T: int, seed: int) -> None:
    torch.manual_seed(seed)
    returns = torch.randn(B, T)
    mask = _make_mask(B, T)
    ref = _compute_advantage_vanilla_ref(returns, mask)
    got = compute_advantage_vanilla(returns, mask)
    assert got.shape == ref.shape == (B, T)
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)


@pytest.mark.stress
@pytest.mark.parametrize("B", _BS)
@pytest.mark.parametrize("T", _TS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_advantage_with_value_baseline_matches_reference(
    B: int, T: int, seed: int
) -> None:
    torch.manual_seed(seed)
    returns = torch.randn(B, T)
    value_pred = torch.randn(B, T)
    mask = _make_mask(B, T)
    ref = _compute_advantage_with_value_baseline_ref(returns, value_pred, mask)
    got = compute_advantage_with_value_baseline(returns, value_pred, mask)
    assert got.shape == ref.shape == (B, T)
    assert got.dtype == ref.dtype
    assert torch.allclose(got, ref, atol=1e-6)


@pytest.mark.stress
@pytest.mark.parametrize("B", _BS)
@pytest.mark.parametrize("T", _TS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_advantage_with_batch_baseline_matches_reference(
    B: int, T: int, seed: int
) -> None:
    torch.manual_seed(seed)
    returns = torch.randn(B, T)
    mask = _make_mask(B, T)
    ref = _compute_advantage_with_batch_baseline_ref(returns, mask)
    got = compute_advantage_with_batch_baseline(returns, mask)
    assert got.shape == ref.shape == (B, T)
    assert got.dtype == ref.dtype
    assert torch.allclose(got, ref, atol=1e-6)


@pytest.mark.stress
@pytest.mark.parametrize("B", _BS)
@pytest.mark.parametrize("T", _TS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_policy_loss_matches_reference(B: int, T: int, seed: int) -> None:
    torch.manual_seed(seed)
    log_probs = torch.randn(B, T)
    advantages = torch.randn(B, T)
    mask = _make_mask(B, T)
    ref = _policy_loss_ref(log_probs, advantages, mask)
    got = policy_loss(log_probs, advantages, mask)
    assert got.dim() == 0
    assert got.dtype == ref.dtype
    assert torch.isclose(got, ref, atol=1e-6) or torch.equal(got, ref), (
        f"policy_loss mismatch B={B} T={T} seed={seed}: got={got} ref={ref}"
    )


# ---------------------------------------------------------------------------
# Mask edge cases
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize("mask_kind", ["all_false", "all_true", "half_half"])
def test_advantage_and_loss_mask_edge_cases(mask_kind: str) -> None:
    torch.manual_seed(0)
    B, T = 8, 16
    returns = torch.randn(B, T)
    value_pred = torch.randn(B, T)
    log_probs = torch.randn(B, T)

    if mask_kind == "all_false":
        mask = torch.zeros(B, T, dtype=torch.bool)
    elif mask_kind == "all_true":
        mask = torch.ones(B, T, dtype=torch.bool)
    else:  # half_half
        mask = torch.cat(
            [
                torch.ones(B, T // 2, dtype=torch.bool),
                torch.zeros(B, T - T // 2, dtype=torch.bool),
            ],
            dim=1,
        )

    for adv_fn, args in [
        (compute_advantage_vanilla, (returns, mask)),
        (compute_advantage_with_value_baseline, (returns, value_pred, mask)),
        (compute_advantage_with_batch_baseline, (returns, mask)),
    ]:
        A = adv_fn(*args)
        assert A.shape == (B, T)
        assert torch.isfinite(A).all(), (
            f"{adv_fn.__name__} produced NaN/Inf with mask={mask_kind}"
        )

    A = compute_advantage_vanilla(returns, mask)
    L = policy_loss(log_probs, A, mask)
    assert L.dim() == 0
    assert torch.isfinite(L).item(), (
        f"policy_loss produced NaN/Inf with mask={mask_kind}"
    )
    if mask_kind == "all_false":
        # Σ mask = 0 → clamp(min=1) returns 0/1 = 0 by construction.
        assert L.item() == 0.0


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


@pytest.mark.stress
def test_policy_loss_grad_flows_through_log_probs_and_advantages() -> None:
    torch.manual_seed(0)
    log_probs = torch.randn(8, 16, requires_grad=True)
    advantages = torch.randn(8, 16, requires_grad=True)
    mask = torch.ones(8, 16, dtype=torch.bool)
    L = policy_loss(log_probs, advantages, mask)
    L.backward()
    assert log_probs.grad is not None
    assert advantages.grad is not None
    assert torch.isfinite(log_probs.grad).all()
    assert torch.isfinite(advantages.grad).all()


@pytest.mark.stress
def test_value_baseline_advantage_grad_flows_through_value_pred() -> None:
    """We do NOT detach inside ``compute_advantage_with_value_baseline``; any
    detaching is the caller's responsibility (SPEC §5.4 docstring). So
    ``∂A/∂value_pred = -1``."""
    torch.manual_seed(0)
    returns = torch.randn(8, 16)
    value_pred = torch.randn(8, 16, requires_grad=True)
    mask = torch.ones(8, 16, dtype=torch.bool)
    A = compute_advantage_with_value_baseline(returns, value_pred, mask)
    A.sum().backward()
    assert value_pred.grad is not None
    assert torch.all(value_pred.grad == -1.0)
