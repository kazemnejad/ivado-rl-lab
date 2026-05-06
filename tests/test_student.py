"""Tests for student-facing fill-ins — SPEC §5.4.

Task 7: ``compute_returns_to_go``.
Task 8: ``compute_advantage_vanilla``.
Task 9: ``compute_advantage_with_value_baseline``.
Task 10: ``compute_advantage_with_batch_baseline``.
Task 11: ``policy_loss``.
"""

from __future__ import annotations

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


def test_returns_shape():
    rewards = torch.zeros(4, 10)
    mask = torch.ones(4, 10, dtype=torch.bool)
    G = compute_returns_to_go(rewards, mask, gamma=1.0)
    assert G.shape == (4, 10) and G.dtype == torch.float32


def test_returns_known_2x3_case():
    rewards = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    G = compute_returns_to_go(rewards, mask, gamma=1.0)
    expected = torch.tensor([[1.0, 0.0, 0.0],
                             [1.0, 1.0, 1.0]])
    assert torch.allclose(G, expected), f"got {G.tolist()}"


def test_returns_gamma_decay():
    # rewards = [1, 1, 1] gamma=0.5
    # G_2 = 1
    # G_1 = 1 + 0.5*1 = 1.5
    # G_0 = 1 + 0.5*1.5 = 1.75
    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    G = compute_returns_to_go(rewards, mask, gamma=0.5)
    expected = torch.tensor([[1.75, 1.5, 1.0]])
    assert torch.allclose(G, expected, atol=1e-6)


def test_returns_matches_reference():
    torch.manual_seed(0)
    rewards = torch.randn(8, 50)
    mask = torch.bernoulli(torch.full((8, 50), 0.7)).bool()
    for gamma in [0.5, 0.99, 1.0]:
        ref = _compute_returns_to_go_ref(rewards, mask, gamma=gamma)
        got = compute_returns_to_go(rewards, mask, gamma=gamma)
        assert torch.allclose(got, ref, atol=1e-6), f"mismatch at gamma={gamma}"


def test_advantage_vanilla_returns_unchanged():
    returns = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
    mask = torch.ones_like(returns, dtype=torch.bool)
    A = compute_advantage_vanilla(returns, mask)
    assert torch.equal(A, returns), f"vanilla advantage must equal returns, got {A.tolist()}"
    # Document the aliasing contract: vanilla returns the input as-is, so
    # callers must NOT mutate the returned tensor in-place. Future fill-ins
    # for value/batch baselines must produce a NEW tensor.
    assert A is returns, "compute_advantage_vanilla should return the input by reference"


def test_advantage_vanilla_mask_argument_unused():
    # Same returns + DIFFERENT masks must produce identical advantage tensors,
    # i.e. mask is not consumed by the function body.
    returns = torch.randn(4, 8)
    mask_a = torch.ones(4, 8, dtype=torch.bool)
    mask_b = torch.zeros(4, 8, dtype=torch.bool)
    A_a = compute_advantage_vanilla(returns, mask_a)
    A_b = compute_advantage_vanilla(returns, mask_b)
    assert torch.equal(A_a, A_b), "mask should not affect vanilla advantage output"


def test_advantage_vanilla_matches_reference():
    torch.manual_seed(7)
    returns = torch.randn(8, 50)
    mask = torch.bernoulli(torch.full((8, 50), 0.7)).bool()
    ref = _compute_advantage_vanilla_ref(returns, mask)
    got = compute_advantage_vanilla(returns, mask)
    assert torch.equal(got, ref)


def test_advantage_value_baseline_equals_returns_minus_value():
    returns = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
    value_pred = torch.tensor([[0.5, 1.0, 2.0],
                                [3.0, 4.5, 5.5]])
    mask = torch.ones_like(returns, dtype=torch.bool)
    A = compute_advantage_with_value_baseline(returns, value_pred, mask)
    expected = torch.tensor([[0.5, 1.0, 1.0],
                              [1.0, 0.5, 0.5]])
    assert torch.allclose(A, expected, atol=1e-6), f"got {A.tolist()}"


def test_advantage_value_baseline_no_grad_through_value_pred():
    # If caller passes a detached value_pred, the advantage carries no grad.
    returns = torch.randn(2, 3, requires_grad=False)
    value_pred = torch.randn(2, 3, requires_grad=True).detach()  # detached
    mask = torch.ones(2, 3, dtype=torch.bool)
    A = compute_advantage_with_value_baseline(returns, value_pred, mask)
    assert not A.requires_grad, "detached value_pred should yield non-grad advantage"


def test_advantage_value_baseline_grad_flows_through_inputs():
    # Body must NOT call .detach() internally (detach is the caller's
    # responsibility per SPEC). If both inputs carry grad, the advantage
    # must remain differentiable w.r.t. both.
    returns_g = torch.randn(2, 3, requires_grad=True)
    value_pred_g = torch.randn(2, 3, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    A = compute_advantage_with_value_baseline(returns_g, value_pred_g, mask)
    assert A.requires_grad, "advantage must carry grad when both inputs require grad"
    A.sum().backward()
    assert returns_g.grad is not None and torch.all(returns_g.grad == 1.0)
    assert value_pred_g.grad is not None and torch.all(value_pred_g.grad == -1.0)


def test_advantage_value_baseline_shape_preserved():
    for B, T in [(1, 1), (4, 8), (16, 50)]:
        returns = torch.randn(B, T)
        value_pred = torch.randn(B, T)
        mask = torch.ones(B, T, dtype=torch.bool)
        A = compute_advantage_with_value_baseline(returns, value_pred, mask)
        assert A.shape == (B, T)
        assert A.dtype == returns.dtype


def test_advantage_value_baseline_matches_reference():
    torch.manual_seed(11)
    returns = torch.randn(8, 50)
    value_pred = torch.randn(8, 50)
    mask = torch.bernoulli(torch.full((8, 50), 0.7)).bool()
    ref = _compute_advantage_with_value_baseline_ref(returns, value_pred, mask)
    got = compute_advantage_with_value_baseline(returns, value_pred, mask)
    assert torch.allclose(got, ref, atol=0)  # exact match expected


def test_advantage_batch_baseline_per_t_mean_subtracted():
    # B=2, T=3, mask all True. mean_t over batch: [0.5*(2+4), 0.5*(3+5), 0.5*(1+1)] = [3, 4, 1]
    # adv = returns - mean_t broadcast
    returns = torch.tensor([[2.0, 3.0, 1.0],
                            [4.0, 5.0, 1.0]])
    mask = torch.ones_like(returns, dtype=torch.bool)
    A = compute_advantage_with_batch_baseline(returns, mask)
    expected = torch.tensor([[2 - 3.0, 3 - 4.0, 1 - 1.0],
                              [4 - 3.0, 5 - 4.0, 1 - 1.0]])
    assert torch.allclose(A, expected, atol=1e-6), f"got {A.tolist()}"


def test_advantage_batch_baseline_mask_aware_mean():
    # mask = [[T, T, F], [T, F, T]]; alive counts per t: [2, 1, 1]
    # mean_t (alive only) = [(2+4)/2, 3/1, 1/1] = [3, 3, 1]
    # NB: masked-out rows in `returns` are still in the tensor, but the
    # mean ignores them.
    returns = torch.tensor([[2.0, 3.0, 999.0],
                            [4.0, 999.0, 1.0]])
    mask = torch.tensor([[True, True, False],
                          [True, False, True]])
    A = compute_advantage_with_batch_baseline(returns, mask)
    # adv at masked positions is "garbage minus mean_t" — we don't care
    # about its value, only that the alive positions came out right.
    # alive (0,0): 2 - 3 = -1 ;  alive (0,1): 3 - 3 = 0 ; alive (1,0): 4 - 3 = 1 ; alive (1,2): 1 - 1 = 0
    assert torch.isclose(A[0, 0], torch.tensor(-1.0))
    assert torch.isclose(A[0, 1], torch.tensor(0.0))
    assert torch.isclose(A[1, 0], torch.tensor(1.0))
    assert torch.isclose(A[1, 2], torch.tensor(0.0))


def test_advantage_batch_baseline_centering():
    # Over alive cells, (adv * mask).sum(dim=0) ≈ 0 per timestep.
    torch.manual_seed(3)
    returns = torch.randn(8, 20)
    mask = torch.bernoulli(torch.full((8, 20), 0.7)).bool()
    A = compute_advantage_with_batch_baseline(returns, mask)
    # require denom>0 for centering check (skip timesteps with 0 alive)
    m = mask.float()
    denom = m.sum(dim=0)
    weighted = (A * m).sum(dim=0)
    nonzero = denom > 0
    assert torch.allclose(weighted[nonzero], torch.zeros_like(weighted[nonzero]), atol=1e-5)


def test_advantage_batch_baseline_T1_edge():
    # T=1 with all-True mask: mean_0 = mean of column → adv = returns - mean → centered, sum=0 across batch.
    returns = torch.tensor([[1.0], [2.0], [3.0]])
    mask = torch.ones(3, 1, dtype=torch.bool)
    A = compute_advantage_with_batch_baseline(returns, mask)
    # mean = 2.0; adv = [-1, 0, 1]
    expected = torch.tensor([[-1.0], [0.0], [1.0]])
    assert torch.allclose(A, expected, atol=1e-6)


def test_advantage_batch_baseline_matches_reference():
    torch.manual_seed(13)
    returns = torch.randn(8, 50)
    mask = torch.bernoulli(torch.full((8, 50), 0.7)).bool()
    ref = _compute_advantage_with_batch_baseline_ref(returns, mask)
    got = compute_advantage_with_batch_baseline(returns, mask)
    assert torch.allclose(got, ref, atol=0)


def test_policy_loss_scalar_output():
    log_probs = torch.randn(4, 8)
    advantages = torch.randn(4, 8)
    mask = torch.ones(4, 8, dtype=torch.bool)
    loss = policy_loss(log_probs, advantages, mask)
    assert loss.dim() == 0, f"expected scalar, got shape {loss.shape}"
    assert loss.dtype == torch.float32


def test_policy_loss_known_2x2_value():
    # logp = [[a, b], [c, d]], adv = [[1, 1], [1, 1]], mask all True
    # loss = -(a + b + c + d) / 4
    log_probs = torch.tensor([[0.1, 0.2],
                               [0.3, 0.4]])
    advantages = torch.ones_like(log_probs)
    mask = torch.ones_like(log_probs, dtype=torch.bool)
    loss = policy_loss(log_probs, advantages, mask)
    expected = -(0.1 + 0.2 + 0.3 + 0.4) / 4
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6), (
        f"got {loss.item()}, expected {expected}"
    )


def test_policy_loss_mask_zeros_dont_contribute():
    # Garbage values in masked-out cells must not change loss.
    log_probs_a = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                 [0.5, 0.6, 0.7, 0.8]])
    advantages = torch.ones_like(log_probs_a)
    mask = torch.tensor([[True, True, False, False],
                          [True, True, False, False]])
    log_probs_b = log_probs_a.clone()
    log_probs_b[:, 2:] = 1e6  # garbage
    advantages_b = advantages.clone()
    advantages_b[:, 2:] = 1e6
    loss_a = policy_loss(log_probs_a, advantages, mask)
    loss_b = policy_loss(log_probs_b, advantages_b, mask)
    assert torch.isclose(loss_a, loss_b, atol=1e-6), (
        f"mask not honored: {loss_a.item()} vs {loss_b.item()}"
    )


def test_policy_loss_differentiable_wrt_log_probs():
    log_probs = torch.randn(2, 3, requires_grad=True)
    advantages = torch.randn(2, 3)  # detached effectively (no requires_grad)
    mask = torch.ones(2, 3, dtype=torch.bool)
    loss = policy_loss(log_probs, advantages, mask)
    loss.backward()
    assert log_probs.grad is not None
    assert log_probs.grad.shape == log_probs.shape


def test_policy_loss_matches_reference():
    torch.manual_seed(17)
    log_probs = torch.randn(8, 50)
    advantages = torch.randn(8, 50)
    mask = torch.bernoulli(torch.full((8, 50), 0.7)).bool()
    ref = _policy_loss_ref(log_probs, advantages, mask)
    got = policy_loss(log_probs, advantages, mask)
    # student.py and utils.py refs run identical Python expressions on the
    # same tensors, so exact bit-equality is expected (matches the pattern
    # used by the batch-baseline ref-match test).
    assert torch.equal(got, ref)
