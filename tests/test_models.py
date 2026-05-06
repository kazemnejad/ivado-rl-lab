"""Tests for MLPPolicy and ValueNetwork (SPEC §5.2)."""

import torch

from rl_basics.models import MLPPolicy, ValueNetwork


def test_policy_forward_shape():
    policy = MLPPolicy()
    states = torch.zeros(8, dtype=torch.long)
    logits = policy(states)
    assert logits.shape == (8, 4)
    assert logits.dtype == torch.float32


def test_policy_one_hot_invariant():
    # Forward is deterministic given fixed weights: same input -> same output.
    # Toggle train/eval modes to lock in "no stochastic layers" as a regression
    # guard (a future Dropout / BatchNorm would diverge across modes).
    torch.manual_seed(0)
    policy = MLPPolicy()
    s = torch.tensor([5, 17, 200], dtype=torch.long)
    policy.train()
    out_train = policy(s)
    out_train_again = policy(s)
    policy.eval()
    out_eval = policy(s)
    assert torch.allclose(out_train, out_train_again), "non-deterministic forward in train mode"
    assert torch.allclose(out_train, out_eval), "train != eval (stochastic layer present?)"


def test_value_forward_shape():
    v = ValueNetwork()
    # B=8 case
    out8 = v(torch.zeros(8, dtype=torch.long))
    assert out8.shape == (8,)
    assert out8.dtype == torch.float32
    # B=1 case: squeeze(-1) must preserve the batch dim (squeeze() would not).
    out1 = v(torch.zeros(1, dtype=torch.long))
    assert out1.shape == (1,), f"B=1 shape collapsed: {out1.shape}"


def test_param_count_matches_spec():
    policy = MLPPolicy()
    p_count = sum(p.numel() for p in policy.parameters())
    assert p_count == 18820, f"MLPPolicy total params = {p_count}, expected 18820"
    v = ValueNetwork()
    v_count = sum(p.numel() for p in v.parameters())
    assert v_count == 18625, f"ValueNetwork total params = {v_count}, expected 18625"
