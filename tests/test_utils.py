"""Tests for rl_basics.utils — SPEC §5.3 helpers."""

import torch
import torch.nn as nn

from rl_basics.utils import (
    grad_norm,
    log_probs,
    reset,
    sample_action,
    shape,
    value_loss,
)


def test_sample_action_shape_and_range():
    torch.manual_seed(0)
    logits = torch.randn(8, 4)
    a = sample_action(logits)
    assert a.shape == (8,)
    assert a.dtype == torch.long
    assert a.min().item() >= 0 and a.max().item() <= 3


def test_log_probs_matches_categorical():
    torch.manual_seed(0)
    logits = torch.randn(16, 4)
    actions = torch.randint(0, 4, (16,))
    expected = torch.distributions.Categorical(logits=logits).log_prob(actions)
    got = log_probs(logits, actions)
    assert torch.allclose(got, expected)


def test_value_loss_zero_when_pred_eq_returns():
    pred = torch.randn(4, 10, requires_grad=True)
    returns = pred.detach().clone()
    mask = torch.ones(4, 10, dtype=torch.bool)
    loss = value_loss(pred, returns, mask)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_value_loss_mask_aware():
    pred = torch.zeros(2, 4)
    returns_a = torch.zeros(2, 4)
    returns_b = returns_a.clone()
    returns_b[:, 2:] = 1e6  # garbage in masked-out cells
    mask = torch.tensor(
        [[True, True, False, False], [True, True, False, False]]
    )
    la = value_loss(pred, returns_a, mask)
    lb = value_loss(pred, returns_b, mask)
    assert torch.allclose(la, lb), f"mask not honored: {la.item()} vs {lb.item()}"


def test_grad_norm_after_backward():
    # tiny linear: y = w * x, x=2, w=0.5 -> grad_w = x = 2.0
    lin = nn.Linear(1, 1, bias=False)
    lin.weight.data.fill_(0.5)
    x = torch.tensor([[2.0]])
    y = lin(x)
    y.sum().backward()
    gn = grad_norm(lin)
    assert abs(gn - 2.0) < 1e-6, f"grad_norm = {gn}, expected 2.0"


def test_grad_norm_no_grads_returns_zero():
    # all params have None grad -> grad_norm should be 0.0 (not error)
    lin = nn.Linear(2, 2)
    gn = grad_norm(lin)
    assert gn == 0.0


def test_shape_prints_format(capsys):
    x = torch.zeros(2, 3)
    shape(x, "x")
    out = capsys.readouterr().out
    assert "x:" in out
    assert "(2, 3)" in out
    assert "dtype=" in out


def test_reset_no_op_on_empty(tmp_path):
    # Smoke: reset() against an empty runs/ dir should not raise.
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    reset(verbose=False, runs_dir=runs_dir)
