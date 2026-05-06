"""Bootstrap policy-gradient variance measurement — SPEC §C / §7.5.

Public API:

    measure_grad_variance(env_class, env_kwargs, batch_size, device,
                          policy, value_net, advantage_kind, gamma, n_boot)
        -> dict with keys
            n_boot, mean_norm_sq, mean_g_norm_sq, tr_var, snr.

Computes ``g_hat = ∇_θ policy_loss`` over ``n_boot`` independent rollouts at
FROZEN parameters and bootstraps the trace-variance:

    tr_var          = E[||g_hat||²] − ||E[g_hat]||²   = Σ_k Var[g_hat[k]]
    snr             = ||E[g_hat]||² / tr_var

This is precisely the Var[g_hat] (in the trace sense) that the baseline is
supposed to reduce. No proxies. The function is BASELINE-AWARE: it computes
the same advantage that the trainer would use given the requested
``advantage_kind`` and the (frozen) value_net.

Transcribed (and adapted) from
``$PROTO/baseline-search-fourrooms/grad_variance.py`` (5/5 verified).

CPU-only — accepts the ``device`` parameter for API parity but uses default
torch device throughout.
"""

from __future__ import annotations

import torch

from rl_basics.utils import (
    _compute_advantage_vanilla_ref,
    _compute_advantage_with_batch_baseline_ref,
    _compute_advantage_with_value_baseline_ref,
    _compute_returns_to_go_ref,
    _policy_loss_ref,
    log_probs as log_probs_fn,
    rollout,
)


def _compute_advantage(
    traj,
    value_net,
    advantage_kind: str,
    gamma: float,
):
    """Compute (advantage, mask) for the requested baseline.

    advantage_kind ∈ {'vanilla', 'value', 'batch'}.
    Mirrors the public ``compute_advantage_*`` API in :mod:`rl_basics.student`.
    """
    G = _compute_returns_to_go_ref(traj.rewards, traj.mask, gamma=gamma)
    mask = traj.mask

    if advantage_kind == "vanilla":
        adv = _compute_advantage_vanilla_ref(G, mask)
    elif advantage_kind == "batch":
        adv = _compute_advantage_with_batch_baseline_ref(G, mask)
    elif advantage_kind == "value":
        if value_net is None:
            raise ValueError("advantage_kind='value' requires a value_net")
        with torch.no_grad():
            V_pred = value_net(traj.states.flatten()).view_as(traj.states)
        adv = _compute_advantage_with_value_baseline_ref(G, V_pred, mask)
    else:
        raise ValueError(f"unknown advantage_kind: {advantage_kind!r}")
    return adv, mask


def _grad_one_batch(
    env,
    policy,
    value_net,
    advantage_kind: str,
    gamma: float,
) -> torch.Tensor:
    """One independent rollout → flattened policy-gradient vector.

    Caller owns ``env`` (which carries the RNG state) — we just roll it out
    once, build the loss, backward, and concat ``p.grad`` for each policy
    param. Policy params are READ ONLY (no optimizer step).

    Returns a 1-D tensor of shape (n_params,).
    """
    # zero existing grads on policy params (only)
    for p in policy.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    traj = rollout(env, policy)
    adv, mask = _compute_advantage(traj, value_net, advantage_kind, gamma)

    # Recompute log_probs under the current policy (SPEC §5.1: not stored).
    logits = policy(traj.states.flatten())
    lp = log_probs_fn(logits, traj.actions.flatten()).view_as(traj.states)

    # Detach advantage so we don't backprop through value_net here.
    ploss = _policy_loss_ref(lp, adv.detach(), mask)
    ploss.backward()

    g = torch.cat([p.grad.detach().flatten() for p in policy.parameters()
                   if p.requires_grad and p.grad is not None])
    assert g.numel() > 0, "no gradient computed"
    return g


def _measure_grad_variance_raw(
    env_class,
    env_kwargs: dict,
    batch_size: int,
    device: str,
    policy,
    value_net,
    advantage_kind: str,
    gamma: float,
    n_boot: int,
) -> torch.Tensor:
    """Run ``n_boot`` independent rollouts, return the raw (n_boot, P) grad
    matrix. Each rollout uses a FRESH env (fresh RNG seeded from the global
    torch RNG inside ``FourRoomsTL.__init__``), so successive calls are
    statistically independent.

    Helper for tests that need the raw bootstrap samples (split-half,
    independence). ``measure_grad_variance`` is a thin summarizer.

    Restores ``policy.training`` / ``value_net.training`` after the run.
    """
    was_train = policy.training
    policy.eval()
    was_v_train = None
    if value_net is not None:
        was_v_train = value_net.training
        value_net.eval()

    grads = []
    for _ in range(n_boot):
        env = env_class(batch_size, device=device, **env_kwargs)
        g = _grad_one_batch(env, policy, value_net, advantage_kind, gamma)
        grads.append(g)

    # leave policy / value_net grads as None (matches set_to_none=True)
    for p in policy.parameters():
        p.grad = None
    if value_net is not None:
        for p in value_net.parameters():
            p.grad = None

    if was_train:
        policy.train()
    if value_net is not None and was_v_train:
        value_net.train()

    return torch.stack(grads, dim=0)  # (n_boot, P)


def measure_grad_variance(
    env_class,
    env_kwargs: dict,
    batch_size: int,
    device: str,
    policy,
    value_net,
    advantage_kind: str,
    gamma: float,
    n_boot: int,
) -> dict:
    """Bootstrap the policy-gradient estimator's trace-variance.

    See module docstring for definitions. Returns a dict with keys::

        n_boot          number of bootstrap samples
        mean_norm_sq    E[||g_hat||²]
        mean_g_norm_sq  ||E[g_hat]||²
        tr_var          E[||g_hat||²] − ||E[g_hat]||²
        snr             ||E[g_hat]||² / tr_var (or +inf if tr_var == 0)

    Freezes ``policy.eval()`` (and ``value_net.eval()`` if present) for the
    duration; restores prior training mode on exit. Does not step any
    optimizer.
    """
    grads = _measure_grad_variance_raw(
        env_class, env_kwargs, batch_size, device, policy, value_net,
        advantage_kind, gamma, n_boot,
    )
    mean_norm_sq = float((grads * grads).sum(dim=1).mean().item())
    mean_g = grads.mean(dim=0)
    mean_g_norm_sq = float((mean_g * mean_g).sum().item())
    tr_var = max(0.0, mean_norm_sq - mean_g_norm_sq)
    snr = mean_g_norm_sq / tr_var if tr_var > 0 else float("inf")
    return {
        "n_boot": n_boot,
        "mean_norm_sq": mean_norm_sq,
        "mean_g_norm_sq": mean_g_norm_sq,
        "tr_var": tr_var,
        "snr": snr,
    }
