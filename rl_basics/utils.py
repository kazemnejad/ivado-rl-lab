"""Free-floating RL helpers — SPEC §5.3.

Pedagogical primitives shared by REINFORCE / A2C / PPO walkthroughs.
Kept as plain functions (not policy methods) to mirror the LLM-RL idiom
``log_probs(model_logits, target_token_ids)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from rl_basics.env import FourRoomsTL
    from rl_basics.models import MLPPolicy


def sample_action(logits: torch.Tensor) -> torch.Tensor:
    """logits: (..., n_actions). Returns (...,) Categorical sample (long)."""
    return torch.distributions.Categorical(logits=logits).sample()


def log_probs(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """logits: (..., n_actions). actions: (...,) long. Returns (...,) log p(a|s).

    Free-floating utility — NOT a policy method. Mirrors LLM RL idiom:
    log_probs(model_logits, target_token_ids).
    """
    return torch.distributions.Categorical(logits=logits).log_prob(actions)


def value_loss(
    v_pred: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """v_pred, returns: (B, T). mask: (B, T) bool. Returns scalar masked MSE."""
    m = mask.float()
    return ((v_pred - returns.detach()) ** 2 * m).sum() / m.sum().clamp(min=1)


def grad_norm(model: nn.Module) -> float:
    """Global L2 norm of param grads. Returns 0.0 if no params have a grad."""
    sq = torch.tensor(0.0)
    for p in model.parameters():
        if p.grad is not None:
            sq = sq + (p.grad.detach() ** 2).sum()
    return float(torch.sqrt(sq))


def shape(x: torch.Tensor, name: str = "") -> None:
    """Print '<name>: (16, 400) dtype=float32 min=-3.21 max=4.10'.

    Use liberally in walkthrough cells. Locked-in pedagogical primitive.
    """
    dt = str(x.dtype).removeprefix("torch.")
    print(
        f"{name}: {tuple(x.shape)} dtype={dt} "
        f"min={x.min().item():.3g} max={x.max().item():.3g}"
    )


@dataclass
class Trajectory:
    """Vectorized rollout buffer — SPEC §5.1. All tensors are (B, T).

    Fields:
      states:  (B, T) long  — state index at the START of step t.
      actions: (B, T) long  — action sampled at step t (the policy's choice;
                              env may slip-substitute internally, but we log
                              the policy's intent).
      rewards: (B, T) float — reward emitted by env.step at step t.
      mask:    (B, T) bool  — True where env was alive at the START of step t.

    NB: log_probs are deliberately NOT recorded. Recompute under the (current)
    policy in the loss step — same convention as LLM RL: redo a forward pass
    at backward time.
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    mask: torch.Tensor


def sample_initial_states(
    env: "FourRoomsTL",
    B: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Returns (B,) long. Uniform over env.valid_start_idx (the 64 TL cells).

    Mirrors the LLM Lab 2 ``sample_prompts(B)`` interface — distinct entry
    point so the training loop reads similarly across the two labs.
    """
    n = env.valid_start_idx.shape[0]
    if generator is None:
        sample = torch.randint(0, n, (B,), device=env.device)
    else:
        sample = torch.randint(0, n, (B,), generator=generator, device=env.device)
    return env.valid_start_idx[sample]


def rollout(
    env: "FourRoomsTL",
    policy: "MLPPolicy",
    init_states: torch.Tensor | None = None,
) -> Trajectory:
    """Sync-free vectorized rollout for ``env.max_steps`` timesteps.

    Done envs no-op internally (env.step masks them out). Returns a
    Trajectory dataclass with shapes (B, T) for all four fields.

    log_probs are NOT recorded — recompute under the current policy in the
    loss step (LLM-RL convention).
    """
    states = env.reset(init_states=init_states)
    B = env.B
    T = env.max_steps
    dev = env.device

    S_buf = torch.zeros((B, T), dtype=torch.long, device=dev)
    A_buf = torch.zeros((B, T), dtype=torch.long, device=dev)
    R_buf = torch.zeros((B, T), dtype=torch.float32, device=dev)
    M_buf = torch.zeros((B, T), dtype=torch.bool, device=dev)

    for t in range(T):
        was_alive = ~env.done  # alive at START of step t (SPEC semantics)
        with torch.no_grad():
            logits = policy(states)
            actions = sample_action(logits)
        next_states, rewards, _ = env.step(actions)
        S_buf[:, t] = states
        A_buf[:, t] = actions
        R_buf[:, t] = rewards
        M_buf[:, t] = was_alive
        states = next_states

    return Trajectory(states=S_buf, actions=A_buf, rewards=R_buf, mask=M_buf)


def _compute_returns_to_go_ref(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Private reference for ``compute_returns_to_go`` — same body as the
    student-facing impl in :mod:`rl_basics.student`.

    Used as an independent oracle in ``tests/test_student.py`` and the
    Task 12 student stress suite. Kept here so the test reference survives
    when the lab notebook stubs the bodies in ``student.py``.

    NOTE: kept bit-for-bit identical to ``student.py::compute_returns_to_go``.
    Update BOTH locations if the logic ever changes.
    """
    B, T = rewards.shape
    assert mask.shape == (B, T), f"mask shape mismatch: {mask.shape}"
    G = torch.zeros_like(rewards)
    running = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    for t in range(T - 1, -1, -1):
        running = rewards[:, t] + gamma * running
        G[:, t] = running
    return G


def _compute_advantage_vanilla_ref(
    returns: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Private reference for ``compute_advantage_vanilla`` — same body as the
    student-facing impl in :mod:`rl_basics.student`.

    A_t = G_t. (B, T) → (B, T). Mask is unused here (kept for signature
    parity with the value-baseline / batch-baseline siblings).

    NOTE: kept bit-for-bit identical to ``student.py::compute_advantage_vanilla``.
    Update BOTH locations if the logic ever changes.
    """
    return returns


def _compute_advantage_with_value_baseline_ref(
    returns: torch.Tensor,
    value_pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Private reference for ``compute_advantage_with_value_baseline`` — same
    body as the student-facing impl in :mod:`rl_basics.student`.

    A_t = G_t − V_φ(s_t). All (B, T). ``value_pred`` should be detached
    upstream (we don't backprop through value_net for the policy gradient).
    Mask is unused here (kept for signature parity with the vanilla /
    batch-baseline siblings).

    NOTE: kept bit-for-bit identical to
    ``student.py::compute_advantage_with_value_baseline``. Update BOTH
    locations if the logic ever changes.
    """
    return returns - value_pred


def _compute_advantage_with_batch_baseline_ref(
    returns: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Private reference for ``compute_advantage_with_batch_baseline`` — same
    body as the student-facing impl in :mod:`rl_basics.student`.

    A_t = G_t − μ_t  where μ_t = mean over the batch axis at timestep t,
    averaged ONLY over alive trajectories at time t (mask-aware). Shape
    (B, T) → (B, T).

    NOTE: kept bit-for-bit identical to
    ``student.py::compute_advantage_with_batch_baseline``. Update BOTH
    locations if the logic ever changes.
    """
    m = mask.float()
    denom = m.sum(dim=0).clamp(min=1.0)                 # (T,)
    mean_t = (returns * m).sum(dim=0) / denom           # (T,)
    return returns - mean_t.unsqueeze(0)                # (B, T)


def _policy_loss_ref(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Private reference for ``policy_loss`` — same body as the
    student-facing impl in :mod:`rl_basics.student`.

    REINFORCE policy-gradient loss:
        L = − (1/Σ mask) · Σ_{(b,t) alive} A_{b,t} · log π(a|s)
    All inputs (B, T); returns a scalar.

    NOTE: kept bit-for-bit identical to ``student.py::policy_loss``.
    Update BOTH locations if the logic ever changes.
    """
    m = mask.float()
    return -(advantages * log_probs * m).sum() / m.sum().clamp(min=1)


def reset(verbose: bool = True, runs_dir: "Path | None" = None) -> None:
    """Idempotent. Safe to run anytime — SPEC §8.

    Steps:
      1. Kill all PIDs in any ``runs/*/pids.json`` still alive (SIGTERM, then
         SIGKILL after ~2 s).
      2. Stop the Dash app subprocess (best-effort via
         :func:`rl_basics.dash_app.stop`).
      3. ``rmtree`` every child of ``runs/`` (preserving ``runs/.gitkeep``
         and the ``runs/`` dir itself).
      4. ``plt.close('all')`` to clear matplotlib's global figure cache.
      5. Print a one-line summary if ``verbose``.

    Parameters
    ----------
    verbose:
        If True, print a summary of what was killed/removed.
    runs_dir:
        Test-only additive parameter. Defaults to ``Path.cwd() / "runs"``
        — matching the SPEC §8 contract. Tests pass an isolated tmp dir to
        avoid touching the user's real runs/.
    """
    import json
    import os
    import shutil
    import signal
    import time
    from pathlib import Path

    if runs_dir is None:
        runs_dir = Path.cwd() / "runs"
    runs_dir = Path(runs_dir)

    # ---- 1. Kill PIDs from runs/*/pids.json --------------------------------
    killed: list[int] = []
    if runs_dir.exists():
        for pids_path in runs_dir.glob("*/pids.json"):
            try:
                pid_map = json.loads(pids_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for pid in pid_map.values():
                if pid is None:
                    continue
                try:
                    pid = int(pid)
                except (TypeError, ValueError):
                    continue
                # Best-effort SIGTERM.
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                except PermissionError:
                    # Not ours — skip.
                    continue
                killed.append(pid)

        # Wait up to ~2 s for graceful exit, then SIGKILL stragglers.
        deadline = time.monotonic() + 2.0
        stragglers = list(killed)
        while stragglers and time.monotonic() < deadline:
            stragglers = [p for p in stragglers if _pid_alive(p)]
            if stragglers:
                time.sleep(0.05)
        for pid in stragglers:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass

    # ---- 2. Stop Dash subprocess (best-effort) -----------------------------
    dash_stopped = False
    try:
        from rl_basics import dash_app

        dash_app.stop()
        dash_stopped = True
    except Exception as exc:  # pragma: no cover - defensive
        if verbose:
            print(f"[rl_basics.reset] dash_app.stop() failed: {exc!r}")

    # ---- 3. rmtree children of runs/ (preserve runs/ + .gitkeep) -----------
    removed_dirs = 0
    if runs_dir.exists():
        for child in runs_dir.iterdir():
            if child.name == ".gitkeep":
                continue
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                    removed_dirs += 1
                else:
                    child.unlink()
            except OSError as exc:  # pragma: no cover - defensive
                if verbose:
                    print(f"[rl_basics.reset] could not remove {child}: {exc!r}")

    # ---- 4. matplotlib figure cache ----------------------------------------
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:  # pragma: no cover - matplotlib optional at runtime
        pass

    # ---- 5. Summary --------------------------------------------------------
    if verbose:
        print(
            f"[rl_basics.reset] killed {len(killed)} pid(s); "
            f"dash_stopped={dash_stopped}; removed {removed_dirs} run dir(s)."
        )


def _pid_alive(pid: int) -> bool:
    """True iff ``pid`` is still in the process table (best-effort)."""
    import os

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
