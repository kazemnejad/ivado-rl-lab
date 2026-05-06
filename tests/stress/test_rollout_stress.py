"""Stress suite for ``rollout`` (Task 6).

Three regression-guard tests gated behind ``@pytest.mark.stress`` so the
default ``pytest tests/`` run skips them.

1. ``test_b64_single_vs_4xb16_same_stats`` — per-element vectorization is
   intact: aggregate return distributions agree whether we run B=64 in one
   shot or B=16 four times. KS-test if scipy is available, else 2-sigma +
   std-ratio fallback.
2. ``test_no_NaN_anywhere`` — 50 rollouts, no NaN/Inf in states/actions/
   rewards, mask not entirely False.
3. ``test_mask_aligns_with_done`` — monkey-patches ``env.step`` to record
   ground-truth ``done`` after each step, then cross-checks ``traj.mask``
   exactly: ``mask[:, t] == ~done_history[t-1]`` for ``t > 0``.

Run via:
    .venv/bin/python -m pytest tests/stress/test_rollout_stress.py -v -m stress
"""

from __future__ import annotations

import statistics

import pytest
import torch

from rl_basics.env import FourRoomsTL
from rl_basics.models import MLPPolicy
from rl_basics.utils import rollout


def _make_policy() -> MLPPolicy:
    """Fresh policy with deterministic random init."""
    torch.manual_seed(0)
    return MLPPolicy()


# ---------------------------------------------------------------------------
# Test 1: B=64 single batch vs 4 x B=16 batches — same statistics
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_b64_single_vs_4xb16_same_stats():
    """Per-element rollouts: aggregate stats invariant to batch slicing.

    A regression that shares state across the batch (e.g. shared slip RNG
    inside the policy or env hot-path) would make a single B=64 rollout
    differ statistically from 4 independent B=16 rollouts.

    Recipe:
      * env.max_steps = 50 with goal-adjacent starts gives meaningful
        return variation (some hit the goal, some time out, some wander).
        Random TL starts would all uniformly time out (BFS distance >= 19,
        random policy is hopeless), collapsing the distribution to a
        delta at -2.5 and making the stats check vacuous.
      * Same fresh policy (manual_seed(0)) for every rollout.
      * Different env seeds across the four B=16 slices.
      * KS test (scipy) if available, else 2-sigma + std-ratio fallback.
    """
    M = 50  # max_steps
    # Goal-adjacent: (15, 14) is one step W of the goal (15, 15).
    INIT_CELL = 15 * 17 + 14

    # --- single B=64 ---
    env_full = FourRoomsTL(batch_size=64, seed=42)
    env_full.max_steps = M
    pol_full = _make_policy()
    init_full = torch.full((64,), INIT_CELL, dtype=torch.long)
    traj_full = rollout(env_full, pol_full, init_states=init_full)
    totals_full = traj_full.rewards.sum(dim=1).tolist()

    # --- 4 x B=16 ---
    totals_split: list[float] = []
    init_split = torch.full((16,), INIT_CELL, dtype=torch.long)
    for s in (1, 2, 3, 4):
        e = FourRoomsTL(batch_size=16, seed=42 + s)
        e.max_steps = M
        p = _make_policy()
        tr = rollout(e, p, init_states=init_split)
        totals_split.extend(tr.rewards.sum(dim=1).tolist())

    assert len(totals_full) == 64, f"expected 64 totals, got {len(totals_full)}"
    assert len(totals_split) == 64, f"expected 64 totals, got {len(totals_split)}"

    try:
        from scipy.stats import ks_2samp  # type: ignore[import-not-found]

        result = ks_2samp(totals_full, totals_split)
        # scipy returns a named-tuple KstestResult(statistic, pvalue); cast
        # explicitly to float to keep type-checkers happy.
        stat = float(result.statistic)  # type: ignore[attr-defined]
        p = float(result.pvalue)  # type: ignore[attr-defined]
        print(
            f"\n[test1] scipy KS: stat={stat:.4f} p={p:.4f} "
            f"(mean_full={statistics.mean(totals_full):.2f} "
            f"mean_split={statistics.mean(totals_split):.2f})"
        )
        assert p > 0.05, f"KS p={p:.4f} (must be > 0.05)"
    except ImportError:
        m1 = statistics.mean(totals_full)
        m2 = statistics.mean(totals_split)
        s1 = statistics.stdev(totals_full)
        s2 = statistics.stdev(totals_split)
        # pooled SE for difference of means under independent samples
        # (n1 = n2 = 64), so SE = sqrt((s1^2 + s2^2) / 64).
        pooled_se = ((s1**2 + s2**2) / 64) ** 0.5
        ratio = s1 / s2 if s2 > 0 else float("inf")
        # 3-sigma keeps the false-positive rate ~0.3% (vs ~5% at 2-sigma) while
        # still catching the regression where one batch collapses to delta.
        print(
            f"\n[test1] fallback (no scipy): m_full={m1:.3f} m_split={m2:.3f} "
            f"|dm|={abs(m1 - m2):.3f} 3*SE={3 * pooled_se:.3f} "
            f"s_full={s1:.3f} s_split={s2:.3f} ratio={ratio:.3f}"
        )
        assert abs(m1 - m2) < 3 * pooled_se, (
            f"means differ beyond 3-sigma: {m1:.3f} vs {m2:.3f} "
            f"(3*SE = {3 * pooled_se:.3f})"
        )
        assert 0.5 <= ratio <= 2.0, (
            f"std ratio {ratio:.3f} outside [0.5, 2.0]: "
            f"s_full={s1:.3f} s_split={s2:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 2: No NaN / Inf over 50 rollouts
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_no_NaN_anywhere():
    """50 rollouts (B=32, max_steps=50): no NaN/Inf, mask not all-False."""
    torch.manual_seed(20260504)
    B = 32
    M = 50
    N = 50
    for i in range(N):
        env = FourRoomsTL(batch_size=B, seed=1000 + i)
        env.max_steps = M
        policy = MLPPolicy()
        traj = rollout(env, policy)

        # NaN / Inf are float-only concepts. For long/bool tensors, range
        # checks express the intent more precisely than isfinite.
        assert traj.states.dtype == torch.long
        assert traj.actions.dtype == torch.long
        assert traj.rewards.dtype == torch.float32
        assert traj.mask.dtype == torch.bool

        # long-tensor sanity: no negative or absurdly large values.
        assert (traj.states >= 0).all(), f"negative state index at rollout {i}"
        assert (traj.states < env.n_states).all(), f"oversized state at rollout {i}"
        assert (traj.actions >= 0).all() and (traj.actions < env.n_actions).all(), (
            f"out-of-range action at rollout {i}"
        )

        # rewards are floats: explicit NaN/Inf guard.
        assert torch.isfinite(traj.rewards).all(), (
            f"non-finite reward at rollout {i}"
        )

        # mask: bool — finite-check is meaningless. Just guard against
        # an all-False mask, which would mean the rollout never recorded
        # an alive step (impossible if reset() worked).
        assert traj.mask.any(), f"mask is all-False at rollout {i}"


# ---------------------------------------------------------------------------
# Test 3: mask aligns with env.step's done sequence
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_mask_aligns_with_done():
    """``mask[:, t] == ~done_history[t-1]`` exactly for every t > 0.

    Strategy: monkey-patch env.step to record the ``done`` tensor returned
    at the END of each step. After the rollout finishes, cross-check
    ``traj.mask`` against the recorded sequence:
        mask[:, 0] = True   (just reset; everyone alive at start of step 0)
        mask[:, t] = ~done_history[t - 1]   (alive at start of step t iff
                                             not done at end of step t-1)
    """
    B = 8
    M = 30
    env = FourRoomsTL(batch_size=B, seed=0)
    env.max_steps = M
    policy = MLPPolicy()
    # 1 step W of goal — most agents reach done within a few steps so
    # the cross-check exercises both branches (alive and done).
    init = torch.full((B,), 15 * 17 + 14, dtype=torch.long)

    done_history: list[torch.Tensor] = []
    orig_step = env.step

    def step_wrapper(actions: torch.Tensor):
        s, r, d = orig_step(actions)
        done_history.append(d.clone())
        return s, r, d

    env.step = step_wrapper  # type: ignore[method-assign]

    traj = rollout(env, policy, init_states=init)

    assert len(done_history) == M, (
        f"expected {M} step calls, got {len(done_history)}"
    )

    # mask[:, 0]: alive at start of step 0 — everyone (just reset).
    assert traj.mask[:, 0].all(), (
        f"mask[:, 0] should be all-True post-reset; got {traj.mask[:, 0].tolist()}"
    )

    for t in range(1, M):
        expected = ~done_history[t - 1]
        actual = traj.mask[:, t]
        assert torch.equal(actual, expected), (
            f"mask[:, {t}] mismatch:\n"
            f"  got      = {actual.tolist()}\n"
            f"  expected = {expected.tolist()}\n"
            f"  done[t-1]= {done_history[t - 1].tolist()}"
        )

    # Also confirm the test actually exercised the False branch — i.e.
    # at least one element flipped to done before timeout. Otherwise the
    # cross-check is vacuous.
    assert (~traj.mask).any(), (
        "no element ever transitioned to done; cross-check vacuous"
    )
