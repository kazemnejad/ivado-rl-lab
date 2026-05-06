"""Stress suite for FourRoomsTL.

Mirrors SPEC §7.5 + Appendix A. These tests run thousands of resets / steps
and statistical checks; they are slow (>30s in aggregate) and gated behind
the `stress` pytest marker so the default unit run skips them.

TDD note: this is a *verification* suite for an existing, correct env. The
classical RED phase doesn't apply -- if the impl is right, the tests pass on
first run. The "RED" guarantee is regression: if any of these starts to fail,
the env has drifted from spec (e.g. shared slip noise across the batch,
correlated start sampling, walls blocking BFS unexpectedly).

Run via:
    .venv/bin/python -m pytest tests/stress/test_env_stress.py -v -m stress
"""
from __future__ import annotations

from collections import deque

import pytest
import torch

from rl_basics.env import FourRoomsTL


# ---------------------------------------------------------------------------
# Test 1: per-element independence over 1000 resets (Pearson r <= 0.10)
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_per_element_independence_1000_resets():
    """B=128 x 1000 resets: any two batch positions must be uncorrelated.

    A regression where slip noise / start sampling shares state across the
    batch (Appendix H.2) would manifest as off-diagonal |r| -> 1.

    Threshold note: the plan suggested 0.10, but with R=1000 samples per
    column the per-pair null std of r is ~1/sqrt(R-1) ~ 0.0316; over
    128*127/2 = 8128 pairs the empirical max |r| sits at ~0.12-0.14 across
    seeds (extreme-value of |N(0, 0.0316)| over 8128 draws ~ 0.13). We use
    0.20 -- safely above the noise floor, far below the regression signature
    of ~1.0.
    """
    B = 128
    R = 1000
    env = FourRoomsTL(batch_size=B, seed=0)
    samples = torch.empty((R, B), dtype=torch.long)
    for r in range(R):
        samples[r] = env.reset()

    corr = torch.corrcoef(samples.T.float())  # (B, B)
    off_diag = corr - torch.eye(B, dtype=corr.dtype)
    max_abs = off_diag.abs().max().item()
    print(f"\n[test1] max |off-diagonal correlation| = {max_abs:.4f}")
    assert max_abs <= 0.20, f"max |off-diag r| = {max_abs:.4f} > 0.20"


# ---------------------------------------------------------------------------
# Test 2: uniform coverage chi-square over 64 TL cells
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_uniform_coverage_chi2():
    """200 resets x B=128 = 25 600 samples must look uniform over 64 cells.

    z-score of chi-square w.r.t. chi^2(df=63) approximate normal:
        z = (chi2 - df) / sqrt(2 df)
    must satisfy |z| < 4.
    """
    B = 128
    R = 200
    env = FourRoomsTL(batch_size=B, seed=1)
    counts = torch.zeros(env.n_states, dtype=torch.long)
    for _ in range(R):
        s = env.reset()
        counts.scatter_add_(0, s, torch.ones_like(s))

    valid = env.valid_start_idx
    obs = counts[valid].double()  # (64,)
    n = obs.sum().item()
    k = valid.shape[0]  # number of bins (= 64)
    dof = k - 1  # multinomial chi^2 dof = bins - 1 = 63
    exp = n / k
    chi2 = ((obs - exp) ** 2 / exp).sum().item()
    z = (chi2 - dof) / (2 * dof) ** 0.5
    print(f"\n[test2] n={int(n)} chi2={chi2:.2f} z={z:.3f} (dof={dof})")
    assert abs(z) < 4, f"chi2 z-score = {z:.3f} (|z| must be < 4)"


# ---------------------------------------------------------------------------
# Test 3: fresh-instance decoupling (Appendix H.1)
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_fresh_instance_decoupling():
    """Two FourRoomsTL(seed=None) must NOT share start states.

    Without the H.1 fix (reseed from global RNG when seed is None), every
    fresh instance had identical RNG state and gave 64/64 shared starts.

    Threshold note: with B=64 samples drawn with replacement from 64 cells,
    each instance covers ~40 distinct cells; two independent draws share
    ~40*40/64 ~ 25 cells in expectation (empirically 18-34 over 50 trials).
    The regression signature is 64/64 -- we use 50 as a generous ceiling
    that still catches it.
    """
    # Pin global RNG so this test is order-independent: every fresh
    # FourRoomsTL(seed=None) self-seeds via torch.randint() against the global
    # RNG, so without this pin the per-instance seeds depend on whatever ran
    # before us.
    torch.manual_seed(20260504)
    B = 64
    overlaps = []
    for _ in range(5):
        e1 = FourRoomsTL(batch_size=B, seed=None)
        e2 = FourRoomsTL(batch_size=B, seed=None)
        s1 = set(e1.reset().tolist())
        s2 = set(e2.reset().tolist())
        overlaps.append(len(s1 & s2))
    print(f"\n[test3] overlaps={overlaps} (B={B})")
    # Per-trial: well below the 64/64 regression signature.
    for ov in overlaps:
        assert ov <= 50, f"per-trial overlap {ov}/{B} suspicious (regression -> 64)"
    # Mean across trials must stay below the regression floor.
    mean_ov = sum(overlaps) / len(overlaps)
    assert mean_ov < 45, f"mean overlap {mean_ov:.1f} too high (regression?)"


# ---------------------------------------------------------------------------
# Test 4: per-element slip via env.step (exercises env.py:115)
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_slip_noise_per_element_via_env_step():
    """Slip RNG inside env.step must draw shape (B,) per element (Appendix H.2).

    The regression we guard against is changing
        slip = torch.rand(B, generator=self.gen, device=dev) < self.slip_prob
    to
        slip = torch.rand(1, generator=self.gen, device=dev).expand(B) < ...
    or any other shared-across-batch RNG, which would force every batch
    element to slip (or not) identically.

    Strategy: force every batch element to the same start cell (5, 5) and
    issue the same action (N) over T trials. With per-element slip, the
    final cell distribution across the batch spreads over multiple cells
    every trial. With shared slip (regression), every batch element ends
    up at the SAME cell every trial (since they share start, action, and
    slip outcome). We assert that a strong majority of trials show >1
    distinct outcome cell across the batch.

    Quantitative check (per-element mode):
      Pr[1-step from (5,5) ends at (4,5)] = 0.6 + 0.4/4 = 0.7
      Pr[ends at S/E/W neighbour]         = 0.4/4    = 0.1 each
      For B=64 independent draws, Pr[all 64 same cell]
        ~= 0.7**64 + 3 * 0.1**64 ~= 1e-10  (negligible)
      So under per-element slip ~all T trials show >=2 distinct cells.
    """
    B = 64
    T = 200
    env = FourRoomsTL(batch_size=B, seed=42)
    init = torch.full((B,), 5 * 17 + 5, dtype=torch.long)
    intended = torch.zeros(B, dtype=torch.long)  # action N (=0)

    diverse_trials = 0
    distinct_counts: list[int] = []
    for _ in range(T):
        env.reset(init_states=init)
        env.step(intended)
        unique = env.state_to_index().unique()
        distinct_counts.append(int(unique.numel()))
        if unique.numel() > 1:
            diverse_trials += 1

    avg_distinct = sum(distinct_counts) / T
    print(
        f"\n[test4] env.step slip per-element: {diverse_trials}/{T} trials "
        f"had >1 distinct outcome (avg distinct cells/trial = {avg_distinct:.2f})"
    )
    # Per-element: expect ~all T trials. Allow ~5 for absurd edge cases.
    # Shared-slip regression: 0 trials would have >1 distinct outcome.
    assert diverse_trials >= T - 5, (
        f"only {diverse_trials}/{T} trials showed per-element slip variance; "
        "regression to shared-slip RNG?"
    )
    # Per-element also implies on average ~3 distinct cells (since slip outcomes
    # spread over {N,S,E,W} non-N branches at rate 0.1 each, with B=64 we expect
    # to see all 4 outcomes in most trials).
    assert avg_distinct >= 2.5, (
        f"avg distinct cells per trial = {avg_distinct:.2f}; expected >= 2.5"
    )


# ---------------------------------------------------------------------------
# Test 5: BFS reachability from every TL cell to the goal
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_goal_reachability_BFS():
    """Every TL non-wall cell must reach (15,15) with distance in [19,30]."""
    env = FourRoomsTL(batch_size=1, seed=0)
    walls = env.walls.cpu().numpy()
    size = env.size
    gy, gx = env.goal_y, env.goal_x

    def bfs(sy: int, sx: int) -> int:
        if walls[sy, sx]:
            return -1
        visited = {(sy, sx)}
        q = deque([(sy, sx, 0)])
        while q:
            y, x, d = q.popleft()
            if y == gy and x == gx:
                return d
            for dy, dx in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < size
                    and 0 <= nx < size
                    and not walls[ny, nx]
                    and (ny, nx) not in visited
                ):
                    visited.add((ny, nx))
                    q.append((ny, nx, d + 1))
        return -1

    distances: list[int] = []
    for cell in env.valid_start_idx.tolist():
        sy, sx = cell // size, cell % size
        d = bfs(sy, sx)
        assert d > 0, (
            f"cell ({sy},{sx}) unreachable from goal "
            f"(BFS returned {d}; expected positive distance)"
        )
        distances.append(d)

    dmin, dmax = min(distances), max(distances)
    print(f"\n[test5] BFS distances over {len(distances)} TL cells: min={dmin} max={dmax}")
    assert len(distances) == 64, f"expected 64 TL cells, got {len(distances)}"
    assert dmin >= 19, f"min BFS distance {dmin} < 19 (spec says [19,30])"
    assert dmax <= 30, f"max BFS distance {dmax} > 30 (spec says [19,30])"
