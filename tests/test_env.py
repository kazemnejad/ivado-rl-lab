"""Unit tests for FourRoomsTL (basic correctness).

Mirrors prototypes/baseline-search-fourrooms/verify_env_tl.py.
Stress tests (per-element independence over many fresh seeds, etc.) live
in tests/stress/ and are run via the `stress` pytest marker.
"""
import pytest
import torch

from rl_basics.env import FourRoomsTL


def test_reset_shape_and_dtype():
    env = FourRoomsTL(batch_size=8, seed=0)
    s = env.reset()
    assert s.shape == (8,)
    assert s.dtype == torch.long


def test_reset_starts_in_TL_room():
    env = FourRoomsTL(batch_size=128, seed=42)
    s = env.reset()
    rows = s // 17
    cols = s % 17
    assert (rows < 8).all(), f"row >=8 found: {rows[rows >= 8].tolist()}"
    assert (cols < 8).all(), f"col >=8 found: {cols[cols >= 8].tolist()}"
    # goal (15,15) is not reachable as a start by construction
    assert not ((rows == env.goal_y) & (cols == env.goal_x)).any()


def test_per_element_independence():
    # 128 samples drawn from 64 valid TL cells. Birthday-style upper bound is
    # ~58 distinct on average. Threshold 45 mirrors prototype verify_env_tl.py
    # (note: task description's "70" is a typo -- only 64 cells exist).
    env = FourRoomsTL(batch_size=128, seed=42)
    s = env.reset()
    distinct = len(set(s.tolist()))
    assert distinct >= 45, f"only {distinct} distinct starts out of 128 (need >=45)"


def test_step_shapes():
    env = FourRoomsTL(batch_size=8, seed=0)
    env.reset()
    actions = torch.zeros(8, dtype=torch.long)
    states, rewards, done = env.step(actions)
    assert states.shape == (8,) and states.dtype == torch.long
    assert rewards.shape == (8,) and rewards.dtype == torch.float32
    assert done.shape == (8,) and done.dtype == torch.bool


def test_goal_terminates():
    env = FourRoomsTL(batch_size=1, seed=0)
    init_s = torch.tensor([15 * 17 + 14], dtype=torch.long)  # one E step from goal
    env.reset(init_states=init_s)
    env.slip_prob = 0.0  # deterministic
    action = torch.tensor([2], dtype=torch.long)  # E
    _, reward, done = env.step(action)
    assert reward.item() == pytest.approx(1.0), f"reward={reward.item()}"
    assert bool(done.item()) is True
    # alive_mask follows ~done
    assert bool(env.alive_mask.item()) is False


def test_walls_block_per_element_movement():
    # Appendix A test 10: stepping into a wall must keep the agent in place.
    # Place 4 agents adjacent to walls of the '+' divider and step toward them.
    # All 4 should be blocked (positions unchanged).
    env = FourRoomsTL(batch_size=4, seed=0)
    env.slip_prob = 0.0  # deterministic, isolate wall-blocking from slip
    # Cells just NW of the cross intersection: (7,7), (7,5), (5,7), (3,7).
    # Adjacent walls along row 8 / col 8 (excluding doorways):
    #   (7, 7) action S -> (8, 7) is a wall (col 8? no, wait: row 8 is wall except doorways)
    # Use 4 cells with unambiguous wall neighbors:
    init_cells = torch.tensor(
        [
            7 * 17 + 7,  # (7, 7) — S into (8, 7) wall (row 8 is wall, col 7 ≠ doorway 4)
            7 * 17 + 5,  # (7, 5) — S into (8, 5) wall
            7 * 17 + 0,  # (7, 0) — S into (8, 0) wall
            0 * 17 + 7,  # (0, 7) — E into (0, 8) wall (col 8 is wall, row 0 ≠ doorway 4)
        ],
        dtype=torch.long,
    )
    env.reset(init_states=init_cells)
    actions = torch.tensor([1, 1, 1, 2], dtype=torch.long)  # S, S, S, E
    states, _, _ = env.step(actions)
    # All 4 should still be at their start cells (movement blocked by wall).
    assert torch.equal(states, init_cells), (
        f"expected positions unchanged {init_cells.tolist()}, got {states.tolist()}"
    )
