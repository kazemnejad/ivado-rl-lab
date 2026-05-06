"""Task 5 tests — Trajectory + rollout + sample_initial_states.

Per SPEC §5.1:
  * Trajectory has exactly {states, actions, rewards, mask} — NO log_probs.
  * mask[:, t] = ~env.done at the START of step t.
  * sample_initial_states draws uniformly from env.valid_start_idx.
"""

import torch

from rl_basics.env import FourRoomsTL
from rl_basics.models import MLPPolicy
from rl_basics.utils import Trajectory, rollout, sample_initial_states


def test_rollout_shapes():
    env = FourRoomsTL(batch_size=4, seed=0)
    policy = MLPPolicy()
    traj = rollout(env, policy)
    assert isinstance(traj, Trajectory)
    T = env.max_steps
    assert traj.states.shape == (4, T) and traj.states.dtype == torch.long
    assert traj.actions.shape == (4, T) and traj.actions.dtype == torch.long
    assert traj.rewards.shape == (4, T) and traj.rewards.dtype == torch.float32
    assert traj.mask.shape == (4, T) and traj.mask.dtype == torch.bool


def test_rollout_mask_monotone_falls_to_done():
    # Force agents to start ONE step from the goal so done fires within
    # ~1 step (with slip, ~70% of agents reach goal on step 0). This
    # guarantees the False branch is actually exercised, otherwise the
    # monotone property is never observed (a random policy from the TL
    # room rarely reaches goal within 50 steps).
    B = 16
    env = FourRoomsTL(batch_size=B, seed=0)
    env.max_steps = 50
    policy = MLPPolicy()
    init = torch.full((B,), 15 * 17 + 14, dtype=torch.long)  # (15, 14): 1 step E from goal
    traj = rollout(env, policy, init_states=init)

    # Sanity: with B=16 agents start one E step from the goal, at least one
    # element should hit done within 50 steps (slip-only failure is ~30%
    # per step, so within 50 steps every agent reaches done w.h.p.).
    assert (~traj.mask).any(), "no agents went done; test cannot exercise monotone branch"

    # Real check: for every batch element that has a False, all entries
    # after the first False must also be False.
    flipped_count = 0
    for b in range(B):
        m = traj.mask[b]
        first_false = (~m).nonzero(as_tuple=True)[0]
        if first_false.numel() > 0:
            flipped_count += 1
            t0 = first_false[0].item()
            assert not m[t0:].any(), (
                f"mask flips True->False->True at b={b}, t0={t0}"
            )
    assert flipped_count > 0, "no monotone-flip branch exercised in any element"


def test_rollout_no_log_probs_recorded():
    env = FourRoomsTL(batch_size=2, seed=0)
    env.max_steps = 5
    policy = MLPPolicy()
    traj = rollout(env, policy)
    field_names = {f.name for f in traj.__dataclass_fields__.values()}
    assert "log_probs" not in field_names, (
        f"log_probs leaked into Trajectory: {field_names}"
    )
    assert field_names == {"states", "actions", "rewards", "mask"}, (
        f"unexpected fields: {field_names}"
    )


def test_sample_initial_states_within_TL():
    env = FourRoomsTL(batch_size=8, seed=0)
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    s1 = sample_initial_states(env, B=64, generator=g)
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(99)
    s2 = sample_initial_states(env, B=64, generator=g2)
    valid = set(env.valid_start_idx.tolist())
    assert all(int(s) in valid for s in s1.tolist())
    assert all(int(s) in valid for s in s2.tolist())
    assert s1.tolist() != s2.tolist(), (
        "two different generators produced identical samples"
    )
