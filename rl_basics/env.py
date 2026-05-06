"""FourRoomsTL: 17x17 four-rooms gridworld, fully vectorized, CPU only.

Single-class flattened version of the prototype's
VectorizedFourRoomsTLStart (env-comparison/Hard -> closure -> TLStart chain),
with all closure machinery stripped (closure_prob=0 is dead code in this lab).

Key properties (SPEC §5.1):
  * 17x17 grid; '+'-shaped wall divider (row 8, col 8) with 4 doorways at
    (8,4), (8,13), (4,8), (13,8).
  * Goal at (15, 15) -> bottom-right room.
  * Per-episode random start, sampled INDEPENDENTLY per batch element from
    the 64 cells of the top-left room (rows 0..7, cols 0..7). Per-element
    independence was a major bug source -- see Appendix H.
  * Slip noise drawn per-element via torch.rand(B, generator=self.gen).
  * Action map: 0=N, 1=S, 2=E, 3=W, matching VectorizedFourRoomsHard._DY/_DX.

Reset accepts an optional `init_states: (B,) long Tensor | None` to force
specific starting cells (used in tests, value-iteration probing, etc.).
"""
from __future__ import annotations

import torch


class FourRoomsTL:
    """17x17 four-rooms gridworld with TL random start. Fully vectorized over
    the batch. CPU only.
    """

    # --- class-level constants (SPEC §5.1) -----------------------------------
    size: int = 17
    n_states: int = 17 * 17  # 289
    n_actions: int = 4  # N=0, S=1, E=2, W=3
    max_steps: int = 400
    step_penalty: float = -0.05
    goal_reward: float = 1.0
    slip_prob: float = 0.4
    goal_y: int = 15
    goal_x: int = 15

    # action deltas (N, S, E, W). Match VectorizedFourRoomsHard._DY/_DX.
    _DY = torch.tensor([-1, 1, 0, 0], dtype=torch.long)
    _DX = torch.tensor([0, 0, 1, -1], dtype=torch.long)

    def __init__(self, batch_size: int, device: str = "cpu", seed: int | None = None):
        self.B = int(batch_size)
        self.device = device

        # --- walls: '+'-divider with 4 doorways -------------------------------
        size = self.size
        walls = torch.zeros((size, size), dtype=torch.bool, device=device)
        mid = size // 2  # = 8
        walls[mid, :] = True  # horizontal divider
        walls[:, mid] = True  # vertical divider
        walls[mid, mid // 2] = False  # (8, 4)
        walls[mid, mid + mid // 2 + 1] = False  # (8, 13)
        walls[mid // 2, mid] = False  # (4, 8)
        walls[mid + mid // 2 + 1, mid] = False  # (13, 8)
        self.walls = walls

        # action delta vectors on the right device
        self._dy = self._DY.to(device)
        self._dx = self._DX.to(device)

        # --- RNG ---------------------------------------------------------------
        # Appendix H.1 fix: torch.Generator() with no manual_seed is deterministic
        # across instances, so without this every fresh env had the same RNG state.
        # Reseed from the global torch RNG when no explicit seed is given.
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(int(seed))
        else:
            self.gen.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))

        # --- valid TL-room start cells: rows 0..mid-1, cols 0..mid-1, no walls --
        mask = torch.zeros((size, size), dtype=torch.bool, device=device)
        mask[:mid, :mid] = True
        mask &= ~self.walls  # TL has no walls, but be explicit
        self.valid_start_idx = mask.flatten().nonzero(as_tuple=True)[0]

        # initialize state buffers
        self.reset()

    # --- core API ------------------------------------------------------------
    def reset(self, init_states: torch.Tensor | None = None) -> torch.Tensor:
        # init_states: optional (B,) long tensor of state indices to force.
        # returns: (B,) long state indices.
        B, dev = self.B, self.device
        if init_states is not None:
            assert init_states.shape == (B,), (
                f"init_states must be (B,)={B}, got {tuple(init_states.shape)}"
            )
            init_long = init_states.to(device=dev, dtype=torch.long)
            self.pos_y = init_long // self.size
            self.pos_x = init_long % self.size
        else:
            n = self.valid_start_idx.shape[0]
            sample = torch.randint(0, n, (B,), generator=self.gen, device=dev)
            chosen = self.valid_start_idx[sample]
            self.pos_y = chosen // self.size
            self.pos_x = chosen % self.size
        self.steps = torch.zeros(B, dtype=torch.long, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)
        return self.state_to_index()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # actions: (B,) long in {0..3}.
        # returns: (states, rewards, done) -- all (B,), dtypes long/float32/bool.
        B, dev = self.B, self.device
        alive = ~self.done

        # per-element slip noise (Appendix H.2: NOT shared across batch).
        slip = torch.rand(B, generator=self.gen, device=dev) < self.slip_prob
        rand_a = torch.randint(0, self.n_actions, (B,), generator=self.gen, device=dev)
        actions = torch.where(slip, rand_a, actions)

        dy = self._dy[actions]
        dx = self._dx[actions]
        ny = (self.pos_y + dy).clamp(0, self.size - 1)
        nx = (self.pos_x + dx).clamp(0, self.size - 1)

        # walls block movement (stay in place)
        is_wall = self.walls[ny, nx]
        ny = torch.where(is_wall, self.pos_y, ny)
        nx = torch.where(is_wall, self.pos_x, nx)
        # done envs don't move
        ny = torch.where(alive, ny, self.pos_y)
        nx = torch.where(alive, nx, self.pos_x)
        self.pos_y = ny
        self.pos_x = nx

        at_goal = (
            (self.pos_y == self.goal_y) & (self.pos_x == self.goal_x) & alive
        )
        reward = torch.where(
            ~alive,
            torch.zeros_like(self.pos_y, dtype=torch.float32),
            torch.where(
                at_goal,
                torch.full_like(self.pos_y, self.goal_reward, dtype=torch.float32),
                torch.full_like(self.pos_y, self.step_penalty, dtype=torch.float32),
            ),
        )

        self.steps = torch.where(alive, self.steps + 1, self.steps)
        timed_out = self.steps >= self.max_steps
        self.done = self.done | (at_goal & alive) | timed_out
        # clone() prevents callers (e.g. rollout buffers) from aliasing the
        # internal `self.done` tensor that we mutate next step.
        return self.state_to_index(), reward, self.done.clone()

    # --- helpers -------------------------------------------------------------
    def state_to_index(self) -> torch.Tensor:
        # (B,) long state indices.
        return self.pos_y * self.size + self.pos_x

    @property
    def alive_mask(self) -> torch.Tensor:
        # (B,) bool.
        return ~self.done
