"""Build the Lab-1 notebooks — IVADO Bootcamp 2026.

Single source of truth for the IVADO Bootcamp Lab-1 notebooks. Run::

    # Build both at once (default — what CI / the README expects):
    .venv/bin/python scripts/build_lab1_v2_nb.py

    # Just the student stub:
    .venv/bin/python scripts/build_lab1_v2_nb.py --mode stub --out notebooks/lab1_reinforce_fourrooms_student.ipynb

    # Just the answer key:
    .venv/bin/python scripts/build_lab1_v2_nb.py --mode answer --out notebooks/lab1_reinforce_fourrooms.ipynb

Section structure follows the lab spec verbatim (§0 Setup → §11 Bridge to
Lab 2). STUDENT cells (§4–§8) source from the real ``rl_basics.student``
module; in ``--mode stub`` the generator finds the
``# === STUDENT TODO === #`` sentinel and replaces the rest of the body
with ``raise NotImplementedError("Fill me in!")``, preserving the
signature, docstring, and shape asserts before the sentinel.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
STUDENT_PY = REPO / "rl_basics" / "student.py"


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    if text.startswith("\n"):
        text = text[1:]
    return nbf.v4.new_code_cell(text.rstrip("\n"))


def _section_md(anchor: str, title: str, body: str = "") -> nbf.NotebookNode:
    out = f'<a name="{anchor}"></a>\n## {title}'
    if body:
        out += "\n\n" + body
    return md(out)


# ---------------------------------------------------------------------------
# Student-function extraction (sourced from rl_basics.student)
# ---------------------------------------------------------------------------


def _student_source() -> str:
    return STUDENT_PY.read_text()


def _extract_student_function(name: str) -> str:
    src = _student_source()
    pattern = rf"def {name}\(.*?(?=\n(?:def |class |\Z))"
    m = re.search(pattern, src, re.DOTALL)
    if not m:
        raise ValueError(f"function {name!r} not found in {STUDENT_PY}")
    return m.group(0).rstrip("\n") + "\n"


def stub_student_function(name: str) -> str:
    fn_src = _extract_student_function(name)
    sentinel = "# === STUDENT TODO === #"
    if sentinel not in fn_src:
        return fn_src
    before, _ = fn_src.split(sentinel, 1)
    return (
        before.rstrip("\n")
        + "\n    "
        + sentinel
        + "\n"
        + '    raise NotImplementedError("Fill me in!")\n'
    )


def student_cell(name: str, mode: str) -> str:
    if mode == "answer":
        return _extract_student_function(name)
    if mode == "stub":
        return stub_student_function(name)
    raise ValueError(f"mode must be 'answer' or 'stub', got {mode!r}")


_INLINE_TRAINING_PREFIX = """
# §6 — Transparent training-loop fill-in (5 updates, see the mechanics).
torch.manual_seed(0)
env_demo_loop = FourRoomsTL(batch_size=8, seed=0)
policy_demo   = MLPPolicy(env_demo_loop.n_states, env_demo_loop.n_actions, hidden=32)
opt_demo      = torch.optim.Adam(policy_demo.parameters(), lr=3e-3)
N_UPDATES_INLINE = 5

for upd in range(N_UPDATES_INLINE):
    init_states = sample_initial_states(env_demo_loop, env_demo_loop.B)
    traj = rollout(env_demo_loop, policy_demo, init_states)
    # === STUDENT TODO === #
    # Assemble: returns_to_go -> advantage -> log_probs -> policy_loss -> backward -> step
"""

_INLINE_TRAINING_ANSWER = """    G = compute_returns_to_go(traj.rewards, traj.mask, gamma=1.0)
    A = compute_advantage_vanilla(G, traj.mask)
    logits = policy_demo(traj.states.flatten()).view(*traj.states.shape, -1)
    lp = log_probs(logits, traj.actions)
    loss = policy_loss(lp, A, traj.mask)
    opt_demo.zero_grad()
    loss.backward()
    opt_demo.step()
    print(f"upd={upd:2d}  loss={loss.item():+.3f}  "
          f"ep_return={traj.rewards.sum(dim=1).mean().item():+.3f}")"""

_INLINE_TRAINING_STUB = (
    '    raise NotImplementedError(\n'
    '        "Fill in the per-update step: returns_to_go -> advantage -> "\n'
    '        "log_probs -> policy_loss -> opt.zero_grad / loss.backward / opt.step"\n'
    '    )'
)


def _inline_training_loop_src(mode: str) -> str:
    body = _INLINE_TRAINING_ANSWER if mode == "answer" else _INLINE_TRAINING_STUB
    return _INLINE_TRAINING_PREFIX + body


def _toc_md() -> str:
    return (
        "# IVADO Bootcamp 2026 — Lab 1: REINFORCE on FourRooms\n"
        "\n"
        "**Goal.** Implement the policy-gradient primitives end-to-end on a\n"
        "tiny 17×17 gridworld. Compare three advantage rules — *vanilla*,\n"
        "*value baseline*, *batch baseline* — and see which one yields a\n"
        "lower-variance gradient estimator. The same loss form scales\n"
        "straight into Lab 2's LLM setup; the only thing that changes is the\n"
        "alphabet (states → tokens).\n"
        "\n"
        "## Table of contents\n"
        "\n"
        "[§0 Setup](#sec0) · "
        "[§1 The FourRooms env](#sec1) · "
        "[§2 The policy](#sec2) · "
        "[§3 Rollouts & Trajectory](#sec3) · "
        "[§4 STUDENT: returns_to_go](#sec4) · "
        "[§5 STUDENT: policy_loss](#sec5) · "
        "[§6 STUDENT: vanilla advantage + RUN](#sec6) · "
        "[§7 STUDENT: value-baseline + RUN](#sec7) · "
        "[§8 STUDENT: batch-baseline + RUN](#sec8) · "
        "[§9 Compare baselines](#sec9) · "
        "[§10 Why it works (grad-var)](#sec10) · "
        "[§11 Bridge to Lab 2](#sec11)"
    )


def build(mode: str = "answer") -> nbf.NotebookNode:
    if mode not in ("answer", "stub"):
        raise ValueError(f"mode must be 'answer' or 'stub', got {mode!r}")
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    cells.append(md(_toc_md()))

    # ------------------------------------------------------------------
    # §0 Setup
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec0",
        "§0 Setup",
        "Idempotent. Re-run any time. Installs the package from GitHub on a\n"
        "fresh Colab runtime and loads everything you'll need below.\n"
        "\n"
        "**Wall-time knobs.** ``N_UPDATES_DEMO``, ``N_BOOT_DEMO`` and\n"
        "``SEEDS_DEMO`` are set tight so the full notebook completes in\n"
        "<5 min on Colab CPU. Scale them up (e.g. ``n_updates=400``,\n"
        "``n_boot=64``, ``seeds=[0, 1, 2]``) for the curves the slides show.",
    ))
    cells.append(code("""
# §0 — Setup. Idempotent.
import os, sys, subprocess, time

try:
    import rl_basics  # noqa: F401
except ImportError:  # pragma: no cover  (exercised in Colab, not nbmake)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/kazemnejad/ivado-rl-lab", "ipytest"],
        check=True,
    )
    import rl_basics  # noqa: F401

import numpy as np
import torch
import matplotlib.pyplot as plt
import ipytest
from IPython.display import clear_output
ipytest.autoconfig()

from rl_basics import render
from rl_basics.env import FourRoomsTL
from rl_basics.models import MLPPolicy, ValueNetwork
from rl_basics.utils import (
    Trajectory, rollout, sample_initial_states, sample_action,
    log_probs, value_loss, grad_norm, shape, reset as _rl_reset,
)
from rl_basics.runs import RunConfig, launch
from rl_basics.viz import (
    plot_curves, value_map, advantage_overlay, grad_var_panel,
    render_gif, play_traj,
)

# Wall-time knobs — tighten/expand to taste.
N_UPDATES_DEMO  = 400       # full SPEC-Appendix-B run; ~5 min per launch on Colab CPU
N_BOOT_DEMO     = 16        # n_boot for the §10 final-ckpt bootstrap
SEEDS_DEMO      = [0, 1]    # two seeds for variance bands; bump to [0, 1, 2] to match SPEC
GRAD_VAR_EVERY  = max(1, N_UPDATES_DEMO // 10)  # in-training grad-var measurement cadence (~4/run)
GRAD_VAR_NBOOT  = 50                            # n_boot for the in-training measurement


def wait_with_progress(rg, label, timeout=1800, poll=2.0):
    \"\"\"Block on rg until all seeds finish; show a live ASCII progress bar
    by line-counting each seed's metrics.jsonl on disk every `poll` seconds.\"\"\"
    n_total = rg.cfg.n_updates * len(rg.cfg.seeds)
    t0 = time.time()
    while rg.is_running():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{label} did not finish in {timeout}s")
        n_done = 0
        for h in rg.runs:
            try:
                with open(h.metrics_path) as f:
                    n_done += sum(1 for _ in f)
            except FileNotFoundError:
                pass
        bar_w = 32
        frac = min(1.0, n_done / max(n_total, 1))
        filled = int(bar_w * frac)
        bar = "█" * filled + "·" * (bar_w - filled)
        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-3)
        eta = (n_total - n_done) / max(rate, 1e-3) if rate > 0 else float("inf")
        clear_output(wait=True)
        print(f"[{label}] {bar} {n_done}/{n_total} updates  "
              f"· {elapsed:5.0f}s elapsed · ~{eta:5.0f}s remaining")
        time.sleep(poll)
    elapsed = time.time() - t0
    clear_output(wait=True)
    print(f"[{label}] done · {n_total}/{n_total} updates · {elapsed:.0f}s")


try:
    _rl_reset(verbose=False)
except Exception as _e:  # pragma: no cover  (Colab on first run has no runs/)
    print(f"reset skipped: {_e}")

print(f"Setup OK · rl_basics v{rl_basics.__version__} · python {sys.version.split()[0]}")
print(f"N_UPDATES_DEMO={N_UPDATES_DEMO}, N_BOOT_DEMO={N_BOOT_DEMO}, "
      f"SEEDS_DEMO={SEEDS_DEMO}, GRAD_VAR_EVERY={GRAD_VAR_EVERY}")
"""))

    # ------------------------------------------------------------------
    # §1 The FourRooms env
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec1",
        "§1 The FourRooms env",
        "A 17×17 gridworld split by a `+`-shaped wall into four rooms with\n"
        "one-cell doorways at `(8, 4)`, `(8, 13)`, `(4, 8)`, `(13, 8)`.\n"
        "Reward = `−0.05` per step, `+1.0` on reaching the goal at `(15,15)`.\n"
        "\n"
        "**Per-element randomness.**\n"
        "- Each `reset()` samples a fresh start uniformly from the 64 cells\n"
        "  of the **top-left** room (the `TL` in `FourRoomsTL`).\n"
        "- Each `step` flips to a uniformly random action with probability\n"
        "  `slip_prob = 0.4`.\n"
        "\n"
        "Action map: `0=N, 1=S, 2=W, 3=E`. The visualization uses the\n"
        "framework's pixel-art renderer (`rl_basics.render`) — same one\n"
        "we'll use for trajectory GIFs in §9.",
    ))
    cells.append(code("""
# §1 — Spin up an env, peek at the wall layout + a fresh batch of starts.
demo_env = FourRoomsTL(batch_size=8, seed=0)
walls = demo_env.walls.cpu().numpy()
goal_xy = (demo_env.goal_y, demo_env.goal_x)

# Pixel-art bg (Farama-style) — same renderer we'll use for trajectory GIFs.
bg = render.build_static_bg(walls, goal_xy=goal_xy, start_xy=None)
bg_arr = np.asarray(bg.convert("RGB"))

# Map each batch start (state idx) to pixel coords on the bg canvas.
init_states = demo_env.state_to_index().cpu().numpy()
iy, ix = init_states // demo_env.size, init_states % demo_env.size
CELL = render.CELL
px = ix * CELL + CELL // 2
py = iy * CELL + CELL // 2

fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.imshow(bg_arr, interpolation="nearest")
ax.scatter(px, py, marker="o", s=120, c="#1f77b4",
           edgecolors="white", linewidths=2.0, zorder=3, label="batch starts")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"FourRoomsTL · {demo_env.size}×{demo_env.size} grid · 8 random starts")
ax.legend(loc="upper right", framealpha=0.95)
plt.show()
print(f"states={demo_env.n_states}, actions={demo_env.n_actions}, "
      f"max_steps={demo_env.max_steps}, slip={demo_env.slip_prob}")
"""))

    # ------------------------------------------------------------------
    # §2 The policy
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec2",
        "§2 The policy",
        "A 1-hidden-layer MLP that takes a one-hot state index and emits 4\n"
        "logits:\n"
        "\n"
        "$$\\pi(a \\mid s) = \\mathrm{softmax}\\big(\\, W_2\\, "
        "\\mathrm{ReLU}(W_1\\, \\mathrm{onehot}(s))\\, \\big)_a$$\n"
        "\n"
        "Same shape as Lab 2 — swap one-hot for token embeddings and the\n"
        "rest is identical.",
    ))
    cells.append(code("""
# §2 — Build a fresh policy and probe its output shapes.
torch.manual_seed(0)
policy = MLPPolicy(n_states=demo_env.n_states, n_actions=demo_env.n_actions, hidden=64)

states = demo_env.state_to_index()
logits = policy(states)
shape(logits, "logits")               # (B, n_actions)
actions = sample_action(logits)
shape(actions, "actions")             # (B,)
lp = log_probs(logits, actions)
shape(lp, "log_probs")                # (B,)
"""))

    # ------------------------------------------------------------------
    # §3 Rollouts & Trajectory
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec3",
        "§3 Rollouts & Trajectory",
        "`rollout(env, policy, init_states)` returns a `Trajectory` with\n"
        "four `(B, T)` tensors:\n"
        "\n"
        "- `states`  — agent position (state index)\n"
        "- `actions` — chosen action\n"
        "- `rewards` — scalar reward at each step\n"
        "- `mask`    — `True` while the env is alive at step `t`; flips to\n"
        "  `False` once the episode ends and stays `False`.\n"
        "\n"
        "We do **not** record `log_probs` — they're recomputed under the\n"
        "*current* policy at backward time, exactly like in LLM RL.",
    ))
    cells.append(code("""
# §3a — Generate one batch of trajectories with a fresh (random) policy.
torch.manual_seed(0)
env_demo  = FourRoomsTL(batch_size=8, seed=0)
policy_demo = MLPPolicy(env_demo.n_states, env_demo.n_actions, hidden=64)
init = sample_initial_states(env_demo, env_demo.B)
traj = rollout(env_demo, policy_demo, init)

shape(traj.states,  "states")
shape(traj.actions, "actions")
shape(traj.rewards, "rewards")
shape(traj.mask,    "mask")
print(f"alive ratio (mask.float().mean) = {traj.mask.float().mean().item():.3f}")
print(f"episode returns = {traj.rewards.sum(dim=1).tolist()}")
"""))
    cells.append(code("""
# §3b — Sanity: `mask` is monotone non-increasing per row.
m = traj.mask.float().cpu().numpy()
diff = np.diff(m, axis=1)
assert (diff <= 0).all(), "mask must be monotone non-increasing once an env is done"
print("mask monotone OK across all", m.shape[0], "trajectories")
"""))

    # ------------------------------------------------------------------
    # §4 STUDENT — returns_to_go
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec4",
        "§4 STUDENT — returns_to_go",
        "**Math.**\n"
        "\n"
        "$$G_t = \\sum_{k \\geq t} \\gamma^{\\,k - t}\\, r_k$$\n"
        "\n"
        "**Implementation.** Sweep **backwards** with a running sum: keep\n"
        "`running = 0`, then for `t` from `T-1` down to `0`:\n"
        "\n"
        "$$\\text{running} \\,\\leftarrow\\, r_t + \\gamma \\cdot \\text{running}, "
        "\\qquad G_t \\,\\leftarrow\\, \\text{running}$$\n"
        "\n"
        "**Mask awareness.** If `mask[b, t] = False`, the env was already\n"
        "done at step `t`. Downstream loss multiplies advantage by `mask`\n"
        "so dead timesteps don't actually move the gradient — just produce\n"
        "a finite tensor.",
    ))
    cells.append(code(student_cell("compute_returns_to_go", mode)))
    cells.append(code("""
%%ipytest -q

def test_returns_to_go_mc():
    rewards = torch.tensor([[1., 0., 0.],
                            [0., 0., 1.]])
    mask = torch.ones_like(rewards, dtype=torch.bool)
    G = compute_returns_to_go(rewards, mask, gamma=1.0)
    expected = torch.tensor([[1., 0., 0.],
                             [1., 1., 1.]])
    assert torch.allclose(G, expected), f"expected {expected.tolist()}, got {G.tolist()}"

def test_returns_to_go_discount():
    rewards = torch.tensor([[0., 0., 1.]])
    mask = torch.ones_like(rewards, dtype=torch.bool)
    G = compute_returns_to_go(rewards, mask, gamma=0.9)
    # G_0 = 0.81, G_1 = 0.9, G_2 = 1.0
    expected = torch.tensor([[0.81, 0.9, 1.0]])
    assert torch.allclose(G, expected, atol=1e-6), f"got {G.tolist()}"

def test_returns_to_go_shape():
    rewards = torch.zeros(4, 7)
    mask = torch.ones(4, 7, dtype=torch.bool)
    assert compute_returns_to_go(rewards, mask, gamma=1.0).shape == (4, 7)
"""))

    # ------------------------------------------------------------------
    # §5 STUDENT — policy_loss
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec5",
        "§5 STUDENT — policy_loss",
        "**The REINFORCE objective.** Maximize expected return by following\n"
        "the policy gradient. Equivalently, minimize:\n"
        "\n"
        "$$L \\;=\\; -\\,\\frac{1}{\\sum \\mathrm{mask}} "
        "\\sum_{(b,\\,t)\\,\\text{alive}} A_{b,t}\\, \\log \\pi(a_t \\mid s_t)$$\n"
        "\n"
        "This is the **same expression** Lab 2 uses for tokens. One-liner\n"
        "using element-wise multiplication; clamp the denominator so an\n"
        "all-dead batch returns a finite zero.",
    ))
    cells.append(code(student_cell("policy_loss", mode)))
    cells.append(code("""
%%ipytest -q

def test_policy_loss_scalar():
    lp  = torch.tensor([[-0.1, -2.0, -0.5]])
    adv = torch.tensor([[ 1.0,  0.0,  2.0]])
    m   = torch.tensor([[True, True, True]])
    L = policy_loss(lp, adv, m)
    # mean over alive of -A*lp = -mean(-0.1, 0, -1.0) = 0.3666...
    assert L.dim() == 0, f"expected scalar, got dim={L.dim()}"
    assert torch.allclose(L, torch.tensor(0.3667), atol=1e-3), L.item()

def test_policy_loss_mask_aware():
    lp  = torch.tensor([[-1.0, -1.0]])
    adv = torch.tensor([[ 1.0,  9.0]])  # 9.0 should NOT contribute
    m   = torch.tensor([[True, False]])
    L = policy_loss(lp, adv, m)
    assert torch.allclose(L, torch.tensor(1.0)), L.item()

def test_policy_loss_all_dead_safe():
    lp  = torch.zeros(2, 3)
    adv = torch.zeros(2, 3)
    m   = torch.zeros(2, 3, dtype=torch.bool)
    assert torch.isfinite(policy_loss(lp, adv, m)).item()
"""))

    # ------------------------------------------------------------------
    # §6 STUDENT — vanilla advantage + RUN
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec6",
        "§6 STUDENT — vanilla REINFORCE + RUN",
        "**Vanilla advantage.** $A_t = G_t$. One-liner; mask is unused here.\n"
        "\n"
        "Two cells follow:\n"
        "1. A *transparent* training loop you write by hand (5 updates)\n"
        "   — `returns_to_go → advantage → log_probs → policy_loss → backward → step`.\n"
        "   This is the algorithm; everything else is bookkeeping.\n"
        "2. A `launch()` call that runs the same loop across seeds in\n"
        "   subprocesses. We measure `tr(Var[ĝ])` every `GRAD_VAR_EVERY`\n"
        "   updates so §10 can plot how variance evolves during training.\n"
        "\n"
        "If §4 / §5 tests are red, expect divergence — fix those first.",
    ))
    cells.append(code(student_cell("compute_advantage_vanilla", mode)))
    cells.append(code("""
%%ipytest -q
def test_advantage_vanilla_passthrough():
    G = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    m = torch.tensor([[True, True], [True, False]])
    A = compute_advantage_vanilla(G, m)
    assert torch.allclose(A, G), (A, G)
"""))
    cells.append(code(_inline_training_loop_src(mode)))
    cells.append(code("""
# §6 — Launch vanilla REINFORCE: one subprocess per seed.
import shutil
from pathlib import Path
runs_dir_demo = Path.cwd() / "runs"
shutil.rmtree(runs_dir_demo, ignore_errors=True)
runs_dir_demo.mkdir(parents=True, exist_ok=True)

cfg_van = RunConfig(
    advantage_kind="vanilla",
    n_updates=N_UPDATES_DEMO,
    seeds=SEEDS_DEMO,
    log_every=1,
    grad_var_every=GRAD_VAR_EVERY,
    grad_var_n_boot=GRAD_VAR_NBOOT,
)
rg_van = launch(cfg_van, advantage_fn=compute_advantage_vanilla,
                workers=2, runs_dir=runs_dir_demo)
wait_with_progress(rg_van, "vanilla")
print("vanilla metrics rows per seed:", [len(h.metrics) for h in rg_van.runs])
"""))
    cells.append(code("""
# §6 visual (a) — vanilla learning curve.
fig = plot_curves([rg_van], metric="ep_return_mean", smooth=20,
                  title="vanilla — ep_return_mean (mean ± std over seeds)")
plt.show()
"""))
    cells.append(code("""
# §6 visual (b) — vanilla pixel-art rollout GIF.
from IPython.display import Image as _IPyImage
gif_van = render_gif(rg_van.runs[0], out_path="/tmp/lab1_vanilla_seed0.gif", fps=8)
_IPyImage(filename=str(gif_van))
"""))

    # ------------------------------------------------------------------
    # §7 STUDENT — value baseline + RUN
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec7",
        "§7 STUDENT — value baseline + RUN",
        "**Math.** $A_t = G_t - V_\\phi(s_t)$.\n"
        "\n"
        "$V_\\phi$ is trained with masked MSE against $G_t$ inside\n"
        "`train.py`; `value_pred` is **detached** before being used in the\n"
        "policy gradient (no backprop through $V_\\phi$ for the policy\n"
        "update — it has its own optimizer).",
    ))
    cells.append(code(student_cell("compute_advantage_with_value_baseline", mode)))
    cells.append(code("""
%%ipytest -q
def test_advantage_value_baseline_subtracts():
    G = torch.tensor([[1.0, 2.0]])
    V = torch.tensor([[0.2, 0.5]])
    m = torch.ones_like(G, dtype=torch.bool)
    A = compute_advantage_with_value_baseline(G, V, m)
    assert torch.allclose(A, torch.tensor([[0.8, 1.5]]))
"""))
    cells.append(code("""
# §7 — Launch value-baseline REINFORCE.
cfg_val = RunConfig(
    advantage_kind="value",
    use_value_baseline=True,
    n_updates=N_UPDATES_DEMO,
    seeds=SEEDS_DEMO,
    log_every=1,
    grad_var_every=GRAD_VAR_EVERY,
    grad_var_n_boot=GRAD_VAR_NBOOT,
)
rg_val = launch(cfg_val, advantage_fn=compute_advantage_with_value_baseline,
                workers=2, runs_dir=runs_dir_demo)
wait_with_progress(rg_val, "value")
print("value metrics rows per seed:", [len(h.metrics) for h in rg_val.runs])
"""))
    cells.append(code("""
# §7 visual (a) — value-baseline learning curve.
fig = plot_curves([rg_val], metric="ep_return_mean", smooth=20,
                  title="value baseline — ep_return_mean (mean ± std over seeds)")
plt.show()
"""))
    cells.append(code("""
# §7 visual (b) — value-baseline pixel-art rollout GIF.
from IPython.display import Image as _IPyImage
gif_val = render_gif(rg_val.runs[0], out_path="/tmp/lab1_value_seed0.gif", fps=8)
_IPyImage(filename=str(gif_val))
"""))

    # ------------------------------------------------------------------
    # §8 STUDENT — batch baseline + RUN
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec8",
        "§8 STUDENT — batch baseline + RUN",
        "**Math.** $A_t = G_t - \\mu_t$, where $\\mu_t$ is the **mean over\n"
        "the batch axis at timestep $t$**, restricted to alive trajectories:\n"
        "\n"
        "$$\\mu_t \\;=\\; \\frac{\\sum_{b}\\, G_{b,t}\\, \\mathrm{mask}_{b,t}}"
        "{\\sum_{b}\\, \\mathrm{mask}_{b,t}}$$\n"
        "\n"
        "Cheap, no value network, time-varying — and (spoiler) very\n"
        "competitive in this gridworld.",
    ))
    cells.append(code(student_cell("compute_advantage_with_batch_baseline", mode)))
    cells.append(code("""
%%ipytest -q
def test_advantage_batch_baseline_zero_mean_per_t():
    G = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    m = torch.ones_like(G, dtype=torch.bool)
    A = compute_advantage_with_batch_baseline(G, m)
    assert torch.allclose(A.sum(dim=0), torch.zeros(2), atol=1e-6)

def test_advantage_batch_baseline_mask_aware():
    G = torch.tensor([[1.0, 5.0],
                      [3.0, 9.0]])
    m = torch.tensor([[True,  True ],
                      [True,  False]])
    A = compute_advantage_with_batch_baseline(G, m)
    expected = torch.tensor([[-1.0, 0.0],
                             [ 1.0, 4.0]])
    assert torch.allclose(A, expected, atol=1e-6), (A, expected)
"""))
    cells.append(code("""
# §8 — Launch batch-baseline REINFORCE.
cfg_bat = RunConfig(
    advantage_kind="batch",
    n_updates=N_UPDATES_DEMO,
    seeds=SEEDS_DEMO,
    log_every=1,
    grad_var_every=GRAD_VAR_EVERY,
    grad_var_n_boot=GRAD_VAR_NBOOT,
)
rg_bat = launch(cfg_bat, advantage_fn=compute_advantage_with_batch_baseline,
                workers=2, runs_dir=runs_dir_demo)
wait_with_progress(rg_bat, "batch")
print("batch metrics rows per seed:", [len(h.metrics) for h in rg_bat.runs])
"""))
    cells.append(code("""
# §8 visual (a) — batch-baseline learning curve.
fig = plot_curves([rg_bat], metric="ep_return_mean", smooth=20,
                  title="batch baseline — ep_return_mean (mean ± std over seeds)")
plt.show()
"""))
    cells.append(code("""
# §8 visual (b) — batch-baseline pixel-art rollout GIF.
from IPython.display import Image as _IPyImage
gif_bat = render_gif(rg_bat.runs[0], out_path="/tmp/lab1_batch_seed0.gif", fps=8)
_IPyImage(filename=str(gif_bat))
"""))

    # ------------------------------------------------------------------
    # §9 Compare baselines
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec9",
        "§9 Compare baselines",
        "Four quick views (per-baseline rollout GIFs already appear after\n"
        "each §6 / §7 / §8 launch):\n"
        "1. **Learning curves** — `plot_curves` overlays the three groups\n"
        "   with mean ± std bands across seeds.\n"
        "2. **Value map** — what $V_\\phi$ has learned (value baseline only).\n"
        "3. **Advantage overlay** — single-traj comparison of $r_t$, $G_t$,\n"
        "   and the three advantage rules.\n"
        "4. **Interactive player** — `play_traj` returns a Plotly figure\n"
        "   with Play / Pause + a scrubbable slider.",
    ))
    cells.append(code("""
# §9a — Learning-curve overlay across the three baselines.
fig = plot_curves([rg_van, rg_val, rg_bat],
                  metric="ep_return_mean",
                  smooth=max(1, N_UPDATES_DEMO // 5),
                  title=f"Episode return (n_updates={N_UPDATES_DEMO})")
plt.show()
"""))
    cells.append(code("""
# §9b — Value map: what V_phi(s) has learned about the grid.
fig = value_map(rg_val.runs[0])
plt.show()
"""))
    cells.append(code("""
# §9c — Advantage overlay on a single trajectory.
# Requires a ckpt with both 'policy' and 'value_net' state-dicts → use rg_val.
fig = advantage_overlay(rg_val.runs[0], n_traj=1)
plt.show()
"""))
    cells.append(code("""
# §9e — Interactive trajectory player (Plotly Play + slider) on the batch policy.
play_traj(rg_bat.runs[0])
"""))

    # ------------------------------------------------------------------
    # §10 Why it works — gradient variance
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec10",
        "§10 Why it works — gradient variance",
        "**Goal.** Show that the *batch* baseline yields a lower-variance\n"
        "gradient estimator than vanilla, with the *value* baseline\n"
        "somewhere in between (depending on how well $V_\\phi$ has been\n"
        "trained).\n"
        "\n"
        "**Mechanism.** For an estimator\n"
        "$\\hat{g} = \\nabla_\\theta L(\\theta)$ over a batch of $B$\n"
        "trajectories,\n"
        "\n"
        "$$\\mathrm{tr}\\big(\\mathrm{Var}[\\hat{g}]\\big) \\;=\\; "
        "\\mathbb{E}\\big[\\|\\hat{g}\\|^2\\big] \\;-\\; "
        "\\big\\| \\mathbb{E}[\\hat{g}] \\big\\|^2$$\n"
        "\n"
        "Lower is better — the gradient signal isn't drowned in noise.\n"
        "\n"
        "We show the in-training trajectory of `tr(Var[g])` (already logged\n"
        "every `GRAD_VAR_EVERY` updates by `train.py`) and a final-ckpt\n"
        "bootstrap.",
    ))
    cells.append(code("""
# §10a — In-training: return curves + tr(Var[g]) on linear y-axis.
fig = grad_var_panel([rg_van, rg_val, rg_bat], panels=("return", "tr_var"))
plt.show()
"""))
    cells.append(code("""
# §10b — Final-ckpt bootstrap: re-roll under each frozen policy and measure tr(Var[g]).
from rl_basics.grad_variance import measure_grad_variance

def _load_policy_value(handle, hidden=64):
    ckpt = torch.load(handle.ckpt_path, map_location="cpu", weights_only=True)
    pol = MLPPolicy(n_states=demo_env.n_states, n_actions=demo_env.n_actions, hidden=hidden)
    pol.load_state_dict(ckpt["policy"])
    pol.eval()
    val = None
    if "value_net" in ckpt:
        val = ValueNetwork(n_states=demo_env.n_states, hidden=hidden)
        val.load_state_dict(ckpt["value_net"])
        val.eval()
    return pol, val

results = {}
for kind, rg in [("vanilla", rg_van), ("value", rg_val), ("batch", rg_bat)]:
    pol, val = _load_policy_value(rg.runs[0])
    res = measure_grad_variance(
        env_class=FourRoomsTL, env_kwargs={},
        batch_size=rg.cfg.batch_size, device="cpu",
        policy=pol, value_net=val,
        advantage_kind=kind, gamma=rg.cfg.gamma,
        n_boot=N_BOOT_DEMO,
    )
    results[kind] = res
    print(f"{kind:>7s} | tr_var={res['tr_var']:.3e} | "
          f"||E[g]||²={res['mean_g_norm_sq']:.3e} | snr={res['snr']:.3e}")
"""))

    # ------------------------------------------------------------------
    # §11 Bridge to Lab 2
    # ------------------------------------------------------------------
    cells.append(_section_md(
        "sec11",
        "§11 Bridge to Lab 2",
        "Same loss form, swap the alphabet. In the LLM lab `s` becomes a\n"
        "token context, `a` becomes the next token, and\n"
        "$\\log \\pi(a \\mid s)$ is the LM head's log-softmax over the\n"
        "vocabulary. The mask becomes a per-token alive mask\n"
        "(prompt-vs-completion). Returns flow back the same way; advantages\n"
        "are computed the same way.\n"
        "\n"
        "**The expression you wrote in §5 is the expression Lab 2 calls\n"
        "`policy_loss` — verbatim.**\n"
        "\n"
        "Below: the same machinery handling a partial-observability detour\n"
        "task — included as \"look what this same code can do\" (no\n"
        "inference cell; static gif).\n"
        "\n"
        "![POMDP detour rollout](https://raw.githubusercontent.com/kazemnejad/ivado-rl-lab/main/pretrained/v7_pomdp_tr_detour.gif)",
    ))
    cells.append(code("""
# §11 — Cleanup. Idempotent — safe to skip.
_rl_reset(verbose=True, runs_dir=runs_dir_demo)
"""))

    nb.cells = cells
    nb.metadata.update({
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    })
    return nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for a single notebook. If omitted, both notebooks "
             "(lab1_reinforce_fourrooms_student.ipynb + lab1_reinforce_fourrooms.ipynb) are built.",
    )
    parser.add_argument(
        "--mode",
        choices=("answer", "stub"),
        default=None,
        help="'answer' = function bodies kept (nbmake-runnable), "
             "'stub' = bodies replaced with NotImplementedError (learner copy). "
             "If omitted, both notebooks are built.",
    )
    args = parser.parse_args(argv)

    nb_dir = REPO / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    if args.out is None and args.mode is None:
        targets = [
            ("stub",   nb_dir / "lab1_reinforce_fourrooms_student.ipynb"),
            ("answer", nb_dir / "lab1_reinforce_fourrooms.ipynb"),
        ]
    else:
        mode = args.mode or "answer"
        out = Path(args.out) if args.out else (
            nb_dir / ("lab1_reinforce_fourrooms_student.ipynb" if mode == "stub" else "lab1_reinforce_fourrooms.ipynb")
        )
        targets = [(mode, out)]

    for mode, out in targets:
        nb = build(mode=mode)
        with open(out, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"wrote {out} · {len(nb.cells)} cells · mode={mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
