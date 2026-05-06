"""Visualization helpers — SPEC §5.7.

Tasks 20–21 ship `plot_curves` and `play_traj`. Subsequent tasks (22-26) add
`play_traj_compare`, `value_map`, `advantage_overlay`, `grad_var_panel`,
`show_tensor`, etc.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt  # noqa: F401
    import plotly.graph_objects as go  # noqa: F401

    from rl_basics.runs import RunGroup, RunHandle


def plot_curves(
    rgs: "RunGroup | list[RunGroup]",
    metric: str = "ep_return_mean",
    smooth: int = 10,
    title: str | None = None,
    ax: "plt.Axes | None" = None,
) -> "plt.Figure":
    """Static matplotlib curve. Mean ± std band over seeds.

    Auto-handles different RunGroups overlaid (one color per advantage_kind).
    """
    import matplotlib.pyplot as plt
    from rl_basics.runs import RunGroup

    if isinstance(rgs, RunGroup):
        rgs = [rgs]
    rgs = list(rgs)
    if smooth < 1:
        raise ValueError(f"smooth must be >= 1, got {smooth}")

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    try:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        ])

        for idx, rg in enumerate(rgs):
            curves: list[np.ndarray] = []
            for h in rg.runs:
                df = h.metrics
                if df is None or len(df) == 0 or metric not in df.columns:
                    continue
                vals = np.asarray(df[metric].to_numpy(), dtype=float)
                if vals.size == 0:
                    continue
                curves.append(vals)
            if not curves:
                continue

            n = min(c.size for c in curves)
            if n == 0:
                continue
            mat = np.stack([c[:n] for c in curves], axis=0)

            w = max(1, min(int(smooth), n))
            if w > 1:
                kernel = np.ones(w) / w
                smoothed = np.stack(
                    [np.convolve(row, kernel, mode="valid") for row in mat],
                    axis=0,
                )
            else:
                smoothed = mat

            mean = smoothed.mean(axis=0)
            std = smoothed.std(axis=0, ddof=0)

            offset = w - 1
            x = np.arange(offset, offset + mean.size)

            label = rg.cfg.advantage_kind
            color = color_cycle[idx % len(color_cycle)]

            ax.plot(x, mean, color=color, label=str(label))
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel("update")
        ax.set_ylabel(metric)
        if title is not None:
            ax.set_title(title)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        return fig
    except Exception:
        if owns_fig:
            plt.close(fig)
        raise


# ---------------------------------------------------------------------------
# Task 21: play_traj (Plotly + slider)
# ---------------------------------------------------------------------------


def _import_poc():
    """Back-compat shim: returns the in-package renderer (rl_basics.render).

    The pixel-art sprite pipeline used to live at ``scripts/poc_pixel_render.py``
    with a hardcoded REPO_ROOT and external sprite dir; it now ships inside
    the package as ``rl_basics.render`` (assets at ``rl_basics/_assets/farama/``)
    so a pip-installed wheel works on Colab without a source clone.
    """
    from rl_basics import render
    return render


def play_traj(
    run: "RunHandle",
    init: tuple[int, int] | None = None,
    max_frames: int = 120,
    display_max_dim: int = 384,
) -> "go.Figure":
    """Render a single trajectory on the FourRooms grid as Plotly with
    Play + slider. If ``init`` given, forces the start cell.

    Performance knobs (Plotly figures embed every frame as a base64-encoded
    image; without these caps a 400-step episode at the renderer's native
    1088×1088 resolution would ship a ~700 MB JSON payload and freeze the
    browser):
      * ``max_frames`` — if the rolled-out episode has more steps than
        this, frames are stride-sampled down (start, stop, every Kth
        intermediate) so the player stays scrubbable.
      * ``display_max_dim`` — each rendered frame is downscaled (NEAREST,
        aspect-preserving) so its largest side is ≤ this many pixels
        before being embedded in the figure.

    Returns a ``plotly.graph_objects.Figure`` whose:
      - ``frames`` list has one ``go.Frame`` per (sampled) timestep,
      - ``layout.sliders`` has a step-index slider,
      - ``layout.updatemenus`` has Play/Pause buttons,
      - ``layout.meta`` exposes ``{"states": [(y,x), ...], "bump_idx": [...]}``
        for downstream inspection (and tests).
    """
    import plotly.graph_objects as go
    import torch
    from PIL import Image

    from rl_basics.env import FourRoomsTL
    from rl_basics.models import MLPPolicy
    from rl_basics.utils import sample_action

    poc = _import_poc()

    # ---- 1. Roll out a single episode on FourRoomsTL(B=1) -----------------
    seed = int(getattr(run, "seed", 0) or 0)
    cfg = run.config if isinstance(run.config, dict) else {}
    hidden = int(cfg.get("hidden", 64))

    env = FourRoomsTL(batch_size=1, seed=seed)
    if init is not None:
        iy, ix = int(init[0]), int(init[1])
        init_state = torch.tensor([iy * env.size + ix], dtype=torch.long)
        s = env.reset(init_states=init_state)
    else:
        s = env.reset()

    policy = MLPPolicy(env.n_states, env.n_actions, hidden=hidden)
    ckpt = torch.load(run.ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
    policy.load_state_dict(sd)
    policy.eval()

    torch.manual_seed(seed)
    states_yx: list[tuple[int, int]] = [
        (int(env.pos_y[0].item()), int(env.pos_x[0].item()))
    ]
    bump_idx: list[int] = []
    cum_returns: list[float] = [0.0]
    for _ in range(env.max_steps):
        prev = states_yx[-1]
        with torch.no_grad():
            logits = policy(s)
            a = sample_action(logits)
        s, r, done = env.step(a)
        new = (int(env.pos_y[0].item()), int(env.pos_x[0].item()))
        states_yx.append(new)
        cum_returns.append(cum_returns[-1] + float(r[0].item()))
        # Bump = tried to move but didn't (blocked by wall or grid edge).
        if new == prev:
            bump_idx.append(len(states_yx) - 1)
        if bool(done[0].item()):
            break

    # ---- 2. Build sprite frames via the poc helpers -----------------------
    walls = env.walls.cpu().numpy()
    goal_xy = (env.goal_y, env.goal_x)
    bg = poc.build_static_bg(walls, goal_xy, states_yx[0])
    elf_keys = poc.infer_elf_keys(states_yx)
    bump_set = set(bump_idx)

    n = len(states_yx)

    # Stride-sample if too many frames (always keep the first and last so
    # learners see start state + final state).
    if max_frames is not None and n > max_frames:
        stride = max(1, (n - 1) // (max_frames - 1))
        keep_idx = list(range(0, n, stride))
        if keep_idx[-1] != n - 1:
            keep_idx.append(n - 1)
    else:
        keep_idx = list(range(n))

    # Downscale ratio for embedding in Plotly (one full sprite-resolution
    # canvas is ~1 MB raw; cap so the JSON payload stays browser-friendly).
    canvas_max = max(bg.size)
    if display_max_dim is not None and canvas_max > display_max_dim:
        scale = display_max_dim / canvas_max
        new_size = (int(bg.size[0] * scale), int(bg.size[1] * scale))
    else:
        new_size = bg.size

    frames: list[go.Frame] = []
    init_arr: np.ndarray | None = None
    initial_title = ""
    for t in keep_idx:
        is_bump = t in bump_set
        trailed = poc.composite_trail(bg, states_yx, t)
        composed = poc.composite_elf(
            trailed, states_yx[t], elf_keys[t], bump=is_bump
        )
        if new_size != bg.size:
            composed = composed.resize(new_size, Image.NEAREST)
        arr = np.asarray(composed.convert("RGB"))
        flag = "  ⚠ BUMP" if is_bump else ""
        title = (
            f"step {t:>3}/{n - 1}    cum return {cum_returns[t]:+.2f}{flag}"
        )
        if t == keep_idx[0]:
            init_arr = arr
            initial_title = title
        frames.append(
            go.Frame(
                data=[go.Image(z=arr)],
                name=str(t),
                layout=go.Layout(title=dict(text=title)),
            )
        )

    # ---- 3. Assemble figure with Play button + slider ---------------------
    if init_arr is None:
        init_arr = np.zeros((1, 1, 3), dtype=np.uint8)

    slider_steps = [
        dict(
            method="animate",
            label=str(t),
            args=[
                [str(t)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
        )
        for t in keep_idx
    ]

    fig = go.Figure(
        data=[go.Image(z=init_arr)],
        frames=frames,
    )
    fig.update_layout(
        title=dict(text=initial_title),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="step: "),
            pad=dict(t=30),
            steps=slider_steps,
        )],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.0, y=-0.05, xanchor="left", yanchor="top",
            pad=dict(t=20, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=150, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode="immediate",
                    )],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        meta=dict(
            states=states_yx,
            bump_idx=bump_idx,
            n_frames=n,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Task 22: play_traj_compare (side-by-side trajectory player)
# ---------------------------------------------------------------------------


def _rollout_one_traj(run: "RunHandle", init: tuple[int, int] | None):
    """Single B=1 rollout for a run. Returns (states_yx, bump_idx, cum_returns, env).

    Helper for ``play_traj_compare``. Mirrors the rollout block in
    ``play_traj`` so the two stay decoupled (per task brief: don't touch the
    existing play_traj body).
    """
    import torch

    from rl_basics.env import FourRoomsTL
    from rl_basics.models import MLPPolicy
    from rl_basics.utils import sample_action

    seed = int(getattr(run, "seed", 0) or 0)
    cfg = run.config if isinstance(run.config, dict) else {}
    hidden = int(cfg.get("hidden", 64))

    env = FourRoomsTL(batch_size=1, seed=seed)
    if init is not None:
        iy, ix = int(init[0]), int(init[1])
        init_state = torch.tensor([iy * env.size + ix], dtype=torch.long)
        s = env.reset(init_states=init_state)
    else:
        s = env.reset()

    policy = MLPPolicy(env.n_states, env.n_actions, hidden=hidden)
    ckpt = torch.load(run.ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
    policy.load_state_dict(sd)
    policy.eval()

    torch.manual_seed(seed)
    states_yx: list[tuple[int, int]] = [
        (int(env.pos_y[0].item()), int(env.pos_x[0].item()))
    ]
    bump_idx: list[int] = []
    cum_returns: list[float] = [0.0]
    for _ in range(env.max_steps):
        prev = states_yx[-1]
        with torch.no_grad():
            logits = policy(s)
            a = sample_action(logits)
        s, r, done = env.step(a)
        new = (int(env.pos_y[0].item()), int(env.pos_x[0].item()))
        states_yx.append(new)
        cum_returns.append(cum_returns[-1] + float(r[0].item()))
        if new == prev:
            bump_idx.append(len(states_yx) - 1)
        if bool(done[0].item()):
            break
    return states_yx, bump_idx, cum_returns, env


def play_traj_compare(
    rgs: "list[RunGroup]",
    init: tuple[int, int] | None = None,
    max_frames: int = 80,
    display_max_dim: int = 320,
) -> "go.Figure":
    """Side-by-side trajectory player. One subplot per RunGroup (uses
    ``runs[0]`` per group), shared step slider + Play button, same ``init``.

    Frames are right-padded by repeating each panel's last sprite so all
    panels stay in lockstep on a single global step axis.

    Performance knobs (matter even more than for ``play_traj`` because the
    per-frame payload is ``n_panels`` × the per-frame size):
      * ``max_frames`` — cap on the global frame count after stride-sampling.
      * ``display_max_dim`` — each panel's frames are downscaled to fit.
    """
    import plotly.graph_objects as go
    from PIL import Image
    from plotly.subplots import make_subplots

    poc = _import_poc()
    rgs = list(rgs)
    if not rgs:
        raise ValueError("play_traj_compare needs at least one RunGroup")

    panels = []
    for rg in rgs:
        if not rg.runs:
            raise ValueError(f"RunGroup {rg.cfg.resolve_name()} has no runs")
        run = rg.runs[0]
        states, bumps, cums, env = _rollout_one_traj(run, init)
        walls = env.walls.cpu().numpy()
        goal_xy = (env.goal_y, env.goal_x)
        bg = poc.build_static_bg(walls, goal_xy, states[0])
        elf_keys = poc.infer_elf_keys(states)
        n = len(states)

        canvas_max = max(bg.size)
        if display_max_dim is not None and canvas_max > display_max_dim:
            scale = display_max_dim / canvas_max
            new_size = (int(bg.size[0] * scale), int(bg.size[1] * scale))
        else:
            new_size = bg.size

        per_step_imgs: list[np.ndarray] = []
        bump_set = set(bumps)
        for t in range(n):
            is_bump = t in bump_set
            trailed = poc.composite_trail(bg, states, t)
            composed = poc.composite_elf(
                trailed, states[t], elf_keys[t], bump=is_bump
            )
            if new_size != bg.size:
                composed = composed.resize(new_size, Image.NEAREST)
            per_step_imgs.append(np.asarray(composed.convert("RGB")))
        panels.append(dict(
            label=rg.cfg.resolve_name(),
            states=states,
            bump_idx=bumps,
            cum_returns=cums,
            imgs=per_step_imgs,
        ))

    n_panels = len(panels)
    n_global_full = max(len(p["imgs"]) for p in panels)
    if max_frames is not None and n_global_full > max_frames:
        stride = max(1, (n_global_full - 1) // (max_frames - 1))
        keep_idx = list(range(0, n_global_full, stride))
        if keep_idx[-1] != n_global_full - 1:
            keep_idx.append(n_global_full - 1)
    else:
        keep_idx = list(range(n_global_full))
    n_global = len(keep_idx)

    fig = make_subplots(
        rows=1, cols=n_panels,
        subplot_titles=[p["label"] for p in panels],
        horizontal_spacing=0.02,
    )

    # Initial frame = step 0 image of each panel.
    for col_idx, p in enumerate(panels, start=1):
        fig.add_trace(go.Image(z=p["imgs"][0]), row=1, col=col_idx)

    frames: list[go.Frame] = []
    for t in keep_idx:
        data = []
        for p in panels:
            idx = min(t, len(p["imgs"]) - 1)
            data.append(go.Image(z=p["imgs"][idx]))
        frames.append(go.Frame(
            data=data,
            name=str(t),
            traces=list(range(n_panels)),
        ))
    fig.frames = frames

    slider_steps = [
        dict(
            method="animate",
            label=str(t),
            args=[
                [str(t)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
        )
        for t in keep_idx
    ]

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="step: "),
            pad=dict(t=30),
            steps=slider_steps,
        )],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.0, y=-0.05, xanchor="left", yanchor="top",
            pad=dict(t=20, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=150, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode="immediate",
                    )],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        meta=dict(
            n_frames=n_global,
            panels=[
                dict(
                    label=p["label"],
                    states=p["states"],
                    bump_idx=p["bump_idx"],
                )
                for p in panels
            ],
        ),
    )
    # Hide axis ticks/grid on every subplot.
    for i in range(1, n_panels + 1):
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, row=1, col=i)
    return fig


# ---------------------------------------------------------------------------
# render_gif: framework-canonical trajectory GIF (Farama pixel-art)
# ---------------------------------------------------------------------------


def render_gif(
    run: "RunHandle",
    out_path: "str | Path | None" = None,
    init: tuple[int, int] | None = None,
    fps: int = 6,
    title: str | None = None,
) -> Path:
    """Roll out one episode under ``run.ckpt_path`` and save a GIF.

    Uses the in-package Farama-style pixel-art renderer
    (``rl_basics.render``). A wall-bump frame (i.e. a step where the agent
    tried to move but didn't) gets a yellow/orange burst overlay; free
    movement is rendered without the burst, so the highlight remains a
    clear "I tried and got blocked" signal.

    Parameters
    ----------
    run : RunHandle
        A trained seed (carries ckpt_path + config).
    out_path : str | Path | None, optional
        Destination .gif. If None, writes to
        ``<run.exp_dir>/<run.seed>_traj.gif`` so it sits next to the run.
    init : (y, x), optional
        Force the start cell.
    fps : int
        GIF frame rate.
    title : str, optional
        Top-left text title; defaults to ``"<exp_name> seed=<n> · return=…"``.

    Returns
    -------
    Path
        The written GIF path.
    """
    from rl_basics import render

    states, bumps, cums, env = _rollout_one_traj(run, init)
    walls = env.walls.cpu().numpy()
    goal_xy = (env.goal_y, env.goal_x)
    start_xy = states[0]

    if out_path is None:
        exp_dir = Path(run.exp_dir)
        out_path = exp_dir / f"seed_{run.seed}_traj.gif"
    out_path = Path(out_path)

    if title is None:
        title = (
            f"{Path(run.exp_dir).name} · seed {run.seed} · "
            f"return={cums[-1]:+.2f} · {len(states) - 1} steps · "
            f"{len(bumps)} wall bumps"
        )

    return render.render_traj_gif(
        walls=walls,
        goal_xy=goal_xy,
        start_xy=start_xy,
        states=states,
        bump_idx=bumps,
        out_path=out_path,
        title=title,
        fps=fps,
    )


# ---------------------------------------------------------------------------
# Task 23: value_map (heatmap of V_phi(s))
# ---------------------------------------------------------------------------


def value_map(run: "RunHandle") -> "plt.Figure":
    """Static heatmap of V_phi(s) across the 17x17 grid.

    Loads ``value_net`` from ``run.ckpt_path`` (raises if absent), runs a
    forward pass over all 289 cell indices, reshapes to (17, 17), and
    renders a matplotlib heatmap with a colorbar. Wall cells are masked to
    NaN so the heatmap reads as the underlying value field; the goal cell
    is annotated with a star.
    """
    import matplotlib.pyplot as plt
    import torch

    from rl_basics.env import FourRoomsTL
    from rl_basics.models import ValueNetwork

    ckpt = torch.load(run.ckpt_path, map_location="cpu", weights_only=True)
    if not (isinstance(ckpt, dict) and "value_net" in ckpt):
        raise ValueError(
            f"value_map requires a ckpt with a 'value_net' key; got "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)} "
            f"at {run.ckpt_path}"
        )

    cfg = run.config if isinstance(run.config, dict) else {}
    hidden = int(cfg.get("hidden", 64))

    env = FourRoomsTL(batch_size=1, seed=0)  # only used for walls/goal/size
    n_states = env.n_states
    size = env.size

    vnet = ValueNetwork(n_states=n_states, hidden=hidden)
    vnet.load_state_dict(ckpt["value_net"])
    vnet.eval()

    with torch.no_grad():
        all_states = torch.arange(n_states, dtype=torch.long)
        v = vnet(all_states).cpu().numpy().reshape(size, size)

    walls = env.walls.cpu().numpy()
    v_masked = np.array(v, dtype=float)
    v_masked[walls] = np.nan

    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="black")
        im = ax.imshow(v_masked, cmap=cmap, origin="upper")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")
        # Goal star
        ax.plot(env.goal_x, env.goal_y, marker="*", color="gold",
                markersize=14, markeredgecolor="black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("V_phi(s)")
        fig.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        raise


# ---------------------------------------------------------------------------
# Task 24: advantage_overlay (the "killer plot")
# ---------------------------------------------------------------------------


def advantage_overlay(
    run: "RunHandle",
    n_traj: int = 1,
) -> "plt.Figure":
    """The killer plot. Pick ONE rollout, render the FourRooms grid 5 times,
    color the visited cells by:

        [1] reward r_t          — single bright +1 cell at the goal,
        [2] return-to-go G_t    — bright near goal, fading away,
        [3] advantage (vanilla) A_t = G_t,
        [4] advantage (value)   A_t = G_t - V_phi(s_t),
        [5] advantage (batch)   A_t = G_t - mu_t (mean across batch at time t).

    The advantage panel from SPEC §5.7 is split into 3 side-by-side sub-views
    (vanilla / value / batch). To make the batch-baseline sub-view non-trivial
    we roll out with B=4 and visualise traj 0; with B=1, mu_t == G_t makes
    every batch-baseline cell zero (which is the spec point — but only after
    explanation). Walls are masked black; the goal is starred.

    Requires a checkpoint with both 'policy' and 'value_net' state-dicts
    (raises ValueError otherwise — same convention as value_map).

    Parameters
    ----------
    run
        RunHandle whose ``ckpt_path`` carries 'policy' + 'value_net'.
    n_traj
        Number of rollouts to render. Currently only ``1`` is supported
        (multi-traj averaging is out of scope).

    Returns
    -------
    plt.Figure
        Figure with 5 imshow panels in a row.
    """
    import matplotlib.pyplot as plt
    import torch

    from rl_basics.env import FourRoomsTL
    from rl_basics.models import MLPPolicy, ValueNetwork
    from rl_basics.utils import (
        _compute_advantage_with_batch_baseline_ref,
        _compute_advantage_with_value_baseline_ref,
        _compute_returns_to_go_ref,
        sample_action,
    )

    if n_traj != 1:
        raise NotImplementedError(
            f"advantage_overlay currently supports n_traj=1 only, got {n_traj}"
        )

    # ---- 1. Load ckpt (policy + value_net) -------------------------------
    ckpt = torch.load(run.ckpt_path, map_location="cpu", weights_only=True)
    if not (isinstance(ckpt, dict) and "policy" in ckpt and "value_net" in ckpt):
        raise ValueError(
            f"advantage_overlay requires a ckpt with both 'policy' and "
            f"'value_net' keys; got "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)} "
            f"at {run.ckpt_path}"
        )

    cfg = run.config if isinstance(run.config, dict) else {}
    hidden = int(cfg.get("hidden", 64))
    gamma = float(cfg.get("gamma", 1.0))
    seed = int(getattr(run, "seed", 0) or 0)

    # ---- 2. Vectorized rollout (B=4 so batch baseline is non-trivial) ----
    B = 4
    env = FourRoomsTL(batch_size=B, seed=seed)
    s = env.reset()

    policy = MLPPolicy(env.n_states, env.n_actions, hidden=hidden)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    vnet = ValueNetwork(n_states=env.n_states, hidden=hidden)
    vnet.load_state_dict(ckpt["value_net"])
    vnet.eval()

    torch.manual_seed(seed)
    T = env.max_steps
    S_buf = torch.zeros((B, T), dtype=torch.long)
    R_buf = torch.zeros((B, T), dtype=torch.float32)
    M_buf = torch.zeros((B, T), dtype=torch.bool)

    for t in range(T):
        was_alive = ~env.done
        with torch.no_grad():
            logits = policy(s)
            a = sample_action(logits)
        next_s, r, _ = env.step(a)
        S_buf[:, t] = s
        R_buf[:, t] = r
        M_buf[:, t] = was_alive
        s = next_s

    # ---- 3. Returns-to-go and the 3 advantages ---------------------------
    G = _compute_returns_to_go_ref(R_buf, M_buf, gamma=gamma)  # (B, T)
    with torch.no_grad():
        V = vnet(S_buf.reshape(-1)).reshape(B, T)
    A_vanilla = G  # _compute_advantage_vanilla_ref is identity
    A_value = _compute_advantage_with_value_baseline_ref(G, V, M_buf)
    A_batch = _compute_advantage_with_batch_baseline_ref(G, M_buf)

    # ---- 4. Pick traj 0 and unwind to per-cell quantities ---------------
    b = 0
    mask_b = M_buf[b].numpy().astype(bool)
    states_b = S_buf[b].numpy()
    rewards_b = R_buf[b].numpy()
    G_b = G[b].numpy()
    Av_b = A_vanilla[b].numpy()
    Avv_b = A_value[b].numpy()
    Ab_b = A_batch[b].numpy()

    size = env.size
    walls = env.walls.cpu().numpy()

    def _grid_from_per_step(values: np.ndarray) -> np.ndarray:
        """Build a (size, size) NaN grid; for each alive step t, set
        grid[y, x] = values[t]. Later visits overwrite earlier ones — this
        is fine for the spec ("color = some quantity at visited cell").
        Walls stay NaN (masked black via cmap.set_bad).
        """
        grid = np.full((size, size), np.nan, dtype=float)
        for t in range(len(values)):
            if not mask_b[t]:
                continue
            sy, sx = int(states_b[t]) // size, int(states_b[t]) % size
            grid[sy, sx] = float(values[t])
        # Re-mask walls (defensive — visited cells should never be walls).
        grid[walls] = np.nan
        return grid

    grids = {
        "reward": _grid_from_per_step(rewards_b),
        "G": _grid_from_per_step(G_b),
        "A_vanilla": _grid_from_per_step(Av_b),
        "A_value": _grid_from_per_step(Avv_b),
        "A_batch": _grid_from_per_step(Ab_b),
    }

    # ---- 5. Render: 5 panels in a row -----------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    try:
        viridis = plt.get_cmap("viridis").copy()
        viridis.set_bad(color="black")
        rdbu = plt.get_cmap("RdBu_r").copy()
        rdbu.set_bad(color="black")

        panels = [
            ("reward",    "r_t",            grids["reward"],   viridis),
            ("G",         "G_t",            grids["G"],        viridis),
            ("A_vanilla", "A_t (vanilla)",  grids["A_vanilla"], rdbu),
            ("A_value",   "A_t (value)",    grids["A_value"],  rdbu),
            ("A_batch",   "A_t (batch)",    grids["A_batch"],  rdbu),
        ]

        for ax, (key, title, grid, cmap) in zip(axes, panels):
            # Symmetrise diverging cmaps about 0 so sign reads clearly.
            finite = grid[np.isfinite(grid)]
            if cmap is rdbu and finite.size > 0:
                vmax = float(np.max(np.abs(finite)))
                if vmax == 0.0:
                    vmax = 1.0
                im = ax.imshow(grid, cmap=cmap, origin="upper",
                               vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(grid, cmap=cmap, origin="upper")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.plot(env.goal_x, env.goal_y, marker="*", color="gold",
                    markersize=12, markeredgecolor="black")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)

        fig.suptitle(f"advantage_overlay  (run seed={seed}, traj 0 of B={B})",
                     y=1.02)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        raise


# ---------------------------------------------------------------------------
# Task 25: grad_var_panel (4-panel returns / tr_var / tr_var-log / SNR)
# ---------------------------------------------------------------------------


def _read_grad_var_records(run: "RunHandle") -> "list[dict]":
    """Read raw per-update records (including nested grad_var) from a seed's metrics.jsonl."""
    import json as _json

    if not run.metrics_path.exists():
        return []
    out: list[dict] = []
    with open(run.metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


_PANEL_KEYS = ("return", "tr_var", "tr_var_log", "snr")


def grad_var_panel(
    rgs: "list[RunGroup]",
    panels: "tuple[str, ...] | list[str] | None" = None,
) -> "plt.Figure":
    """Grad-variance overview across one or more RunGroups.

    Default = the full 4-panel layout (mirrors the prototype's
    ``plot_tl_gv.py``). Pass ``panels=("return", "tr_var")`` for the compact
    2-panel view recommended for the lab notebook (return curves alongside
    tr(Var[g]) on a linear y-axis).

    Panel keys:
      * ``"return"``     — episode return mean ± std over seeds
      * ``"tr_var"``     — tr(Var[ĝ]) on a linear y-axis
      * ``"tr_var_log"`` — tr(Var[ĝ]) on a log y-axis
      * ``"snr"``        — SNR = ||E[ĝ]||² / tr(Var[ĝ])

    RunGroups missing grad_var data simply don't contribute to the variance
    panels.
    """
    import matplotlib.pyplot as plt

    rgs = list(rgs)
    if panels is None:
        panels = _PANEL_KEYS
    panels = tuple(panels)
    bad = [p for p in panels if p not in _PANEL_KEYS]
    if bad:
        raise ValueError(
            f"unknown panel keys {bad!r}; valid keys are {_PANEL_KEYS}"
        )
    if not panels:
        raise ValueError("grad_var_panel needs at least one panel")

    n_panels = len(panels)
    if n_panels == 1:
        nrows, ncols = 1, 1
        figsize = (5.5, 3.6)
    elif n_panels == 2:
        nrows, ncols = 1, 2
        figsize = (10, 3.8)
    elif n_panels <= 4:
        nrows, ncols = 2, 2
        figsize = (10, 7)
    else:
        nrows = (n_panels + 1) // 2
        ncols = 2
        figsize = (10, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    panel_axes: dict[str, "plt.Axes"] = {}
    for ax, key in zip(axes_flat, panels):
        panel_axes[key] = ax
    for unused in axes_flat[n_panels:]:
        unused.set_visible(False)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    ])

    try:
        need_var = any(k in panels for k in ("tr_var", "tr_var_log"))
        need_snr = "snr" in panels

        for idx, rg in enumerate(rgs):
            color = color_cycle[idx % len(color_cycle)]
            label = rg.cfg.advantage_kind

            # ---- "return" panel: episode return curves (mean ± std) ----
            if "return" in panels:
                ret_curves: list[np.ndarray] = []
                for h in rg.runs:
                    df = h.metrics
                    if df is None or len(df) == 0 or "ep_return_mean" not in df.columns:
                        continue
                    ret_curves.append(np.asarray(df["ep_return_mean"].to_numpy(), dtype=float))
                if ret_curves:
                    n = min(c.size for c in ret_curves)
                    if n > 0:
                        mat = np.stack([c[:n] for c in ret_curves], axis=0)
                        mean = mat.mean(axis=0)
                        std = mat.std(axis=0, ddof=0)
                        x = np.arange(n)
                        ax = panel_axes["return"]
                        ax.plot(x, mean, color=color, label=str(label))
                        ax.fill_between(x, mean - std, mean + std,
                                        alpha=0.2, color=color)

            # ---- variance / snr panels: pull grad_var from raw JSONL ----
            seeds_tv: list[list[tuple[int, float]]] = []
            seeds_snr: list[list[tuple[int, float]]] = []
            if need_var or need_snr:
                for h in rg.runs:
                    tv_pts: list[tuple[int, float]] = []
                    snr_pts: list[tuple[int, float]] = []
                    for rec in _read_grad_var_records(h):
                        gv = rec.get("grad_var")
                        if not isinstance(gv, dict):
                            continue
                        upd = rec.get("upd")
                        if upd is None:
                            continue
                        if need_var and "tr_var" in gv:
                            tv_pts.append((int(upd), float(gv["tr_var"])))
                        if need_snr and "snr" in gv:
                            snr_pts.append((int(upd), float(gv["snr"])))
                    if tv_pts:
                        seeds_tv.append(tv_pts)
                    if snr_pts:
                        seeds_snr.append(snr_pts)

            if seeds_tv:
                shared = sorted(
                    set.intersection(*[{u for u, _ in s} for s in seeds_tv])
                )
                if shared:
                    mat = np.array(
                        [[dict(s)[u] for u in shared] for s in seeds_tv],
                        dtype=float,
                    )
                    mean = mat.mean(axis=0)
                    std = mat.std(axis=0, ddof=0)
                    if "tr_var" in panels:
                        ax = panel_axes["tr_var"]
                        ax.plot(shared, mean, color=color, label=str(label))
                        ax.fill_between(shared, mean - std, mean + std,
                                        alpha=0.2, color=color)
                    if "tr_var_log" in panels:
                        ax = panel_axes["tr_var_log"]
                        ax.plot(shared, mean, color=color, label=str(label))
                        ax.fill_between(shared, np.maximum(mean - std, 1e-12),
                                        mean + std, alpha=0.2, color=color)

            if seeds_snr and "snr" in panels:
                shared = sorted(
                    set.intersection(*[{u for u, _ in s} for s in seeds_snr])
                )
                if shared:
                    mat = np.array(
                        [[dict(s)[u] for u in shared] for s in seeds_snr],
                        dtype=float,
                    )
                    mean = mat.mean(axis=0)
                    std = mat.std(axis=0, ddof=0)
                    ax = panel_axes["snr"]
                    ax.plot(shared, mean, color=color, label=str(label))
                    ax.fill_between(shared, mean - std, mean + std,
                                    alpha=0.2, color=color)

        if "return" in panels:
            ax = panel_axes["return"]
            ax.set_title("episode return")
            ax.set_xlabel("update")
            ax.set_ylabel("ep_return_mean")
        if "tr_var" in panels:
            ax = panel_axes["tr_var"]
            ax.set_title("tr(Var[g])  (linear)")
            ax.set_xlabel("update")
            ax.set_ylabel("tr_var")
        if "tr_var_log" in panels:
            ax = panel_axes["tr_var_log"]
            ax.set_title("tr(Var[g])  (log)")
            ax.set_xlabel("update")
            ax.set_ylabel("tr_var")
            ax.set_yscale("log")
        if "snr" in panels:
            ax = panel_axes["snr"]
            ax.set_title("SNR = ||E[g]||² / tr(Var[g])")
            ax.set_xlabel("update")
            ax.set_ylabel("snr")

        for ax in panel_axes.values():
            if ax.get_legend_handles_labels()[1]:
                ax.legend(loc="best", frameon=False, fontsize=8)

        fig.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        raise


# ---------------------------------------------------------------------------
# Task 26: show_tensor (walkthrough primitive)
# ---------------------------------------------------------------------------


def show_tensor(x, name: str = "") -> None:
    """Print ``shape/dtype/min/max`` line + show a 3x8 corner of the tensor
    as a styled DataFrame for walkthrough cells. Returns ``None``.

    Best-effort styled display — falls back silently if pandas styling
    isn't available (e.g. headless tests). The printed line is always
    emitted so tests can capture it.
    """
    import torch

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float()
        shape = tuple(arr.shape)
        dtype = x.dtype
        try:
            xmin = float(arr.min().item()) if arr.numel() > 0 else float("nan")
            xmax = float(arr.max().item()) if arr.numel() > 0 else float("nan")
        except (RuntimeError, ValueError):
            xmin, xmax = float("nan"), float("nan")
        np_arr = arr.numpy()
    else:
        np_arr = np.asarray(x)
        shape = tuple(np_arr.shape)
        dtype = np_arr.dtype
        if np_arr.size > 0:
            xmin = float(np.nanmin(np_arr))
            xmax = float(np.nanmax(np_arr))
        else:
            xmin, xmax = float("nan"), float("nan")

    label = f"{name}: " if name else ""
    print(
        f"{label}shape={shape} dtype={dtype} "
        f"min={xmin:.3g} max={xmax:.3g}"
    )

    # Best-effort top-left preview (Jupyter rich display when available).
    try:
        import pandas as pd
        from IPython.display import display  # type: ignore

        if np_arr.ndim == 0:
            return None
        if np_arr.ndim == 1:
            preview = np_arr[:8].reshape(1, -1)
        else:
            flat2d = np_arr.reshape(np_arr.shape[0], -1)
            preview = flat2d[:3, :8]
        df = pd.DataFrame(preview)
        try:
            styled = df.style.background_gradient(cmap="viridis").format("{:.3g}")
            display(styled)
        except Exception:
            display(df)
    except Exception:
        pass
    return None
