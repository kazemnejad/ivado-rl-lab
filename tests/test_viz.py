"""Tests for `rl_basics.viz` — Tasks 20 (`plot_curves`) and 21 (`play_traj`).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # noqa: E402  — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from rl_basics.models import MLPPolicy  # noqa: E402
from rl_basics.runs import RunConfig, RunGroup, RunHandle  # noqa: E402
from rl_basics.viz import (  # noqa: E402
    advantage_overlay,
    grad_var_panel,
    play_traj,
    play_traj_compare,
    plot_curves,
    show_tensor,
    value_map,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mini_run"


def _load_run_group(exp_name: str) -> RunGroup:
    """Build a RunGroup from a fixture exp dir without spawning processes."""
    exp_dir = FIXTURE_DIR / exp_name
    cfg_dict = json.loads((exp_dir / "config.json").read_text())
    cfg = RunConfig(**cfg_dict)
    handles = []
    for seed in cfg.seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        handles.append(
            RunHandle(
                seed=seed,
                exp_dir=exp_dir,
                pid=None,
                config={**cfg_dict, "seed": seed},
                ckpt_path=seed_dir / "ckpt.pt",
                event_log_path=seed_dir / "event.log",
                metrics_path=seed_dir / "metrics.jsonl",
            )
        )
    # Bypass __init__ — we don't need live procs, only the .runs view.
    rg = RunGroup.__new__(RunGroup)
    rg.cfg = cfg
    rg.exp_dir = exp_dir
    rg._run_handles = handles
    rg._live = {}
    rg._pending = []
    return rg


def test_plot_curves_returns_figure():
    """plot_curves should return a Figure with a mean line and a std band."""
    rg = _load_run_group("vanilla_b4_g1.0_h8_aaaaaa")
    fig = plot_curves(rg)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    lines = [c for c in ax.get_children() if isinstance(c, Line2D)]
    bands = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
    # Filter out axis spines/etc — only data lines have non-empty xdata.
    data_lines = [ln for ln in lines if len(ln.get_xdata()) > 0]
    assert len(data_lines) >= 1, "expected >= 1 mean line"
    assert len(bands) >= 1, "expected >= 1 std fill_between band"
    plt.close(fig)


def test_overlay_two_run_groups_distinct_colors():
    """Two overlaid RunGroups should plot with distinct colors."""
    rg1 = _load_run_group("vanilla_b4_g1.0_h8_aaaaaa")
    rg2 = _load_run_group("value_b4_g1.0_h8_bbbbbb")
    fig = plot_curves([rg1, rg2])
    ax = fig.axes[0]
    data_lines = [
        c for c in ax.get_children()
        if isinstance(c, Line2D) and len(c.get_xdata()) > 0
    ]
    assert len(data_lines) >= 2, "expected >= 2 mean lines (one per RunGroup)"
    color_a = data_lines[0].get_color()
    color_b = data_lines[1].get_color()
    assert color_a != color_b, f"colors must differ: {color_a!r} vs {color_b!r}"
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 21: play_traj (Plotly + slider)
# ---------------------------------------------------------------------------


def _make_dummy_run(tmp_path: Path, *, hidden: int = 8, seed: int = 0) -> RunHandle:
    """Build a minimal RunHandle with a tiny untrained MLPPolicy ckpt.

    No subprocess; ckpt is a torch.save({'policy': state_dict}) under
    tmp_path so play_traj can load_state_dict from it.
    """
    torch.manual_seed(seed)
    policy = MLPPolicy(n_states=289, n_actions=4, hidden=hidden)
    exp_dir = tmp_path / "exp"
    seed_dir = exp_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    ckpt_path = seed_dir / "ckpt.pt"
    torch.save({"policy": policy.state_dict()}, ckpt_path)
    cfg_dict = {"hidden": hidden, "seed": seed}
    return RunHandle(
        seed=seed,
        exp_dir=exp_dir,
        pid=None,
        config=cfg_dict,
        ckpt_path=ckpt_path,
        event_log_path=seed_dir / "event.log",
        metrics_path=seed_dir / "metrics.jsonl",
    )


def _short_episode_env(monkeypatch):
    """Cap FourRoomsTL.max_steps to 10 so test rollouts terminate fast.

    An untrained policy almost never reaches the goal, so without this cap
    every play_traj test would render 400 sprite frames (~6s each).
    """
    from rl_basics import env as env_mod
    monkeypatch.setattr(env_mod.FourRoomsTL, "max_steps", 10, raising=True)


def test_play_traj_returns_plotly_figure(tmp_path, monkeypatch):
    """play_traj returns a plotly.go.Figure with frames + a slider."""
    import plotly.graph_objects as go

    _short_episode_env(monkeypatch)
    run = _make_dummy_run(tmp_path)
    fig = play_traj(run)
    assert isinstance(fig, go.Figure)
    assert hasattr(fig, "frames")
    assert len(fig.frames) > 0, "expected >= 1 animation frame"
    sliders = fig.layout.sliders
    assert sliders is not None and len(sliders) >= 1, "expected a slider"


def test_bump_idx_are_valid_frame_indices(tmp_path, monkeypatch):
    """bump_idx values are valid step indices (< T) for the rendered traj.

    PoC's `rollout_real_policy` uses an old pkl/MLPPolicy interface tied to
    the prototype env, so we can't directly co-call it here. Structural
    check: every bump idx is a non-negative int strictly less than the
    number of rendered frames.
    """
    _short_episode_env(monkeypatch)
    run = _make_dummy_run(tmp_path)
    fig = play_traj(run)
    # play_traj attaches the bump_idx list to the figure for inspection.
    bump_idx = fig.layout.meta.get("bump_idx") if fig.layout.meta else None
    assert bump_idx is not None, "expected bump_idx surfaced via layout.meta"
    n_frames = len(fig.frames)
    for b in bump_idx:
        assert isinstance(b, int)
        assert 0 <= b < n_frames, f"bump idx {b} out of range [0, {n_frames})"


def test_init_override_forces_start_cell(tmp_path, monkeypatch):
    """Passing init=(2,3) makes the rendered first state == (2,3)."""
    _short_episode_env(monkeypatch)
    run = _make_dummy_run(tmp_path)
    fig = play_traj(run, init=(2, 3))
    states = fig.layout.meta.get("states") if fig.layout.meta else None
    assert states is not None, "expected states surfaced via layout.meta"
    assert states[0] == (2, 3), f"expected start (2,3), got {states[0]}"


# ---------------------------------------------------------------------------
# Task 22: play_traj_compare (3-up subplot grid)
# ---------------------------------------------------------------------------


def _make_dummy_rg(tmp_path: Path, name: str, *, hidden: int = 8, seed: int = 0) -> RunGroup:
    """Build a single-seed RunGroup pointing at a tiny untrained MLPPolicy ckpt."""
    exp_dir = tmp_path / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = exp_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    policy = MLPPolicy(n_states=289, n_actions=4, hidden=hidden)
    torch.save({"policy": policy.state_dict()}, seed_dir / "ckpt.pt")
    cfg = RunConfig(
        name=name,
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=1,
        hidden=hidden,
        seeds=[seed],
    )
    handle = RunHandle(
        seed=seed,
        exp_dir=exp_dir,
        pid=None,
        config={"hidden": hidden, "seed": seed},
        ckpt_path=seed_dir / "ckpt.pt",
        event_log_path=seed_dir / "event.log",
        metrics_path=seed_dir / "metrics.jsonl",
    )
    rg = RunGroup.__new__(RunGroup)
    rg.cfg = cfg
    rg.exp_dir = exp_dir
    rg._run_handles = [handle]
    rg._live = {}
    rg._pending = []
    return rg


def test_play_traj_compare_smoke(tmp_path, monkeypatch):
    """play_traj_compare returns a Figure with >=1 frame and >=2 panel traces."""
    import plotly.graph_objects as go

    _short_episode_env(monkeypatch)
    rg1 = _make_dummy_rg(tmp_path, "a", seed=0)
    rg2 = _make_dummy_rg(tmp_path, "b", seed=0)
    fig = play_traj_compare([rg1, rg2])
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) >= 1, "expected >= 1 animation frame"
    assert len(fig.data) >= 2, "expected >= 2 panel traces"


def test_play_traj_compare_same_init_across_panels(tmp_path, monkeypatch):
    """init=(2,3) -> every panel's first state == (2,3)."""
    _short_episode_env(monkeypatch)
    rg1 = _make_dummy_rg(tmp_path, "a", seed=0)
    rg2 = _make_dummy_rg(tmp_path, "b", seed=1)
    fig = play_traj_compare([rg1, rg2], init=(2, 3))
    panels = fig.layout.meta.get("panels") if fig.layout.meta else None
    assert panels is not None, "expected per-panel meta in layout.meta['panels']"
    assert len(panels) == 2
    for p in panels:
        assert p["states"][0] == (2, 3), f"panel start {p['states'][0]} != (2,3)"


# ---------------------------------------------------------------------------
# Task 23: value_map (17x17 heatmap of V_phi)
# ---------------------------------------------------------------------------


def _make_dummy_run_with_value(tmp_path: Path, *, hidden: int = 8, seed: int = 0) -> RunHandle:
    """Build a RunHandle whose ckpt contains both 'policy' and 'value_net' keys."""
    from rl_basics.models import ValueNetwork

    torch.manual_seed(seed)
    policy = MLPPolicy(n_states=289, n_actions=4, hidden=hidden)
    value_net = ValueNetwork(n_states=289, hidden=hidden)
    exp_dir = tmp_path / "exp_v"
    seed_dir = exp_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    ckpt_path = seed_dir / "ckpt.pt"
    torch.save(
        {"policy": policy.state_dict(), "value_net": value_net.state_dict()},
        ckpt_path,
    )
    return RunHandle(
        seed=seed,
        exp_dir=exp_dir,
        pid=None,
        config={"hidden": hidden, "seed": seed},
        ckpt_path=ckpt_path,
        event_log_path=seed_dir / "event.log",
        metrics_path=seed_dir / "metrics.jsonl",
    )


def test_value_map_returns_17x17_heatmap(tmp_path):
    """value_map should imshow a (17, 17) heatmap of V_phi(s)."""
    from matplotlib.image import AxesImage

    run = _make_dummy_run_with_value(tmp_path)
    fig = value_map(run)
    assert isinstance(fig, plt.Figure)
    images = [a for ax in fig.axes for a in ax.get_images() if isinstance(a, AxesImage)]
    assert len(images) >= 1, "expected at least one AxesImage on the figure"
    arr = images[0].get_array()
    assert arr.shape == (17, 17), f"expected (17,17) image, got {arr.shape}"
    plt.close(fig)


def test_value_map_colorbar_present(tmp_path):
    """value_map should attach a colorbar."""
    run = _make_dummy_run_with_value(tmp_path)
    fig = value_map(run)
    # A colorbar adds an extra Axes to the figure.
    assert len(fig.axes) >= 2, "expected >= 2 axes (heatmap + colorbar)"
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 25: grad_var_panel (4-panel)
# ---------------------------------------------------------------------------


def _make_grad_var_rg(tmp_path: Path, name: str, *, seed: int = 0) -> RunGroup:
    """Build a RunGroup with a hand-rolled metrics.jsonl carrying grad_var entries."""
    exp_dir = tmp_path / name
    seed_dir = exp_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = seed_dir / "metrics.jsonl"
    rows = []
    for upd in range(5):
        rows.append({
            "upd": upd,
            "wall": float(upd),
            "ep_return_mean": 0.1 + 0.05 * upd,
            "p_loss": 1.0 - 0.05 * upd,
            "grad_var": {
                "tr_var": 1.0 / (1 + upd),
                "snr": 0.1 + 0.02 * upd,
                "mean_norm_sq": 0.5,
            },
        })
    with open(metrics_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cfg = RunConfig(
        name=name,
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=5,
        hidden=8,
        seeds=[seed],
    )
    handle = RunHandle(
        seed=seed,
        exp_dir=exp_dir,
        pid=None,
        config={"hidden": 8, "seed": seed},
        ckpt_path=seed_dir / "ckpt.pt",
        event_log_path=seed_dir / "event.log",
        metrics_path=metrics_path,
    )
    rg = RunGroup.__new__(RunGroup)
    rg.cfg = cfg
    rg.exp_dir = exp_dir
    rg._run_handles = [handle]
    rg._live = {}
    rg._pending = []
    return rg


def test_grad_var_panel_returns_4_axes(tmp_path):
    rg = _make_grad_var_rg(tmp_path, "gv_a")
    fig = grad_var_panel([rg])
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4, f"expected 4 axes, got {len(fig.axes)}"
    plt.close(fig)


def test_grad_var_panel_log_y_on_panel_3(tmp_path):
    rg = _make_grad_var_rg(tmp_path, "gv_b")
    fig = grad_var_panel([rg])
    assert fig.axes[2].get_yscale() == "log", "panel 3 should use log y-scale"
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 26: show_tensor (walkthrough primitive)
# ---------------------------------------------------------------------------


def test_show_tensor_prints_shape_dtype_min_max(capsys):
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    out = show_tensor(x, name="x")
    assert out is None
    captured = capsys.readouterr().out
    for needle in ("shape=", "dtype=", "min=", "max="):
        assert needle in captured, f"expected {needle!r} in stdout, got: {captured!r}"


# ---------------------------------------------------------------------------
# Task 24: advantage_overlay (the killer plot — 3 panels + 3 advantage subviews)
# ---------------------------------------------------------------------------


def test_advantage_overlay_three_panels(tmp_path, monkeypatch):
    """advantage_overlay returns a Figure with >= 5 axes (reward + G_t +
    3 advantage sub-views).
    """
    _short_episode_env(monkeypatch)
    run = _make_dummy_run_with_value(tmp_path)
    fig = advantage_overlay(run)
    assert isinstance(fig, plt.Figure)
    # 5 main panels (each may add a colorbar axes — so total >= 5).
    assert len(fig.axes) >= 5, (
        f"expected >= 5 axes (reward + G + 3 advantage subviews), "
        f"got {len(fig.axes)}"
    )
    plt.close(fig)


def test_advantage_overlay_advantage_panel_has_three_sub_views(tmp_path, monkeypatch):
    """The 3 advantage sub-views (vanilla / value / batch) render distinct
    image arrays — i.e. they're not just the same heatmap copied 3 times.
    """
    from matplotlib.image import AxesImage

    _short_episode_env(monkeypatch)
    run = _make_dummy_run_with_value(tmp_path)
    fig = advantage_overlay(run)
    images = [
        a for ax in fig.axes for a in ax.get_images() if isinstance(a, AxesImage)
    ]
    assert len(images) >= 5, (
        f"expected >= 5 imshow images on the figure, got {len(images)}"
    )
    # Last 3 images correspond to the 3 advantage sub-views (vanilla, value,
    # batch). They should NOT all be identical — at minimum vanilla differs
    # from the value-baselined view (same finite cells, different numbers).
    arrs = [np.asarray(im.get_array(), dtype=float) for im in images[-3:]]

    def _diff(a, b):
        # Compare on cells where BOTH are finite (visited cells); if they
        # disagree anywhere, they're different.
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.any():
            return False
        return not np.allclose(a[finite], b[finite])

    assert _diff(arrs[0], arrs[1]), (
        "vanilla and value-baseline advantage panels look identical"
    )
    assert _diff(arrs[0], arrs[2]), (
        "vanilla and batch-baseline advantage panels look identical"
    )


def test_advantage_overlay_handles_short_trajectory_T_le_5(tmp_path, monkeypatch):
    """advantage_overlay survives a degenerate short trajectory (T<=5)."""
    from rl_basics import env as env_mod
    monkeypatch.setattr(env_mod.FourRoomsTL, "max_steps", 5, raising=True)
    run = _make_dummy_run_with_value(tmp_path)
    fig = advantage_overlay(run)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
