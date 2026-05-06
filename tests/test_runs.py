import json
import os
import re
import time

import pandas as pd
import pytest

from rl_basics.runs import RunConfig, _cfg_hash


def test_runconfig_defaults_match_spec():
    c = RunConfig()
    assert c.name is None
    assert c.env == "FourRoomsTL"
    assert c.advantage_kind == "vanilla"
    assert c.use_value_baseline is False
    assert c.batch_size == 16
    assert c.n_updates == 400
    assert c.lr == 3e-3
    assert c.hidden == 64
    assert c.gamma == 1.0
    assert c.seeds == [0, 1, 2]
    assert c.log_every == 10
    assert c.grad_var_every == 0


def test_auto_name_from_hash_stable():
    c1 = RunConfig(advantage_kind="vanilla", batch_size=16)
    c2 = RunConfig(advantage_kind="vanilla", batch_size=16)
    assert c1.auto_name() == c2.auto_name(), "same cfg → same name"
    c3 = RunConfig(advantage_kind="vanilla", batch_size=32)
    assert c1.auto_name() != c3.auto_name(), "different batch → different name"
    # Hash must ignore the name field itself
    c4 = RunConfig(advantage_kind="vanilla", batch_size=16, name="my-run")
    assert _cfg_hash(c1) == _cfg_hash(c4), "name should not affect hash"


def test_auto_name_pattern():
    pattern = r"^(vanilla|value|batch)_b\d+_g[\d.]+_h\d+_[a-f0-9]{6}$"
    cases = [
        RunConfig(advantage_kind="vanilla"),
        RunConfig(advantage_kind="value", use_value_baseline=True),
        RunConfig(advantage_kind="batch"),
    ]
    for c in cases:
        assert re.match(pattern, c.auto_name()), (
            f"name doesn't match pattern: {c.auto_name()}"
        )


def test_post_init_rejects_inconsistent_value_baseline():
    # use_value_baseline must be True iff advantage_kind == 'value'.
    with pytest.raises(ValueError):
        RunConfig(advantage_kind="vanilla", use_value_baseline=True)
    with pytest.raises(ValueError):
        RunConfig(advantage_kind="value", use_value_baseline=False)
    with pytest.raises(ValueError):
        RunConfig(advantage_kind="bogus")
    # The 3 consistent cases must NOT raise:
    RunConfig(advantage_kind="vanilla", use_value_baseline=False)
    RunConfig(advantage_kind="value", use_value_baseline=True)
    RunConfig(advantage_kind="batch", use_value_baseline=False)


def test_resolve_name_uses_supplied():
    c = RunConfig(name="my_special_run")
    assert c.resolve_name() == "my_special_run"
    c2 = RunConfig()  # no name
    assert c2.resolve_name() == c2.auto_name()


def test_train_smoke(tmp_path):
    """Smoke test: invoke train.py via subprocess, verify metrics.jsonl ends at upd=4."""
    import json
    import subprocess
    import sys

    import cloudpickle

    from rl_basics.student import compute_advantage_vanilla

    cfg = RunConfig(
        advantage_kind="vanilla",
        use_value_baseline=False,
        batch_size=4,
        n_updates=5,
        lr=3e-3,
        hidden=16,  # tiny for speed
        seeds=[0],
        log_every=1,
    )
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({k: v for k, v in cfg.__dict__.items()}))

    advfn_path = tmp_path / "advfn.pkl"
    advfn_path.write_bytes(cloudpickle.dumps(compute_advantage_vanilla))

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    env_for_subproc = {**os.environ, "OMP_NUM_THREADS": "1"}

    result = subprocess.run(
        [
            sys.executable, "-m", "rl_basics.train",
            "--cfg", str(cfg_path.resolve()),
            "--advfn-pkl", str(advfn_path.resolve()),
            "--seed", "0",
            "--out", str(out_dir.resolve()),
        ],
        env=env_for_subproc,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    metrics_path = out_dir / "metrics.jsonl"
    assert metrics_path.exists(), f"no metrics.jsonl at {metrics_path}"
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) == 5, f"expected 5 metric lines, got {len(lines)}"
    last = json.loads(lines[-1])
    assert last["upd"] == 4, f"last upd = {last['upd']}, expected 4"
    # Check ckpt and per-seed config snapshot were saved
    assert (out_dir / "ckpt.pt").exists()
    assert (out_dir / "config.json").exists()


def test_launch_spawns_one_proc_per_seed(tmp_path):
    """3 seeds → 3 entries in pids.json (some may be null while pending)."""
    from rl_basics.runs import launch

    cfg = RunConfig(
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=3,
        hidden=16,
        seeds=[0, 1, 2],
    )
    rg = launch(cfg, workers=3, runs_dir=tmp_path)
    try:
        pids_path = rg.exp_dir / "pids.json"
        assert pids_path.exists()
        pids = json.loads(pids_path.read_text())
        assert set(pids.keys()) == {"0", "1", "2"}
        # With workers=3 and 3 seeds, all 3 must be live (PID populated).
        assert all(v is not None for v in pids.values())
    finally:
        rg.kill()


def test_launch_writes_run_dirs(tmp_path):
    """Directory structure matches SPEC §5.5."""
    from rl_basics.runs import launch

    cfg = RunConfig(
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=2,
        hidden=16,
        seeds=[0, 1],
    )
    rg = launch(cfg, workers=2, runs_dir=tmp_path)
    try:
        assert rg.exp_dir.exists()
        assert (rg.exp_dir / "config.json").exists()
        assert (rg.exp_dir / "advfn.pkl").exists()
        assert (rg.exp_dir / "pids.json").exists()
        for seed in [0, 1]:
            seed_dir = rg.exp_dir / f"seed_{seed}"
            assert seed_dir.exists(), f"missing seed_{seed} dir"
            assert (seed_dir / "event.log").exists()
            # metrics.jsonl is created by the subprocess; may not exist immediately.
    finally:
        rg.kill()


def test_kill_terminates_all(tmp_path):
    """All children die after kill()."""
    from rl_basics.runs import launch

    cfg = RunConfig(
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=1000,  # Long enough that subprocess won't finish naturally.
        hidden=16,
        seeds=[0, 1],
    )
    rg = launch(cfg, workers=2, runs_dir=tmp_path)
    pids_before = [proc.pid for proc in rg._live.values()]
    assert len(pids_before) == 2

    rg.kill()
    # All Popens must be terminated (poll() returns non-None).
    for proc in rg._live.values() if rg._live else []:
        assert proc.poll() is not None
    # Wait briefly for OS cleanup, then verify no surviving children.
    time.sleep(0.5)
    # Use os.kill(pid, 0) to check; raises OSError if process gone.
    for pid in pids_before:
        with pytest.raises((ProcessLookupError, OSError)):
            os.kill(pid, 0)


# ---------------------------------------------------------------------------
# Task 16: RunGroup.snapshot — incremental JSONL reader (SPEC §5.5, H.10)
# ---------------------------------------------------------------------------


def _make_runs_skeleton(tmp_path, seeds=(0, 1)):
    """Build a RunGroup with no live procs but real seed dirs/metrics paths.

    We don't actually launch — we just synthesize the run structure on disk
    so we can hand-write metrics.jsonl files and exercise snapshot() in
    isolation. Returns (rg, seed_dirs).
    """
    from rl_basics.runs import RunGroup, RunHandle

    cfg = RunConfig(
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=3,
        hidden=16,
        seeds=list(seeds),
    )
    exp_dir = tmp_path / cfg.resolve_name()
    exp_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    for s in seeds:
        seed_dir = exp_dir / f"seed_{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        handles.append(
            RunHandle(
                seed=s,
                exp_dir=exp_dir,
                pid=None,
                config={**cfg.__dict__, "seed": s},
                ckpt_path=seed_dir / "ckpt.pt",
                event_log_path=seed_dir / "event.log",
                metrics_path=seed_dir / "metrics.jsonl",
            )
        )
    rg = RunGroup(
        cfg=cfg,
        exp_dir=exp_dir,
        run_handles=handles,
        live={},
        pending=[],
        cfg_path=exp_dir / "config.json",
        advfn_path=exp_dir / "advfn.pkl",
        workers=2,
    )
    return rg, [h.metrics_path for h in handles]


def test_snapshot_returns_long_dataframe(tmp_path):
    """snapshot() returns long-format [seed, update, metric_name, value]."""
    rg, [m0, m1] = _make_runs_skeleton(tmp_path, seeds=(0, 1))
    rec0 = {"upd": 0, "wall": 0.1, "ep_return_mean": -19.0, "p_loss": 0.5}
    rec1 = {"upd": 0, "wall": 0.2, "ep_return_mean": -18.0, "p_loss": 0.6}
    m0.write_text(json.dumps(rec0) + "\n")
    m1.write_text(json.dumps(rec1) + "\n")

    df = rg.snapshot()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"seed", "update", "metric_name", "value"}
    # Each line has 3 non-upd fields → 3 rows; 2 seeds → 6 rows total.
    assert len(df) == 6
    # Spot-check a value.
    row = df[(df.seed == 0) & (df.metric_name == "ep_return_mean")]
    assert len(row) == 1
    assert float(row.iloc[0]["value"]) == pytest.approx(-19.0)
    assert int(row.iloc[0]["update"]) == 0


def test_snapshot_incremental(tmp_path):
    """Second call returns ONLY rows for newly-appended lines."""
    rg, [m0, m1] = _make_runs_skeleton(tmp_path, seeds=(0, 1))
    m0.write_text(json.dumps({"upd": 0, "p_loss": 0.5}) + "\n")
    m1.write_text(json.dumps({"upd": 0, "p_loss": 0.6}) + "\n")

    df1 = rg.snapshot()
    assert len(df1) == 2  # 1 metric per line × 2 seeds

    # Append more lines.
    with open(m0, "a") as f:
        f.write(json.dumps({"upd": 1, "p_loss": 0.4}) + "\n")
    df2 = rg.snapshot()
    assert len(df2) == 1, f"expected only new row, got {len(df2)}"
    assert int(df2.iloc[0]["update"]) == 1
    assert int(df2.iloc[0]["seed"]) == 0

    # Third call with no further writes: empty.
    df3 = rg.snapshot()
    assert len(df3) == 0


def test_snapshot_skips_partial_lines(tmp_path):
    """Half-flushed line is skipped; appears once newline lands (H.10)."""
    rg, [m0] = _make_runs_skeleton(tmp_path, seeds=(0,))
    # Write partial line (no trailing newline).
    m0.write_text('{"upd": 0, "wal')

    df = rg.snapshot()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0, "partial line must be skipped"

    # Now finish writing the line.
    with open(m0, "a") as f:
        f.write('l": 0.1, "p_loss": 0.5}\n')
    df2 = rg.snapshot()
    # ep_return_mean isn't here, only wall+p_loss → 2 rows.
    assert len(df2) == 2
    metric_names = set(df2["metric_name"])
    assert metric_names == {"wall", "p_loss"}


# ---------------------------------------------------------------------------
# Task 17: wait + is_running + RunHandle.metrics (SPEC §5.5)
# ---------------------------------------------------------------------------


def test_wait_returns_when_subprocesses_exit(tmp_path):
    """wait() blocks until all children exit, then returns."""
    from rl_basics.runs import launch

    cfg = RunConfig(
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=3,  # tiny — will finish quickly
        hidden=16,
        seeds=[0, 1],
        log_every=1,
    )
    rg = launch(cfg, workers=2, runs_dir=tmp_path)
    try:
        # Capture Popens BEFORE wait() — wait() reaps via `del self._live[seed]`,
        # so iterating rg._live afterwards would be vacuously empty.
        procs_before = list(rg._live.values())
        assert len(procs_before) > 0, "no procs to wait on"
        rg.wait(timeout=60)
        for proc in procs_before:
            assert proc.poll() is not None, "proc still alive after wait()"
    finally:
        rg.kill()


def test_is_running_flips_correctly(tmp_path):
    """is_running True before wait, False after."""
    from rl_basics.runs import launch

    cfg = RunConfig(
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=3,
        hidden=16,
        seeds=[0, 1],
        log_every=1,
    )
    rg = launch(cfg, workers=2, runs_dir=tmp_path)
    try:
        # Immediately after launch, at least one proc should be live.
        assert rg.is_running() is True
        rg.wait(timeout=60)
        assert rg.is_running() is False
    finally:
        rg.kill()


def test_runhandle_metrics_lazy_dataframe(tmp_path):
    """handle.metrics returns a DataFrame after the seed completes."""
    from rl_basics.runs import launch

    cfg = RunConfig(
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=3,
        hidden=16,
        seeds=[0, 1],
        log_every=1,
    )
    rg = launch(cfg, workers=2, runs_dir=tmp_path)
    try:
        rg.wait(timeout=60)
        for handle in rg.runs:
            df = handle.metrics
            assert isinstance(df, pd.DataFrame)
            assert "upd" in df.columns
            assert "ep_return_mean" in df.columns
            assert len(df) == 3  # n_updates rows (log_every=1)
            assert int(df["upd"].iloc[-1]) == 2
    finally:
        rg.kill()
