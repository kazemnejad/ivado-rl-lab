"""Stress suite for ``rl_basics.runs`` — SPEC §7.5 + Appendix H.6/H.9/H.10.

Four tests:

1. ``test_50_consecutive_launches`` — 50 sequential ``launch`` + ``wait`` cycles
   with cfg=vanilla, batch=16, n_updates=10, hidden=16. All 50 runs must
   produce a valid ``metrics.jsonl`` ending at upd=9. No zombie children
   left behind.

2. ``test_no_zombies_after_kill`` — long-running cfg, capture child PIDs,
   ``RunGroup.kill()``, then verify every PID is gone (``os.kill(pid, 0)``
   raises ``ProcessLookupError`` and ``pgrep -P <pytest_pid>`` does not list
   any of those PIDs).

3. ``test_jsonl_writer_reader_race`` — concurrent writer thread + reader
   poll-loop hammering ``RunGroup.snapshot``. Zero ``JSONDecodeError``;
   every written ``upd`` value (0..99) eventually appears in the reader's
   collected rows.

4. ``test_cloudpickle_advfn_closure_roundtrip`` — pickle a function that
   closes over a Python local; deserialize in a fresh subprocess; confirm
   identical output (Appendix H.9 pre-emptive guard for the eventual
   notebook-global / closure flow).

Run via:
    .venv/bin/python -m pytest tests/stress/test_runs_stress.py -v -m stress
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import cloudpickle
import pytest

from rl_basics.runs import RunConfig, RunGroup, RunHandle, launch


# Module-level constant — simulates a notebook global captured by an
# advantage-fn closure (SPEC Appendix H.9 pre-emptive case).
_HIDDEN_HYPERPARAM = 0.42


def _pid_gone(pid: int) -> bool:
    """Return True if `pid` is no longer a live process we can signal.

    PermissionError => PID exists but is owned by another user (recycled);
    we treat that as "gone from our perspective" — same as ProcessLookupError.
    """
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return True
    return False


# ---------------------------------------------------------------------------
# Test 1: 50 consecutive launches
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_50_consecutive_launches(tmp_path):
    """50× launch + wait. Every metrics.jsonl must end at upd=9.

    Wall budget: aim for < 5 min on a healthy box. Allow up to 10 min
    headroom on shared/contended hardware before we declare regression.
    """
    N = 50
    overall_start = time.time()
    last_lines: list[str] = []

    for i in range(N):
        iter_dir = tmp_path / f"iter_{i:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        cfg = RunConfig(
            advantage_kind="vanilla",
            batch_size=16,
            n_updates=10,
            hidden=16,
            seeds=[0],
        )
        rg = launch(cfg, runs_dir=iter_dir)
        try:
            rg.wait(timeout=60)
            # Should have completed naturally — no live procs.
            assert not rg.is_running(), (
                f"iter {i}: rg still running after wait(timeout=60)"
            )
            handle = rg.runs[0]
            assert handle.metrics_path.exists(), (
                f"iter {i}: metrics.jsonl missing at {handle.metrics_path}"
            )
            # Read last non-empty line.
            with open(handle.metrics_path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f if ln.endswith("\n") and ln.strip()]
            assert lines, f"iter {i}: metrics.jsonl is empty"
            last_rec = json.loads(lines[-1].strip())
            assert last_rec.get("upd") == 9, (
                f"iter {i}: last upd={last_rec.get('upd')!r} (expected 9); "
                f"line={lines[-1]!r}"
            )
            last_lines.append(lines[-1].strip())
        finally:
            # Idempotent — no-op if wait() finished cleanly. Guards both the
            # timeout path and assertion-failure paths from leaking subprocs
            # into the next iteration (which would contaminate the pgrep
            # check at the end of the test).
            rg.kill()

    elapsed = time.time() - overall_start
    print(
        f"\n[test_50_consecutive_launches] {N} launches in {elapsed:.1f}s "
        f"(avg {elapsed / N:.2f}s/launch)"
    )

    # No leaked child procs of *this* test process.
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Returncode 1 = no matches; 0 = matches found. Anything we own should
        # have been waited on by rg.wait() above.
        leaked = [pid for pid in out.stdout.split() if pid.strip()]
        assert not leaked, (
            f"after 50 launches, pgrep -P {os.getpid()} listed leftover "
            f"children: {leaked}"
        )
    except FileNotFoundError:
        pytest.skip("pgrep unavailable — skipping zombie scan")

    # Soft wall-budget warning (the assertion just keeps the test honest on
    # contended hardware; the spec target is 5 min).
    assert elapsed < 600, (
        f"50 launches took {elapsed:.1f}s (> 600s budget) — likely a "
        "subprocess hang or unexpected per-launch overhead regression"
    )


# ---------------------------------------------------------------------------
# Test 2: no zombies after kill
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_no_zombies_after_kill(tmp_path):
    """Launch a long-running cfg, kill, verify every captured PID is gone.

    The cfg sets n_updates=10000 so the children never finish naturally
    inside the test window — they only exit via the SIGTERM/SIGKILL path
    in ``RunGroup.kill``.
    """
    cfg = RunConfig(
        advantage_kind="vanilla",
        batch_size=16,
        n_updates=10000,
        hidden=16,
        seeds=[0, 1, 2],
    )
    rg = launch(cfg, workers=3, runs_dir=tmp_path)
    # Give Popen a moment to exec the python interpreter (pid is set
    # immediately, but a freshly forked child may still be initializing).
    time.sleep(0.5)
    captured_pids = [h.pid for h in rg.runs if h.pid is not None]
    assert len(captured_pids) == 3, (
        f"expected 3 live PIDs after launch, got {captured_pids}"
    )
    # Sanity: every captured PID is alive right now.
    for pid in captured_pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pytest.fail(
                f"pid {pid} already gone before kill() — child died early?"
            )

    rg.kill()
    # Bounded poll for OS reaping after SIGTERM/SIGKILL — replaces a blind
    # sleep. Exits the loop as soon as all PIDs are gone, or after 3 s.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if all(_pid_gone(pid) for pid in captured_pids):
            break
        time.sleep(0.05)

    # 1) Each captured PID must be gone (or, if PIDs got recycled, at least
    #    not visible as our child anymore — _pid_gone handles both cases via
    #    ProcessLookupError and PermissionError).
    still_alive = [pid for pid in captured_pids if not _pid_gone(pid)]
    assert not still_alive, (
        f"after kill(), these PIDs are still alive: {still_alive}"
    )

    # 2) pgrep -P <pytest> must not list any of our captured PIDs.
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=5,
        )
        listed = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
        leaked = [pid for pid in captured_pids if pid in listed]
        assert not leaked, (
            f"pgrep -P {os.getpid()} still lists captured PIDs {leaked} "
            f"after kill(); full pgrep output: {out.stdout!r}"
        )
    except FileNotFoundError:
        pytest.skip("pgrep unavailable — skipping orphan-children scan")

    # 3) is_running() must report False after kill().
    assert not rg.is_running(), "rg.is_running() True after kill()"


# ---------------------------------------------------------------------------
# Test 3: writer/reader race on metrics.jsonl
# ---------------------------------------------------------------------------
def _make_runs_skeleton(tmp_path: Path, seed: int = 0):
    """Build a RunGroup with one synthetic seed (no real subprocess).

    Mirrors the helper used in tests/test_runs.py so we can exercise
    snapshot() against a hand-written metrics.jsonl.
    """
    cfg = RunConfig(
        advantage_kind="vanilla",
        batch_size=4,
        n_updates=3,
        hidden=16,
        seeds=[seed],
    )
    exp_dir = tmp_path / cfg.resolve_name()
    exp_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = exp_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    handle = RunHandle(
        seed=seed,
        exp_dir=exp_dir,
        pid=None,
        config={**cfg.__dict__, "seed": seed},
        ckpt_path=seed_dir / "ckpt.pt",
        event_log_path=seed_dir / "event.log",
        metrics_path=seed_dir / "metrics.jsonl",
    )
    rg = RunGroup(
        cfg=cfg,
        exp_dir=exp_dir,
        run_handles=[handle],
        live={},
        pending=[],
        cfg_path=exp_dir / "config.json",
        advfn_path=exp_dir / "advfn.pkl",
        workers=1,
    )
    return rg, handle.metrics_path


@pytest.mark.stress
def test_jsonl_writer_reader_race(tmp_path):
    """Concurrent writer + reader: zero JSONDecodeError, every line seen.

    Writer thread writes 100 rows, one at a time, with explicit ``flush``
    after each (matching H.10 4-line discipline) and a 10 ms sleep between
    rows. The reader polls ``rg.snapshot`` every 10 ms and aggregates seen
    ``upd`` values into a set. Test passes when:

    * Writer ran to completion without crashing.
    * Reader collected ``{0, 1, ..., 99}`` (every row eventually visible).
    * Zero ``JSONDecodeError`` raised inside ``rg.snapshot``.

    Note: ``rg.snapshot`` already swallows ``JSONDecodeError`` per H.10, so
    "zero" is verified by patching ``json.loads`` to count raises (we wrap
    the original to track exceptions internally).
    """
    rg, metrics_path = _make_runs_skeleton(tmp_path, seed=0)

    # Hook: count JSONDecodeError raises during snapshot reading.
    decode_errors: list[Exception] = []
    orig_loads = json.loads

    def counting_loads(s, *args, **kwargs):
        try:
            return orig_loads(s, *args, **kwargs)
        except json.JSONDecodeError as e:
            decode_errors.append(e)
            raise

    # Patch the json.loads symbol used inside runs.py via the runs module.
    import rl_basics.runs as runs_mod

    # IMPORTANT: runs.py uses `import json` (module-level), so patching the
    # attribute on the module's `json` binding intercepts all json.loads calls
    # inside snapshot(). If runs.py is ever refactored to `from json import loads`,
    # this patch silently stops intercepting and decode_errors stays empty
    # (false-pass). Re-verify this patch if runs.py import style changes.
    runs_mod.json.loads = counting_loads  # type: ignore[attr-defined]
    # Defense: catch the "patch didn't take" case immediately.
    assert runs_mod.json.loads is counting_loads, (
        "monkey-patch on runs_mod.json.loads did not take effect — "
        "runs.py import style may have changed"
    )

    N_LINES = 100
    writer_done = threading.Event()
    writer_exc: list[BaseException] = []

    def writer():
        try:
            with open(metrics_path, "a", encoding="utf-8") as f:
                for i in range(N_LINES):
                    rec = {
                        "upd": i,
                        "wall": float(i) * 0.01,
                        "ep_return_mean": -19.0 + 0.1 * i,
                        "p_loss": 0.5 - 0.001 * i,
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    time.sleep(0.01)
        except BaseException as e:  # noqa: BLE001
            writer_exc.append(e)
        finally:
            writer_done.set()

    seen_upds: set[int] = set()

    try:
        t = threading.Thread(target=writer, daemon=True)
        t.start()

        deadline = time.time() + 30.0
        while time.time() < deadline:
            df = rg.snapshot()
            if not df.empty:
                for u in df["update"].tolist():
                    seen_upds.add(int(u))
            if writer_done.is_set() and len(seen_upds) == N_LINES:
                break
            time.sleep(0.01)

        # Final drain after writer finishes (in case writer beat the loop's
        # last poll).
        t.join(timeout=5.0)
        for _ in range(5):
            df = rg.snapshot()
            if not df.empty:
                for u in df["update"].tolist():
                    seen_upds.add(int(u))
            if len(seen_upds) == N_LINES:
                break
            time.sleep(0.05)
    finally:
        # Restore json.loads no matter what.
        runs_mod.json.loads = orig_loads  # type: ignore[attr-defined]

    assert not writer_exc, f"writer thread crashed: {writer_exc[0]!r}"
    assert not decode_errors, (
        f"snapshot() saw {len(decode_errors)} JSONDecodeError(s); "
        "H.10 reader discipline regression"
    )
    missing = sorted(set(range(N_LINES)) - seen_upds)
    assert not missing, (
        f"reader missed {len(missing)} rows out of {N_LINES}: "
        f"first missing={missing[:5]!r}"
    )
    print(
        f"\n[test_jsonl_writer_reader_race] saw all {N_LINES} rows; "
        f"0 JSONDecodeError; writer ok"
    )


# ---------------------------------------------------------------------------
# Test 4: cloudpickle closure roundtrip across subprocess (H.9)
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_cloudpickle_advfn_closure_roundtrip(tmp_path):
    """Pickle advantage fns closing over (a) a local, (b) a module-global.

    Models the eventual flow where a learner defines an advantage_fn in a
    notebook cell that closes over a notebook-global. cloudpickle should
    capture the closure variable so the fresh subprocess gets the SAME
    output without raising NameError.

    Two sub-cases:
      * **local**: closure captures a function-local. cloudpickle handles
        this via cell objects — structurally easiest path.
      * **module-global**: closure captures a module-level constant
        (`_HIDDEN_HYPERPARAM`). This is the SPEC Appendix H.9 / plan-line-328
        failure mode — notebook globals may resolve differently in a fresh
        subprocess, so cloudpickle must serialize the value, not the name.
    """
    inputs = [1.0, 2.0, 3.0, 4.0]

    def _roundtrip(fn, label: str, expected):
        """Pickle fn, load + call in a fresh subprocess, assert equality."""
        pkl_path = tmp_path / f"advfn_{label}.pkl"
        pkl_path.write_bytes(cloudpickle.dumps(fn))
        code = textwrap.dedent(
            f"""
            import json, sys
            import cloudpickle
            with open({str(pkl_path)!r}, "rb") as f:
                fn = cloudpickle.load(f)
            out = fn({inputs!r})
            sys.stdout.write(json.dumps(out))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"[{label}] subprocess crashed: rc={result.returncode}\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        sub_out = json.loads(result.stdout)
        assert sub_out == pytest.approx(expected), (
            f"[{label}] subprocess output {sub_out!r} != in-process "
            f"{expected!r}; cloudpickle failed to capture the closure"
        )
        return sub_out

    # ---- Sub-case 1: local closure --------------------------------------
    scale = 0.42  # local closure variable

    def my_advantage_fn_local(returns):
        # Closes over `scale` (function-local). No module-level import needed.
        return [r * scale for r in returns]

    expected_local = my_advantage_fn_local(inputs)
    sub_local = _roundtrip(my_advantage_fn_local, "local", expected_local)

    # ---- Sub-case 2: module-global closure (H.9 pre-emptive case) -------
    def my_advantage_fn_global(returns):
        # Closes over `_HIDDEN_HYPERPARAM` (module-level constant). This
        # mimics a notebook-global captured by a user-defined advantage fn.
        return [r * _HIDDEN_HYPERPARAM for r in returns]

    expected_global = my_advantage_fn_global(inputs)
    sub_global = _roundtrip(my_advantage_fn_global, "global", expected_global)

    print(
        f"\n[test_cloudpickle_advfn_closure_roundtrip] "
        f"local: in-process={expected_local} subprocess={sub_local}; "
        f"global: in-process={expected_global} subprocess={sub_global}"
    )
