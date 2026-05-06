"""Tests for rl_basics.reset() — SPEC §8.

Stress suite (100-cycle) skipped per user directive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rl_basics.utils import reset


def _spawn_sleeper() -> subprocess.Popen:
    """Spawn a long-sleep child we can kill safely."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_reset_kills_tracked_pids(tmp_path: Path):
    """Register a fake long-sleep child via pids.json; reset; child gone."""
    runs_dir = tmp_path / "runs"
    exp_dir = runs_dir / "exp"
    exp_dir.mkdir(parents=True)

    proc = _spawn_sleeper()
    try:
        pids_path = exp_dir / "pids.json"
        pids_path.write_text(json.dumps({"0": proc.pid}))

        # Sanity: child is alive.
        assert _pid_alive(proc.pid)

        reset(verbose=False, runs_dir=runs_dir)

        # Popen.wait() reaps the zombie left after SIGTERM/SIGKILL.
        # If reset() didn't actually kill it, this hangs → use a timeout.
        try:
            rc = proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
            pytest.fail("reset() failed to kill tracked PID within 3s")
        # Killed by signal → negative returncode in Popen convention.
        assert rc != 0, f"sleeper exited cleanly (rc={rc}); reset() didn't signal it"
    finally:
        # Belt-and-suspenders cleanup so a failing test doesn't leak the sleeper.
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait(timeout=2)


def test_reset_removes_runs_dir(tmp_path: Path):
    """tmp_path/runs/exp/ with files vanishes; .gitkeep preserved."""
    runs_dir = tmp_path / "runs"
    exp_dir = runs_dir / "exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text("{}")
    (exp_dir / "pids.json").write_text(json.dumps({}))
    gitkeep = runs_dir / ".gitkeep"
    gitkeep.write_text("")

    reset(verbose=False, runs_dir=runs_dir)

    assert not exp_dir.exists(), "exp dir should be removed"
    assert runs_dir.exists(), "runs/ itself should still exist"
    assert gitkeep.exists(), ".gitkeep should be preserved"


def test_reset_idempotent(tmp_path: Path):
    """5 calls in a row, no exception."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for _ in range(5):
        reset(verbose=False, runs_dir=runs_dir)
