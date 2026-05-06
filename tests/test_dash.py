"""Tests for `rl_basics.dash_app` — Task 28 (live JSONL polling + kill button).

Per the implementation plan's 3-test contract:
  1. `test_dash_subprocess_starts_and_serves_root` — subprocess starts; `/`
     returns HTTP 200 within 5 s; `stop()` cleans up.
  2. `test_dashboard_picks_up_new_jsonl_lines` — appending a line to a tmp
     metrics.jsonl is reflected by the read helper. (We exercise the read
     helper directly rather than the live HTTP loop to keep the test fast.)
  3. `test_kill_button_endpoint_calls_kill` — the kill callback drops a
     `.kill_requested` marker file under `runs/<exp>/`.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Subprocess starts and serves '/'
# ---------------------------------------------------------------------------

def test_dash_subprocess_starts_and_serves_root(tmp_path):
    """`start()` must launch a child Dash process that serves HTTP within 5s.

    The PID must be tracked at /tmp/rl_basics_dash.pid so `stop()` can find
    and terminate it cleanly.
    """
    from rl_basics import dash_app

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    port = dash_app.start(port=None, runs_dir=runs_dir)
    try:
        assert isinstance(port, int) and port > 0

        # PID file must exist with a live PID.
        pid_path = Path("/tmp/rl_basics_dash.pid")
        assert pid_path.exists(), "expected PID file at /tmp/rl_basics_dash.pid"
        pid = int(pid_path.read_text().strip())
        assert pid > 0

        # Poll '/' for HTTP 200 within 5 s.
        deadline = time.time() + 5.0
        last_err = None
        ok = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=0.5
                ) as resp:
                    if resp.status == 200:
                        ok = True
                        break
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                time.sleep(0.1)
        assert ok, f"Dash didn't serve / within 5s; last_err={last_err!r}"
    finally:
        dash_app.stop()

    # After stop(), pid file is gone.
    assert not Path("/tmp/rl_basics_dash.pid").exists()


# ---------------------------------------------------------------------------
# 2. Read helper picks up newly-appended JSONL lines
# ---------------------------------------------------------------------------

def test_dashboard_picks_up_new_jsonl_lines(tmp_path):
    """After appending a line to a metrics.jsonl, the read helper must
    return the new row. We exercise `_read_runs_dir` directly (no HTTP) so
    the test is fast and deterministic.
    """
    from rl_basics import dash_app

    runs_dir = tmp_path / "runs"
    exp_dir = runs_dir / "vanilla_demo" / "seed_0"
    exp_dir.mkdir(parents=True)
    mfile = exp_dir / "metrics.jsonl"
    mfile.write_text(
        json.dumps({"upd": 0, "wall": 0.0, "ep_return_mean": 0.1}) + "\n"
    )

    df1 = dash_app._read_runs_dir(runs_dir)
    assert len(df1) == 1
    assert "vanilla_demo" in set(df1["exp"])
    assert df1["upd"].tolist() == [0]

    # Append one more line.
    with mfile.open("a") as f:
        f.write(
            json.dumps({"upd": 1, "wall": 1.0, "ep_return_mean": 0.2}) + "\n"
        )

    df2 = dash_app._read_runs_dir(runs_dir)
    assert len(df2) == 2
    assert sorted(df2["upd"].tolist()) == [0, 1]


# ---------------------------------------------------------------------------
# 3. Kill button drops a kill marker file
# ---------------------------------------------------------------------------

def test_kill_button_endpoint_calls_kill(tmp_path):
    """The kill callback must drop `runs/<exp>/.kill_requested`.

    The framework's `RunGroup.kill()` lives in the parent kernel process; the
    Dash subprocess can only signal a kill request via a marker file on disk.
    """
    from rl_basics import dash_app

    runs_dir = tmp_path / "runs"
    exp = "vanilla_demo"
    (runs_dir / exp / "seed_0").mkdir(parents=True)

    marker = dash_app._request_kill(runs_dir, exp)
    assert marker.exists()
    assert marker == runs_dir / exp / ".kill_requested"

    # Idempotent: calling again is fine.
    marker2 = dash_app._request_kill(runs_dir, exp)
    assert marker2 == marker
    assert marker2.exists()

    # Refuses unknown exp (no such directory).
    with pytest.raises((FileNotFoundError, ValueError)):
        dash_app._request_kill(runs_dir, "no_such_exp")
