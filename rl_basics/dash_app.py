"""Live Dash dashboard — SPEC §5.8.

Tiny Plotly Dash app that reads ``runs/*/seed_*/metrics.jsonl`` from disk
every 2 s, displays per-`<exp_name>` learning curves (mean ± std band)
with a "kill" button per experiment.

Public API
----------
``start(port=None, runs_dir=None) -> int``
    Spawn the Dash app as a `subprocess.Popen` child and return the chosen
    port. Tracks the child PID at ``/tmp/rl_basics_dash.pid`` so a later
    ``stop()`` (or ``rl_basics.reset()``) can find and terminate it.

``stop() -> None``
    Kill the Dash subprocess via the PID file and remove the file.

Internal helpers (also unit-tested)
-----------------------------------
``_read_runs_dir(runs_dir) -> pd.DataFrame``
    Scan ``<runs_dir>/<exp>/seed_*/metrics.jsonl`` and return a long-format
    DataFrame with columns ``[exp, seed, upd, ep_return_mean, ...]``.

``_request_kill(runs_dir, exp) -> Path``
    Drop a ``<runs_dir>/<exp>/.kill_requested`` marker file. The parent
    kernel watches for this and invokes the matching ``RunGroup.kill()``.

The Dash subprocess is spawned via ``python -m rl_basics.dash_app
--port <p> --runs-dir <d>`` so it is fully self-contained — the parent
kernel only needs to supervise the child PID.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PID_FILE = Path("/tmp/rl_basics_dash.pid")


# ---------------------------------------------------------------------------
# Read helper — used both by the Dash callback and by unit tests.
# ---------------------------------------------------------------------------

def _read_runs_dir(runs_dir: Path) -> pd.DataFrame:
    """Scan ``<runs_dir>/<exp>/seed_*/metrics.jsonl`` -> long DataFrame.

    Implements the H.10 4-line JSONL discipline on the reader side:
      * skip lines that don't end with ``\\n`` (partial flushes);
      * wrap ``json.loads`` in ``try/except JSONDecodeError`` -> continue.

    Returns an empty DataFrame with no columns if no metrics files exist.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp = exp_dir.name
        for seed_dir in sorted(exp_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            mfile = seed_dir / "metrics.jsonl"
            if not mfile.exists():
                continue
            try:
                raw = mfile.read_text()
            except OSError:
                continue
            for line in raw.splitlines(keepends=True):
                if not line.endswith("\n"):
                    continue  # partial flush, skip
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = dict(rec)
                rec["exp"] = exp
                rec["seed"] = seed
                rows.append(rec)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Kill marker — Dash subprocess can only signal; parent kernel acts.
# ---------------------------------------------------------------------------

def _request_kill(runs_dir: Path, exp: str) -> Path:
    """Drop a ``.kill_requested`` marker under ``<runs_dir>/<exp>/``.

    Parent kernel polls these and invokes the matching ``RunGroup.kill()``
    in-process (the Dash subprocess can't reach those Python objects).

    Raises ``FileNotFoundError`` if ``<runs_dir>/<exp>`` doesn't exist.
    """
    runs_dir = Path(runs_dir)
    exp_dir = runs_dir / exp
    if not exp_dir.is_dir():
        raise FileNotFoundError(
            f"experiment directory not found: {exp_dir}"
        )
    marker = exp_dir / ".kill_requested"
    marker.touch()
    return marker


# ---------------------------------------------------------------------------
# Dash app builder + figure helper
# ---------------------------------------------------------------------------

def _build_figure(df: pd.DataFrame):
    """Mean ± std band per exp on ``ep_return_mean`` vs ``upd``.

    Falls back to an empty placeholder figure if no data yet.
    """
    import plotly.graph_objects as go  # local — avoid heavy import at parse

    fig = go.Figure()
    if df.empty or "ep_return_mean" not in df.columns:
        fig.update_layout(
            title="awaiting metrics…",
            xaxis_title="update",
            yaxis_title="ep_return_mean",
        )
        return fig

    for exp, sub in df.groupby("exp"):
        agg = (
            sub.groupby("upd")["ep_return_mean"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("upd")
        )
        agg["std"] = agg["std"].fillna(0.0)
        x = agg["upd"].tolist()
        m = agg["mean"].tolist()
        lo = (agg["mean"] - agg["std"]).tolist()
        hi = (agg["mean"] + agg["std"]).tolist()

        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=hi + lo[::-1],
                fill="toself",
                fillcolor="rgba(120,120,200,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{exp}-band",
            )
        )
        fig.add_trace(
            go.Scatter(x=x, y=m, mode="lines", name=exp)
        )
    fig.update_layout(
        title="learning curves (mean ± std)",
        xaxis_title="update",
        yaxis_title="ep_return_mean",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _build_app(runs_dir: Path):
    """Construct the Dash app object. Wired with a 2-second poll interval.

    Layout: sidebar of `<exp_name>` entries with kill buttons + main learning
    curve plot. Shows ``tr_var`` / ``snr`` columns when present.
    """
    from dash import Dash, Input, Output, State, dcc, html, no_update, ALL

    app = Dash(__name__)
    app.title = "rl_basics — live"

    app.layout = html.Div(
        [
            dcc.Store(id="runs-dir", data=str(runs_dir)),
            dcc.Interval(id="tick", interval=2000, n_intervals=0),
            html.Div(
                [
                    html.H3("experiments"),
                    html.Div(id="sidebar"),
                ],
                style={
                    "width": "260px",
                    "float": "left",
                    "padding": "12px",
                    "borderRight": "1px solid #ddd",
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="curves"),
                    html.Div(id="grad-var-line", style={"fontSize": "13px"}),
                ],
                style={"marginLeft": "280px", "padding": "12px"},
            ),
        ]
    )

    @app.callback(
        Output("sidebar", "children"),
        Output("curves", "figure"),
        Output("grad-var-line", "children"),
        Input("tick", "n_intervals"),
        State("runs-dir", "data"),
    )
    def _refresh(_n, runs_dir_str):
        rd = Path(runs_dir_str)
        df = _read_runs_dir(rd)

        # Sidebar: one row per exp with seed PIDs (best-effort) + kill button.
        rows = []
        if not df.empty:
            for exp, sub in df.groupby("exp"):
                seeds = sorted(sub["seed"].unique().tolist())
                rows.append(
                    html.Div(
                        [
                            html.Strong(exp),
                            html.Div(
                                f"seeds: {seeds}",
                                style={"fontSize": "11px", "color": "#555"},
                            ),
                            html.Button(
                                "kill",
                                id={"type": "kill-btn", "exp": exp},
                                n_clicks=0,
                                style={"marginTop": "4px"},
                            ),
                        ],
                        style={
                            "padding": "6px 0",
                            "borderBottom": "1px solid #eee",
                        },
                    )
                )
        else:
            rows.append(
                html.Div(
                    "no runs yet…",
                    style={"color": "#888", "fontStyle": "italic"},
                )
            )

        fig = _build_figure(df)

        # Grad-var line (tr_var/snr) if those columns exist.
        gv_text = ""
        if not df.empty and "tr_var" in df.columns:
            gv = df.dropna(subset=["tr_var"])
            if not gv.empty:
                last = (
                    gv.sort_values("upd")
                    .groupby("exp")
                    .tail(1)[["exp", "upd", "tr_var", "snr"]]
                )
                gv_text = " | ".join(
                    f"{r.exp}@{int(r.upd)}: tr_var={r.tr_var:.3g} "
                    f"snr={r.snr:.3g}"
                    for r in last.itertuples()
                )

        return rows, fig, gv_text

    @app.callback(
        Output({"type": "kill-btn", "exp": ALL}, "n_clicks"),
        Input({"type": "kill-btn", "exp": ALL}, "n_clicks"),
        State("runs-dir", "data"),
        prevent_initial_call=True,
    )
    def _on_kill(n_clicks_list, runs_dir_str):
        # Drop a kill marker for any button that was actually clicked.
        from dash import callback_context

        rd = Path(runs_dir_str)
        triggered = callback_context.triggered_id
        if triggered and isinstance(triggered, dict):
            exp = triggered.get("exp")
            if exp:
                try:
                    _request_kill(rd, exp)
                except FileNotFoundError:
                    pass
        return n_clicks_list  # pass-through (no_update would also work)

    return app


# ---------------------------------------------------------------------------
# Subprocess lifecycle
# ---------------------------------------------------------------------------

def _pick_port(preferred: int = 8050) -> int:
    """Find a free TCP port. Try ``preferred`` first, fall back to ephemeral.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    finally:
        s.close()


def start(port: int | None = None, runs_dir: Path | None = None) -> int:
    """Spawn the Dash app as a child subprocess; return chosen port.

    PID is recorded at ``/tmp/rl_basics_dash.pid`` so ``stop()`` (or
    ``rl_basics.reset()``) can terminate it later. If a previous PID file
    exists for a process that's still alive, we kill that one first to
    avoid double-binding the port.
    """
    if runs_dir is None:
        runs_dir = Path.cwd() / "runs"
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any stale dash from a previous run.
    if PID_FILE.exists():
        stop()

    if port is None:
        port = _pick_port(8050)

    cmd = [
        sys.executable,
        "-m",
        "rl_basics.dash_app",
        "--port",
        str(port),
        "--runs-dir",
        str(runs_dir),
    ]
    # Log to /tmp so failures are visible without polluting the cwd.
    log_path = Path("/tmp/rl_basics_dash.log")
    log_fh = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    return port


def stop() -> None:
    """Kill the Dash subprocess via the PID file (idempotent)."""
    if not PID_FILE.exists():
        return
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        try:
            PID_FILE.unlink()
        except OSError:
            pass
        return

    # SIGTERM, wait briefly, then SIGKILL the process group as a backstop.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        # Give it up to ~1s to exit cleanly before escalating.
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            continue  # still alive after SIGTERM -> escalate to SIGKILL
        break

    try:
        PID_FILE.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Subprocess entrypoint: `python -m rl_basics.dash_app --port P --runs-dir D`
# ---------------------------------------------------------------------------

def _serve(port: int, runs_dir: Path) -> None:
    """Build the Dash app and serve forever. Called by the child process."""
    app = _build_app(Path(runs_dir))
    # `host=127.0.0.1` keeps it bound to localhost; debug=False to avoid the
    # reloader spawning a second process (which would orphan our PID file).
    app.run(host="127.0.0.1", port=int(port), debug=False)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--runs-dir", type=str, required=True)
    args = parser.parse_args()
    _serve(args.port, Path(args.runs_dir))


if __name__ == "__main__":
    _main()
