"""Subprocess-per-seed run management — SPEC §5.5.

Shipped: RunConfig + auto-naming + launch + RunGroup (snapshot, wait,
is_running, kill) + RunHandle (lazy metrics property).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cloudpickle


@dataclass
class RunConfig:
    """Hyperparameter bundle for one experiment (one advantage rule, multiple seeds).

    All defaults match SPEC §3 (LOCKED Lab-1 hyperparameters) and SPEC §5.5.
    """

    name: str | None = None
    env: str = "FourRoomsTL"
    advantage_kind: str = "vanilla"  # 'vanilla' | 'value' | 'batch'
    use_value_baseline: bool = False  # True only for advantage_kind='value'
    batch_size: int = 16
    n_updates: int = 400
    lr: float = 3e-3
    hidden: int = 64
    gamma: float = 1.0
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    log_every: int = 10
    grad_var_every: int = 0
    grad_var_n_boot: int = 32

    def __post_init__(self):
        # Catch silent misconfigurations early. SPEC §5.5: use_value_baseline
        # must be True iff advantage_kind == 'value'. Without this guard,
        # train.py (Task 14) silently runs the wrong code path.
        if self.advantage_kind not in ("vanilla", "value", "batch"):
            raise ValueError(
                f"advantage_kind must be 'vanilla' | 'value' | 'batch', "
                f"got {self.advantage_kind!r}"
            )
        if self.use_value_baseline and self.advantage_kind != "value":
            raise ValueError(
                f"use_value_baseline=True requires advantage_kind='value', "
                f"got advantage_kind={self.advantage_kind!r}"
            )
        if self.advantage_kind == "value" and not self.use_value_baseline:
            raise ValueError(
                "advantage_kind='value' requires use_value_baseline=True"
            )

    def auto_name(self) -> str:
        """Deterministic name from cfg fields. Pattern:
        ``{advantage_kind}_b{batch}_g{gamma}_h{hidden}_{cfg_hash[:6]}``.
        """
        return (
            f"{self.advantage_kind}_b{self.batch_size}_g{self.gamma}"
            f"_h{self.hidden}_{_cfg_hash(self)[:6]}"
        )

    def resolve_name(self) -> str:
        """Return the user-supplied name or the auto-generated one."""
        return self.name if self.name is not None else self.auto_name()


def _cfg_hash(cfg: RunConfig) -> str:
    """Stable SHA1 hash of the cfg's serialized fields (excluding ``name``).

    Same cfg dict -> same hash; differing cfg -> different hash. The ``name``
    field is excluded so two configs that differ only in their (possibly auto-)
    name still produce the same hash — otherwise auto-name would be circular.
    """
    d = asdict(cfg)
    d.pop("name", None)
    serialized = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()


# ---------------------------------------------------------------------------
# Task 15: launch + RunGroup + RunHandle (SPEC §5.5)
# ---------------------------------------------------------------------------


@dataclass
class RunHandle:
    """Per-seed handle to a launched training subprocess.

    All paths are absolute; ``pid`` is None until the subprocess for this seed
    is admitted by the workers cap. ``metrics`` is a lazy property landing in
    Task 17 — for now it raises NotImplementedError.
    """

    seed: int
    exp_dir: Path
    pid: int | None
    config: dict
    ckpt_path: Path
    event_log_path: Path
    metrics_path: Path

    @property
    def metrics(self) -> Any:
        """Lazy-load this seed's ``metrics.jsonl`` as a DataFrame.

        One row per update; columns are the JSONL field names (``upd``,
        ``wall``, ``ep_return_mean``, ``p_loss``, ...). Returns an empty
        DataFrame if the file doesn't exist (still pending). Each access
        re-reads the file — this is intentional (cheap, eyeball-readable
        files; no caching to keep semantics simple). For incremental cross-
        seed reads use ``RunGroup.snapshot`` instead.
        """
        import pandas as pd

        if not self.metrics_path.exists():
            return pd.DataFrame()
        # H.10 reader discipline: skip partial lines + JSONDecodeError.
        records: list[dict] = []
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n"):
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
        return pd.DataFrame.from_records(records)


def _default_advantage_fn(cfg: RunConfig) -> Callable:
    """Resolve the framework's shipped advantage reference based on cfg.advantage_kind.

    Used when ``launch(..., advantage_fn=None)`` — falls back to the locked
    reference impls in utils.py so the framework can run without a learner-
    supplied function (e.g. CI sanity launches).
    """
    from rl_basics.utils import (
        _compute_advantage_vanilla_ref,
        _compute_advantage_with_batch_baseline_ref,
        _compute_advantage_with_value_baseline_ref,
    )
    return {
        "vanilla": _compute_advantage_vanilla_ref,
        "value": _compute_advantage_with_value_baseline_ref,
        "batch": _compute_advantage_with_batch_baseline_ref,
    }[cfg.advantage_kind]


class RunGroup:
    """Handle for a multi-seed training launch.

    Created by ``launch``. Owns the subprocess.Popen children, the event-log
    file handles, and the pids.json bookkeeping. Tasks 16/17 add ``snapshot``,
    ``wait``, ``is_running``, and lazy metrics loading.
    """

    def __init__(
        self,
        cfg: RunConfig,
        exp_dir: Path,
        run_handles: list[RunHandle],
        live: dict[int, subprocess.Popen],
        pending: list[int],
        cfg_path: Path,
        advfn_path: Path,
        workers: int,
    ):
        self.cfg = cfg
        self.exp_dir = exp_dir
        self._run_handles = run_handles
        self._live: dict[int, subprocess.Popen] = live
        self._pending: list[int] = pending
        self._cfg_path = cfg_path
        self._advfn_path = advfn_path
        self._workers = workers
        self._event_logs: dict[int, Any] = {}
        # Per-seed file position for incremental snapshot() reads (H.10).
        self._snapshot_pos: dict[int, int] = {h.seed: 0 for h in run_handles}

    # ---- public API ------------------------------------------------------

    @property
    def runs(self) -> list[RunHandle]:
        return list(self._run_handles)

    def snapshot(self) -> "pd.DataFrame":
        """Read each seed's metrics.jsonl up to current end-of-file.

        Returns a long-format ``pd.DataFrame`` with columns
        ``[seed, update, metric_name, value]``. Per-seed file positions are
        tracked across calls so subsequent invocations are incremental
        (only newly-appended bytes are parsed).

        Implements the H.10 4-line JSONL discipline on the reader side:
          * Skip lines that don't end with ``\\n`` (partial flushes).
          * Wrap ``json.loads`` in ``try/except JSONDecodeError`` → continue.

        Nested-dict fields (e.g. ``grad_var``) and the ``upd`` key itself
        are not melted into rows — only scalar metric fields become rows.
        """
        import pandas as pd

        records: list[dict] = []
        for handle in self._run_handles:
            seed = handle.seed
            path = handle.metrics_path
            if not path.exists():
                continue
            last_pos = self._snapshot_pos.get(seed, 0)
            with open(path, "rb") as f:
                f.seek(last_pos)
                chunk = f.read()
                end_pos = f.tell()
            if not chunk:
                continue
            text = chunk.decode("utf-8", errors="replace")
            # Split keeping awareness of the final partial line. If the chunk
            # ends with "\n" the trailing element after split is "" (skipped).
            lines = text.split("\n")
            consumed_bytes = 0
            # All but the last element are guaranteed to be complete (split
            # produces n+1 chunks for n separators); the last is partial iff
            # it's non-empty (no trailing newline).
            complete_lines = lines[:-1]
            for line in complete_lines:
                # Each consumed complete line takes len(line) + 1 bytes (the \n).
                consumed_bytes += len(line.encode("utf-8")) + 1
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                upd = rec.get("upd")
                if upd is None:
                    continue
                for k, v in rec.items():
                    if k == "upd":
                        continue
                    # Skip nested dicts (e.g. grad_var); scalars only here.
                    if isinstance(v, (dict, list)):
                        continue
                    records.append({
                        "seed": seed,
                        "update": int(upd),
                        "metric_name": k,
                        "value": v,
                    })
            # Advance position by exactly the bytes we accepted as complete.
            # If the chunk had a trailing partial fragment, those bytes stay
            # unread and will be re-read (now hopefully complete) next call.
            self._snapshot_pos[seed] = last_pos + consumed_bytes
            # Sanity: never exceed end_pos.
            if self._snapshot_pos[seed] > end_pos:
                self._snapshot_pos[seed] = end_pos

        cols = ["seed", "update", "metric_name", "value"]
        if not records:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame.from_records(records, columns=cols)

    def wait(self, timeout: float | None = None) -> None:
        """Block until all subprocesses exit (or timeout elapses).

        Drains pending seeds as live procs finish (so workers cap is
        honored across the wait). Surfaces negative return codes (OOM /
        SIGKILL) via a printed warning so silent failures don't go
        unnoticed.

        Plain polling loop — no threading, matching SPEC §5.5.
        """
        deadline = (time.time() + timeout) if timeout is not None else None
        warned: set[int] = set()
        while True:
            # Reap finished procs and admit pending in their place.
            finished = [
                seed for seed, proc in self._live.items()
                if proc.poll() is not None
            ]
            for seed in finished:
                proc = self._live[seed]
                rc = proc.returncode
                if rc is not None and rc < 0 and seed not in warned:
                    # Negative rc → killed by signal (OOM / SIGKILL / etc).
                    print(
                        f"[RunGroup.wait] seed={seed} pid={proc.pid} "
                        f"exited with signal {-rc} (likely OOM or external "
                        f"kill); check {self.exp_dir}/seed_{seed}/event.log",
                        flush=True,
                    )
                    warned.add(seed)
                # Drop from live so admit_pending can spawn a replacement.
                del self._live[seed]
                # Close that seed's event log handle.
                fh = self._event_logs.pop(seed, None)
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
            # Spawn waiting seeds if there's slack.
            if self._pending:
                self._admit_pending()
                self._write_pids()
            # Done condition: nothing live and nothing pending.
            if not self._live and not self._pending:
                return
            if deadline is not None and time.time() >= deadline:
                return
            time.sleep(0.05)

    def is_running(self) -> bool:
        """True iff any subprocess is alive or any seed is still pending."""
        if self._pending:
            return True
        return any(proc.poll() is None for proc in self._live.values())

    def kill(self) -> None:
        """SIGTERM all children, wait briefly, SIGKILL stragglers, clean up.

        After this call:
          * Every Popen in self._live has poll() != None.
          * No further pending seeds will be admitted.
          * Open event-log file handles are closed.
          * pids.json is rewritten with empty pid map.
        """
        for proc in self._live.values():
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if all(p.poll() is not None for p in self._live.values()):
                break
            time.sleep(0.05)
        for proc in self._live.values():
            if proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        self._close_event_logs()
        # Drop pending so future _admit_pending() calls (if any) are no-ops.
        self._pending.clear()
        # Drop live so is_running() (Task 17) and len(rg._live)==0 checks
        # are unambiguous after kill.
        self._live.clear()
        # Rewrite pids.json with cleared map.
        pids_path = self.exp_dir / "pids.json"
        try:
            pids_path.write_text(json.dumps({}, indent=2))
        except OSError:
            pass

    # ---- internals -------------------------------------------------------

    def _admit_pending(self) -> None:
        """Spawn pending seeds until pending is empty or workers cap is hit.

        Plain Python loop, no threading. Each spawn opens its event log file
        for line-buffered append and routes child stdout+stderr there.
        """
        while self._pending and len(self._live) < self._workers:
            seed = self._pending.pop(0)
            handle = next(h for h in self._run_handles if h.seed == seed)
            event_log_fh = open(handle.event_log_path, "w", buffering=1)
            env = os.environ.copy()
            # Defense in depth (H.6): train.py also self-enforces this.
            env.setdefault("OMP_NUM_THREADS", "1")
            try:
                proc = subprocess.Popen(
                    [
                        sys.executable, "-m", "rl_basics.train",
                        "--cfg", str(self._cfg_path),
                        "--advfn-pkl", str(self._advfn_path),
                        "--seed", str(seed),
                        "--out", str(handle.exp_dir / f"seed_{seed}"),
                    ],
                    stdout=event_log_fh,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    env=env,
                )
            except Exception:
                # Popen failed (bad executable, OS limits) — close the
                # event log we just opened so it doesn't leak.
                event_log_fh.close()
                raise
            self._event_logs[seed] = event_log_fh
            handle.pid = proc.pid
            self._live[seed] = proc

    def _write_pids(self) -> None:
        """Persist current {seed: pid} map (or null while pending) to pids.json."""
        pids_path = self.exp_dir / "pids.json"
        pid_map = {
            str(h.seed): (h.pid if h.pid is not None else None)
            for h in self._run_handles
        }
        pids_path.write_text(json.dumps(pid_map, indent=2))

    def _close_event_logs(self) -> None:
        for fh in self._event_logs.values():
            try:
                fh.close()
            except Exception:
                pass
        self._event_logs.clear()


def launch(
    cfg: RunConfig,
    advantage_fn: Callable | None = None,
    workers: int = 2,
    runs_dir: Path | None = None,
) -> RunGroup:
    """Launch one ``subprocess.Popen`` per seed (capped at ``workers`` alive).

    Steps:
      1. Resolve ``cfg.name`` to an exp dir under ``runs_dir`` (or ``cwd/runs``).
      2. Persist the shared cfg snapshot (``config.json``) and the pickled
         ``advantage_fn`` (``advfn.pkl``) at the exp dir.
      3. Create per-seed ``RunHandle`` objects + their ``seed_<i>/`` dirs.
      4. Admit up to ``workers`` Popens immediately; remaining seeds stay pending.
      5. Write ``pids.json`` mapping seed → pid (null while pending).

    Returns a ``RunGroup`` that owns the children and exposes ``.kill()``.

    Notes:
      * No threading. Plain os subprocesses; ``close_fds=True`` for hygiene.
      * H.5: All paths resolved to absolute via ``Path.resolve()``.
      * H.6: ``OMP_NUM_THREADS=1`` set in subprocess env; ``train.py`` also
        self-enforces this.
    """
    runs_dir = (runs_dir if runs_dir is not None else Path.cwd() / "runs").resolve()
    exp_name = cfg.resolve_name()
    exp_dir = (runs_dir / exp_name).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Persist shared cfg + pickled advfn at exp dir (not /tmp) so they survive
    # cleanup and are inspectable post-mortem.
    cfg_dict = asdict(cfg)
    cfg_path = exp_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg_dict, indent=2, default=str))

    if advantage_fn is None:
        advantage_fn = _default_advantage_fn(cfg)
    advfn_path = exp_dir / "advfn.pkl"
    advfn_path.write_bytes(cloudpickle.dumps(advantage_fn))

    # Per-seed RunHandle + seed dir creation.
    run_handles: list[RunHandle] = []
    for seed in cfg.seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        run_handles.append(
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

    rg = RunGroup(
        cfg=cfg,
        exp_dir=exp_dir,
        run_handles=run_handles,
        live={},
        pending=list(cfg.seeds),
        cfg_path=cfg_path,
        advfn_path=advfn_path,
        workers=workers,
    )
    rg._admit_pending()
    rg._write_pids()
    return rg
