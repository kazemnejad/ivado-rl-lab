"""Refresh the cached `tests/fixtures/mini_run/` fixture used by viz tests.

Run this ONCE manually to regenerate the fixture; it is NOT invoked by pytest.

The fixture mimics two `RunGroup` exp dirs (vanilla + value advantage), each
with 3 seeds and 10 updates of synthetic metrics. We synthesize the JSONL
records directly rather than calling `launch()` — the viz tests only need
DataFrames with the right columns, and synthesized bytes are far cheaper than
spawning subprocesses.

Usage:
    .venv/bin/python tests/fixtures/build_mini_run.py
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def main() -> None:
    base = Path(__file__).parent / "mini_run"
    base.mkdir(parents=True, exist_ok=True)

    exps = {
        "vanilla_b4_g1.0_h8_aaaaaa": {
            "advantage_kind": "vanilla",
            "use_value_baseline": False,
            "base_return": 0.2,
        },
        "value_b4_g1.0_h8_bbbbbb": {
            "advantage_kind": "value",
            "use_value_baseline": True,
            "base_return": 0.4,
        },
    }
    cfg_common = {
        "name": None,
        "env": "FourRoomsTL",
        "batch_size": 4,
        "n_updates": 10,
        "lr": 3e-3,
        "hidden": 8,
        "gamma": 1.0,
        "seeds": [0, 1, 2],
        "log_every": 1,
        "grad_var_every": 0,
    }
    for exp_name, info in exps.items():
        exp_dir = base / exp_name
        exp_dir.mkdir(exist_ok=True)
        cfg = {**cfg_common, **{k: v for k, v in info.items() if k != "base_return"}}
        (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        for seed in [0, 1, 2]:
            seed_dir = exp_dir / f"seed_{seed}"
            seed_dir.mkdir(exist_ok=True)
            stable_hash = int(hashlib.md5(exp_name.encode()).hexdigest(), 16)
            rng = random.Random(seed + stable_hash % 1000)
            lines = []
            for upd in range(10):
                rec = {
                    "upd": upd,
                    "wall": float(upd),
                    "ep_return_mean": float(
                        info["base_return"] + 0.05 * upd + rng.gauss(0, 0.02)
                    ),
                    "p_loss": float(1.0 - 0.05 * upd + rng.gauss(0, 0.05)),
                }
                lines.append(json.dumps(rec))
            (seed_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")
            (seed_dir / "event.log").write_text("synthetic fixture\n")
            (seed_dir / "ckpt.pt").write_bytes(b"")
    print(f"Wrote fixture under {base}")


if __name__ == "__main__":
    main()
