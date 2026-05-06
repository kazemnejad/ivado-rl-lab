"""Subprocess entrypoint for one (cfg, seed) training run — SPEC §5.6.

Invoked via:
    python -m rl_basics.train --cfg <path> --advfn-pkl <path> --seed <int> --out <dir>

Writes:
  <out>/config.json     — cfg snapshot for this seed
  <out>/metrics.jsonl   — one JSONL line per update (line-flushed)
  <out>/ckpt.pt         — final policy + value state-dicts after last update
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cloudpickle
import torch
import torch.optim as optim

from rl_basics.env import FourRoomsTL
from rl_basics.grad_variance import measure_grad_variance
from rl_basics.models import MLPPolicy, ValueNetwork
from rl_basics.runs import RunConfig

# IMPORTANT: framework subprocess uses the SHIPPED references in utils.py,
# NOT the student.py versions (SPEC §5.6 footnote: "we call the SHIPPED
# reference for the framework's own internal training"). When Task 32's
# notebook builder stubs student.py to NotImplementedError, this import
# guarantees train.py keeps running. Only `student_advantage_fn` (loaded
# from advfn.pkl) routes through the learner's notebook namespace.
from rl_basics.utils import (
    _compute_returns_to_go_ref as compute_returns_to_go,
    _policy_loss_ref as policy_loss,
    grad_norm,
    log_probs,
    rollout,
    sample_initial_states,
    value_loss,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--advfn-pkl", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=str)
    args = parser.parse_args(argv)

    # 1. Niceness — yield CPU so the kernel's live-plot cell can re-render.
    try:
        os.nice(10)
    except (PermissionError, AttributeError, OSError):
        pass  # Windows / restricted envs
    # Self-enforce H.6 (OMP thread cap) so direct invocations from a
    # notebook or shell can't accidentally spawn 6+ OpenBLAS threads per
    # subprocess and starve the kernel. setdefault respects an explicit
    # override from the caller.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # 2. Cfg
    cfg_path = Path(args.cfg).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = RunConfig(**cfg_dict)

    # Snapshot the per-seed config for downstream tooling.
    seed_cfg = {**cfg_dict, "seed": args.seed}
    (out_dir / "config.json").write_text(json.dumps(seed_cfg, indent=2, default=str))

    # 3. Advfn (pickled student callable).
    advfn_path = Path(args.advfn_pkl).resolve()
    with open(advfn_path, "rb") as f:
        student_advantage_fn = cloudpickle.load(f)

    # 4. Seed + instances.
    torch.manual_seed(args.seed)
    env = FourRoomsTL(batch_size=cfg.batch_size, seed=args.seed)
    policy = MLPPolicy(
        n_states=env.n_states, n_actions=env.n_actions, hidden=cfg.hidden
    )
    value_net = (
        ValueNetwork(n_states=env.n_states, hidden=cfg.hidden)
        if cfg.use_value_baseline
        else None
    )
    opt = optim.Adam(policy.parameters(), lr=cfg.lr)
    v_opt = (
        optim.Adam(value_net.parameters(), lr=cfg.lr)
        if value_net is not None
        else None
    )

    # 5. Loop.
    metrics_path = out_dir / "metrics.jsonl"
    with open(metrics_path, "w", buffering=1) as fout:
        t0 = time.time()
        for upd in range(cfg.n_updates):
            init = sample_initial_states(env, cfg.batch_size)
            traj = rollout(env, policy, init)

            returns = compute_returns_to_go(
                traj.rewards, traj.mask, gamma=cfg.gamma
            )

            v_pred_full = None
            if cfg.advantage_kind == "value":
                assert value_net is not None
                v_pred_full = value_net(traj.states.flatten()).view_as(traj.states)
                v_pred_detached = v_pred_full.detach()
                adv = student_advantage_fn(returns, v_pred_detached, traj.mask)
            else:
                adv = student_advantage_fn(returns, traj.mask)

            logits = policy(traj.states.flatten()).view(*traj.states.shape, -1)
            logp = log_probs(logits, traj.actions)

            p_loss = policy_loss(logp, adv, traj.mask)

            opt.zero_grad()
            v_loss = None
            if v_opt is not None:
                v_opt.zero_grad()
                v_loss = value_loss(v_pred_full, returns, traj.mask)
                total = p_loss + 0.5 * v_loss
                total.backward()
            else:
                p_loss.backward()

            # Grad norm BEFORE opt.step() — captures the gradient that drove
            # the update.
            gnorm = grad_norm(policy)

            opt.step()
            if v_opt is not None:
                v_opt.step()

            ep_total = traj.rewards.sum(dim=1)
            metric = {
                "upd": upd,
                "wall": round(time.time() - t0, 4),
                "ep_return_mean": float(ep_total.mean().item()),
                "ep_return_std": (
                    float(ep_total.std().item()) if cfg.batch_size > 1 else 0.0
                ),
                "p_loss": float(p_loss.item()),
                "grad_norm": gnorm,
                "adv_abs_mean": float(adv.abs().mean().item()),
            }
            if v_loss is not None:
                metric["v_loss"] = float(v_loss.item())

            if cfg.grad_var_every > 0 and (upd + 1) % cfg.grad_var_every == 0:
                gv = measure_grad_variance(
                    env_class=FourRoomsTL,
                    env_kwargs={},
                    batch_size=cfg.batch_size,
                    device="cpu",
                    policy=policy,
                    value_net=value_net,
                    advantage_kind=cfg.advantage_kind,
                    gamma=cfg.gamma,
                    n_boot=cfg.grad_var_n_boot,
                )
                metric["grad_var"] = {k: float(v) for k, v in gv.items()}

            # JSONL discipline (Appendix H.10): single write w/ trailing \n,
            # explicit flush.
            fout.write(json.dumps(metric) + "\n")
            fout.flush()

    # 8. Final ckpt.
    ckpt: dict = {"policy": policy.state_dict()}
    if value_net is not None:
        ckpt["value_net"] = value_net.state_dict()
    torch.save(ckpt, out_dir / "ckpt.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
