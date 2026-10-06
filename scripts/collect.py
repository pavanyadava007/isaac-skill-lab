"""Collect scripted-expert demonstrations in Isaac Sim across parallel envs (robomimic-style HDF5).

The expert's clean action is recorded as the label while a noisy version is executed (DART-style noise
injection), so the demos also cover small recoveries. Only successful episodes are kept.

    OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PY scripts/collect.py --headless --rand dr --num_envs 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--rand", choices=["dr", "no_dr"], default="dr")
ap.add_argument("--num_envs", type=int, default=1024)
ap.add_argument("--batches", type=int, default=1)
ap.add_argument("--noise_pos", type=float, default=0.006, help="std of executed TCP delta noise, m/step")
ap.add_argument("--noise_yaw", type=float, default=0.03, help="std of executed yaw delta noise, rad/step")
ap.add_argument("--seed_base", type=int, default=0)
ap.add_argument("--size_seed", type=int, default=11)
ap.add_argument("--tail_steps", type=int, default=20, help="steps kept after success (retreat)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from skilllab.config import (  # noqa: E402
    DR,
    EPISODE_STEPS,
    NO_DR,
    POLICY_DT,
    SUCCESS_HOLD_STEPS,
    Condition,
    to_dict,
)
from skilllab.dataset import write_demos  # noqa: E402
from skilllab.expert import ScriptedExpert  # noqa: E402
from skilllab.io import write_json  # noqa: E402
from skilllab.runner import run  # noqa: E402
from skilllab.sim.env import PickPlaceEnv  # noqa: E402
from skilllab.sim.rollout import rollout  # noqa: E402


def main() -> None:
    rand = DR if args.rand == "dr" else NO_DR
    cond = Condition(f"collect_{rand.name}", rand)
    t0 = time.time()
    env = PickPlaceEnv(args.num_envs, rand, size_seed=args.size_seed)
    expert = ScriptedExpert(env.n, env.device)
    gen = torch.Generator(device=env.device).manual_seed(args.seed_base + 123)

    def perturb(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        a[:, :3] += args.noise_pos * torch.randn(a[:, :3].shape, generator=gen, device=a.device)
        a[:, 3] += args.noise_yaw * torch.randn(a[:, 3].shape, generator=gen, device=a.device)
        return a

    demos, total, wall = [], 0, 0.0
    lengths, times = [], []
    for b in range(args.batches):
        seeds = [args.seed_base + b * args.num_envs + i for i in range(args.num_envs)]
        out = rollout(env, seeds, expert.act, cond, record=True, reset_fn=expert.reset, exec_perturb=perturb)
        wall += out["wall_s"]
        total += env.n
        for i in np.flatnonzero(out["success"]):
            t_end = min(EPISODE_STEPS, int(out["t_success"][i]) + SUCCESS_HOLD_STEPS + args.tail_steps)
            ep = out["episodes"][i]
            demos.append({
                "obs": {k: v[i, :t_end] for k, v in out["obs"].items()},
                "actions": out["actions"][i, :t_end],
                "attrs": {**ep, "t_success": int(out["t_success"][i]), "retries": int(expert.retries[i])},
            })
            lengths.append(t_end)
            times.append(int(out["t_success"][i]) * POLICY_DT)
        print(f"[collect] batch {b}: kept {int(out['success'].sum())}/{env.n}", flush=True)

    path = ROOT / "data" / f"demos_{rand.name}.hdf5"
    env_args = {"env_name": "IsaacSkillLab-CubeOnCoaster-Franka", "type": "isaaclab_custom",
                "env_kwargs": {"randomization": to_dict(rand), "policy_dt": POLICY_DT,
                               "noise_pos": args.noise_pos, "noise_yaw": args.noise_yaw}}
    write_demos(path, demos, env_args)
    write_json(ROOT / "results" / f"collect_{rand.name}.json", {
        "randomization": to_dict(rand),
        "attempted": total,
        "kept": len(demos),
        "kept_rate": len(demos) / total,
        "num_envs": args.num_envs,
        "exec_noise": {"pos_std_m_per_step": args.noise_pos, "yaw_std_rad_per_step": args.noise_yaw},
        "demo_length_steps": {"mean": float(np.mean(lengths)), "min": int(np.min(lengths)),
                              "max": int(np.max(lengths))},
        "expert_time_to_success_s": {"mean": float(np.mean(times)), "median": float(np.median(times))},
        "demos_with_regrasp": int(sum(d["attrs"]["retries"] > 0 for d in demos)),
        "samples": int(sum(lengths)),
        "rollout_wall_s": wall,
        "total_wall_s": time.time() - t0,
        "file": str(path.relative_to(ROOT)),
    })
    print(f"[collect] kept {len(demos)}/{total} -> {path}", flush=True)


if __name__ == "__main__":
    run(main, app)
