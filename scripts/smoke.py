"""Smoke test in Isaac Sim: build the scene, run the scripted expert on a few envs, print outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--debug", action="store_true")
ap.add_argument("--cond", default="nominal")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import time  # noqa: E402

import torch  # noqa: E402

from skilllab.config import CONDITIONS  # noqa: E402
from skilllab.expert import PHASE_NAMES, ScriptedExpert  # noqa: E402
from skilllab.runner import run  # noqa: E402
from skilllab.sim.env import PickPlaceEnv  # noqa: E402
from skilllab.sim.rollout import rollout  # noqa: E402


def main() -> None:

    t0 = time.time()
    env = PickPlaceEnv(args.num_envs, CONDITIONS["nominal"].rand, size_seed=1)
    print(f"[smoke] scene built in {time.time() - t0:.1f}s", flush=True)
    expert = ScriptedExpert(env.n, env.device)
    log = []

    def act(obs):
        a = expert.act(obs)
        if args.debug:
            log.append((expert.phase[0].item(), obs["tcp_pos"][0].tolist(), obs["cube_pos"][0].tolist(),
                        obs["gripper_width"][0].item(), obs["yaw_err"][0].item()))
        return a

    out = rollout(env, list(range(args.num_envs)), act, CONDITIONS[args.cond], reset_fn=expert.reset)
    print("[smoke] success", out["success"].astype(int).tolist())
    print("[smoke] t_success", out["t_success"].tolist())
    print("[smoke] final phases", [PHASE_NAMES[p] for p in expert.phase.tolist()])
    print("[smoke] retries", expert.retries.tolist())
    print(f"[smoke] wall {out['wall_s']:.1f}s, {out['env_steps_per_s']:.0f} env-steps/s")
    if args.debug:
        for i, row in enumerate(log):
            if i % 5 == 0:
                print(i, PHASE_NAMES[row[0]], [round(x, 3) for x in row[1]], [round(x, 3) for x in row[2]],
                      round(row[3], 4), round(row[4], 3))
    print("[smoke] home q", [round(x, 3) for x in env._home_q.tolist()])
    torch.cuda.synchronize()


if __name__ == "__main__":
    run(main, app)
