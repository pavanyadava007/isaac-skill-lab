"""Classify why episodes fail (Isaac Sim): never grasped, dropped/misplaced, placed but not released; stalled.

Runs selected policies on the same held-out seeds as evaluate.py (DR world) and writes
results/failure_modes.json.

    OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PY scripts/diagnose_failures.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--episodes", type=int, default=200)
ap.add_argument("--policies", nargs="+", default=["mlp_dr_n100_s0", "mlp_dr_n100_s1", "mlp_dr_n100_s2",
                                                  "mlp_no_dr_n400_s0", "mlp_no_dr_n400_s1", "mlp_no_dr_n400_s2"])
ap.add_argument("--conditions", nargs="+", default=["nominal", "obs_noise_5mm"])
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from skilllab import models  # noqa: E402
from skilllab.config import CONDITIONS, SUCCESS_RADIUS  # noqa: E402
from skilllab.io import write_json  # noqa: E402
from skilllab.runner import run  # noqa: E402
from skilllab.sim.env import PickPlaceEnv  # noqa: E402
from skilllab.sim.rollout import rollout  # noqa: E402
from skilllab.task import flatten_obs  # noqa: E402

EVAL_SEED_BASE, EVAL_SIZE_SEED = 1_000_000, 777
STALL_WINDOW = 40  # last 2 s


def main() -> None:
    env = PickPlaceEnv(args.episodes, CONDITIONS["nominal"].rand, size_seed=EVAL_SIZE_SEED)
    seeds = [EVAL_SEED_BASE + i for i in range(args.episodes)]
    out: dict = {}
    for name in args.policies:
        model, _ = models.load(ROOT / "checkpoints" / f"{name}.pt", env.device)
        runner = models.PolicyRunner(model, env.n, env.device)
        out[name] = {}
        for cname in args.conditions:
            r = rollout(env, seeds, lambda o, rn=runner: rn.act(flatten_obs(o)), CONDITIONS[cname], record=True,
                        reset_fn=runner.reset, noise_seed=99)
            o = r["obs"]  # (n, T, d), true (noise-free) observations
            cube_z = torch.tensor(o["cube_pos"][..., 2])
            size = torch.tensor(o["cube_size"][..., 0])
            lifted = ((cube_z - size / 2) > 0.02).any(dim=1)
            final_xy_err = torch.linalg.norm(torch.tensor(o["target_rel_cube"][:, -1]), dim=-1)
            tcp = torch.tensor(o["tcp_pos"])
            speed = torch.linalg.norm(tcp[:, 1:] - tcp[:, :-1], dim=-1)[:, -STALL_WINDOW:].mean(dim=1)
            fail = ~torch.tensor(r["success"])
            cats = {"never_lifted": fail & ~lifted,
                    "lifted_not_on_coaster": fail & lifted & (final_xy_err >= SUCCESS_RADIUS),
                    "on_coaster_not_released_or_not_resting": fail & lifted & (final_xy_err < SUCCESS_RADIUS)}
            stalled = fail & (speed < 0.001)
            out[name][cname] = {"episodes": env.n, "failures": int(fail.sum()),
                                **{k: int(v.sum()) for k, v in cats.items()},
                                "failures_stalled_last_2s": int(stalled.sum()),
                                "stall_definition": "mean TCP motion < 1 mm per policy step over the last 2 s"}
            print("[diag]", name, cname, out[name][cname], flush=True)
    write_json(ROOT / "results" / "failure_modes.json", {"world": "dr", "results": out})


if __name__ == "__main__":
    run(main, app)
