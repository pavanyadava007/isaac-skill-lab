"""Evaluation harness in Isaac Sim: fixed unseen seeds, every policy and the scripted expert, all conditions.

One simulator process per "world" (the cube sizes are baked into the USD stage before the simulation
starts, so a size shift needs its own process). Within a world every policy sees exactly the same
episodes (same seeds, same env slots), which makes the comparisons paired.

    OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PY scripts/evaluate.py --headless --world dr
    OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PY scripts/evaluate.py --headless --world small_cubes
    OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PY scripts/evaluate.py --headless --world no_dr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--world", choices=["dr", "small_cubes", "no_dr"], default="dr")
ap.add_argument("--episodes", type=int, default=200)
ap.add_argument("--policies", default="*", help="glob over checkpoints/*.pt")
ap.add_argument("--no_expert", action="store_true")
ap.add_argument("--merge", action="store_true", help="update an existing results/eval/<world>.json in place")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import time  # noqa: E402

from skilllab import models  # noqa: E402
from skilllab.config import (  # noqa: E402
    CONDITIONS,
    NO_DR,
    POLICY_DT,
    SIZE_CONDITIONS,
    Condition,
    to_dict,
)
from skilllab.expert import ScriptedExpert  # noqa: E402
from skilllab.io import read_json, write_json  # noqa: E402
from skilllab.runner import run  # noqa: E402
from skilllab.sim.env import PickPlaceEnv  # noqa: E402
from skilllab.sim.rollout import rollout  # noqa: E402
from skilllab.stats import success_summary  # noqa: E402
from skilllab.task import flatten_obs  # noqa: E402

EVAL_SEED_BASE = 1_000_000  # training demos use seeds < 10_000
EVAL_SIZE_SEED = 777  # training collection uses size seed 11


def world_conditions(world: str) -> list[Condition]:
    if world == "dr":
        return list(CONDITIONS.values())
    if world == "small_cubes":
        return [SIZE_CONDITIONS["small_cubes"]]
    return [Condition("no_dr_world", NO_DR, note="fixed nominal size, mass and friction (no-DR training world)")]


def main() -> None:
    conds = world_conditions(args.world)
    env = PickPlaceEnv(args.episodes, conds[0].rand, size_seed=EVAL_SIZE_SEED)
    seeds = [EVAL_SEED_BASE + i for i in range(args.episodes)]
    ckpts = sorted((ROOT / "checkpoints").glob(f"{args.policies}.pt"))
    agents = {} if args.no_expert else {"expert": None}
    agents.update({p.stem: p for p in ckpts})
    out_path = ROOT / "results" / "eval" / f"{args.world}.json"
    results: dict = read_json(out_path)["results"] if args.merge and out_path.exists() else {}
    t_all = time.time()
    for name, path in agents.items():
        results[name] = {}
        if path is None:
            expert = ScriptedExpert(env.n, env.device)
            act, reset = expert.act, expert.reset
        else:
            model, _ = models.load(path, env.device)
            runner = models.PolicyRunner(model, env.n, env.device)

            def act(obs, runner=runner):
                return runner.act(flatten_obs(obs))

            reset = runner.reset
        for cond in conds:
            out = rollout(env, seeds, act, cond, reset_fn=reset, noise_seed=99)
            s = success_summary(out["success"].tolist(), out["t_success"].tolist(), POLICY_DT)
            results[name][cond.name] = {**s, "success": out["success"].astype(int).tolist(),
                                        "t_success": out["t_success"].tolist()}
            print(f"[eval] {args.world} {name:28s} {cond.name:16s} {s['successes']:4d}/{s['n']} "
                  f"({s['rate']:.3f}) {out['wall_s']:.1f}s", flush=True)
    write_json(out_path, {
        "world": args.world,
        "episodes": args.episodes,
        "eval_seeds": [seeds[0], seeds[-1]],
        "size_seed": EVAL_SIZE_SEED,
        "cube_sizes_m": env.sizes_np.tolist(),
        "conditions": {c.name: {**to_dict(c), "rand": to_dict(c.rand)} for c in conds},
        "results": results,
        "wall_s": time.time() - t_all,
    })


if __name__ == "__main__":
    run(main, app)
