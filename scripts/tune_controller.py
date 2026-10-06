"""Controller tuning in Isaac Sim: TCP step and tracking response of the DLS differential-IK + joint PD loop.

Each env gets one joint PD gain pair (gains are written to PhysX at runtime), each process pass uses one DLS
damping value. Protocol, starting from the home pose with the gripper open and no object contact:
  1. step: TCP target jumps 10 cm in +x and 10 cm down, then held for 1.5 s;
  2. ramp: TCP target moves 20 cm in y at 0.4 m/s (the maximum speed the policy can command), then held.
Metrics: 10-90% rise time, overshoot past the target, settled error after 1.5 s, RMS tracking error on the ramp.
  3. closed loop: the scripted expert (an outer proportional loop on TCP increments) runs the full skill on
     REPS episodes per gain pair (DR ranges, DLS lambda 0.05); success rate with Wilson 95% CI.

    OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PY scripts/tune_controller.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from skilllab.config import CONTROLLER, DR, HOME_TCP, PHYSICS_DT, POLICY_DT, Condition, to_dict  # noqa: E402
from skilllab.expert import ScriptedExpert  # noqa: E402
from skilllab.io import write_json  # noqa: E402
from skilllab.runner import run  # noqa: E402
from skilllab.sim.env import PickPlaceEnv  # noqa: E402
from skilllab.sim.rollout import rollout  # noqa: E402
from skilllab.stats import success_summary  # noqa: E402

GAINS = [(80.0, 4.0), (400.0, 40.0), (400.0, 80.0), (1000.0, 100.0), (1000.0, 200.0)]
LAMBDAS = [0.01, 0.05, 0.2]
REPS = 100


def main() -> None:
    n = len(GAINS) * REPS  # env i uses gain pair i // REPS
    env = PickPlaceEnv(n, DR, size_seed=0)
    # move the cubes out of reach so the arm moves freely
    env.reset(list(range(n)))
    far = torch.tensor([[0.0, 0.6, 0.03, 1, 0, 0, 0]], device=env.device).repeat(n, 1)
    far[:, :3] += env.scene.env_origins
    env.cube.write_root_pose_to_sim(far)
    stiff = torch.tensor([g[0] for g in GAINS], device=env.device).repeat_interleave(REPS)[:, None]
    damp = torch.tensor([g[1] for g in GAINS], device=env.device).repeat_interleave(REPS)[:, None]
    stiff, damp = stiff.repeat(1, len(env.arm_ids)), damp.repeat(1, len(env.arm_ids))
    env.robot.write_joint_stiffness_to_sim(stiff, joint_ids=env.arm_ids)
    env.robot.write_joint_damping_to_sim(damp, joint_ids=env.arm_ids)
    home = torch.tensor(HOME_TCP, device=env.device)
    rows = []
    for lam in LAMBDAS:
        env.ik.cfg.ik_params = {"lambda_val": lam}
        q = env._home_q.repeat(n, 1)
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.target_pos = home.repeat(n, 1)
        env.target_yaw = torch.zeros(n, device=env.device)
        for _ in range(100):
            env._physics_step()
        # 1) step
        start = env.tcp_pose()[0].clone()
        goal = home + torch.tensor([0.10, 0.0, -0.10], device=env.device)
        env.target_pos = goal.repeat(n, 1)
        dist0 = torch.linalg.norm(goal - start, dim=-1)
        prog, errs = [], []
        for _ in range(150):
            env._physics_step()
            tcp = env.tcp_pose()[0]
            d = goal - tcp
            prog.append(((goal - start) * (tcp - start)).sum(-1) / dist0**2)  # fraction of the step covered
            errs.append(torch.linalg.norm(d, dim=-1))
        prog_t = torch.stack(prog)  # (T, n)
        t10 = (prog_t >= 0.1).float().argmax(0)
        t90 = (prog_t >= 0.9).float().argmax(0)
        overshoot = (prog_t.max(0).values - 1.0).clamp_min(0) * dist0
        settled = errs[-1]
        # 2) ramp at 0.4 m/s along y
        ramp_err = []
        y0 = goal[1].item()
        for k in range(1, 51):  # 0.5 s at 100 Hz, 0.2 m
            tgt = goal.clone()
            tgt[1] = y0 + 0.004 * k
            env.target_pos = tgt.repeat(n, 1)
            env._physics_step()
            ramp_err.append(torch.linalg.norm(tgt - env.tcp_pose()[0], dim=-1))
        ramp = torch.stack(ramp_err)
        for gi, (kp, kd) in enumerate(GAINS):
            i = gi * REPS
            rows.append({"dls_lambda": lam, "stiffness": kp, "damping": kd,
                         "rise_time_10_90_s": float((t90[i] - t10[i]).item() * PHYSICS_DT),
                         "overshoot_mm": float(overshoot[i] * 1000), "settled_error_mm": float(settled[i] * 1000),
                         "ramp_rms_error_mm": float(torch.sqrt((ramp[:, i] ** 2).mean()) * 1000),
                         "ramp_final_lag_mm": float(ramp[-1, i] * 1000)})
            print("[tune]", rows[-1], flush=True)
    # 3) closed loop with the scripted expert
    env.ik.cfg.ik_params = {"lambda_val": CONTROLLER.dls_lambda}
    expert = ScriptedExpert(n, env.device)
    out = rollout(env, [500_000 + i for i in range(n)], expert.act, Condition("tune", DR), reset_fn=expert.reset)
    closed = []
    for gi, (kp, kd) in enumerate(GAINS):
        sl = slice(gi * REPS, (gi + 1) * REPS)
        summ = success_summary(out["success"][sl].tolist(), out["t_success"][sl].tolist(), POLICY_DT)
        closed.append({"stiffness": kp, "damping": kd, "dls_lambda": CONTROLLER.dls_lambda,
                       **{k: summ[k] for k in ("n", "successes", "rate", "ci95", "time_to_success_s")}})
        print("[tune] closed loop", closed[-1], flush=True)
    write_json(ROOT / "results" / "controller_tuning.json", {
        "closed_loop_expert": closed,
        "protocol": __doc__.split("\n\n")[0].splitlines()[1:],
        "physics_dt_s": PHYSICS_DT,
        "chosen": to_dict(CONTROLLER),
        "rows": rows,
    })


if __name__ == "__main__":
    run(main, app)
