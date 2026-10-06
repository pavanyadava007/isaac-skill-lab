"""Synchronous batched rollouts with optional test-time perturbations (import only after AppLauncher)."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch

from skilllab.config import EPISODE_STEPS, SUCCESS_HOLD_STEPS, Condition
from skilllab.sim.env import PickPlaceEnv


def rollout(env: PickPlaceEnv, seeds: list[int], act_fn: Callable[[dict], torch.Tensor], cond: Condition,
            record: bool = False, noise_seed: int = 0, frame_cb: Callable[[int], None] | None = None,
            reset_fn: Callable[[], None] | None = None, steps: int = EPISODE_STEPS,
            exec_perturb: Callable[[torch.Tensor], torch.Tensor] | None = None) -> dict:
    """Run one batch of episodes (one per env) and return per-episode outcomes (+ trajectories if record)."""
    obs = env.reset(seeds, cond.rand)
    if reset_fn is not None:
        reset_fn()
    n = env.n
    gen = torch.Generator(device=env.device).manual_seed(noise_seed)
    hold = torch.zeros(n, dtype=torch.long, device=env.device)
    t_success = torch.full((n,), -1, dtype=torch.long, device=env.device)
    queue: list[torch.Tensor] = []
    obs_log, act_log = [], []
    t0 = time.perf_counter()
    for t in range(steps):
        seen = obs
        if cond.obs_noise_std > 0:
            seen = dict(obs)
            for k in ("cube_pos", "target_xy"):
                seen[k] = obs[k] + cond.obs_noise_std * torch.randn(obs[k].shape, generator=gen, device=env.device)
            seen["cube_rel_tcp"] = seen["cube_pos"] - obs["tcp_pos"]
            seen["target_rel_cube"] = seen["target_xy"] - seen["cube_pos"][:, :2]
        a = act_fn(seen)
        if record:
            obs_log.append({k: v.detach().cpu().numpy().copy() for k, v in obs.items()})
            act_log.append(a.detach().cpu().numpy().copy())
        queue.append(a)
        if len(queue) > cond.action_delay:
            exec_a = queue.pop(0)
        else:  # hold still with the gripper open until the first delayed action arrives
            exec_a = torch.zeros_like(a)
            exec_a[:, 4] = 1.0
        if exec_perturb is not None:  # e.g. DART-style noise injection: execute noisy, record the clean label
            exec_a = exec_perturb(exec_a)
        obs = env.step(exec_a, render=frame_cb is not None)
        if frame_cb is not None:
            frame_cb(t)
        ok = env.success_now()
        hold = torch.where(ok, hold + 1, torch.zeros_like(hold))
        newly = (hold >= SUCCESS_HOLD_STEPS) & (t_success < 0)
        t_success[newly] = t - SUCCESS_HOLD_STEPS + 1
    wall = time.perf_counter() - t0
    out = {
        "success": (t_success >= 0).cpu().numpy(),
        "t_success": t_success.cpu().numpy(),
        "episodes": env.episodes,
        "wall_s": wall,
        "env_steps_per_s": n * steps / wall,
    }
    if record:
        out["obs"] = {k: np.stack([o[k] for o in obs_log], axis=1) for k in obs_log[0]}  # (n, T, d)
        out["actions"] = np.stack(act_log, axis=1)
    return out
