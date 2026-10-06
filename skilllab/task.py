"""Episode sampling, observation layout and the success check (numpy/torch, no simulator import)."""

from __future__ import annotations

import math

import numpy as np
import torch

from skilllab.config import SUCCESS_RADIUS, SUCCESS_REST_TOL, Randomization

OBS_KEYS = [  # (name, width), concatenated in this order into the flat policy observation
    ("tcp_pos", 3),
    ("gripper_width", 1),
    ("cube_pos", 3),
    ("cube_rel_tcp", 3),
    ("yaw_err", 1),
    ("target_xy", 2),
    ("target_rel_cube", 2),
    ("cube_size", 1),
]
OBS_DIM = sum(w for _, w in OBS_KEYS)


def sample_episode(seed: int, rand: Randomization) -> dict:
    """Per-episode randomization from one integer seed: identical seeds give identical episodes."""
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(*rand.cube_x), rng.uniform(*rand.cube_y)
    for _ in range(1000):
        tx, ty = rng.uniform(*rand.target_x), rng.uniform(*rand.target_y)
        if math.hypot(tx - cx, ty - cy) >= rand.min_cube_target_dist:
            break
    return {
        "seed": int(seed),
        "cube_x": float(cx),
        "cube_y": float(cy),
        "cube_yaw": float(math.radians(rng.uniform(*rand.cube_yaw_deg))),
        "target_x": float(tx),
        "target_y": float(ty),
        "cube_mass": float(rng.uniform(*rand.cube_mass)),
        "friction": float(rng.uniform(*rand.friction)),
    }


def sample_sizes(n: int, seed: int, rand: Randomization) -> np.ndarray:
    """Cube edge length per env slot (applied to the USD scale once, before the simulation starts)."""
    return np.random.default_rng(seed).uniform(*rand.cube_size, size=n)


def wrap_quarter(a: torch.Tensor) -> torch.Tensor:
    q = math.pi / 2
    return torch.remainder(a + q / 2, q) - q / 2


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw (rotation about world z) of the body x axis, quaternions in (w, x, y, z) order."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_obs(tcp_pos, tcp_yaw, grip_width, cube_pos, cube_yaw, target_xy, cube_size) -> dict[str, torch.Tensor]:
    """Observation dictionary (robomimic style: one array per key)."""
    return {
        "tcp_pos": tcp_pos,
        "gripper_width": grip_width.reshape(-1, 1),
        "cube_pos": cube_pos,
        "cube_rel_tcp": cube_pos - tcp_pos,
        "yaw_err": wrap_quarter(cube_yaw - tcp_yaw).reshape(-1, 1),
        "target_xy": target_xy,
        "target_rel_cube": target_xy - cube_pos[:, :2],
        "cube_size": cube_size.reshape(-1, 1),
    }


def flatten_obs(obs: dict) -> torch.Tensor | np.ndarray:
    parts = [obs[k] for k, _ in OBS_KEYS]
    if isinstance(parts[0], np.ndarray):
        return np.concatenate(parts, axis=-1)
    return torch.cat(parts, dim=-1)


def placed(cube_pos: torch.Tensor, target_xy: torch.Tensor, cube_size: torch.Tensor,
           grip_width: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
    """Instantaneous success: on the coaster, resting on the table and not held by the gripper."""
    on_target = torch.linalg.norm(cube_pos[:, :2] - target_xy, dim=-1) < SUCCESS_RADIUS
    resting = torch.abs(cube_pos[:, 2] - cube_size / 2) < SUCCESS_REST_TOL
    released = (grip_width > cube_size + 0.004) | (torch.linalg.norm(tcp_pos - cube_pos, dim=-1) > 0.05)
    return on_target & resting & released
