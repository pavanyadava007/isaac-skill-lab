"""Scripted pick-and-place expert: a vectorized finite state machine over the state observation.

It outputs the same action as the learned policy (TCP position/yaw deltas + gripper), so its demos can be
used for behaviour cloning directly. It reads only the observation dictionary, no privileged simulator state.
"""

from __future__ import annotations

import torch

from skilllab.config import MAX_DPOS, MAX_DYAW

APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, LOWER, RELEASE, RETREAT = range(8)
PHASE_NAMES = ["approach", "descend", "grasp", "lift", "transport", "lower", "release", "retreat"]

HOVER = 0.10  # m above the cube top while approaching
CARRY_Z = 0.15  # TCP height while carrying
GRASP_DEPTH = 0.4  # TCP at this fraction of the cube height above the table (finger pads around the middle)
GRASP_STEPS = 8  # 0.4 s for the fingers to close
RELEASE_STEPS = 6
GAIN = 0.6  # fraction of the remaining error commanded per step, before clipping


class ScriptedExpert:
    def __init__(self, num_envs: int, device: str | torch.device):
        self.n = num_envs
        self.device = device
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.timer = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.retries = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self) -> None:
        self.phase.zero_()
        self.timer.zero_()
        self.retries.zero_()

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        tcp = obs["tcp_pos"]
        cube = obs["cube_pos"]
        size = obs["cube_size"][:, 0]
        grip = obs["gripper_width"][:, 0]
        yaw_err = obs["yaw_err"][:, 0]
        target = obs["target_xy"]
        p = self.phase

        # --- transitions (evaluated on the current observation) ---
        err_xy_cube = torch.linalg.norm(cube[:, :2] - tcp[:, :2], dim=-1)
        grasp_z = size * GRASP_DEPTH
        aligned = (err_xy_cube < 0.008) & (torch.abs(yaw_err) < 0.04)
        hover_z = size + HOVER
        nxt = p.clone()
        nxt[(p == APPROACH) & aligned & (torch.abs(tcp[:, 2] - hover_z) < 0.03)] = DESCEND
        nxt[(p == DESCEND) & (torch.abs(tcp[:, 2] - grasp_z) < 0.006)] = GRASP
        nxt[(p == GRASP) & (self.timer >= GRASP_STEPS)] = LIFT
        nxt[(p == LIFT) & (tcp[:, 2] > CARRY_Z - 0.02)] = TRANSPORT
        nxt[(p == TRANSPORT) & (torch.linalg.norm(cube[:, :2] - target, dim=-1) < 0.006)] = LOWER
        nxt[(p == LOWER) & (cube[:, 2] - size / 2 < 0.004)] = RELEASE
        nxt[(p == RELEASE) & (self.timer >= RELEASE_STEPS)] = RETREAT
        # descending while misaligned (cube was nudged): go back up
        nxt[(p == DESCEND) & (err_xy_cube > 0.02)] = APPROACH
        # lost the cube while carrying: open and start over
        dropped = ((p == LIFT) | (p == TRANSPORT) | (p == LOWER)) & (torch.linalg.norm(cube - tcp, dim=-1) > 0.04)
        nxt[dropped] = APPROACH
        self.retries += dropped.long()
        self.timer = torch.where(nxt == p, self.timer + 1, torch.zeros_like(self.timer))
        self.phase = p = nxt

        # --- waypoint per phase ---
        goal = tcp.clone()
        goal_xy_cube = cube[:, :2]
        goal_xy_target = target + (tcp[:, :2] - cube[:, :2])  # move the cube, not the TCP, onto the target
        m = p == APPROACH
        goal[m, :2] = goal_xy_cube[m]
        goal[m, 2] = hover_z[m]
        m = p == DESCEND
        goal[m, :2] = goal_xy_cube[m]
        goal[m, 2] = grasp_z[m]
        m = p == LIFT
        goal[m, 2] = CARRY_Z
        m = p == TRANSPORT
        goal[m, :2] = goal_xy_target[m]
        goal[m, 2] = CARRY_Z
        m = p == LOWER
        goal[m, :2] = goal_xy_target[m]
        goal[m, 2] = grasp_z[m]
        m = p == RETREAT
        goal[m, 2] = CARRY_Z

        dpos = torch.clamp(GAIN * (goal - tcp), -MAX_DPOS, MAX_DPOS)
        # during the approach, do not descend before being roughly above the cube
        far = (p == APPROACH) & (err_xy_cube > 0.03)
        dpos[far, 2] = torch.clamp(GAIN * (torch.maximum(hover_z[far], tcp[far, 2]) - tcp[far, 2]), -MAX_DPOS, MAX_DPOS)
        dyaw = torch.where((p == APPROACH) | (p == DESCEND), torch.clamp(GAIN * yaw_err, -MAX_DYAW, MAX_DYAW),
                           torch.zeros_like(yaw_err))
        closed = (p == GRASP) | (p == LIFT) | (p == TRANSPORT) | (p == LOWER)
        gripper = torch.where(closed, -torch.ones_like(grip), torch.ones_like(grip))
        return torch.cat([dpos, dyaw[:, None], gripper[:, None]], dim=-1)
