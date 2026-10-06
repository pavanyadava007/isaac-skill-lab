"""Isaac Lab scene + task-space controller for the "cube onto coaster" skill (import only after AppLauncher).

Own scene, not a registered task: Franka Panda on a box table, a randomized cube, a kinematic coaster disc
(visual only, no collision). Arm control: differential IK (damped least squares) from a TCP pose target
to joint position targets, tracked by the joint PD loop of the implicit actuators. See docs/CONTROLLER.md.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import numpy as np
import omni.usd
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, skew_symmetric_matrix
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from pxr import Gf, Sdf, UsdGeom, Vt

from skilllab.config import (
    CONTROLLER,
    DECIMATION,
    HOME_TCP,
    MAX_DPOS,
    MAX_DYAW,
    NOMINAL_FRICTION,
    NOMINAL_SIZE,
    PHYSICS_DT,
    TCP_MAX_LEAD,
    TCP_OFFSET,
    WORKSPACE_HI,
    WORKSPACE_LO,
    ControllerCfg,
    Randomization,
)
from skilllab.task import build_obs, placed, sample_episode, sample_sizes, yaw_from_quat


def robot_cfg(ctrl: ControllerCfg) -> ArticulationCfg:
    cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # links float without gravity, like the gravity compensation of the real arm's torque controller
    cfg.spawn.rigid_props.disable_gravity = True
    cfg.actuators = {
        "panda_shoulder": ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-4]"], effort_limit_sim=87.0,
                                              stiffness=ctrl.arm_stiffness, damping=ctrl.arm_damping),
        "panda_forearm": ImplicitActuatorCfg(joint_names_expr=["panda_joint[5-7]"], effort_limit_sim=12.0,
                                             stiffness=ctrl.arm_stiffness, damping=ctrl.arm_damping),
        "panda_hand": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], effort_limit_sim=200.0,
                                          stiffness=ctrl.finger_stiffness, damping=ctrl.finger_damping),
    }
    return cfg


@configclass
class SkillSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.75)))
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)))
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.50, 0.38)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, -0.025)),
    )
    robot: ArticulationCfg = robot_cfg(CONTROLLER)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(NOMINAL_SIZE,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16,
                                                         solver_velocity_iteration_count=1,
                                                         max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=NOMINAL_FRICTION,
                                                            dynamic_friction=NOMINAL_FRICTION),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.80)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, NOMINAL_SIZE / 2)),
    )
    coaster = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Coaster",
        spawn=sim_utils.CylinderCfg(
            radius=0.045, height=0.002, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.20)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.2, 0.001)),
    )


class PickPlaceEnv:
    """Batched env: all envs reset together with one episode seed each and step synchronously."""

    def __init__(self, num_envs: int, rand: Randomization, size_seed: int, device: str = "cuda:0",
                 ctrl: ControllerCfg = CONTROLLER, env_spacing: float = 2.0, camera: bool = False):
        self.n = num_envs
        self.rand = rand
        self.ctrl = ctrl
        self.device = device
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=device))
        scene_cfg = SkillSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=False)
        scene_cfg.robot = robot_cfg(ctrl)
        if camera:
            from isaaclab.sensors import CameraCfg

            scene_cfg.camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Cam", update_period=0.0, height=480, width=640, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 10.0)),
                offset=CameraCfg.OffsetCfg(pos=(1.25, 0.75, 0.65), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            )
        self.scene = InteractiveScene(scene_cfg)
        self.sizes_np = sample_sizes(num_envs, size_seed, rand)
        self._scale_cubes(self.sizes_np / NOMINAL_SIZE)
        self.sim.reset()
        if camera:
            self.scene["camera"].set_world_poses_from_view(
                torch.tensor([[1.15, 0.95, 0.85]], device=device).repeat(num_envs, 1) + self.scene.env_origins,
                torch.tensor([[0.38, 0.0, 0.18]], device=device).repeat(num_envs, 1) + self.scene.env_origins)
        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        self.coaster = self.scene["coaster"]
        self.sizes = torch.tensor(self.sizes_np, dtype=torch.float32, device=device)
        self.arm_ids = self.robot.find_joints("panda_joint.*")[0]
        self.finger_ids = self.robot.find_joints("panda_finger_joint.*")[0]
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        self.jac_id = self.hand_id - 1  # fixed base: the root body has no Jacobian row
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method=ctrl.ik_method,
                                        ik_params={"lambda_val": ctrl.dls_lambda} if ctrl.ik_method == "dls" else None),
            num_envs=num_envs, device=device)
        self.offset = torch.tensor([0.0, 0.0, TCP_OFFSET], device=device).repeat(num_envs, 1)
        self.q_down = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.ws_lo = torch.tensor(WORKSPACE_LO, device=device)
        self.ws_hi = torch.tensor(WORKSPACE_HI, device=device)
        self.target_pos = torch.zeros(num_envs, 3, device=device)
        self.target_yaw = torch.zeros(num_envs, device=device)
        self.grip_cmd = torch.ones(num_envs, device=device)
        self.target_xy = torch.zeros(num_envs, 2, device=device)
        self.episodes: list[dict] = []
        self._home_q = None

    # ------------------------------------------------------------------ setup helpers
    def _scale_cubes(self, scales: np.ndarray) -> None:
        stage = omni.usd.get_context().get_stage()
        with Sdf.ChangeBlock():
            for i, s in enumerate(scales):
                path = f"/World/envs/env_{i}/Cube"
                spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), path)
                attr = spec.GetAttributeAtPath(path + ".xformOp:scale")
                new = attr is None
                if new:
                    attr = Sdf.AttributeSpec(spec, path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
                attr.default = Gf.Vec3f(float(s), float(s), float(s))
                if new:
                    order = spec.GetAttributeAtPath(path + ".xformOpOrder")
                    if order is None:
                        order = Sdf.AttributeSpec(spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
                    order.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    # ------------------------------------------------------------------ kinematics
    def tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """TCP position (env frame) and orientation; the robot base sits at the env origin, unrotated."""
        hand_pos = self.robot.data.body_pos_w[:, self.hand_id] - self.scene.env_origins
        hand_quat = self.robot.data.body_quat_w[:, self.hand_id]
        return hand_pos + quat_apply(hand_quat, self.offset), hand_quat

    def tcp_jacobian(self) -> torch.Tensor:
        jac = self.robot.root_physx_view.get_jacobians()[:, self.jac_id, :, :][:, :, self.arm_ids].clone()
        hand_quat = self.robot.data.body_quat_w[:, self.hand_id]
        r = quat_apply(hand_quat, self.offset)  # hand origin -> TCP, world frame
        # v_tcp = v_hand + w x r = v_hand - [r]x w
        jac[:, 0:3, :] -= torch.bmm(skew_symmetric_matrix(r), jac[:, 3:6, :])
        return jac

    def gripper_width(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.finger_ids].sum(dim=-1)

    def _home_joint_positions(self) -> torch.Tensor:
        """Joint configuration with the TCP at HOME_TCP pointing down (solved once with the IK controller)."""
        q = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        goal = torch.tensor(HOME_TCP, device=self.device).repeat(self.n, 1)
        self.ik.set_command(torch.cat([goal, self.q_down], dim=-1))
        for _ in range(300):
            q_arm = self.ik.compute(*self.tcp_pose(), self.tcp_jacobian(), self.robot.data.joint_pos[:, self.arm_ids])
            self.robot.set_joint_position_target(q_arm, joint_ids=self.arm_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)
        return self.robot.data.joint_pos[0].clone()

    # ------------------------------------------------------------------ episode API
    def reset(self, seeds: list[int], rand: Randomization | None = None) -> dict[str, torch.Tensor]:
        rand = rand or self.rand
        assert len(seeds) == self.n
        eps = [sample_episode(s, rand) for s in seeds]
        if self._home_q is None:
            self._home_q = self._home_joint_positions()
        ids = torch.arange(self.n, device=self.device)
        q = self._home_q.repeat(self.n, 1)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        self.robot.reset()

        f = lambda k: torch.tensor([e[k] for e in eps], dtype=torch.float32, device=self.device)  # noqa: E731
        zeros = torch.zeros(self.n, device=self.device)
        cube_quat = quat_from_euler_xyz(zeros, zeros, f("cube_yaw"))
        cube_pos = torch.stack([f("cube_x"), f("cube_y"), self.sizes / 2 + 0.001], dim=-1)
        self.cube.write_root_pose_to_sim(torch.cat([cube_pos + self.scene.env_origins, cube_quat], dim=-1), ids)
        self.cube.write_root_velocity_to_sim(torch.zeros(self.n, 6, device=self.device), ids)
        self.target_xy = torch.stack([f("target_x"), f("target_y")], dim=-1)
        coaster_pos = torch.cat([self.target_xy, torch.full((self.n, 1), 0.001, device=self.device)], dim=-1)
        ident = torch.tensor([1.0, 0, 0, 0], device=self.device).repeat(self.n, 1)
        self.coaster.write_root_pose_to_sim(torch.cat([coaster_pos + self.scene.env_origins, ident], dim=-1), ids)
        self.cube.reset()

        # physics randomization through the PhysX tensor API (CPU tensors)
        cpu_ids = torch.arange(self.n)
        masses = self.cube.root_physx_view.get_masses()
        masses[:, 0] = torch.tensor([e["cube_mass"] for e in eps])
        self.cube.root_physx_view.set_masses(masses, cpu_ids)
        mats = self.cube.root_physx_view.get_material_properties()
        fr = torch.tensor([e["friction"] for e in eps])
        mats[:, :, 0] = fr[:, None]
        mats[:, :, 1] = fr[:, None]
        self.cube.root_physx_view.set_material_properties(mats, cpu_ids)

        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(PHYSICS_DT)
        self.ik.reset()
        tcp, quat = self.tcp_pose()
        self.target_pos = tcp.clone()
        self.target_yaw = yaw_from_quat(quat_mul(quat, self._q_down_inv()))
        self.grip_cmd = torch.ones(self.n, device=self.device)
        for e, s in zip(eps, self.sizes_np, strict=True):
            e["cube_size"] = float(s)
        self.episodes = eps
        # let the cube settle on the table for 0.1 s with the arm holding still
        for _ in range(10):
            self._physics_step()
        return self.observe()

    def _q_down_inv(self) -> torch.Tensor:
        return torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device).repeat(self.n, 1)

    def observe(self) -> dict[str, torch.Tensor]:
        tcp, quat = self.tcp_pose()
        tcp_yaw = yaw_from_quat(quat_mul(quat, self._q_down_inv()))
        cube_pos = self.cube.data.root_pos_w - self.scene.env_origins
        cube_yaw = yaw_from_quat(self.cube.data.root_quat_w)
        return build_obs(tcp, tcp_yaw, self.gripper_width(), cube_pos, cube_yaw, self.target_xy, self.sizes)

    def success_now(self) -> torch.Tensor:
        tcp, _ = self.tcp_pose()
        cube_pos = self.cube.data.root_pos_w - self.scene.env_origins
        return placed(cube_pos, self.target_xy, self.sizes, self.gripper_width(), tcp)

    def _physics_step(self, render: bool = False) -> None:
        tcp, quat = self.tcp_pose()
        yaw = self.target_yaw
        zeros = torch.zeros_like(yaw)
        q_cmd = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), self.q_down)
        self.ik.set_command(torch.cat([self.target_pos, q_cmd], dim=-1))
        q_arm = self.ik.compute(tcp, quat, self.tcp_jacobian(), self.robot.data.joint_pos[:, self.arm_ids])
        self.robot.set_joint_position_target(q_arm, joint_ids=self.arm_ids)
        width = torch.where(self.grip_cmd > 0, 0.04, 0.0)[:, None].repeat(1, len(self.finger_ids))
        self.robot.set_joint_position_target(width, joint_ids=self.finger_ids)
        self.scene.write_data_to_sim()
        self.sim.step(render=render)
        self.scene.update(PHYSICS_DT)

    def step(self, action: torch.Tensor, render: bool = False) -> dict[str, torch.Tensor]:
        """action: (N, 5) = [dx, dy, dz, dyaw, gripper]; deltas are clipped, gripper is open if > 0."""
        a = action.to(self.device)
        dpos = torch.clamp(a[:, :3], -MAX_DPOS, MAX_DPOS)
        dyaw = torch.clamp(a[:, 3], -MAX_DYAW, MAX_DYAW)
        tcp, _ = self.tcp_pose()
        tgt = torch.clamp(self.target_pos + dpos, self.ws_lo, self.ws_hi)
        lead = tgt - tcp
        self.target_pos = tcp + torch.clamp(lead, -TCP_MAX_LEAD, TCP_MAX_LEAD)
        self.target_yaw = torch.clamp(self.target_yaw + dyaw, -math.pi / 2, math.pi / 2)
        self.grip_cmd = torch.where(a[:, 4] > 0, 1.0, -1.0)
        for i in range(DECIMATION):
            self._physics_step(render=render and i == DECIMATION - 1)
        return self.observe()

    def camera_rgb(self, env_id: int = 0) -> np.ndarray:
        cam = self.scene["camera"]
        cam.update(PHYSICS_DT * DECIMATION)
        return cam.data.output["rgb"][env_id, ..., :3].cpu().numpy().astype(np.uint8)
