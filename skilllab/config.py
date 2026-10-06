"""Task, randomization and controller settings (pure Python, no simulator import).

Everything is expressed in the robot base frame: x forward, y left, z up, table top at z = 0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

PHYSICS_DT = 0.01  # 100 Hz PhysX step
DECIMATION = 5  # IK + PD run every physics step, the policy runs at 20 Hz
POLICY_DT = PHYSICS_DT * DECIMATION
EPISODE_STEPS = 200  # 10 s at 20 Hz

# action layout: [dx, dy, dz, dyaw, gripper]; gripper > 0 means open
ACTION_DIM = 5
MAX_DPOS = 0.02  # m per policy step (0.4 m/s)
MAX_DYAW = 0.10  # rad per policy step

# TCP = point between the finger pads, measured from the panda_hand frame along its z axis
TCP_OFFSET = 0.1034
TCP_MAX_LEAD = 0.05  # anti wind-up: commanded TCP target never leads the measured TCP by more than this
WORKSPACE_LO = (0.25, -0.40, 0.005)
WORKSPACE_HI = (0.75, 0.40, 0.35)
HOME_TCP = (0.45, 0.0, 0.25)

# success: cube centre within this radius of the coaster centre, resting on the table, released
SUCCESS_RADIUS = 0.03
SUCCESS_REST_TOL = 0.01
SUCCESS_HOLD_STEPS = 5  # condition must hold for 0.25 s


@dataclass(frozen=True)
class Randomization:
    """Ranges sampled per episode (pose, mass, friction) or per env slot at startup (cube size)."""

    name: str = "dr"
    cube_x: tuple[float, float] = (0.40, 0.60)
    cube_y: tuple[float, float] = (-0.20, 0.20)
    cube_yaw_deg: tuple[float, float] = (-45.0, 45.0)
    target_x: tuple[float, float] = (0.40, 0.60)
    target_y: tuple[float, float] = (-0.25, 0.25)
    min_cube_target_dist: float = 0.12
    cube_size: tuple[float, float] = (0.040, 0.060)  # edge length, m
    cube_mass: tuple[float, float] = (0.05, 0.30)  # kg
    friction: tuple[float, float] = (0.5, 1.2)  # static = dynamic coefficient of the cube


NOMINAL_SIZE = 0.05
NOMINAL_MASS = 0.10
NOMINAL_FRICTION = 0.8

DR = Randomization()
# "no DR": the cube pose and coaster pose still vary (otherwise the skill is trivial), physics is fixed
NO_DR = replace(DR, name="no_dr", cube_size=(NOMINAL_SIZE, NOMINAL_SIZE), cube_mass=(NOMINAL_MASS, NOMINAL_MASS),
                friction=(NOMINAL_FRICTION, NOMINAL_FRICTION))


@dataclass(frozen=True)
class Condition:
    """An evaluation condition: a randomization plus test-time perturbations applied by the harness."""

    name: str
    rand: Randomization = DR
    obs_noise_std: float = 0.0  # m, Gaussian noise on the cube and coaster positions the policy sees
    action_delay: int = 0  # policy steps between the observation and the executed action
    note: str = ""


CONDITIONS: dict[str, Condition] = {c.name: c for c in [
    Condition("nominal", note="training distribution (DR ranges)"),
    Condition("wide_pose", replace(DR, name="wide_pose", cube_x=(0.35, 0.68), cube_y=(-0.30, 0.30),
                                   target_x=(0.35, 0.68), target_y=(-0.32, 0.32)),
              note="cube and coaster sampled from a 1.6x larger area"),
    Condition("heavy_slippery", replace(DR, name="heavy_slippery", cube_mass=(0.40, 0.60), friction=(0.25, 0.40)),
              note="mass and friction outside the training ranges"),
    Condition("obs_noise_5mm", obs_noise_std=0.005, note="sigma 5 mm on cube and coaster positions"),
    Condition("action_delay_1", action_delay=1, note="actions applied one policy step (50 ms) late"),
    Condition("action_delay_2", action_delay=2, note="actions applied two policy steps (100 ms) late"),
]}
# size shifts need a different cube USD scale, so they run in their own simulator process
SIZE_CONDITIONS: dict[str, Condition] = {
    "small_cubes": Condition("small_cubes", replace(DR, name="small_cubes", cube_size=(0.032, 0.040)),
                             note="edge 32 to 40 mm, below the 40 to 60 mm training range"),
}


@dataclass(frozen=True)
class ControllerCfg:
    """Differential IK (damped least squares) on top of joint PD, see docs/CONTROLLER.md."""

    ik_method: str = "dls"
    dls_lambda: float = 0.05
    arm_stiffness: float = 400.0
    arm_damping: float = 80.0
    finger_stiffness: float = 2000.0
    finger_damping: float = 100.0
    extra: dict = field(default_factory=dict)


CONTROLLER = ControllerCfg()


def wrap_quarter_turn(a: float) -> float:
    """Wrap an angle to [-pi/4, pi/4): a cube looks the same after a 90 degree turn."""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


def to_dict(obj) -> dict:
    return asdict(obj)
