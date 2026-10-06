# Controller ("behavioural model") of the Franka in this scene

The learned policy and the scripted expert never command joints directly. They output a
task-space action at 20 Hz:

| index | meaning | limit |
|---|---|---|
| 0-2 | TCP position increment dx, dy, dz in the robot base frame | 0.02 m per step (0.4 m/s) |
| 3 | yaw increment of the top-down gripper | 0.10 rad per step |
| 4 | gripper: open if > 0, closed otherwise | binary |

`skilllab/sim/env.py` turns this into joint targets in three layers:

1. **Target integration with anti wind-up.** The increment is added to the stored TCP target,
   the target is clipped to a workspace box (x 0.25 to 0.75 m, y -0.40 to 0.40 m, z 0.005 to 0.35 m)
   and is never allowed to lead the measured TCP by more than 5 cm per axis. Without the lead limit,
   pushing against the table or the cube would accumulate an unbounded target.
2. **Differential IK, damped least squares** (`isaaclab.controllers.DifferentialIKController`,
   `command_type="pose"`, `ik_method="dls"`), run at the 100 Hz physics rate. The controlled point is
   the TCP between the finger pads, 0.1034 m along the `panda_hand` z axis. The PhysX Jacobian of
   `panda_hand` is shifted to that point with v_tcp = v_hand - [r]x w, where r is the hand-to-TCP
   vector expressed in the world frame. The orientation target is always "pointing down" rotated by the
   commanded yaw, so the gripper cannot tilt.
3. **Joint PD** (implicit actuators of PhysX) tracks the IK joint targets. Arm links have gravity
   disabled, which stands in for the gravity compensation of the real arm's torque controller.

## Gains and why

Two kinds of measurement went into the choice (`scripts/tune_controller.py`, tables in
`docs/RESULTS.md` section 2):

- **open loop**: TCP step and 0.4 m/s ramp responses of the IK + PD loop for 5 gain pairs x 3 DLS values;
- **closed loop**: success rate of the scripted expert running the whole skill, 100 episodes per gain pair.

| parameter | value | reason |
|---|---|---|
| arm joint stiffness / damping | 400 Nm/rad / 80 Nms/rad | in the open-loop sweep this pair is overdamped (slower rise, more ramp lag than 400 / 40), so 400 / 40 was tried first. In closed loop it is the other way round: the expert's proportional loop on TCP increments acts as an outer integrator, and with the faster, lightly damped inner loop the combination oscillates above the cube, so the expert success drops sharply. 400 / 80 and 1000 / 200 both reach full closed-loop success; 400 / 80 does it at 40% of the stiffness, which keeps cube contacts softer and the forearm joints further from their 12 Nm effort limit. The Isaac Lab default Franka gains (80 / 4) overshoot, leave a centimetre-level error and fail the skill |
| DLS damping lambda | 0.05 | 0.01 and 0.05 give practically identical responses away from singularities; 0.05 keeps joint steps bounded near the workspace edge; 0.2 clearly slows the response |
| finger stiffness / damping | 2000 / 100 | Isaac Lab defaults for the Panda hand; with the 200 N effort limit they hold a 600 g cube |
| physics / control rate | 100 Hz / 20 Hz | IK and PD every physics step, policy every 5th step |

Lesson: a step response alone picked the wrong gains. The inner controller has to be judged together
with the outer loop that drives it, here by the skill success rate. (A full collection run with
400 / 40 kept only about three quarters of the expert episodes; those data were discarded and every
reported number uses 400 / 80.)

The tuning numbers (rise time, overshoot, settled error, ramp tracking error for 5 gain pairs x 3 DLS
values) are produced by `scripts/tune_controller.py` into `results/controller_tuning.json` and
rendered in `docs/RESULTS.md` section 2.

## Known weakness found by the evaluation

The delta-position interface integrates the commanded increments. A pure delay in the action path
(test conditions `action_delay_1` and `action_delay_2`) therefore turns the proportional
state machine (gain 0.6 per step) into an oscillating loop: the scripted expert circles above the cube
and never meets its alignment threshold. The BC policies inherit the same behaviour. A fix would be
either absolute TCP targets as the action, or delay-randomized training; neither is part of this
project yet.
