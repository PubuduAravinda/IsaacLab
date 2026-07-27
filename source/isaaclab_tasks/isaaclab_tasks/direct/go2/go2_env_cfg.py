# go2_env_cfg.py — Go2 EDU Flat Terrain Config
#
# Follows go1_env_cfg.py structure exactly.
# No PACE calibration (clean hardware, no faults).
# No joint masking.
# Calibration values from Go2 hardware tests (Jun 2026):
#   Test 1 (IMU):     gyro_std = [0.0064, 0.0082, 0.0061] rad/s
#   Test 2 (encoder): jpos_std_max = 0.000223 rad (knee)
#                     jvel_std_max = 0.040 rad/s (hip)
#   Test 3 (spike):   τ_comm = 9.7ms → action_delay = 5 steps
#   Test 5 (sweep):   f_3dB ≈ 2.1Hz → lag_alpha = 0.21 (uniform)
#
# Joint ordering (Isaac Lab type-grouped):
#   [0]=FL_hip  [1]=FR_hip  [2]=RL_hip  [3]=RR_hip
#   [4]=FL_th   [5]=FR_th   [6]=RL_th   [7]=RR_th
#   [8]=FL_kn   [9]=FR_kn  [10]=RL_kn  [11]=RR_kn
#
# PD gains at deployment: KP=60 Kd=5 uniform (measured on hardware)
# Go2 is heavier (15kg vs 12kg), lower BW than Go1 — uniform gains appropriate.
#
# SINGLE GROUP PATTERN (same as Go1):
#   One "legs" group for all 12 joints → one delay buffer.

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
from gymnasium import spaces
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import DCMotorCfg

# Go2 URDF from isaaclab_assets — confirmed present in unitree.py
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
_GO2_BASE_CFG = UNITREE_GO2_CFG


# =============================================================================
# ACTUATOR CONFIG
# ImplicitActuator — no PACE, no fault modelling.
# Uses hardware-measured PD gains: KP=60, KD=5 uniform.
# Delay is handled via FIFO ring buffer in go2_env.py (max_delay=0 here).
# =============================================================================
GO2_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    saturation_effort=23.5,
    effort_limit=23.5,
    velocity_limit=30.0,
    stiffness={
        ".*_hip_joint":   60.0,   # calibrated KP (Test 5) uniform
        ".*_thigh_joint": 60.0,
        ".*_calf_joint":  60.0,
    },
    damping={
        ".*_hip_joint":   5.0,    # calibrated KD uniform
        ".*_thigh_joint": 5.0,
        ".*_calf_joint":  5.0,
    },
    friction=0.0,  # PACE-equivalent tau_f = 0 (brand-new platform, no
                   # PACE identification run — see go2_env.py header for
                   # the full Go1<->Go2 PACE-equivalent parameter mapping)
)


# =============================================================================
# FRICTION / CONTACT DR
# =============================================================================
@configclass
class EventCfg:
    """Foot friction domain randomisation.
    Go2 EDU operates on lab floors, grass, and gravel — wide friction range.
    """
    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg":              SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range":  (0.7, 1.3),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range":      (0.0, 0.05),
            "num_buckets":            32,
        },
    )
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg":              SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range":  (0.4, 1.2),
            "dynamic_friction_range": (0.35, 0.95),
            "restitution_range":      (0.0, 0.05),
            "num_buckets":            16,
        },
    )


# =============================================================================
# SCENE
# =============================================================================
@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Flat terrain scene with contact sensors."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(400.0, 400.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=500.0,
                                         color=(1.0, 1.0, 1.0)),
    )

    robot: ArticulationCfg = _GO2_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=_GO2_BASE_CFG.spawn.replace(activate_contact_sensors=True),
        init_state=_GO2_BASE_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.42),   # slightly higher spawn to prevent ground clip
            joint_pos={
                # Go2 standing pose from go2_stand_example.py TARGET_2
                # All hips = 0.0 (symmetric) — avoids immediate tilt at spawn
                # UNITREE_GO2_CFG default ±0.1 caused tilt=1.0 at step 1
                ".*_hip_joint":    0.0,   # ALL hips neutral (symmetric)
                ".*_thigh_joint":  0.67,  # from go2_stand_example TARGET_2
                ".*_calf_joint":  -1.3,   # from go2_stand_example TARGET_2
            },
            joint_vel={".*": 0.0},
        ),
        actuators={"legs": GO2_ACTUATOR_CFG},
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.005,
        history_length=3,
        debug_vis=True,
        track_air_time=True,
    )


# =============================================================================
# ENVIRONMENT CONFIG
# =============================================================================
@configclass
class Go2FlatEnvCfg(DirectRLEnvCfg):
    """
    45-D observation, 50Hz policy, flat terrain.
    Clean hardware — no PACE, no fault modelling.

    Obs vector (45D, same structure as Go1):
      [0:3]   velocity command (vx, vy, wz)
      [3:15]  joint position offsets from default (12)
      [15:27] joint velocities (12)
      [27:30] base angular velocity from IMU (3)
      [30:33] projected gravity vector (3)
      [33:45] previous actions (12)

    Action space (12D): joint position offsets, tanh-bounded.

    Calibrated noise (from Go2 hardware tests Jun 2026):
      gyro:  σ=[0.0064, 0.0082, 0.0061] rad/s  (Test 1 Phase 1)
      jpos:  σ_max=0.000223 rad                  (Test 2 Phase 1)
      jvel:  σ_max=0.040 rad/s                   (Test 2 Phase 1)

    Lag:
      action_delay = 5 steps (9.7ms τ_comm, Test 3)
      lag_alpha    = 0.21    (f_3dB=2.1Hz, Test 5)
      DR range     = [0.19, 0.23] ± 10%
    """

    episode_length_s = 20.0
    decimation       = 10         # 500Hz sim → 50Hz policy
    num_envs         = 1000
    env_spacing      = 4.0

    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
    action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    state_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    # Velocity command — same structure as Go1 flat
    commands = mdp.commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        debug_vis=False,
        ranges=mdp.commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.0001, 0.0001),
            heading=(-np.pi / 8, np.pi / 8),
        ),
        heading_command=False,
    )

    sim = SimulationCfg(
        dt=0.002,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.7,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
    )

    scene: Go2SceneCfg = Go2SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    events: EventCfg   = EventCfg()