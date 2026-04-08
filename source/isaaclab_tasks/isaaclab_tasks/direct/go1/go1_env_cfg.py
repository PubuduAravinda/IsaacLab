# go1_env_cfg.py — PaceDCMotorCfg baseline (walking reproduction)
#
# PURPOSE: Reproduce walking experiment using PaceDCMotorCfg,
#          following exact PACE documentation pattern.
#
# KEY FIXES FROM PREVIOUS BROKEN ATTEMPTS:
#   ✗ WRONG: Three separate actuator groups (hip/thigh/calf split)
#            → three independent delay buffers → corrupted delay model
#   ✗ WRONG: friction={".*": 0.001} in constructor
#            → applies PD deadzone even at near-zero → causes hyper-extension
#   ✓ RIGHT: Single actuator group, all 12 joints
#            → one delay buffer, correct per-type stiffness/damping dicts
#   ✓ RIGHT: No friction params in constructor
#            → Ia/d/τf applied via write_joint_*_to_sim in go1_env.py
#
# PACE DOCS (pace.filipbjelonic.com/tutorials/basics):
#   "Prior to training, configure your articulation object with the
#    optimized parameters for joint_armature, joint_viscous_friction,
#    and joint_friction"
#   → These go in go1_env.py _setup_scene(), NOT in PaceDCMotorCfg cfg
#
# PHASE PLAN:
#   Phase 1 (this): max_delay=0, encoder_bias=0 → reproduce ImplicitActuator
#                   Confirm walking. If reward > 25 by iter 500 → success.
#   Phase 2:        max_delay=8 (16ms, Test 3)
#   Phase 3:        Add Ia/d/τf via write methods in go1_env.py
#
# ACTUATOR SINGLE GROUP (PACE ANYmal pattern):
#   ANYDRIVE_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
#       joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
#       stiffness={".*": 85.0},   ← single wildcard (uniform type)
#       ...
#   )
#   Go1 has 3 joint types → stiffness dict with per-type patterns

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
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG
from pace_sim2real.utils import PaceDCMotorCfg


# =============================================================================
# PACE DC MOTOR CONFIG — PHASE 1 (walking reproduction)
#
# Follows exact PACE documentation pattern (pace.filipbjelonic.com):
#   - Single actuator group covering all 12 joints
#   - stiffness/damping as per-type dicts (Go1 has 3 types)
#   - encoder_bias as list of 12 floats (not dict)
#   - max_delay=0 → Phase 1: reproduce ImplicitActuator behaviour
#   - NO friction params → Ia/d/τf applied in go1_env.py via write methods
#
# Compare to PACE ANYmal (single drive type):
#   stiffness={".*": 85.0}           ← uniform
# Go1 (3 drive types):
#   stiffness={".*_hip_joint": 35.0, ...}  ← per-type
# =============================================================================
GO1_PACE_CFG = PaceDCMotorCfg(
    # Single group — ALL 12 joints
    # CRITICAL: must be one group, not three separate groups
    # Three groups = three delay buffers = corrupted delay model
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],

    # Hardware torque/velocity limits
    saturation_effort=28.0,        # calf limit (most permissive)
    effort_limit=28.0,
    velocity_limit=25.0,           # conservative Go1 joint speed limit

    # PD gains — match deployment script exactly
    # Isaac Lab applies these via the PACE actuator torque computation
    stiffness={
        ".*_hip_joint":   35.0,    # KP hip
        ".*_thigh_joint": 65.0,    # KP thigh
        ".*_calf_joint":  80.0,    # KP calf
    },
    damping={
        ".*_hip_joint":   4.0,     # KD hip
        ".*_thigh_joint": 4.5,     # KD thigh
        ".*_calf_joint":  5.0,     # KD calf
    },

    # Encoder bias [rad] — list of 12 floats (PACE ANYmal pattern)
    # Isaac order: FL_hip FR_hip RL_hip RR_hip FL_th FR_th RL_th RR_th FL_kn FR_kn RL_kn RR_kn
    # Phase 1: all zero (near-ImplicitActuator)
    # Phase 3: replace with PACE run 1 identified values
    encoder_bias={".*": 0.0},

    # Command delay buffer
    # Phase 1: 0 steps → no delay (reproduce ImplicitActuator)
    # Phase 2: 8 steps → 16ms (Test 3 measurement)
    max_delay=0,

    # NOTE: friction/dynamic_friction/viscous_friction are NOT set here
    # PACE docs: "configure your articulation object with joint_armature,
    # joint_viscous_friction, and joint_friction" → go in _setup_scene()
    # Setting them here causes PD deadzone even at near-zero values
    # which was the cause of hyper-extension in all previous PACE attempts
)


# =============================================================================
# FRICTION / DOMAIN RANDOMISATION
# =============================================================================
@configclass
class EventCfg:
    """Foot friction DR — covers real floor variation (μ ≈ 0.5–0.6)."""

    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range":  (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range":      (0.0, 0.05),
            "num_buckets": 32,
        },
    )

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range":  (0.5, 1.1),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range":      (0.0, 0.05),
            "num_buckets": 16,
        },
    )


# =============================================================================
# SCENE
# =============================================================================
@configclass
class Go1SceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(400.0, 400.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=500.0, color=(1.0, 1.0, 1.0)),
    )

    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UNITREE_GO1_CFG.spawn.replace(activate_contact_sensors=True),
        init_state=UNITREE_GO1_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.35),
            joint_pos={
                ".*_hip_joint":   0.1,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint":  -1.5,
            },
        ),
        actuators={
            # SINGLE GROUP — exact PACE documentation pattern
            # Key: one group name maps to all 12 joints via GO1_PACE_CFG
            "legs": GO1_PACE_CFG,
        },
    )

    # history_length=3: needed for last_air_time in feet_air_time reward
    # track_air_time=True: enables Rudin 2022 swing phase reward
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.005,
        history_length=3,
        debug_vis=False,
        track_air_time=True,
    )


# =============================================================================
# ENVIRONMENT
# =============================================================================
@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    """
    45-D obs, 50Hz policy.
    PaceDCMotorCfg Phase 1: max_delay=0, encoder_bias=0.
    Target: reproduce walking (reward > 25 at iter 500, ep_len > 900).

    Command: heading_command=True, ang_vel_z=(0,0)
    Confirmed across all training runs to prevent rotation-in-place exploit.
    """

    episode_length_s = 20.0
    decimation       = 10        # 500Hz sim / 10 = 50Hz policy
    num_envs         = 1000
    env_spacing      = 4.0

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
    action_space      = spaces.Box(low=-1.0,    high=1.0,    shape=(12,), dtype=np.float32)
    state_space       = spaces.Box(low=-np.inf, high=np.inf, shape=(0,),  dtype=np.float32)

    commands = mdp.commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        debug_vis=False,
        ranges=mdp.commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.9),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),          # no yaw → prevents spinning exploit
            heading=(-np.pi / 8, np.pi / 8),
        ),
        heading_command=True,
    )

    sim = SimulationCfg(
        dt=0.002,                           # 500Hz — required for max_delay=8 in Phase 2
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

    scene: Go1SceneCfg = Go1SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    events: EventCfg   = EventCfg()