# go1_env_cfg.py — Phase 3: Full PACE calibration
#
# Changes from Phase 2:
#   ✓ viscous_friction added to GO1_PACE_CFG
#       Cfg holds healthy joint means; RL_thigh[6] (d=3.459) applied at runtime
#       in go1_env.py __init__ and _reset_idx Step 5
#   ✓ encoder_bias: cfg stays {".*": 0.0}
#       Per-joint values applied at runtime via actuator tensor (go1_env.py)
#       Avoids PaceDCMotorCfg dict-format uncertainty for per-exact-joint names
#
# Phase plan summary:
#   Phase 1: max_delay=0, encoder_bias=0, Ia/d/τf=0 → ImplicitActuator baseline
#   Phase 2: Ia + τf + delay FIFO curriculum + KP/KD DR + Kim masking
#   Phase 3: + d (viscous) + encoder_bias + per-joint rate weights  ← THIS FILE
#
# JOINT ORDER (Isaac Lab type-grouped, all 12 joints in single "legs" group):
#   [0]=FL_hip  [1]=FR_hip  [2]=RL_hip  [3]=RR_hip
#   [4]=FL_th   [5]=FR_th   [6]=RL_th*  [7]=RR_th   (* stiction fault joint)
#   [8]=FL_kn   [9]=FR_kn  [10]=RL_kn  [11]=RR_kn
#
# SINGLE GROUP PATTERN:
#   One "legs" group for all 12 joints → one delay buffer, one actuator tensor.
#   Three separate groups = three independent buffers = corrupted delay model.

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
# PACE DC MOTOR CONFIG — Phase 3
#
# PaceDCMotorCfg explicit torque model (per PACE paper):
#   τ = KP·(q_des − q) + KD·(−q̇) + d·q̇ + τf·sign(q̇) + Ia·q̈
#
# Where each parameter comes from:
#   KP / KD        → stiffness / damping dicts (match deployment script)
#   d (viscous)    → viscous_friction dict in cfg (healthy means)
#                    RL_thigh[6] override in go1_env.py (fault value d=3.459)
#   τf (Coulomb)   → NOT in cfg — write_joint_friction_coefficient_to_sim in env
#                    Reason: per-env can't be set in cfg; env write is per-joint
#   Ia (armature)  → NOT in cfg — write_joint_armature_to_sim in env
#                    Same reason: per-joint PACE values, env write is cleaner
#   q̃b (bias)     → encoder_bias={".*": 0.0} in cfg (baseline=0)
#                    Per-joint PACE values applied at runtime via actuator tensor
#   Td (delay)     → max_delay=0 in cfg; FIFO ring buffer in go1_env.py
#                    Reason: cfg max_delay = buffer SIZE (compile-time constant)
#                    not a runtime value → can't curriculum-schedule from cfg
# =============================================================================
GO1_PACE_CFG = PaceDCMotorCfg(

    # Single group — ALL 12 joints via three-pattern wildcard
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],

    # Hardware limits
    saturation_effort = 28.0,   # calf limit (most permissive joint type)
    effort_limit      = 28.0,
    velocity_limit    = 25.0,   # conservative Go1 max joint speed [rad/s]

    # PD gains — MUST match deployment SDK script exactly for zero-gap transfer
    stiffness={
        ".*_hip_joint":   35.0,   # KP [Nm/rad]
        ".*_thigh_joint": 65.0,
        ".*_calf_joint":  80.0,
    },
    damping={
        ".*_hip_joint":   4.0,    # KD [Nm·s/rad]
        ".*_thigh_joint": 4.5,
        ".*_calf_joint":  5.0,
    },

    # Phase 3: viscous damping d [Nm·s/rad] — PACE run 26_04_02 healthy means
    # These are the cfg BASELINE values restored after super()._reset_idx().
    # RL_thigh[6] (d=3.459, stiction coupling) overridden at runtime in env:
    #   go1_env.py __init__   → applies 3.459 to all envs on startup
    #   go1_env.py _reset_idx → re-applies 3.459 to reset envs after super()
    viscous_friction={
        ".*_hip_joint":   0.040,  # mean [0.039–0.042] Nm·s/rad, PACE 26_04_02
        ".*_thigh_joint": 0.048,  # healthy thigh mean [0.045–0.050]
                                  # (RL_th excluded; 3.459 set in env at runtime)
        ".*_calf_joint":  0.108,  # mean [0.092–0.124] Nm·s/rad, PACE 26_04_02
    },

    # encoder_bias: baseline 0.0 in cfg; per-joint PACE values applied at runtime
    # in go1_env.py __init__ via actuator.encoder_bias (or .position_offset).
    # Per-joint values from PACE 26_04_02/mean_199.pt:
    #   hips:   [-0.006, -0.002, -0.005, +0.004] rad
    #   thighs: [-0.018, -0.022, +0.063, -0.020] rad  (RL_th offset from fault)
    #   calves: [-0.001, -0.011, +0.012, -0.021] rad
    encoder_bias={".*": 0.0},

    # Delay: FIFO ring buffer in go1_env.py handles curriculum 0→8 steps.
    # cfg max_delay=0 → no additional internal buffer → no double-counting.
    max_delay=0,

    # τf and Ia: applied via write_joint_*_to_sim in go1_env.py __init__
    # NOT set here — setting in cfg creates PD deadzone even at ~0 values
    # (was the cause of hyper-extension in all prior PACE attempts).
)


# =============================================================================
# FRICTION / CONTACT DR
# =============================================================================
@configclass
class EventCfg:
    """Foot friction domain randomisation.

    Startup: full body friction randomised to cover rigid/carpet/rubber floors.
    Reset:   foot-only friction re-randomised each episode.
    Real Go1 floor friction: μ_static ≈ 0.5–0.7 (concrete), 0.8–1.0 (rubber).
    """

    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg":              SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range":  (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range":      (0.0, 0.05),
            "num_buckets":            32,
        },
    )

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg":              SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range":  (0.5, 1.1),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range":      (0.0, 0.05),
            "num_buckets":            16,
        },
    )


# =============================================================================
# SCENE
# =============================================================================
@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    """Flat terrain scene with contact sensors for feet."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(400.0, 400.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=500.0,
                                         color=(1.0, 1.0, 1.0)),
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
            # Single "legs" group — all 12 joints
            # Key rationale: one group = one delay buffer = correct FIFO model
            "legs": GO1_PACE_CFG,
        },
    )

    # Contact sensor: history_length=3 enables feet_air_time reward
    # track_air_time=True: Rudin 2022 swing-phase reward (currently unused
    # but preserved for future gait reward experiments)
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.005,
        history_length=3,
        debug_vis=False,
        track_air_time=True,
    )


# =============================================================================
# ENVIRONMENT CONFIG
# =============================================================================
@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    """
    45-D observation, 50Hz policy, flat terrain.

    Phase 3: Full PACE calibration active in go1_env.py.
    PaceDCMotorCfg max_delay=0 — delay via FIFO ring buffer in env.

    Command: heading_command=True, ang_vel_z=(0,0)
    Prevents rotation-in-place exploit confirmed in all prior training runs.
    """

    episode_length_s = 20.0
    decimation       = 10       # 500Hz sim / 10 = 50Hz policy
    num_envs         = 1000
    env_spacing      = 4.0

    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(46,), dtype=np.float32)  # v12: 46D with f_cmd
    action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    state_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    # Velocity command: forward only, heading-controlled, no yaw rate
    commands = mdp.commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        debug_vis=False,
        ranges=mdp.commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.9),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.0001, 0.0001),      # yaw=0 → prevents spinning exploit
            heading=(-np.pi / 8, np.pi / 8),
        ),
        heading_command=False,
    )

    sim = SimulationCfg(
        dt=0.002,                             # 500Hz — required for max_delay=8
        render_interval=decimation,           # render every policy step
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