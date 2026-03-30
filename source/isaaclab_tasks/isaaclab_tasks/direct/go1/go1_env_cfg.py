# go1_env_cfg.py  — v3
# Flat terrain. 45D obs. 50Hz policy (decimation=10).
# Changes from v2:
#   - EventCfg: terrain_friction added (DR [0.4,1.2]) — covers real floor μ≈0.5-0.6
#   - EventCfg: randomize_actuator_gains removed — go1_env.py handles KP DR
#     per joint type (hips/thighs [0.40,1.20], knees [0.60,1.20])
#   - knee init_state kept at -1.5 (reverted from -1.3 which caused over-extension)

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
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class EventCfg:
    """Friction DR for robot body + feet — covers real floor variation.

    Actuator gain DR is handled in go1_env.py _reset_idx() per-joint-type.

    NOTE: ground plane is an XFormPrim — Isaac Lab cannot apply
    randomize_rigid_body_material to it. Instead we randomize the robot's
    foot friction each episode, which achieves the same physics effect:
    lower foot friction = more slip = same forward pitch as on real floor μ≈0.5-0.6.
    """

    # Robot body friction at startup — baseline for all body links
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

    # Foot friction DR every episode — simulates different floor surfaces.
    # Real floor μ≈0.5-0.6, sim default=1.0. Randomizing foot friction
    # [0.4,1.2] makes policy learn to handle slippery floors, preventing
    # the over-push that caused persistent grav_x≈+0.08 on real hardware.
    # body_names=".*foot" targets only the 4 foot links, not the whole body.
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


@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    """Flat ground + Go1 robot + contact sensor."""

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
                ".*_calf_joint":  -1.5,   # keep at -1.5 (reverted from -1.3)
                                           # knee init=-1.3 caused over-extension
                                           # (policy extends +0.35 → target=-0.95,
                                           #  nearly straight. -1.5 gives -1.15 target
                                           #  which matches real hardware actual)
            },
        ),
        actuators={
            # Nominal KP/KD — go1_env.py applies per-joint DR ranges each episode:
            #   hips/thighs: [0.40,1.20]×nom — covers noisy FR motor (47% nominal)
            #   knees:       [0.60,1.20]×nom — prevents sim knee lag > real hardware
            "hip_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint"],
                stiffness=35.0,
                damping=4.0,
                effort_limit=23.7,
            ),
            "thigh_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_thigh_joint"],
                stiffness=65.0,
                damping=4.5,
                effort_limit=23.7,
            ),
            "calf_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_calf_joint"],
                stiffness=80.0,
                damping=5.0,
                effort_limit=28.0,
            ),
        },
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.005,
        history_length=1,
        debug_vis=False,
        track_air_time=True,
    )


@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    """
    Flat terrain env. 45-D obs. 50Hz policy.

    Obs layout (45-D):
        [0:3]   velocity commands (vx, vy, wz)
        [3:15]  joint pos delta from default (encoders)
        [15:27] joint velocity               (encoders)
        [27:30] base angular velocity        (IMU gyro)
        [30:33] projected gravity            (IMU orientation)
        [33:45] previous actions             (buffer)

    IMPORTANT: @configclass bakes scene(num_envs) at import time.
    Edit num_envs here — do NOT patch at runtime.
    """

    episode_length_s = 20.0
    decimation       = 10       # 500 Hz / 10 = 50 Hz

    num_envs    = 1000  # reduce to 500 if VRAM limited
    env_spacing = 4.0

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
            ang_vel_z=(0.0, 0.0),
            heading=(-np.pi / 8, np.pi / 8),
        ),
        heading_command=True,
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
            static_friction=0.7,    # starting value — terrain_friction DR overrides per episode
            dynamic_friction=0.7,
            restitution=0.0,
        ),
    )

    scene: Go1SceneCfg = Go1SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    events: EventCfg   = EventCfg()