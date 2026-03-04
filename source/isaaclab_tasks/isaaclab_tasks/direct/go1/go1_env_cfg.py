# go1_env_cfg.py
# Flat terrain only. 45D obs (no foot contacts). 50Hz policy (decimation=10).
# Keeps identical @configclass pattern as original working file.
# To change num_envs: edit the class-level field below — do NOT patch at runtime.

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
    """Light friction DR — stable for flat terrain walking."""

    physics_material = EventTerm(
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


@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    """Flat ground + Go1 robot + contact sensor."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(400.0, 400.0)),  # big enough for 4096 envs @ 4m spacing
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
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                stiffness=50.0,   # kp — matches real Go1 SDK
                damping=6.0,      # kd
                effort_limit=23.5,
            ),
        },
    )

    # Used only in r_even reward — NOT in observations (real Go1 has no foot sensors)
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
    Flat terrain env. 45-D obs. 50 Hz policy.

    Obs layout (45-D — all readable from real Go1 hardware):
        [0:3]   velocity commands (vx, vy, wz)
        [3:15]  joint pos delta from default        (encoders)
        [15:27] joint velocity                      (encoders)
        [27:30] base angular velocity               (IMU gyro)
        [30:33] projected gravity                   (IMU orientation)
        [33:45] previous actions                    (buffer)

    No foot contacts — real Go1 has no foot force sensors.

    IMPORTANT: @configclass bakes scene(num_envs, env_spacing) at import time.
    Change num_envs / env_spacing HERE in the class body — do NOT patch scene
    attributes at runtime. For play, pass --num_envs 1 and the cfg default
    is overridden safely because play.py rebuilds a fresh Go1FlatEnvCfg().
    """

    episode_length_s = 20.0
    decimation       = 10       # 500 Hz / 10 = 50 Hz policy — matches real Go1 SDK

    # ── Set training scale here ───────────────────────────────────────────────
    num_envs    = 1000# 4096   # reduce to 500 if VRAM limited
    env_spacing = 4.0    # 4 m spacing → clean grid at 4096 envs (64×64)

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
    action_space      = spaces.Box(low=-1.0,    high=1.0,    shape=(12,), dtype=np.float32)
    state_space       = spaces.Box(low=-np.inf, high=np.inf, shape=(0,),  dtype=np.float32)

    commands = mdp.commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),  # long enough to build momentum
        debug_vis=False,
        ranges=mdp.commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.9),            # slow walk only — learn stability first
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
            static_friction=0.7,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
    )

    scene: Go1SceneCfg = Go1SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    events: EventCfg   = EventCfg()