# go1_env_cfg.py - HIMLoco-aligned for Isaac Lab 2.2.1 (CORRECTED)
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
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


@configclass
class EventCfg:
    """Minimal DR that works without param errors"""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.3),
            "damping_distribution_params": (0.8, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "com_range": {
                "x": (-0.08, 0.08),
                "y": (-0.08, 0.08),
                "z": (-0.04, 0.04),
            },
        },
    )

    # external_force = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",  # Only apply on reset — safe and supported
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
    #         "force_range": (-25.0, 25.0, -25.0, 25.0, -12.0, 12.0),
    #         # flat 6 values: min_x max_x min_y max_y min_z max_z
    #         "torque_range": (-8.0, 8.0, -8.0, 8.0, -8.0, 8.0),  # flat 6 values
    #     },
    # )

# @configclass
# class EventCfg:
#     """Domain randomization events"""
#     physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "static_friction_range": (0.8, 1.25),
#             "dynamic_friction_range": (0.6, 1.0),
#             "restitution_range": (0.0, 0.1),
#             "num_buckets": 64,
#         },
#     )
#
#     add_base_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
#             "mass_distribution_params": (-1.0, 2.0),
#             "operation": "add",
#         },
#     )



@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    """Scene configuration for Go1 quadruped"""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=500.0, color=(1.0, 1.0, 1.0))
    )

    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UNITREE_GO1_CFG.spawn.replace(activate_contact_sensors=True),
        init_state=UNITREE_GO1_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.35),  # Optional: also lower here for consistency
            joint_pos={
                ".*_hip_joint": 0.1,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                stiffness=30.0,  # Align to paper/repo (was 10.0)
                damping=0.75,  # Align to paper/repo (was 0.2)
                effort_limit=23.5,
            ),
        },
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",  # Only feet
        update_period=0.005,
        history_length=1,
        debug_vis=True,
        track_air_time=True,
    )


@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    """HIMLoco environment configuration for flat terrain"""

    # Episode settings
    episode_length_s = 20.0
    decimation = 8  # Updated: 400 Hz physics / 8 = 50 Hz policy

    # Environment settings
    num_envs = 1000 #4096
    env_spacing = 3.0

    # HIMLoco history
    history_length = 5
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    # Action scaling - CRITICAL: legged_gym Go1 uses 0.25
    action_scale = 0.25

    # Velocity commands - LIMITED for small dogs per HIMLoco issue #6
    commands = mdp.commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.5, 0.5),
        debug_vis=False,
        # ranges=mdp.commands.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(0.6, 1.6),  # Strong forward bias — forces learning to walk forward
        #     lin_vel_y=(-0.4, 0.4),  # Small lateral
        #     ang_vel_z=(-1.0, 1.0),  # Moderate yaw
        #     heading=(-np.pi, np.pi),
        # ),
        ranges=mdp.commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.2, 2.2),  # Strong positive forward — no backward
            lin_vel_y=(0.0, 0.0),  # Zero lateral
            ang_vel_z=(0.0, 0.0),  # Zero yaw — straight line only
            heading=(-np.pi / 10, np.pi / 10),  # Small heading
        ),
        heading_command=True,
    )

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 400.0,  # 400 Hz physics for more stability (was 1/200)
        render_interval=decimation,  # Render every policy step
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: Go1SceneCfg = Go1SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    events: EventCfg = EventCfg()


@configclass
class Go1RoughEnvCfg(Go1FlatEnvCfg):
    """HIMLoco configuration for rough terrain"""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    curriculum = True