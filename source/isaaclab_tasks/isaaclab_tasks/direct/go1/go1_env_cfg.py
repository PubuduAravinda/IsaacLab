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
    """Domain randomization events"""
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
            pos=(0.0, 0.0, 0.42),  # Start higher to avoid initial ground contact
            joint_pos={
                # Better initial pose for standing
                ".*_hip_joint": 0.1,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                stiffness=20.0,  # legged_gym Go1 default
                damping=0.5,  # legged_gym Go1 default
                effort_limit=23.5,  # Go1 spec (not 33.5)
            ),
        },
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        track_air_time=True,
    )


@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    """HIMLoco environment configuration for flat terrain"""

    # Episode settings
    episode_length_s = 20.0
    decimation = 4  # 50Hz policy

    # Environment settings
    num_envs = 4096
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
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 2.0),  # Limited to 2m/s max per HIMLoco
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.5, 1.5),
            heading=(-np.pi, np.pi),
        ),
    )

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
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