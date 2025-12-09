# go1_env_cfg.py - HIMLoco-aligned for Isaac Lab 2.2.1
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
    """Domain randomization events (from HIMLoco paper)"""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
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
            "mass_distribution_params": (-2.0, 3.0),  # HIMLoco: ±2-3kg
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
            pos=(0.0, 0.0, 0.35),
            joint_pos={
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.9,
                ".*_calf_joint": -1.8,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                stiffness=50.0,
                damping=2.5,
                effort_limit=33.5,
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
    decimation = 4  # 50Hz policy (200Hz sim / 4)

    # Environment settings
    num_envs = 4096
    env_spacing = 3.0

    # HIMLoco history settings
    history_length = 3  # 3 timesteps of proprioceptive history

    # Observation space: HIMLoco uses stacked proprioceptive obs
    # Base obs (per timestep): cmd(3) + joints(12) + joint_vel(12) +
    #                          ang_vel(3) + gravity(3) + actions(12) + contacts(4) = 49
    # With 3-step history: 49 * 3 = 147
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(147,), dtype=np.float32)

    # Action space: 12D joint position offsets
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    # State space (privileged info - not used in HIMLoco)
    state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    # Action scaling
    action_scale = 0.5  # Conservative for position control

    # Velocity commands (HIMLoco ranges from paper)
    commands = mdp.commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # Fixed 10s curriculum
        debug_vis=False,
        ranges=mdp.commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),  # Forward/backward
            lin_vel_y=(-0.5, 0.5),  # Lateral
            ang_vel_z=(-1.0, 1.0),  # Yaw rate
            heading=(-np.pi, np.pi),
        ),
    )

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,  # 200Hz simulation
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain (flat ground)
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

    # Scene
    scene: Go1SceneCfg = Go1SceneCfg(num_envs=num_envs, env_spacing=env_spacing)

    # Domain randomization events
    events: EventCfg = EventCfg()


@configclass
class Go1RoughEnvCfg(Go1FlatEnvCfg):
    """HIMLoco configuration for rough terrain"""

    # Override terrain with rough terrain generator
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,  # Start with easier terrains
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # Curriculum: gradually increase terrain difficulty
    curriculum = True