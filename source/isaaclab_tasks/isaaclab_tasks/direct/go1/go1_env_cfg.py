# go1_env_cfg.py - Updated for variable impedance control
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

@configclass
class EventCfg:
    """Events configuration for domain randomization"""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    """Scene configuration with Go1 robot and sensors"""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(50.0, 50.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            intensity=400.0,
            color=(1.0, 1.0, 1.0),
            angle=1.5
        )
    )

    ground_light = AssetBaseCfg(
        prim_path="/World/GroundLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=100.0,
            color=(1.0, 0.95, 0.9),
            angle=0.3
        )
    )

    carpet_light = AssetBaseCfg(
        prim_path="/World/CarpetLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=50.0,
            color=(1.0, 0.95, 0.9),
            angle=0.8
        )
    )

    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=100.0,
            color=(0.9, 0.9, 1.0),
            angle=0.5
        )
    )

    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UNITREE_GO1_CFG.spawn.replace(
            activate_contact_sensors=True,
        )
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        debug_vis=False,
    )

    raycaster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.20, 0.0, 0.0),  # 20cm forward under chin
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        ray_alignment="yaw",  # Pattern follows robot yaw – dots rotate with robot
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.4, 0.4]),
        debug_vis=True,
        max_distance=1.0,
        mesh_prim_paths=["/World/ground"],
    )

@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    """Configuration for Go1 on flat terrain with variable impedance control"""
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 36  # 12 positions + 12 KP + 12 KD
    observation_space = 71  # Increased because previous actions are now 36D
    state_space = 0
    num_envs = 512
    env_spacing = 3.0

    # PD gain ranges for scaling policy outputs
    kp_range = (20.0, 100.0)   # Min and max KP values
    kd_range = (1.0, 5.0)      # Min and max KD values

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.3, 0.1, 0.6),
            roughness=0.9,
            metallic=0.0,
        ),
        debug_vis=False,
    )

    scene: Go1SceneCfg = Go1SceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=False
    )

    events: EventCfg = EventCfg()

    # Height curriculum params
    height_curriculum = True
    height_target_min = 0.25
    height_target_max = 0.32
    height_ramp_steps = 2_000_000

@configclass
class Go1RoughEnvCfg(Go1FlatEnvCfg):
    """Configuration for Go1 on rough terrain"""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )