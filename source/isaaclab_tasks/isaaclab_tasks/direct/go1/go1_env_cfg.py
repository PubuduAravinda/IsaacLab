import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg  # Remove CameraCfg if not used
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
        prim_path="{ENV_REGEX_NS}/Robot/trunk",  # Attached to trunk body prim – auto-follows
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.20, 0.0, 0.0),  # 20cm forward under chin in trunk local frame
            rot=(0.0, 0.0, 0.0, 1.0),  # Identity – no additional rotation
        ),
        ray_alignment="yaw",  # Match ANYmal – pattern follows robot yaw (rotation); fixes viz/update bug
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.4, 0.4]),  # 3x3 grid = 9 rays
        debug_vis=True,  # Red dots now rotate with chin on robot yaw
        max_distance=1.0,
        mesh_prim_paths=["/World/ground"],
    )

@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    """Configuration for Go1 on flat terrain with carpet-like ground"""
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 12
    observation_space = 47  # State-only observation
    state_space = 0
    num_envs = 5
    env_spacing = 3.0

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