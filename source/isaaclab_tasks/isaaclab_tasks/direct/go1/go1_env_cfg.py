# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg
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

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(50.0, 50.0)),
    )

    # In your Go1SceneCfg
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,  # Increased for better carpet illumination
            color=(1.0, 1.0, 1.0),
            angle=1.5  # Wider coverage
        )
    )

    # Add additional focused light on the ground
    ground_light = AssetBaseCfg(
        prim_path="/World/GroundLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=1500.0,
            color=(1.0, 0.95, 0.9),  # Warm light for better carpet colors
            angle=0.3  # Focused on ground
        )
    )

    carpet_light = AssetBaseCfg(
        prim_path="/World/CarpetLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=800.0,
            color=(1.0, 0.95, 0.9),  # Slightly warm light
            angle=0.8
        )
    )

    # Add additional fill light
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=500.0,
            color=(0.9, 0.9, 1.0),
            angle=0.5
        )
    )

    # Robot configuration
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UNITREE_GO1_CFG.spawn.replace(
            activate_contact_sensors=True,
        )
    )

    # Contact sensor
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        debug_vis=False,
    )

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk/front_cam",
        update_period=0.0,
        height=320,
        width=320,
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=4.0,  # Wider angle for closer view
            focus_distance=0.3,  # Focus on closer ground
            horizontal_aperture=40.0,  # Wider view
            clipping_range=(0.05, 2.0),
            f_stop=1.8,  # More light for closer view
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.18),  # LOWER: 18cm below trunk instead of 10cm
            rot=(0.0, 0.0, 0.0, 1.0),  # Keep straight down
            convention="ros"
        ),
    )


@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    """Configuration for Go1 on flat terrain with carpet-like ground"""

    # Environment settings
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 12
    observation_space = 47 + 320 * 320 * 3  # 47 state + flattened RGB (HxWx3)
    state_space = 0
    num_envs = 5
    env_spacing = 3.0

    # Simulation settings
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

    # Flat terrain configuration with visual material
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
        # Use a predefined material
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.3, 0.1, 0.6),  # Purple
            roughness=0.9,
            metallic=0.0,
        ),
        debug_vis=False,
    )

    # Scene with robot and sensors
    scene: Go1SceneCfg = Go1SceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=False
    )

    # Domain randomization events
    events: EventCfg = EventCfg()


@configclass
class Go1RoughEnvCfg(Go1FlatEnvCfg):
    """Configuration for Go1 on rough terrain"""

    # Override terrain for rough terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,  # Increased for carpet-like friction
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )