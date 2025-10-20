# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, CameraCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

@configclass
class EventCfg:
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
class Go1FlatEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 12
    observation_space = 47
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
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
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=5,
        env_spacing=3.0,
        replicate_physics=False
    )

    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,  # Update every step
        track_air_time=True,
        force_threshold=0.1,
    )

    # 🔥 FIXED: Camera configuration following Isaac Lab tutorial pattern
    # Key changes:
    # 1. Path: {ENV_REGEX_NS}/Robot/trunk/front_cam (NOT height_camera child)
    # 2. Update period: 0.0 (update every step, not just occasionally)
    # 3. Clipping range: (0.1, 2.0) for ground detection
    height_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk/front_cam",  # 🔥 Correct pattern from Isaac Lab docs
        update_period=0.0,  # 🔥 Update every physics step (0.0 = every step)
        height=64,
        width=64,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,  # Focus at 40cm (typical ground distance)
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),  # 10cm to 2m range
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.35, 0.0, 0.0),  # 35cm forward from trunk center
            rot=(0.7071068, 0.0, 0.0, 0.7071068),  # 90° pitch down (looking straight down)
            convention="ros"
        ),
        debug_vis=True,  # Enable visualization sphere
    )

    events: EventCfg = EventCfg()

@configclass
class Go1RoughEnvCfg(Go1FlatEnvCfg):
    observation_space = 47

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )