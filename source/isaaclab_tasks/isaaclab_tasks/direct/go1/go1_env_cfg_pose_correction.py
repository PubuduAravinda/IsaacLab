# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# ✅ Import Go1 robot config
# from isaaclab_assets.robots.go1 import GO1_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


@configclass
class EventCfg:
    """Randomization config for Go1."""

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
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),  # Go1 uses "trunk" as base
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )


@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12   # Go1 has 12 actuated joints
    observation_space = 48
    state_space = 0

    # simulation
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    # robot: ArticulationCfg = GO1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Robot
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.005,
        track_air_time=True,
        force_threshold=0.1,
    )

    # In go1_env_cfg.py - Positive-only scales
    contact_scale: float = 0.8
    upright_scale: float = 0.7
    stability_scale: float = 0.6
    movement_smoothness_scale: float = 0.5
    stance_duration_scale: float = 0.6
    joint_position_scale: float = 0.4
    standing_bonus_scale: float = 1.0
    energy_efficiency_scale: float = 0.3


@configclass
class Go1RoughEnvCfg(Go1FlatEnvCfg):
    # env
    observation_space = 235

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

    # add height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # override reward
    flat_orientation_reward_scale = 0.0