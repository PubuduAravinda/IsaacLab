# go2_rough_env_cfg.py — Outdoor Terrain Config for Go2 (blind, no LiDAR)
#
# Mirrors go1_rough_env_cfg.py's structure exactly: same env class
# (go2_env.py:Go2Env) reused unchanged, only terrain/scene/friction differ.
# go2_env.py's reward function is ALREADY terrain-relative (ht_rel uses
# self._terrain_z, not raw world z) and already includes the Rudin-style
# gated feet-air-time term — so unlike a from-scratch rough config, there
# is nothing to add to the reward function itself. This file only adds
# what rough terrain actually requires: a terrain generator, a wider
# ground-friction prior, and (optional) push perturbations for robustness.
#
# "MINIMUM, CONFIDENT" REWARD PHILOSOPHY:
# We deliberately do NOT add terrain-specific reward shaping (e.g. foothold
# quality, terrain-slope-aware upright targets, exteroceptive penalties) —
# those require LiDAR/height-scan input this policy doesn't have. For a
# BLIND rough-terrain policy, the literature-standard minimum is:
#   1. Velocity tracking (already present)
#   2. Feet air-time / gait regularity (already present) — Rudin et al.
#      2022 "Learning to walk in minutes" identifies this as the key term
#      that keeps a blind policy's gait well-timed enough to blindly
#      absorb bumps rather than catching a toe on them.
#   3. A wider ground-friction domain-randomization prior than flat terrain
#      needs, since rough/outdoor surfaces vary far more than a lab floor.
#   4. Small external push perturbations (optional, see PUSH_ROBOT below) —
#      standard practice (Rudin et al. 2022, Hwangbo et al. 2019) for
#      teaching recovery from the kind of disturbance a foot catching on
#      an obstacle produces, without needing to model the obstacle itself.
# Nothing else. Extra reward terms for terrain you can't sense are just
# extra hyperparameters to get wrong.
#
# REAL-DEPLOY MOTIVATION FOR THE FRICTION WIDENING:
# The last real Go2 log (real_log_go2_..._115920.npz) showed a recurring
# ~1-2s-period tilt oscillation (mean 17 deg, spikes to 56 deg) with
# front-leg-biased tracking lag — one open hypothesis was foot-ground
# friction not matching sim. This config doesn't claim to fix that
# (would need real friction measurement to know for sure), but widening
# the DR range is the correct, low-risk response to "friction might be
# part of it": it can only make the policy more robust to whatever the
# real friction turns out to be, never less.
#
# TERRAIN SELECTION — Phase 1 only, same conservative starting set Go1
# used (flat, gravel_light, grass_wave). Widen later once flat-analog
# rough terrain is confirmed working on hardware — same phased approach
# as go1_rough_env_cfg.py, not because Go2 needs identical parameters,
# but because there's no reason to deviate from a proven starting point.
#
# USAGE:
#   python train.py --task Isaac-Velocity-Rough-Go2-v0 \
#       --num_envs 1000 --max_iterations 45000 --headless
#
#   Visualisation (one env per patch):
#   python train.py --task Isaac-Velocity-Rough-Go2-v0 --num_envs 160
#
#   Warm-start from flat Go2 policy:
#   python train.py --task Isaac-Velocity-Rough-Go2-v0 \
#       --num_envs 1000 --checkpoint .../go2_flat/.../model_XXXXX.pt --headless
#
# *** train.py WIRING REQUIRED — NOT DONE HERE ***
# go1_rough_env_cfg.py's own comments describe a train.py branch:
#   elif is_rough:
#       env_cfg = Go1RoughEnvCfg()
#       n = args_cli.num_envs or 4096
#       env_cfg.scene = make_rough_scene(n)
# I don't have your actual train.py in this conversation, so I can't
# safely edit it without risking breaking the existing Go1 branch. You
# need to add an equivalent branch here that calls make_go2_rough_scene(n)
# when --task Isaac-Velocity-Rough-Go2-v0 is selected, same shape as the
# Go1 one, before gym.make() is called.

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

# ── Terrain generator imports (same fallback pattern as go1_rough_env_cfg.py) ──
try:
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.height_field import (
        HfRandomUniformTerrainCfg,
        HfWaveTerrainCfg,
    )
    from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
    _TERRAIN_OK = True
except ImportError:
    try:
        from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
        from isaaclab.terrains.height_field.hf_terrains_cfg import (
            HfRandomUniformTerrainCfg,
            HfWaveTerrainCfg,
        )
        from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
        _TERRAIN_OK = True
    except ImportError as e:
        print(f"[go2_rough_env_cfg] terrain imports failed: {e}")
        _TERRAIN_OK = False

from isaaclab_tasks.direct.go2.go2_env_cfg import Go2FlatEnvCfg, Go2SceneCfg

# =============================================================================
# TERRAIN GRID CONSTANTS — same 8x20 grid Go1 uses, no reason to change
# =============================================================================
TERRAIN_NUM_ROWS = 8
TERRAIN_NUM_COLS = 20
TERRAIN_TOTAL_PATCHES = TERRAIN_NUM_ROWS * TERRAIN_NUM_COLS   # 160

# =============================================================================
# TERRAIN SELECTION — Phase 1 only (see header). Extend the same way Go1
# planned Phase 2/3, once Phase 1 is validated on real hardware.
# =============================================================================
ACTIVE_TERRAINS = [
    "flat",          # always keep — policy needs a flat baseline
    "gravel_light",  # gravel path / packed dirt, +-2-4cm
    "grass_wave",    # rolling lawn / undulating ground
]

ALL_TERRAIN_DEFS = {}
if _TERRAIN_OK:
    ALL_TERRAIN_DEFS = {
        "flat": MeshPlaneTerrainCfg(
            proportion=0.50,
        ),
        "gravel_light": HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.02, 0.04),
            noise_step=0.01,
            border_width=0.1,
        ),
        "grass_wave": HfWaveTerrainCfg(
            proportion=0.25,
            amplitude_range=(0.02, 0.05),
            num_waves=5,
            border_width=0.1,
        ),
    }

_active_sub = {k: ALL_TERRAIN_DEFS[k] for k in ACTIVE_TERRAINS
               if k in ALL_TERRAIN_DEFS}
assert len(_active_sub) > 0, f"No valid terrains in ACTIVE_TERRAINS: {ACTIVE_TERRAINS}"
print(f"[TERRAIN] Go2 rough active: {list(_active_sub.keys())}")

if _TERRAIN_OK:
    OUTDOOR_TERRAIN_CFG = TerrainGeneratorCfg(
        seed=42,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=TERRAIN_NUM_ROWS,
        num_cols=TERRAIN_NUM_COLS,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        color_scheme="height",
        sub_terrains=_active_sub,
    )


def make_go2_rough_scene(num_envs: int) -> "Go2RoughSceneCfg":
    """
    Build a Go2RoughSceneCfg with the CORRECT num_envs at construction time.
    Same @configclass-freezing issue Go1 documents — see go1_rough_env_cfg.py
    header. Call this from train.py BEFORE gym.make(), same pattern as
    Go1's make_rough_scene(n).
    """
    return Go2RoughSceneCfg(num_envs=num_envs, env_spacing=0.0)


# =============================================================================
# FRICTION / CONTACT DR — widened for rough terrain
#
# Flat Go2FlatEnvCfg's EventCfg (go2_env_cfg.py) uses:
#   robot_friction static (0.7,1.3) dynamic (0.6,1.0)  -- lab-floor range
#   foot_friction  static (0.5,1.1) dynamic (0.4,0.9)
# Rough terrain needs a wider prior anchored the same way Go1's rough
# config anchors its friction_range_lo/hi = [0.35, 0.80] (mud to dry
# gravel). We widen the LOW end to match that anchor and keep the
# existing upper bound from the flat cfg (already covers dry/rubber).
# =============================================================================
@configclass
class Go2RoughEventCfg:
    """Foot/body friction DR widened for rough terrain, plus optional push."""

    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg":              SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range":  (0.35, 1.3),   # mud .. dry/rubber
            "dynamic_friction_range": (0.30, 1.0),
            "restitution_range":      (0.0, 0.05),
            "num_buckets":            32,
        },
    )

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg":              SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range":  (0.30, 1.1),
            "dynamic_friction_range": (0.25, 0.95),
            "restitution_range":      (0.0, 0.05),
            "num_buckets":            16,
        },
    )

    # ── Optional: small external push perturbations ────────────────────────
    # Standard practice (Rudin et al. 2022, Hwangbo et al. 2019) for
    # teaching recovery from a foot-catch-on-obstacle-style disturbance
    # without needing to simulate the obstacle. Kept modest per the
    # "minimum, confident" brief -- a small periodic base-velocity kick,
    # not a large shove.
    #
    # *** VERIFY FUNCTION NAME AGAINST YOUR ISAAC LAB VERSION ***
    # mdp.push_by_setting_velocity exists in common Isaac Lab manager-based
    # example configs (e.g. the H1/ANYmal rough-terrain configs), but I
    # can't confirm it's present/named identically in your installed
    # version from here. If mdp.push_by_setting_velocity doesn't exist,
    # either remove this term or grep your isaaclab.envs.mdp package for
    # the equivalent (sometimes named push_robot / apply_external_force_torque).
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
        },
    )


# =============================================================================
# SCENE CONFIG
# =============================================================================
@configclass
class Go2RoughSceneCfg(Go2SceneCfg):
    """Go2 scene with outdoor terrain. Build via make_go2_rough_scene(n)."""
    env_spacing: float = 0.0   # terrain generator controls placement

    terrain = TerrainImporterCfg(
        prim_path              = "/World/ground",
        terrain_type           = "generator" if _TERRAIN_OK else "plane",
        terrain_generator      = OUTDOOR_TERRAIN_CFG,
        max_init_terrain_level = TERRAIN_NUM_ROWS - 1,
        collision_group        = -1,
        physics_material       = sim_utils.RigidBodyMaterialCfg(
            static_friction  = 0.60,   # single terrain-level default;
            dynamic_friction = 0.60,   # per-episode variation comes from
            restitution      = 0.01,   # Go2RoughEventCfg above, not here
        ),
        debug_vis = False,
    )


# =============================================================================
# ENV CONFIG
# =============================================================================
@configclass
class Go2RoughEnvCfg(Go2FlatEnvCfg):
    """
    Go2 rough terrain — inherits all flat settings (obs, action space,
    rewards live in go2_env.py and are unchanged; already terrain-relative
    via self._terrain_z). scene is rebuilt by train.py via
    make_go2_rough_scene(num_envs), same reason as Go1.
    """
    episode_length_s : float = 25.0
    env_spacing      : float = 0.0
    # NOTE: scene is overridden by train.py using make_go2_rough_scene()
    # Default here is just a placeholder with num_envs=1
    scene  : Go2RoughSceneCfg = Go2RoughSceneCfg(num_envs=1, env_spacing=0.0)
    events : Go2RoughEventCfg = Go2RoughEventCfg()
