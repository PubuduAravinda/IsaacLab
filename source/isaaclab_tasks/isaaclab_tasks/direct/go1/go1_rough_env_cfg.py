# go1_rough_env_cfg.py — Outdoor Terrain Config for Go1 v2
#
# STACKING FIX (v2):
#   Root cause: Go1RoughSceneCfg hardcoded num_envs=4096 at class definition.
#   When train.py sets env_cfg.scene.num_envs=160, the @configclass default
#   was already frozen. Terrain importer saw 4096 envs → 4096/160patches = 26
#   envs per patch → stacking.
#
#   Fix: scene created with num_envs=1 (sentinel), train.py rebuilds it with
#   the correct num_envs BEFORE gym.make(). See train.py elif is_rough block.
#
# COLOUR (v2):
#   color_scheme="height" — gradient from dark (low) to bright (high).
#   Makes terrain features clearly visible regardless of lighting.
#
# LIGHTING (v2):
#   Handled in go1_env.py _setup_scene via USD API (most reliable in IL 2.2.1).
#
# USAGE:
#   python train.py --task Isaac-Velocity-Rough-Go1-Direct-v0 \
#       --num_envs 4096 --max_iterations 60000 --headless
#
#   Visualisation (one env per patch):
#   python train.py --task Isaac-Velocity-Rough-Go1-Direct-v0 --num_envs 160
#
#   Warm-start from flat:
#   python train.py --task Isaac-Velocity-Rough-Go1-Direct-v0 \
#       --num_envs 4096 --checkpoint .../model_44000.pt --headless

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

# ── Terrain generator imports ──────────────────────────────────────────────
try:
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.height_field import (
        HfRandomUniformTerrainCfg,
        HfPyramidSlopedTerrainCfg,
        HfDiscreteObstaclesTerrainCfg,
        HfWaveTerrainCfg,
    )
    from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
    _TERRAIN_OK = True
except ImportError:
    try:
        from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
        from isaaclab.terrains.height_field.hf_terrains_cfg import (
            HfRandomUniformTerrainCfg,
            HfPyramidSlopedTerrainCfg,
            HfDiscreteObstaclesTerrainCfg,
            HfWaveTerrainCfg,
        )
        from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
        _TERRAIN_OK = True
    except ImportError as e:
        print(f"[go1_rough_env_cfg] terrain imports failed: {e}")
        _TERRAIN_OK = False

from isaaclab_tasks.direct.go1.go1_env_cfg import Go1FlatEnvCfg, Go1SceneCfg

# =============================================================================
# TERRAIN GRID CONSTANTS
# Referenced by train.py to rebuild scene with correct num_envs.
# =============================================================================
TERRAIN_NUM_ROWS = 8    # difficulty levels: 0=flat, 7=steep+obstacles
TERRAIN_NUM_COLS = 20   # patches per difficulty level
TERRAIN_TOTAL_PATCHES = TERRAIN_NUM_ROWS * TERRAIN_NUM_COLS   # 160

# =============================================================================
# OUTDOOR TERRAIN GENERATOR
# =============================================================================
# =============================================================================
# TERRAIN SELECTION — change this list to switch training terrain set
# =============================================================================
# Phase 1 (current): 3 basic outdoor terrains → prove sim-to-real works
# Phase 2 (later):   add slope_gentle, obstacles
# Phase 3 (final):   full suite including slope_steep
#
# ALL terrain definitions are kept below — just comment/uncomment here.
# Proportions are renormalised automatically by Isaac Lab.
# =============================================================================
ACTIVE_TERRAINS = [
    "flat",           # ← always keep — policy needs flat baseline
    "gravel_light",   # ← gravel path / packed dirt
    "grass_wave",     # ← rolling lawn / undulating ground
    # ── Phase 2 ─────────────────────────────────────────────
    # "gravel_heavy",   # loose stone / coarse gravel
    # "slope_gentle",   # ramps 3-12°
    # ── Phase 3 ─────────────────────────────────────────────
    # "obstacles",      # roots / rocks 3-12cm
    # "slope_steep",    # embankments 12-20°
]

# =============================================================================
# ALL TERRAIN DEFINITIONS (full library — selection controlled by ACTIVE_TERRAINS)
# =============================================================================
ALL_TERRAIN_DEFS = {}

if _TERRAIN_OK:
    ALL_TERRAIN_DEFS = {
        # ── Phase 1 — outdoor basics ──────────────────────────────────────
        "flat": MeshPlaneTerrainCfg(
            proportion = 0.50,   # proportion will be renormalised
        ),

        "gravel_light": HfRandomUniformTerrainCfg(
            proportion   = 0.25,   # ±2-4cm — gravel path / packed dirt
            noise_range  = (0.02, 0.04),
            noise_step   = 0.01,
            border_width = 0.1,
        ),

        "grass_wave": HfWaveTerrainCfg(
            proportion      = 0.25,   # rolling lawn / undulating ground
            amplitude_range = (0.02, 0.05),
            num_waves       = 5,
            border_width    = 0.1,
        ),

        # ── Phase 2 — add roughness ────────────────────────────────────────
        "gravel_heavy": HfRandomUniformTerrainCfg(
            proportion   = 0.20,   # ±5-8cm — loose stone
            noise_range  = (0.05, 0.08),
            noise_step   = 0.01,
            border_width = 0.1,
        ),

        "slope_gentle": HfPyramidSlopedTerrainCfg(
            proportion     = 0.20,   # ramps 3-12°
            slope_range    = (0.052, 0.213),   # tan(3°) to tan(12°)
            platform_width = 1.5,
            border_width   = 0.1,
        ),

        # ── Phase 3 — full outdoor suite ──────────────────────────────────
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion            = 0.15,   # roots / rocks 3-12cm
            obstacle_height_range = (0.03, 0.12),
            obstacle_width_range  = (0.10, 0.50),
            num_obstacles         = 20,
            platform_width        = 1.0,
            border_width          = 0.1,
        ),

        "slope_steep": HfPyramidSlopedTerrainCfg(
            proportion     = 0.15,   # embankments 12-20°
            slope_range    = (0.213, 0.364),   # tan(12°) to tan(20°)
            platform_width = 1.0,
            border_width   = 0.1,
        ),
    }

# Build the active subset — only terrains in ACTIVE_TERRAINS list
# Proportions are renormalised automatically by Isaac Lab's terrain generator
_active_sub = {k: ALL_TERRAIN_DEFS[k] for k in ACTIVE_TERRAINS
               if k in ALL_TERRAIN_DEFS}
assert len(_active_sub) > 0, f"No valid terrains in ACTIVE_TERRAINS: {ACTIVE_TERRAINS}"
print(f"[TERRAIN] Active: {list(_active_sub.keys())}")

if _TERRAIN_OK:
    OUTDOOR_TERRAIN_CFG = TerrainGeneratorCfg(
        seed             = 42,
        size             = (8.0, 8.0),
        border_width     = 20.0,
        num_rows         = 8,
        num_cols         = 20,
        horizontal_scale = 0.1,
        vertical_scale   = 0.005,
        slope_threshold  = 0.75,
        use_cache        = False,
        curriculum       = True,
        color_scheme     = "height",
        sub_terrains     = _active_sub,
    )


def make_rough_scene(num_envs: int) -> "Go1RoughSceneCfg":
    """
    Build a Go1RoughSceneCfg with the CORRECT num_envs at construction time.

    WHY THIS FUNCTION EXISTS:
    @configclass freezes default field values at class definition.
    If Go1RoughSceneCfg has 'scene = Go1RoughSceneCfg(num_envs=4096)'
    and train.py later does 'env_cfg.scene.num_envs = 160', the terrain
    importer may have already cached env_origins for 4096 envs.
    Calling make_rough_scene(160) builds the scene with correct num_envs
    from the start, ensuring 1 env per patch with no stacking.

    Used in train.py:
        elif is_rough:
            env_cfg = Go1RoughEnvCfg()
            n = args_cli.num_envs or 4096
            env_cfg.scene = make_rough_scene(n)
    """
    return Go1RoughSceneCfg(num_envs=num_envs, env_spacing=0.0)


# =============================================================================
# SCENE CONFIG
# =============================================================================
@configclass
class Go1RoughSceneCfg(Go1SceneCfg):
    """Go1 scene with outdoor terrain. Build via make_rough_scene(n)."""
    env_spacing: float = 0.0   # terrain generator controls placement

    terrain = TerrainImporterCfg(
        prim_path              = "/World/ground",
        terrain_type           = "generator" if _TERRAIN_OK else "plane",
        terrain_generator      = OUTDOOR_TERRAIN_CFG,
        max_init_terrain_level = TERRAIN_NUM_ROWS - 1,  # all rows available
        collision_group        = -1,
        physics_material       = sim_utils.RigidBodyMaterialCfg(
            static_friction  = 0.65,
            dynamic_friction = 0.65,
            restitution      = 0.01,
        ),
        debug_vis = False,
    )


# =============================================================================
# ENV CONFIG
# =============================================================================
@configclass
class Go1RoughEnvCfg(Go1FlatEnvCfg):
    """
    Go1 rough terrain — inherits all v15 flat settings.
    scene is rebuilt by train.py via make_rough_scene(num_envs) to avoid
    configclass freezing num_envs before terrain importer runs.
    """
    episode_length_s : float = 25.0
    env_spacing      : float = 0.0
    # NOTE: scene is overridden by train.py using make_rough_scene()
    # Default here is just a placeholder with num_envs=1
    scene : Go1RoughSceneCfg = Go1RoughSceneCfg(num_envs=1, env_spacing=0.0)

    # Friction DR range (mud=0.35 to dry gravel=0.80)
    friction_range_lo : float = 0.35
    friction_range_hi : float = 0.80