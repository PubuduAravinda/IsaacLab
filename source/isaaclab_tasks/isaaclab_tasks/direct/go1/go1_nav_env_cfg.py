# go1_nav_env_cfg.py — High-Level Navigation Policy Config v8
#
# ═══════════════════════════════════════════════════════════════════════════
# WHY wz RANGE IS NOW ±0.10 (was ±0.35)
# ═══════════════════════════════════════════════════════════════════════════
# The LL normalizer std for obs[2] (wz) is very small because the LL was
# trained with heading_command=True which generates small, smooth wz
# corrections (not large constant turns).
#
# Evidence from training:
#   wz commanded = 0.35 rad/s
#   yaw observed = 1.2  rad/s  ← 3.4× overreaction
#   → normalizer amplified wz=0.35 to huge normalized value → fall
#
# Safe rule: HL wz ≤ 2 × normalizer_std[2]
#   normalizer_std[2] printed at startup — adjust wz range to match.
#
# HL ACTION SPACE: 2D [vx, wz]  — vy=0 always (LL never trained with vy≠0)
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.go1.go1_env_cfg import Go1SceneCfg, EventCfg


@configclass
class Go1NavEnvCfg(DirectRLEnvCfg):
    """
    High-level navigation policy config.
    HL: 10Hz, obs=5D, action=2D [vx, wz]
    LL: FROZEN 50Hz, obs=46D, action=12D
    """

    episode_length_s = 30.0
    decimation       = 50       # 500Hz / 50 = 10Hz HL
    num_envs         = 4096
    env_spacing      = 12.0

    # ── HL spaces ─────────────────────────────────────────────────────────
    # obs: [dx_robot, dy_robot, dist, cos_err, sin_err]
    observation_space = spaces.Box(
        low  = np.array([-10., -10.,  0., -1., -1.], dtype=np.float32),
        high = np.array([ 10.,  10., 10.,  1.,  1.], dtype=np.float32),
    )

    # action: [vx, wz] — vy permanently zero
    # wz ±0.10: conservative range to avoid LL normalizer overreaction
    # After startup, check "[LL NORMALIZER STD]" print → safe range = ±2×wz_std
    # If wz_std > 0.05, you can safely widen to ±(2×wz_std)
    action_space = spaces.Box(
        low=np.array([0.3, -0.08], dtype=np.float32),  # vx min 0.0→0.3
        high=np.array([0.8, 0.08], dtype=np.float32),  # wz ±0.08 UNCHANGED
    )

    state_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    # ── Navigation goal ────────────────────────────────────────────────────
    goal_dist_min  : float = 3.0
    goal_dist_max  : float = 8.0
    success_radius : float = 0.5

    # ── Frozen LL checkpoint ───────────────────────────────────────────────
    ll_checkpoint   : str   = ""
    ll_obs_dim      : int   = 46
    ll_action_dim   : int   = 12
    ll_actor_hidden : tuple = (512, 256, 128)
    ll_activation   : str   = "elu"
    ll_f_cmd_fixed  : float = 2.0

    # ── Physics ────────────────────────────────────────────────────────────
    sim = SimulationCfg(
        dt              = 0.002,
        render_interval = decimation,
        gravity         = (0.0, 0.0, -9.81),
        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    terrain = TerrainImporterCfg(
        prim_path    = "/World/ground",
        terrain_type = "plane",
        collision_group = -1,
        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.7, dynamic_friction=0.7, restitution=0.0),
    )
    scene : Go1SceneCfg = Go1SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    events: EventCfg    = EventCfg()