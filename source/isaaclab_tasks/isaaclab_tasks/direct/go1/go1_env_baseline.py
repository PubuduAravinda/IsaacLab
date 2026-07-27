# go1_env_baseline.py — Lag + Default Physics (no PACE calibration)
#
# INHERITS everything from Go1Env — identical structure to go1_env_sparse_rough.py
# ONLY overrides:
#   1. __init__: zeros all PACE physical params (τf, d, Ia, q̃b)
#                disables FR_th cap and RL_th fault masking
#                sets uniform rate weights
#   2. _reset_idx: re-zeros τf/d after parent sets them
#
# EVERYTHING ELSE INHERITED UNCHANGED:
#   delay FIFO curriculum (_DELAY_PHASE1_END same as go1_env.py)
#   KP/KD DR ranges
#   calibrated obs noise
#   full reward structure (same r_trot, r_alive, r_lin_vel etc.)
#   debug prints (step() not overridden → go1_env.py prints work)
#   slog structure (play.py --log works)
#   command_manager, _setup_scene, _get_observations, _get_dones
#
# WHY INHERIT vs STANDALONE:
#   Old standalone baseline had reward accumulation bugs, missing debug
#   prints, and step() bypass. Inheriting from Go1Env fixes all of these.
#
# REGISTER in __init__.py:
#   gym.register(
#       id="Isaac-Velocity-Flat-Go1-Baseline-v0",
#       entry_point="isaaclab_tasks.direct.go1.go1_env_baseline:Go1BaselineEnv",
#       disable_env_checker=True,
#       kwargs={
#           "env_cfg_entry_point":
#               "isaaclab_tasks.direct.go1.go1_env_cfg:Go1FlatEnvCfg",
#           "rsl_rl_cfg_entry_point":
#               f"{agents.__name__}.rsl_rl_ppo_cfg:Go1RslRlPpoCfg",
#       },
#   )
#
# TRAIN:
#   python train.py --task Isaac-Velocity-Flat-Go1-Baseline-v0 \
#       --num_envs 4096 --max_iterations 45000 --headless

import torch
from isaaclab_tasks.direct.go1.go1_env import Go1Env


class Go1BaselineEnv(Go1Env):
    """
    Lag curriculum + default physics. No PACE calibration.
    Inherits ALL infrastructure from Go1Env.
    Only PACE physical parameters and fault-specific shaping differ.

    Comparison table vs go1_env.py (ours):
    ┌─────────────────────┬──────────────┬──────────────┐
    │ Feature             │ go1_env.py   │ baseline     │
    ├─────────────────────┼──────────────┼──────────────┤
    │ Delay FIFO          │ 0→8 steps    │ 0→8 steps ✓  │
    │ KP/KD DR            │ ±25%/±20%    │ ±25%/±20% ✓  │
    │ Obs noise           │ calibrated   │ calibrated ✓ │
    │ Rewards             │ full set     │ full set ✓   │
    │ Debug prints        │ go1_env.py   │ inherited ✓  │
    │ Slog keys           │ all keys     │ inherited ✓  │
    │ τf (Coulomb)        │ PACE values  │ 0.0 Nm  ✗   │
    │ d  (viscous)        │ PACE values  │ 0.0 Nm·s ✗  │
    │ Ia (armature)       │ PACE values  │ 0.0 kg·m² ✗ │
    │ q̃b (encoder bias)  │ PACE values  │ 0.0 rad ✗   │
    │ FR_th cap           │ 0.02 rad     │ disabled ✗   │
    │ Rate weights        │ physics-derived│ uniform 1.0 ✗│
    │ RL_th fault mask    │ p=0.10       │ p=0.0 ✗     │
    └─────────────────────┴──────────────┴──────────────┘
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        # ── Call parent __init__ first ────────────────────────────────────
        # This sets up EVERYTHING: delay FIFO, KP/KD, obs noise, rewards,
        # debug prints, slog, command_manager, PACE writes.
        # We then immediately override the PACE physical params.
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "="*72)
        print("Go1BaselineEnv | LAG + DEFAULT PHYSICS")
        print("Overriding PACE params to zero — all else inherited from Go1Env")
        print("="*72)

        _n   = self.num_envs
        _j   = 12
        _zero = torch.zeros(_n, _j, device=self.device)

        # ── Override 1: Ia → 0 ───────────────────────────────────────────
        self._robot.write_joint_armature_to_sim(_zero)
        print("  τf  = 0 Nm          (was PACE per-joint values)")

        # ── Override 2: τf → 0 ───────────────────────────────────────────
        if self._tau_fn is not None:
            getattr(self._robot, self._tau_fn)(_zero)
        self._tau_f_nominal = torch.zeros(_j, device=self.device)
        print("  Ia  = 0 kg·m²       (was PACE per-joint values)")

        # ── Override 3: d → 0 ────────────────────────────────────────────
        actuator = self._robot.actuators["legs"]
        if hasattr(actuator, 'viscous_friction'):
            actuator.viscous_friction[:] = _zero
        self._d_nominal = torch.zeros(_j, device=self.device)
        print("  d   = 0 Nm·s/rad    (was PACE per-joint values)")

        # ── Override 4: q̃b → 0 ──────────────────────────────────────────
        if self._bias_attr is not None:
            getattr(actuator, self._bias_attr)[:] = _zero
        self._bias_values = torch.zeros(_j, device=self.device)
        print("  q̃b = 0 rad          (was PACE per-joint values)")

        # ── Override 5: FR_th cap → disabled ─────────────────────────────
        self._fr_th_max_delta = 999.0
        print("  FR_th cap disabled  (was 0.02 rad)")

        # ── Override 6: rate weights → uniform ───────────────────────────
        self._rate_weights = torch.ones(_j, device=self.device)
        print("  rate_weights = 1.0  (was physics-derived per-joint)")

        # ── Override 7: action limits → symmetric ────────────────────────
        # self._delta_soft_lo = torch.tensor(
        #     [-0.08, -0.08, -0.08, -0.08,
        #      -0.35, -0.35, -0.35, -0.35,
        #      -0.35, -0.35, -0.35, -0.35],
        #     device=self.device)
        # self._delta_soft_hi = torch.tensor(
        #     [ 0.08,  0.08,  0.08,  0.08,
        #       0.35,  0.35,  0.35,  0.35,
        #       0.35,  0.35,  0.35,  0.35],
        #     device=self.device)
        # print("  action limits symmetric ±0.08 hip / ±0.35 thigh+knee")

        # ── Override 8: fault level → always 0 ───────────────────────────
        self._rl_fault_level = torch.zeros(_n, device=self.device)
        print("  RL_th fault masking: p=0  (was p=0.10)")
        print("="*72 + "\n")

    def _reset_idx(self, env_ids):
        """
        Calls parent _reset_idx then re-zeros PACE params it just set.
        Parent handles: KP/KD DR, delay DR, command resample, robot state.
        We handle: undo τf/d/q̃b/fault_level that parent just wrote.
        """
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        _n_r     = len(env_ids)
        _j       = 12
        actuator = self._robot.actuators["legs"]

        # Re-zero τf (parent _reset_idx wrote PACE fault-level values)
        if self._tau_fn is not None and self._friction_env_ids_ok:
            getattr(self._robot, self._tau_fn)(
                torch.zeros(_n_r, _j, device=self.device),
                env_ids=env_ids)
        elif self._tau_fn is not None:
            getattr(self._robot, self._tau_fn)(
                torch.zeros(self.num_envs, _j, device=self.device))

        # Re-zero d (parent wrote per-episode fault-level viscous values)
        if hasattr(actuator, 'viscous_friction'):
            actuator.viscous_friction[env_ids] = torch.zeros(
                _n_r, _j, device=self.device)

        # Re-zero q̃b (parent re-applied PACE encoder bias)
        if self._bias_attr is not None:
            getattr(actuator, self._bias_attr)[env_ids] = torch.zeros(
                _n_r, _j, device=self.device)

        # Keep fault level 0 (parent set rl_fault_level = U[0,1])
        self._rl_fault_level[env_ids] = 0.0