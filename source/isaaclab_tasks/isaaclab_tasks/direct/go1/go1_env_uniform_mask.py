# go1_env_uniform_mask.py — Kim-style uniform masking + calibrated rate weights
#
# PURPOSE: Isolate ONLY the masking probability contribution.
# Everything else identical to go1_env.py including calibrated rate weights.
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ COMPARED TO go1_env.py (ours):                                          │
# │  ✗ ONLY CHANGE: RL_th fault_level = Bernoulli(0.05) instead of U[0,1] │
# │    Means: policy sees RL_th fully faulted 5% of episodes (Kim-style)    │
# │           instead of continuously degraded every episode (ours)         │
# │                                                                          │
# │ IDENTICAL TO go1_env.py:                                                │
# │  ✓ PACE τf, d, Ia, q̃b — full calibration                              │
# │  ✓ Delay FIFO 0→8 steps                                                │
# │  ✓ KP/KD DR including tight RL_th range [0.60, 0.95]                  │
# │  ✓ Calibrated rate weights: RL_th=0.150, FR_th=1.800                   │
# │  ✓ FR_th cap 0.02 rad + proximity penalty                              │
# │  ✓ All reward structure, debug prints, slog                             │
# └──────────────────────────────────────────────────────────────────────────┘
#
# WHAT THIS PROVES:
#   Compare ours vs this:
#     Both have PACE, both have calibrated rate weights, both have tight KP DR
#     Only difference: fault exposure probability (p=1.0 continuous vs p=0.05 binary)
#     → isolates severity-proportional masking as the single contribution
#
# REGISTER in __init__.py:
#   gym.register(
#       id="Isaac-Velocity-Flat-Go1-UniformMask-v0",
#       entry_point="isaaclab_tasks.direct.go1.go1_env_uniform_mask:Go1UniformMaskEnv",
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
#   python train.py --task Isaac-Velocity-Flat-Go1-UniformMask-v0 \
#       --num_envs 4096 --max_iterations 45000 --headless

import torch
from isaaclab_tasks.direct.go1.go1_env import Go1Env

_KIM_MASK_P = 0.05   # 5% per episode — Kim 2024 uniform probability


class Go1UniformMaskEnv(Go1Env):
    """
    Kim-style uniform masking on fully calibrated PACE simulation.
    Calibrated rate weights KEPT — only masking probability differs from ours.

    This is the clean single-variable comparison:
      go1_env.py:              fault_level = U[0,1]   every episode
      go1_env_uniform_mask.py: fault_level = Bern(0.05) per episode

    Rate weights inherited from Go1Env:
      RL_th = 0.150 (allows impulsive stiction-breaking — same as ours)
      FR_th = 1.800 (suppresses binding oscillation — same as ours)

    The policy will STILL have the incentive to break stiction via rate
    weights — but it only practices it 5% of the time instead of 100%.
    This is the Kim limitation: not that it cannot break stiction in principle,
    but that it rarely sees the fault and does not concentrate practice on it.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Parent sets up EVERYTHING including calibrated rate weights
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "="*72)
        print("Go1UniformMaskEnv | KIM UNIFORM MASK — single variable vs ours")
        print(f"  Masking:     Bernoulli({_KIM_MASK_P}) ← ONLY difference from go1_env.py")
        print(f"  Rate wts:    INHERITED — RL_th=0.150  FR_th=1.800  (calibrated)")
        print(f"  PACE:        INHERITED — full τf/d/Ia/q̃b calibration")
        print(f"  Delay:       INHERITED — 0→8 step curriculum")
        print(f"  KP DR:       INHERITED — RL_th tight [0.60,0.95]")
        print(f"  FR_th cap:   INHERITED — 0.02 rad asymmetric limit")
        print("="*72 + "\n")

        # Verify rate weights are inherited correctly — print for confirmation
        rw = self._rate_weights.cpu().numpy()
        print(f"  [VERIFY] rate_weights:")
        JOINTS = ['FL_h','FR_h','RL_h','RR_h',
                  'FL_th','FR_th','RL_th','RR_th',
                  'FL_kn','FR_kn','RL_kn','RR_kn']
        for i, (n, w) in enumerate(zip(JOINTS, rw)):
            marker = " ← KEY" if i in [5, 6] else ""
            print(f"    {n:<8}: {w:.3f}{marker}")
        print()

    def _reset_idx(self, env_ids):
        """
        Calls parent _reset_idx (sets fault_level=U[0,1] with PACE DR)
        then overrides ONLY the fault level distribution to Kim-style.
        All other DR (KP, delay, τf magnitude, d, q̃b) unchanged.
        """
        # Parent handles everything:
        # - KP/KD DR with tight RL_th range [0.60, 0.95]
        # - τf DR: RL_th = healthy + fault_level × (4.944 - 0.007)
        # - d DR:  RL_th = 0.048 + fault_level × (3.459 - 0.048)
        # - delay DR: U[0, 8] steps
        # - command resample
        # - robot state reset
        # - FR_th cap active
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        n        = len(env_ids)
        actuator = self._robot.actuators["legs"]

        # ── ONLY CHANGE: override fault_level distribution ────────────────
        # Parent set: fault_level[env_ids] = U[0, 1]  (always degraded)
        # We set:     fault_level[env_ids] = Bernoulli(0.05) (Kim-style)
        #
        # p=0.05: fault_level = 1.0 → full τf = 4.944 Nm, d = 3.459
        # p=0.95: fault_level = 0.0 → τf = 0.007 Nm,    d = 0.048 (healthy)
        rl_roll      = torch.rand(n, device=self.device)
        rl_fault_kim = (rl_roll < _KIM_MASK_P).float()
        self._rl_fault_level[env_ids] = rl_fault_kim

        # Re-apply τf with Kim fault level
        # (parent already wrote U[0,1] fault level — override it)
        tau_f_healthy = 0.007
        tau_f_fault   = self._tau_f_nominal[6].item()   # 4.944 Nm
        rl_tau_f      = (tau_f_healthy
                         + rl_fault_kim * (tau_f_fault - tau_f_healthy))

        if self._friction_env_ids_ok and self._tau_fn is not None:
            tau_f_reset       = self._tau_f_nominal.unsqueeze(0).expand(n, -1).clone()
            tau_f_reset[:, 6] = rl_tau_f
            getattr(self._robot, self._tau_fn)(
                tau_f_reset, env_ids=env_ids)
        elif self._tau_fn is not None:
            # Fallback: set all envs
            tau_f_all       = self._tau_f_nominal.unsqueeze(0).expand(
                self.num_envs, -1).clone()
            tau_f_all[env_ids, 6] = rl_tau_f
            getattr(self._robot, self._tau_fn)(tau_f_all)

        # Re-apply d (viscous) with Kim fault level
        d_healthy = 0.048
        d_fault   = self._d_nominal[6].item()   # 3.459 Nm·s/rad
        rl_d      = d_healthy + rl_fault_kim * (d_fault - d_healthy)
        if hasattr(actuator, 'viscous_friction'):
            d_reset       = self._d_nominal.unsqueeze(0).expand(n, -1).clone()
            d_reset[:, 6] = rl_d
            actuator.viscous_friction[env_ids] = d_reset

        # Everything else (KP DR, delay DR, encoder bias, FR_th cap,
        # rate weights) stays exactly as parent set it.
        # This is the minimal single-variable override.