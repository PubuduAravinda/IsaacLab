# go1_env.py — v9: Symmetric Gait Shaping (Air-time 0.20s + Stance-time + Ia fix)
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CALIBRATION STATUS (from PACE run 26_04_02 + manual tests)             │
# │  RL_th stiction: τf=4.944 Nm  d=3.459 Nm·s/rad (PACE confirmed)      │
# │  FR_th mechanical binding at >0.900 rad (ground-contact fault)         │
# │  Both fault parameters are the UPPER BOUND of per-episode DR ranges.  │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════
# DEPLOYMENT RESULTS SUMMARY (real_log_model15000_20260508):
#   CONFIRMED WORKING (keep unchanged):
#     RL_th physics DR:  2 spikes on real vs 59 in sim → confirmed ✓
#     Hip saturation:    0.5% vs 9.5% sim → hip_sat reward ✓
#     Lateral drift:     4.9° lean → r_lat_vel working ✓
#     Knee lift:         FR_kn=+0.272, RL_kn=+0.267 on real hardware ✓
#     FR_th safety:      max=0.848, zero binding events ✓
#   NEW EXPLOIT FOUND (model_15000):
#     Real gait: 9.95 Hz (3.71× faster than sim 2.68 Hz)
#     Root cause: air_time threshold 0.10s = exactly 5 policy steps at 50Hz.
#     Policy swings each foot for exactly 0.10s → barely crosses threshold
#     → earns maximum reward with minimum energy. 28 tilt spike events/12s.
#   SIM ACCURACY GAP:
#     Knee tracking error: sim=0.19-0.27 rad, real=0.08-0.13 rad (2× gap).
#     Cause: calf Ia too high in PACE. Sim knees 2× more compliant than real.
# ═══════════════════════════════════════════════════════════════════════════
#
# v9 CHANGES vs v8 — FOUR TARGETED FIXES:
#
# FIX 1: Air-time threshold 0.10s → 0.20s
#   Real 9.95Hz exploit: swing = 0.10s = exactly old threshold. Earns reward.
#   New threshold 0.20s: swing must be ≥ 10 policy steps (0.20s) to earn reward.
#   True 2Hz trot: swing ≈ 0.25s >> 0.20s → full reward ✓
#   9.95Hz tap:    swing ≈ 0.10s < 0.20s  → zero reward. Exploit BLOCKED.
#   Cap adjusted:  0.40 → 0.30s (0.50s total inclusive of threshold).
#
# FIX 2: r_stance_time — symmetric stance quality reward (NEW)
#   Air-time rewards swing quality. Nothing rewarded proper stance duration.
#   Policy gamed this: short tap (0.05s stance) + long swing earned max reward.
#   r_stance_time: rewards feet that maintain ground contact for ≥ 4 steps (0.08s).
#   Uses data.current_contact_time (accumulates every step during stance).
#   Fires CONTINUOUSLY during proper stance → can't be gamed with single long touch.
#   At 2Hz trot (0.25s stance): earns ~0.51/step → matches air-time signal.
#   At 9.95Hz tap (0.05s stance): never reaches threshold → zero reward.
#   Combined with air-time: BOTH swing AND stance quality required → 2Hz stable trot.
#
# FIX 3: Calf Ia halved (PACE sim accuracy correction)
#   Real knee tracking error 2× better than sim (real 0.08-0.13 vs sim 0.19-0.27).
#   Cause: calf armature Ia=0.013-0.016 too high → sim knees too compliant.
#   Fix: Ia calves 0.013-0.016 → 0.006-0.008. Sim knees now match real response.
#   This reduces the sim-to-real tracking gap and makes knee lift training more accurate.
#
# FIX 4: r_upright weight -3.0 → -3.5
#   v7 used -3.0 (tilt spikes came from 9.95Hz rapid impact, not weak balance).
#   With gait frequency fixed by FIX 1+2, -3.5 provides better tilt resistance.
#   Not back to -4.0 (which caused the static trot exploit in v6).
#
# ALL v8 UNCHANGED:
#   RL_th physics DR (τf/d per episode), rate weights FR=1.800 RL=0.150,
#   FR_th cap 0.820, FR_th proximity penalty, clearance deadband 0.03m,
#   r_hip_sat, r_lat_vel=-3.5, r_foot_drag=-1.0, r_hip_reg=-1.5,
#   KP/KD DR, delay FIFO, obs 45D, PACE Ia hips/thighs
#
# RESEARCH CONTEXT — why air+stance is the right approach:
#   CPG-based (Bellegarda 2022): explicit phase variable per leg, strict schedule.
#   Too rigid for impaired joints — RL_th stiction breaks phase adherence.
#   Reference-motion (Hwangbo 2019): requires healthy reference trajectory.
#   Not available for this impaired hardware configuration.
#   Air+Stance (this work, after Rudin 2022 feet_air_time):
#   Rewards the OUTCOME (proper swing/stance duration) without prescribing
#   the exact timing — gives policy flexibility to compensate for RL_th stiction
#   while still converging to rhythmic 2Hz gait. Closest practical equivalent
#   to biological CPG reward structure for hardware-faulted quadruped.
#
# REWARD SANITY CHECK per policy step (×dt=0.02 before summing):
#   r_air_time  max: 20×4×0.30×dt×8touchdowns/step    ≈ 0.48/step
#   r_stance_time max: 3×4×0.25×dt (all feet in stance) ≈ 0.06/step (moderate)
#   r_lin_vel   max: 1.5/step × dt                     ≈ 0.030/step
#   Walking cost at 10°: r_upright = -3.5×sin²(10°)×dt ≈ -0.0021/step
#   NET: walking >> standing, gait >> static solution ✓
# ═══════════════════════════════════════════════════════════════════════════
#
# v6 CHANGES vs v5 — TWO AVENUES COMBINED:
# ═══════════════════════════════════════════════════════════════════════════
#
# AVENUE 1: RL_th Physics DR + Per-joint Rate Weights
# ───────────────────────────────────────────────────
# Root cause identified from real_log_model24000:
#   RL_th delta STUCK at constant +0.30 (std=0.078) — no gait cycling.
#   Compare FL_th delta std=0.101 — actively cycling with gait phase.
#   Cause: U[0,1] target-scale DR creates degenerate gradient for RL_th.
#   At scale=0 (50% of episodes), any RL_th command = same outcome →
#   policy gives up on coordinating RL_th with gait phase.
#
#  Change 1: RL_th Physics DR (replaces target-scale DR)
#    Per-episode fault level ~ U[0,1]:
#      fault_level=0 → τf=0.007 Nm, d=0.048  (healthy — RL_th moves freely)
#      fault_level=1 → τf=4.944 Nm, d=3.459  (full PACE fault — hardware state)
#    In healthy episodes: RL_th responds fully → policy learns to CYCLE
#    RL_th delta 0→+0.35→0 in phase with FR+RL diagonal.
#    That motor pattern then transfers to fault episodes / real hardware.
#    Target scaling REMOVED — physics DR handles execution variation.
#
#  Change 2: Per-joint rate weights (physics-based)
#    RL_th: 0.900 → 0.150  (stiction-dominated: needs impulsive swing init)
#      Data: 53% of steps already have PD_force > τf on real HW.
#      Smoothness penalty was preventing large delta changes needed to sustain
#      PD_force > τf through the full swing phase.
#    FR_th: 0.900 → 1.800  (binding fault: suppress existing oscillation)
#      Data: FR_th has 280/729 impulsive steps (38%) — already over-correcting
#      because binding fault prevents tracking. Higher penalty suppresses this.
#    All others: unchanged.
#
# AVENUE 2: Balance Rewards + FR_th Protection
# ────────────────────────────────────────────
# From real_log_model24000 analysis:
#   Tilt still 16.1° mean (target <10°)
#   FR_hip saturated at HI limit 77% of time (compensation for RL leg)
#   FR_th actual max 0.860 — only 0.040 rad margin to binding (0.900)
#
#  Change 3: r_upright weight -2.5 → -4.0
#  Change 4: r_lat_vel weight -2.0 → -3.5
#  Change 5: r_hip_sat NEW — bilateral hip saturation penalty
#    -4.0 × Σ clamp(|hip_delta| - 0.06, 0)
#    Activates when hip > 75% of ±0.08 range. Prevents FR_hip at 77% hi-sat.
#  Change 6: r_fr_binding NEW — FR_th proximity penalty on ACTUAL position
#    -2.0 × clamp((fr_th_q - 0.800)/0.070, 0, 1)²
#    Penalises inertial overshoot into binding zone. Based on actual_q
#    (not target_q) because overshoot is the risk, not the command.
#  Change 7: r_foot_drag weight -0.5 → -1.0
#    FR leg in stance 69% but foot dragging (FR_kn mean_lift=-0.154).
#
# UNCHANGED FROM v5:
#   PACE Ia/q̃b, KP/KD DR, delay FIFO, FR_th hard cap 0.820,
#   obs (45D, no foot signals), trot/clearance rewards, _get_dones
# ═══════════════════════════════════════════════════════════════════════════
#
# TROT DIAGONALS: diag1=FL+RR (feet[:,0]+[:,3]), diag2=FR+RL ([:,1]+[:,2])
# CONTACT SENSOR body order (alphabetical): FL=0, FR=1, RL=2, RR=3

import torch
import numpy as np
import os
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand

_DELAY_PHASE1_END = 50_000
_DELAY_MAX        = 8
_DELAY_BUF_SIZE   = 10


class Go1Env(DirectRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "="*72)
        print("Go1Env | v15-terrain: Outdoor terrain + wz rotation")
        print("  WZ MIX:       50% straight / 50% turning  wz~U[-0.30,+0.30]")
        print("  r_ang_vel:    weight 1.0  (stronger wz tracking)")
        print("  r_wz_excess:  penalise yaw overshoot > 1.5×wz_cmd")
        print("  r_fr_binding: wz-scaled  (-2.0 straight, -5.0 at max turn)")
        print("  TERRAIN:      height-relative done/reward checks")
        print("                per-episode friction DR [0.35, 0.80]")
        print(f"  Delay: Phase1 0ms → Phase2 U[0,8] at step {_DELAY_PHASE1_END}")
        print("="*72 + "\n")

        _n    = self.num_envs
        _j    = 12
        _zero = torch.zeros(_n, _j, device=self.device)

        # ── PACE: Ia (armature) ───────────────────────────────────────────
        # FIX 3: Calf Ia halved from original PACE values.
        # Real hardware knee tracking error = 0.084-0.133 rad.
        # Sim knee tracking error = 0.193-0.269 rad (2× worse than real).
        # Root cause: Ia_calf too high → sim knees too compliant (over-damp).
        # Original PACE: [0.0141, 0.0159, 0.0130, 0.0140]
        # v9 corrected:  [0.0070, 0.0080, 0.0065, 0.0070] ← halved
        # This brings sim knee response closer to real hardware behaviour.
        self._robot.write_joint_armature_to_sim(_zero)
        Ia = torch.tensor([
            0.0026, 0.0037, 0.0029, 0.0031,
            0.0052, 0.0045, 0.1343, 0.0064,
            0.0070, 0.0080, 0.0065, 0.0070,   # FIX 3: calves halved
        ], device=self.device).unsqueeze(0).expand(_n, -1)
        self._robot.write_joint_armature_to_sim(Ia)

        # ── PACE: zero PhysX drive damping ────────────────────────────────
        self._robot.write_joint_damping_to_sim(_zero)

        # ── PACE: τf Coulomb friction ─────────────────────────────────────
        # Store 1D nominal tensor — used in _reset_idx for per-episode DR
        self._tau_f_nominal = torch.tensor([
            0.026, 0.028, 0.029, 0.031,   # hips  — healthy
            0.052, 0.044, 4.944, 0.045,   # thighs: RL fault value = DR UPPER BOUND
            0.070, 0.071, 0.078, 0.079,   # calves — healthy
        ], device=self.device)
        tau_f_init = self._tau_f_nominal.unsqueeze(0).expand(_n, -1)
        self._tau_fn = None   # name of the friction write API
        for _fn in ("write_joint_friction_coefficient_to_sim",
                    "write_joint_friction_to_sim"):
            if hasattr(self._robot, _fn):
                getattr(self._robot, _fn)(tau_f_init)
                self._tau_fn = _fn
                break

        # ── API capability check: does friction write support env_ids? ────
        # If yes: per-episode τf DR active. If no: d-only DR (still significant).
        self._friction_env_ids_ok = False
        if self._tau_fn is not None:
            try:
                _t = torch.zeros(1, 12, device=self.device)
                _t[0] = self._tau_f_nominal
                getattr(self._robot, self._tau_fn)(
                    _t, env_ids=torch.tensor([0], device=self.device))
                # Restore correct value for env 0
                getattr(self._robot, self._tau_fn)(tau_f_init)
                self._friction_env_ids_ok = True
                print("  [API] τf per-episode DR: SUPPORTED ✓")
            except TypeError:
                self._friction_env_ids_ok = False
                print("  [API] τf per-episode DR: not supported — d-only DR active")

        # ── PACE: d (viscous damping) ─────────────────────────────────────
        # Store nominal 1D tensor for reset DR; RL_th[6] is the DR upper bound
        self._d_nominal = torch.tensor([
            0.039, 0.040, 0.040, 0.042,   # hips
            0.045, 0.050, 3.459, 0.050,   # thighs: RL fault = DR UPPER BOUND
            0.124, 0.092, 0.108, 0.108,   # calves
        ], device=self.device)
        actuator = self._robot.actuators["legs"]
        if hasattr(actuator, 'viscous_friction'):
            actuator.viscous_friction[:] = (
                self._d_nominal.unsqueeze(0).expand(_n, -1))

        # ── PACE: q̃b (encoder bias) ───────────────────────────────────────
        self._bias_values = torch.tensor([
            -0.006, -0.002, -0.005,  0.004,
            -0.018, -0.022,  0.063, -0.020,
            -0.001, -0.011,  0.012, -0.021,
        ], device=self.device)
        self._bias_attr = None
        for _attr in ("encoder_bias", "position_offset", "bias"):
            if hasattr(actuator, _attr):
                getattr(actuator, _attr)[:] = (
                    self._bias_values.unsqueeze(0).expand(_n, -1))
                self._bias_attr = _attr
                break

        # ── FR_th position cap ────────────────────────────────────────────
        # Hard ceiling on TARGET: default(0.800) + 0.020 = 0.820 rad
        # FR_th proximity penalty (Change 6) handles actual_q overshoot.
        self._fr_th_max_delta = 0.02

        self.command_manager = UniformVelocityCommand(
            cfg=self.cfg.commands, env=self)

        self._actions           = torch.zeros(_n, _j, device=self.device)
        self._prev_actions      = torch.zeros_like(self._actions)
        self._prev_prev_actions = torch.zeros_like(self._actions)
        self._target_pos        = torch.zeros_like(self._actions)

        self._delay_buf  = torch.zeros(_DELAY_BUF_SIZE, _n, _j, device=self.device)
        self._delay_ptr  = 0
        self._env_delays = torch.zeros(_n, dtype=torch.long, device=self.device)

        self._kp_nominal = torch.tensor(
            [35., 35., 35., 35., 65., 65., 65., 65., 80., 80., 80., 80.],
            device=self.device)
        self._kd_nominal = torch.tensor(
            [4.0, 4.0, 4.0, 4.0, 4.5, 4.5, 4.5, 4.5, 5.0, 5.0, 5.0, 5.0],
            device=self.device)
        self._kp_live = self._kp_nominal.unsqueeze(0).expand(_n, -1).clone()

        self._kp_dr_hip_th_lo = 0.75;  self._kp_dr_hip_th_hi = 1.25
        self._kp_dr_kn_lo     = 0.80;  self._kp_dr_kn_hi     = 1.20
        self._kd_dr_hip_th_lo = 0.75;  self._kd_dr_hip_th_hi = 1.25
        self._kd_dr_kn_lo     = 0.80;  self._kd_dr_kn_hi     = 1.20

        # ── Action rate weights — v6: per-joint physics-based ─────────────
        # Basis: Test B Bode plots (PDF) for healthy joints.
        # Fault-joint adjustments:
        #   RL_th [6]: 0.900 → 0.150
        #     Stiction-dominated. Needs impulsive commands to:
        #     (a) break τf=4.944 Nm threshold at swing initiation
        #     (b) maintain PD_force > τf throughout swing to prevent re-engagement
        #     Data: 53% of real HW steps already have PD_force > τf — force
        #     magnitude is OK, but smoothness penalty trained gradual ramping
        #     which drops PD_force below τf after each stiction break.
        #     Low rate weight → policy learns to hold large delta through swing.
        #   FR_th [5]: 0.900 → 1.800
        #     Binding fault causes 280/729 impulsive steps (38%) on real HW.
        #     Higher penalty suppresses oscillation and reduces approach speed
        #     toward the 0.820/0.900 binding zone.
        self._rate_weights = torch.tensor([
            1.500, 1.500, 1.500, 1.500,   # hips:   BW=3.1Hz (Test B)
            0.900, 1.800, 0.150, 0.900,   # thighs: FL healthy | FR binding→HIGH | RL stiction→LOW | RR healthy
            0.564, 0.564, 0.564, 0.564,   # knees:  BW=5.5Hz  (Test B)
        ], device=self.device)

        # ── Action space limits ───────────────────────────────────────────
        self._delta_soft_lo = torch.tensor(
            [-0.08, -0.08, -0.08, -0.08,
             -0.35, -0.35, -0.35, -0.35,
             -0.35, -0.35, -0.35, -0.35], device=self.device)
        self._delta_soft_hi = torch.tensor(
            [ 0.08,  0.08,  0.08,  0.08,
              0.35,  0.35,  0.35,  0.35,
              0.35,  0.35,  0.35,  0.35], device=self.device)

        # ── RL_th fault level tracking (replaces _rl_th_scale) ───────────
        # Stores per-env fault level from last reset (0=healthy, 1=full fault)
        # Used for debug only — physics DR is in _reset_idx
        self._rl_fault_level = torch.ones(_n, device=self.device)

        # ── FIX 1: Previous feet contact for temporal trot reward ──────────
        # shape [N, 4]: FL FR RL RR — stores contact state from previous step.
        # r_trot_switch = |contact(t) - contact(t-1)| rewards actual switching.
        # Static solution (one diagonal always down) earns ZERO switching reward.
        # Reset to 0 in _reset_idx to avoid spurious switches at episode start.
        self._prev_feet_contact = torch.zeros(_n, 4, device=self.device)

        # ── v12: Gait frequency command — narrower range for stable learning ─
        # v11 used U[1.0, 4.0] Hz → 4× range → policy struggled to condition.
        # v12 uses U[1.5, 3.0] Hz → 2× range → achievable for the policy.
        # At deploy: set f_cmd=2.0 → obs[45]=0.333 → target_swing=0.250s → 2Hz.
        # Normalisation: obs[45]=(f_cmd-1.0)/3.0 uses full [0,1] range for
        # future extension to U[1,4], so obs scale stays consistent.
        self._episode_f_cmd = torch.full((_n,), 2.0, device=self.device)
        self._f_cmd_lo = 1.5   # Hz — lower bound of training range
        self._f_cmd_hi = 3.0   # Hz — upper bound (2× ratio, manageable variance)

        # ── v15: Mixed wz command — improved 50/50 bimodal distribution ─────
        # v14 used 70/30 — policy biased toward straight, ignored wz in 30% envs
        # v15 uses 50/50 — equal time on straight and turning
        #
        # Wider range ±0.30 rad/s (was ±0.20):
        #   Gives HL more steering authority on real hardware
        #   RL_th stiction needs higher wz to overcome asymmetry
        #   Expected normalizer std ≈ 0.30/√3/√2 ≈ 0.122 rad/s
        #   On real hardware: wz=0.20 normalises to 0.20/0.122 = 1.64 → safe
        #
        # _episode_wz: separate from _wz_cmd, persists through vx resampling
        #   command_manager resamples vx every 5-10s mid-episode
        #   _wz_cmd would survive (we override in _get_observations)
        #   _episode_wz makes this explicit and immune to any command_manager state
        self._wz_cmd           = torch.zeros(_n, device=self.device)
        self._episode_wz       = torch.zeros(_n, device=self.device)  # persistent
        self._WZ_STRAIGHT_PROB = 0.50    # was 0.70 — equal turning/straight
        self._WZ_MAX           = 0.30    # was 0.20 — wider for real hardware
        self._wz_reset_count   = 0
        expected_std = self._WZ_MAX * (1.0 - self._WZ_STRAIGHT_PROB) ** 0.5 / 3.0**0.5
        print(f"  [WZ MIX v15] STRAIGHT_PROB={self._WZ_STRAIGHT_PROB:.0%}  "
              f"WZ_MAX=±{self._WZ_MAX:.2f} rad/s  "
              f"expected_std≈{expected_std:.3f} rad/s  "
              f"→ deploy safe range ±{2*expected_std:.3f} rad/s")

        # ── Terrain height offsets ──────────────────────────────────────────
        # For flat terrain: all zeros (no effect).
        # For rough terrain (Go1RoughEnvCfg): scene.env_origins[:, 2] gives
        # the Z height of each env's origin ON the terrain.
        # Used in _get_dones and _get_rewards for height-relative checks.
        # Must be read AFTER super().__init__() which populates env_origins.
        self._terrain_z = self.scene.env_origins[:, 2].clone()   # [N]
        is_rough = self._terrain_z.abs().max().item() > 0.01
        print(f"  [TERRAIN] {'ROUGH — height-relative checks active' if is_rough else 'FLAT — terrain_z≈0'}")

        # ── Per-episode friction DR bounds ─────────────────────────────────
        # Flat cfg: friction_range_lo/hi not present → use flat defaults
        # Rough cfg (Go1RoughEnvCfg): uses [0.35, 0.80] for mud→dry grass
        self._friction_lo = float(getattr(cfg, 'friction_range_lo', 0.70))
        self._friction_hi = float(getattr(cfg, 'friction_range_hi', 0.80))
        self._friction_dr_enabled = abs(self._friction_hi - self._friction_lo) > 0.05
        if self._friction_dr_enabled:
            print(f"  [FRICTION DR] range=[{self._friction_lo:.2f}, {self._friction_hi:.2f}]"
                  f"  (per-episode terrain friction)")

        # ── Episode reward sums — v15: added wz_excess ───────────────────
        _rk = ["lin_vel", "ang_vel", "ang_vel_xy", "lin_vel_z", "torques",
               "action_rate", "action_jerk", "upright", "trot", "alive",
               "fall", "hip_reg", "hip_sat", "foot_clear", "foot_drag",
               "lat_vel", "fr_binding", "wz_excess"]   # ← v15
        self._ep_sums = {k: torch.zeros(_n, device=self.device) for k in _rk}

        self._global_step = (_DELAY_PHASE1_END
                             if os.environ.get("GO1_EVAL_PHASE2") == "1" else 0)

        # ── Observation noise — hardware Phase 1 (Table 6+7 PDF) ─────────
        # v11: 46 elements (added 0.0 for obs[45] = f_cmd command, no noise)
        self._obs_noise_std = torch.tensor([
            0.0, 0.0, 0.0,
            0.000132, 0.001016, 0.000132, 0.000132,
            0.000061, 0.000061, 0.000061, 0.000061,
            0.000070, 0.000070, 0.000070, 0.000070,
            0.008715, 0.008715, 0.008715, 0.008715,
            0.008554, 0.008554, 0.008554, 0.008554,
            0.005621, 0.005621, 0.005621, 0.005621,
            0.01689,  0.00606,  0.01315,
            0.03086,  0.06381,  0.04405,
            *([0.0] * 12),   # prev_actions: no noise
            0.0,             # obs[45]: f_cmd command: no noise
        ], device=self.device)

        self._obs_noise_enabled   = True
        self._prev_act_init_scale = 0.3
        self.last_obs             = None

        # ── Foot body IDs for clearance and drag rewards ──────────────────
        try:
            _fl, _ = self._robot.find_bodies("FL_foot")
            _fr, _ = self._robot.find_bodies("FR_foot")
            _rl, _ = self._robot.find_bodies("RL_foot")
            _rr, _ = self._robot.find_bodies("RR_foot")
            self._foot_body_ids = [_fl[0], _fr[0], _rl[0], _rr[0]]
            print(f"  [FOOT IDs] FL={_fl[0]}  FR={_fr[0]}  "
                  f"RL={_rl[0]}  RR={_rr[0]}  ✓")
        except Exception as e:
            self._foot_body_ids = None
            print(f"  [FOOT IDs] WARNING: {e} — clearance/drag rewards zero")

        # ── Sim logging ───────────────────────────────────────────────────
        N = 2000
        self._sim_log_maxsteps = N
        self._slog = {
            k: np.zeros((N, s) if s > 1 else N, np.float32)
            for k, s in [("obs_raw",46),("tanh_delta",12),("raw_net",12),  # obs_raw now 46D
                         ("target_q",12),("actual_q",12),("actual_qd",12),
                         ("proj_grav",3),("ang_vel",3),("lin_vel",3),
                         ("cmd",3),("contact",4),("tilt_deg",1),("reward",1)]
        }
        self._slog_step   = 0
        self._slog_active = False

        print(f"  [PACE]  RL_th: Ia={Ia[0,6]:.4f}  "
              f"τf_range=[0.007,{self._tau_f_nominal[6]:.3f}]  "
              f"d_range=[0.048,{self._d_nominal[6]:.3f}]")
        print(f"  [PACE]  FR_th: Ia={Ia[0,5]:.4f}  τf={self._tau_f_nominal[5]:.3f}  "
              f"(binding fault — mechanical, not PACE)")
        print(f"  [CAP]   FR_th target ≤ {0.800+self._fr_th_max_delta:.3f} rad  "
              f"proximity penalty starts at 0.800")
        if self._bias_attr:
            print(f"  [q̃b]    via actuator.{self._bias_attr}  "
                  f"RL_th={self._bias_values[6]:+.3f}")
        print(f"  [RATE]  RL_th={self._rate_weights[6]:.3f}  "
              f"FR_th={self._rate_weights[5]:.3f}  "
              f"hip={self._rate_weights[0]:.3f}")

    # =========================================================================
    # _setup_scene
    # =========================================================================
    def _setup_scene(self):
        self._robot = self.scene["robot"]

        # ── Lighting ──────────────────────────────────────────────────────
        # Spawn a strong directional sun + fill light so terrain is visible.
        # DomeLightCfg cannot go in scene cfg (InteractiveScene rejects it).
        # spawn_light works here because USD stage is ready at _setup_scene time.
        import isaaclab.sim as sim_utils
        import omni.usd
        try:
            from pxr import UsdLux, Gf
            stage = omni.usd.get_context().get_stage()

            # Sun — distant directional light
            sun_prim = stage.DefinePrim("/World/Lights/SunLight", "DistantLight")
            sun      = UsdLux.DistantLight(sun_prim)
            sun.CreateIntensityAttr(10000.0)
            sun.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.90))
            sun.CreateAngleAttr(0.53)
            from pxr import UsdGeom
            UsdGeom.Xformable(sun_prim).MakeMatrixXform().Set(
                Gf.Matrix4d().SetRotate(
                    Gf.Rotation(Gf.Vec3d(1, 0.2, 0.1), 45.0)))

            # Sky dome — ambient fill
            sky_prim = stage.DefinePrim("/World/Lights/SkyLight", "DomeLight")
            sky      = UsdLux.DomeLight(sky_prim)
            sky.CreateIntensityAttr(1500.0)
            sky.CreateColorAttr(Gf.Vec3f(0.6, 0.7, 0.95))   # sky blue fill
            print("[LIGHT] Sun + SkyDome spawned ✓")
        except Exception as e:
            print(f"[LIGHT] USD lights not available ({e}) — using sim default")
            try:
                # Fallback: use Isaac Lab helper
                _dome = sim_utils.DomeLightCfg(intensity=3000.0,
                                               color=(0.85, 0.88, 1.0))
                _dome.func("/World/Lights/SkyLight", _dome)
                print("[LIGHT] DomeLightCfg fallback spawned ✓")
            except Exception as e2:
                print(f"[LIGHT] All light methods failed ({e2})")

        # ── Terrain colour via sim_utils.PreviewSurfaceCfg ────────────────
        # Raw UsdShade.Material shows white because the terrain importer's
        # own physics material overrides it in RTX mode.
        # sim_utils.PreviewSurfaceCfg is the same mechanism used by
        # VisualizationMarkers (markers show colour → this will too).
        # Each terrain TYPE gets a distinct colour so types are distinguishable.
        try:
            import isaaclab.sim as sim_utils
            import omni.usd
            from pxr import UsdGeom, UsdShade, Sdf

            stage = omni.usd.get_context().get_stage()

            # (R,G,B) linear colour per terrain type keyword in prim path
            COLOR_MAP = {
                "flat"         : (0.72, 0.72, 0.72),   # light grey
                "gravel_light" : (0.75, 0.65, 0.48),   # sandy beige
                "gravel_heavy" : (0.48, 0.44, 0.36),   # dark grey-brown
                "grass_wave"   : (0.15, 0.50, 0.12),   # grass green
                "slope_gentle" : (0.78, 0.68, 0.30),   # sandy yellow
                "slope_steep"  : (0.52, 0.38, 0.18),   # rock brown
                "obstacles"    : (0.42, 0.28, 0.15),   # dark brown
                "_default_"    : (0.60, 0.60, 0.60),   # fallback mid-grey
            }

            _mat_cache = {}
            _mat_root  = "/World/TerrainMats"
            _bound = 0
            _paths_seen = []

            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                # Only target terrain meshes
                if "/World/ground" not in path and "/terrain" not in path.lower():
                    continue
                _paths_seen.append(path)

                # Pick colour by keyword match in path
                path_lower = path.lower()
                chosen_key = "_default_"
                for key in COLOR_MAP:
                    if key != "_default_" and key in path_lower:
                        chosen_key = key
                        break
                col = COLOR_MAP[chosen_key]

                # Create/reuse material using Isaac Lab's PreviewSurfaceCfg
                if chosen_key not in _mat_cache:
                    mat_path   = f"{_mat_root}/{chosen_key.strip('_')}"
                    mat_cfg    = sim_utils.PreviewSurfaceCfg(
                        diffuse_color = col,
                        roughness     = 0.85,
                        metallic      = 0.0,
                    )
                    # Spawn the material — same API that colours markers
                    spawned = mat_cfg.func(mat_path, mat_cfg)
                    _mat_cache[chosen_key] = spawned

                # Bind to this mesh
                if _mat_cache[chosen_key] is not None:
                    UsdShade.MaterialBindingAPI(prim).Bind(
                        _mat_cache[chosen_key],
                        UsdShade.Tokens.weakerThanDescendants)
                    _bound += 1

            print(f"[TERRAIN COLOURS] Bound to {_bound} mesh prims "
                  f"({len(_paths_seen)} terrain prims found)")
            if _bound == 0 and _paths_seen:
                print(f"  ⚠ Prims found but no colour matched. "
                      f"Sample paths: {_paths_seen[:3]}")
                print(f"  → Add matching keywords to COLOR_MAP in _setup_scene")
            elif _bound == 0:
                print(f"  ⚠ No terrain mesh prims found under /World/ground")
                print(f"  → Open Stage panel, expand ground prim, "
                      f"check path and tell me the structure")
        except Exception as e:
            print(f"[TERRAIN COLOURS] Skipped ({e})")

        # ── Terrain type labels — floating coloured poles above each patch ──
        # Visible in viewport as tall thin cylinders, one per terrain type.
        # Colour matches terrain material. Height = unique per type for ID.
        # Legend printed to console for reference.
        #
        # Pole heights (easy to see in viewport):
        #   flat=0.5m  gravel_light=1.0m  gravel_heavy=1.5m  grass_wave=2.0m
        #   slope_gentle=2.5m  obstacles=3.0m  slope_steep=3.5m
        try:
            import isaaclab.sim as sim_utils
            LABEL_INFO = {
                # terrain_key: (height_m, colour_rgb, legend_text)
                "flat"         : (0.5,  (0.72, 0.72, 0.72), "FLAT — smooth"),
                "gravel_light" : (1.0,  (0.75, 0.65, 0.48), "GRAVEL LIGHT — ±2-4cm"),
                "gravel_heavy" : (1.5,  (0.48, 0.44, 0.36), "GRAVEL HEAVY — ±5-8cm"),
                "grass_wave"   : (2.0,  (0.15, 0.50, 0.12), "GRASS WAVE — rolling"),
                "slope_gentle" : (2.5,  (0.78, 0.68, 0.30), "SLOPE GENTLE — 3-12°"),
                "obstacles"    : (3.0,  (0.42, 0.28, 0.15), "OBSTACLES — 3-12cm"),
                "slope_steep"  : (3.5,  (0.52, 0.38, 0.18), "SLOPE STEEP — 12-20°"),
            }
            import omni.usd
            from pxr import UsdGeom, Gf, UsdShade, Sdf
            stage = omni.usd.get_context().get_stage()
            _poles_created = 0

            # Find terrain patch centre positions from stage
            _patch_positions = {}  # terrain_key → first patch XYZ
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if "/World/ground" not in path:
                    continue
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                for key in LABEL_INFO:
                    if key in path.lower() and key not in _patch_positions:
                        # Get world position of this mesh prim
                        xf = UsdGeom.Xformable(prim)
                        if xf:
                            bbox = xf.ComputeLocalToWorldTransform(0)
                            pos  = Gf.Vec3f(float(bbox[3][0]),
                                            float(bbox[3][1]),
                                            float(bbox[3][2]))
                            _patch_positions[key] = pos
                        break

            # Create one thin cylinder pole per terrain type at patch centre
            for key, (height, col, legend) in LABEL_INFO.items():
                if key not in _patch_positions:
                    continue
                pos     = _patch_positions[key]
                pole_path = f"/World/TerrainLabels/{key}_pole"
                pole    = UsdGeom.Cylinder.Define(stage, pole_path)
                pole.GetRadiusAttr().Set(0.05)        # 5cm radius — thin
                pole.GetHeightAttr().Set(float(height))
                pole.GetAxisAttr().Set("Z")
                # Place pole at patch centre, raised to half-height
                from pxr import UsdGeom as _ug
                xform = _ug.XformCommonAPI(pole)
                xform.SetTranslate(Gf.Vec3d(
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]) + height * 0.5 + 0.5))   # 0.5m above ground
                # Colour the pole
                mat_path = f"/World/TerrainMats/{key}_pole_mat"
                mat      = UsdShade.Material.Define(stage, mat_path)
                shader   = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor",
                                   Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*col))
                shader.CreateInput("roughness",
                                   Sdf.ValueTypeNames.Float).Set(0.5)
                mat.CreateSurfaceOutput().ConnectToSource(
                    shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(pole).Bind(mat)
                _poles_created += 1

            if _poles_created:
                print(f"[TERRAIN LABELS] {_poles_created} marker poles created")
                print("  Legend (pole height → terrain type):")
                for key, (h, c, legend) in LABEL_INFO.items():
                    if key in _patch_positions:
                        print(f"    {h:.1f}m pole → {legend}")
            else:
                print("[TERRAIN LABELS] No poles — terrain patches not found "
                      "(only works with rough terrain task)")
        except Exception as e:
            print(f"[TERRAIN LABELS] Skipped ({e})")

        print("─"*60 + "\nACTUATOR VERIFICATION\n" + "─"*60)
        for name, act in (getattr(self._robot, "_actuators", None) or {}).items():
            kp = getattr(act, "stiffness",       None)
            kd = getattr(act, "damping",          None)
            vf = getattr(act, "viscous_friction", None)
            print(f"  [{name}]  KP={kp[0,0].item() if kp is not None else 'N/A':5.1f}"
                  f"  KD={kd[0,0].item() if kd is not None else 'N/A':4.1f}"
                  f"  d[RL_th]={vf[0,6].item() if vf is not None else 'N/A':6.3f}")
        try:
            sensor = self.scene.sensors["contact_sensor"]
            print(f"\n  [CONTACT SENSOR] {sensor.body_names}")
            print(f"  Expected: FL[0] FR[1] RL[2] RR[3]")
        except Exception:
            pass
        print("─"*60 + "\n")

    # =========================================================================
    # _pre_physics_step — v6: RL_th target scaling REMOVED (physics DR handles)
    # =========================================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        a     = actions.clone()
        _mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5
        _half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        a = _mid + _half * torch.tanh(a)
        self._prev_prev_actions[:] = self._prev_actions
        self._prev_actions[:]      = self._actions
        self._actions[:]           = a
        self._target_pos = self._actions + self._robot.data.default_joint_pos

        # RL_th: NO target scaling — physics DR (τf/d per episode) handles
        # execution variation. Policy always sends full commanded delta.
        # Full commanded delta at 0.35 rad → PD force = 65×0.35 = 22.75 Nm
        # >> τf=4.944 Nm → can sustain PD_force above stiction threshold.

        # FR_th position cap — hard ceiling on target (binding fault protection)
        fr_th_max = (self._robot.data.default_joint_pos[:, 5]
                     + self._fr_th_max_delta)
        self._target_pos[:, 5] = torch.minimum(self._target_pos[:, 5], fr_th_max)

        # Delay DR ring buffer (FIFO)
        self._delay_buf[self._delay_ptr] = self._target_pos.clone()
        read_ptrs   = (self._delay_ptr - self._env_delays) % _DELAY_BUF_SIZE
        idx         = read_ptrs.view(1, self.num_envs, 1).expand(1, self.num_envs, 12)
        delayed     = self._delay_buf.gather(0, idx).squeeze(0)
        no_delay    = (self._env_delays == 0).unsqueeze(-1)
        cmd_to_send = torch.where(no_delay, self._target_pos, delayed)
        self._delay_ptr = (self._delay_ptr + 1) % _DELAY_BUF_SIZE
        self._robot.set_joint_position_target(cmd_to_send)

    def _apply_action(self):
        pass

    # =========================================================================
    # _get_observations — v15: uses _episode_wz (persistent, survives resampling)
    # obs[0:3] = [vx_cmd, vy_cmd, wz_cmd]
    #   vx, vy from command_manager (vx resamples every 5-10s, vy=0 always)
    #   wz from self._episode_wz — set at episode reset, constant all episode
    #   This ensures wz training is CONTINUOUS within each episode, not reset
    #   by command_manager's mid-episode vx resampling.
    # obs[45] = gait freq cmd normalised
    # =========================================================================
    def _get_observations(self) -> dict:
        # obs[45] = (f_cmd - 1.5) / 1.5 → [0.0, 1.0] for range [1.5, 3.0]
        f_cmd_norm = ((self._episode_f_cmd - self._f_cmd_lo)
                      / (self._f_cmd_hi - self._f_cmd_lo)).unsqueeze(1)  # [N,1]

        # Build cmd obs: vx/vy from command_manager, wz from _episode_wz
        # _episode_wz is set once at episode reset and stays constant all episode
        # This gives the LL a consistent, uninterrupted wz training signal
        cmd_obs = self.command_manager.command[:, :3].clone()   # [N, 3]
        cmd_obs[:, 2] = self._episode_wz                        # override wz ← v15

        obs = torch.cat([
            cmd_obs,                                                       # [0:3]
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # [3:15]
            torch.clamp(self._robot.data.joint_vel,      -5.0, 5.0),      # [15:27]
            torch.clamp(self._robot.data.root_ang_vel_b, -5.0, 5.0),      # [27:30]
            self._robot.data.projected_gravity_b,                          # [30:33]
            self._prev_actions,                                            # [33:45]
            f_cmd_norm,                                                    # [45]
        ], dim=-1)
        if self._obs_noise_enabled:
            obs = obs + torch.randn_like(obs) * self._obs_noise_std
        return {"policy": obs}

    # =========================================================================
    # _get_rewards — v15: stronger wz tracking + FR_th rotation protection
    # Changes from v14:
    #   r_ang_vel:    weight 0.5 → 1.0  (stronger signal to learn turning)
    #   r_wz_excess:  NEW — penalise yaw rate exceeding 1.5×|wz_cmd|
    #                 Suppresses "soldier-march" overreaction seen in HL play test
    #                 At wz=0:     no penalty (straight walking unchanged)
    #                 At wz=0.30:  yaw_rate > 0.45 rad/s → penalty fires
    #   r_fr_binding: wz-scaled — stricter FR_th protection during turns
    #                 wz=0:    -2.0 × fr_prox²  (unchanged from v13)
    #                 wz=±0.30: -5.0 × fr_prox²  (3× stricter at max turn)
    #                 Prevents FR_th binding zone during aggressive rotation
    # =========================================================================
    def _get_rewards(self):
        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        gravity = self._robot.data.projected_gravity_b
        height  = self._robot.data.root_pos_w[:, 2]
        cmd     = self.command_manager.command
        tilt    = torch.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)

        # ── Contact — needed early for gait quality gate on velocity ─────
        # Body order (alphabetical): FL=0, FR=1, RL=2, RR=3
        contact_fz   = self.scene.sensors["contact_sensor"].data.net_forces_w[:, :, 2]
        feet_contact = (contact_fz > 1.0).float()   # [N,4]
        n_contact    = feet_contact.sum(dim=1)        # [N]

        # ── Gait quality gate — triangular peaked at 2-foot (trot) contact ─
        # n=0: 0.0  n=1: 0.5  n=2: 1.0  n=3: 0.5  n=4: 0.0
        # Multiplies velocity reward → proper trot earns FULL velocity reward.
        # Shuffling (4-foot) or hopping (0,1-foot) earns only 30%.
        #
        # WHY this is the correct fix (from data at iter 1707):
        #   Air-time w=15 gain from trot: ~120 per episode
        #   Velocity cost of slowing for trot: ~776 per episode
        #   Air-time is only 15% of velocity loss → policy correctly ignores gait.
        #   With gate: 4-foot earns 30% of velocity, 2-foot earns 100%.
        #   Velocity gain from proper gait: 0.70 × 1140 = 798 per episode.
        #   Now trot earns MORE than shuffling through velocity alone.
        #   This mirrors biology: animals trot because it's energy-efficient
        #   for covering ground, not because trot is directly rewarded.
        gait_quality  = torch.clamp(1.0 - torch.abs(n_contact - 2.0) / 2.0, 0.0, 1.0)
        gait_gate     = 0.30 + 0.70 * gait_quality   # [0.30, 1.00]

        # ── Velocity tracking — gated by gait quality ─────────────────────
        r_lin_vel_raw = 1.5 * torch.exp(-(lin_vel[:, 0] - cmd[:, 0])**2 / 0.25)
        r_lin_vel     = r_lin_vel_raw * gait_gate   # ← gate applied here

        # v15: r_ang_vel uses _episode_wz (persistent per episode)
        #   Weight 0.5 → 1.0: stronger signal so LL actively learns to turn
        #   _episode_wz is the per-episode wz target (50% zero, 50% ±0.30)
        r_ang_vel = 1.0 * torch.exp(-(ang_vel[:, 2] - self._episode_wz)**2 / 0.25)

        # v15 NEW: yaw overshoot penalty — suppresses soldier-march overreaction
        #   HL play test showed yaw_rate 2.7-8.6× the commanded wz
        #   Root cause: RL_th stiction escape creates impulsive yaw
        #   Fix: penalise yaw_rate that exceeds 1.5× the commanded magnitude
        #   At wz=0:    no penalty (abs(yaw) > 0 gives gentle damping)
        #   At wz=0.30: yaw_rate must stay < 0.45 rad/s (1.5×0.30)
        wz_excess    = torch.clamp(
            ang_vel[:, 2].abs() - self._episode_wz.abs() * 1.5, min=0.0)
        r_wz_excess  = -0.5 * wz_excess**2

        # ── Terrain-relative height ────────────────────────────────────────
        # Flat terrain: _terrain_z=0 → height_rel = height (no change)
        # Rough terrain: height_rel removes the terrain floor variation
        # so alive/fall checks are consistent across all terrain types.
        height_rel = height - self._terrain_z

        # ── Velocity-gated alive — terrain-relative ─────────────────────
        vel_gate = torch.clamp(lin_vel[:, 0] / 0.2, 0.0, 1.0)
        r_alive  = 0.3 * vel_gate * ((height_rel > 0.26) & (tilt < 0.3)).float()

        # ── Standard penalties ────────────────────────────────────────────
        # r_ang_vel_xy: -0.08 (same as v13-15).
        r_ang_vel_xy = -0.08 * (ang_vel[:, 0]**2 + ang_vel[:, 1]**2)
        r_lin_vel_z  = -4.0  * lin_vel[:, 2]**2
        r_torques    = -1e-5 * torch.sum(
            self._robot.data.applied_torque**2, dim=1)
        r_fall       = -10.0 * (height_rel < 0.18).float()   # terrain-relative
        r_upright = -1.5 * (gravity[:, 0]**2 + gravity[:, 1]**2)
        r_lat_vel    = -3.5  * lin_vel[:, 1]**2
        r_hip_reg    = -1.5  * torch.sum(self._actions[:, :4]**2, dim=1)
        hip_excess   = torch.clamp(torch.abs(self._actions[:, :4]) - 0.06, 0.0)
        r_hip_sat    = -4.0  * torch.sum(hip_excess, dim=1)

        d1 = self._actions - self._prev_actions
        d2 = self._actions - 2*self._prev_actions + self._prev_prev_actions
        r_action_rate = -1.0 * torch.sum(self._rate_weights * d1**2, dim=1)
        r_action_jerk = -0.5 * torch.sum(self._rate_weights * d2**2, dim=1)

        # ── Contact sensor — already computed above (feet_contact, n_contact) ─
        # feet_contact and n_contact from gait_gate section above — reuse here.

        # ── v10b/v13: relu air-time with f_cmd target ─────────────────────
        #   v5-v6: Static |diag1-diag2| → FR+RL permanent stance
        #   v7:    per-step switch reward → 24Hz rapid toggle
        #   v8:    clamp(air-0.10, 0, 0.40) → 9.95Hz tap on real hardware
        #   v9:    clamp(air-0.20, 0, 0.15) → r_stance_time caused all-4-static
        #   v9fix: clamp(air-0.20, 0, 0.15) + r_stance → 1-leg permanent swing (FL 0.38s)
        #
        # ROOT CAUSE OF ALL EXPLOITS: clamp cap makes one long swing earn more
        # reward-rate than multiple short swings.
        #
        # RUDIN 2022 (legged_gym) original formula:
        #   feet_air_time += dt            (accumulates during swing)
        #   reward = (air_time - target) × first_contact  (fires at touchdown)
        #   feet_air_time *= ~contact      (resets on landing)
        #
        # KEY PROPERTY (mathematically proven):
        #   With no cap, reward_rate = (air - target) × frequency.
        #   For 1 leg at swing S: rate = w(S-T)/(2S×50)  → bounded by w/100
        #   For N legs at period P: rate = wN(P/2-T)/P×50 → grows with N
        #   ∴ 4-leg trot ALWAYS beats 1-leg cycling regardless of swing duration.
        #
        # VELOCITY GATE:
        #   Multiplied by clamp(|v_horizontal|/0.2, 0, 1).
        #   When standing still (v=0): r_air_time = 0 → no gait reward for static.
        #   Policy MUST generate forward motion to earn gait reward.
        #   Blocks: permanent-lift-while-standing, all-4-static, any static pattern.
        #
        # v10b FIX — CATCH-22 IDENTIFIED AND RESOLVED:
        # Original Rudin (air-0.20) penalises short hops with NEGATIVE reward.
        # At training start, policy explores with brief lifts (0.05-0.10s)
        # → gets penalised → learns to keep ALL feet down → trot_penalty fires
        # → CATCH-22: lifting is penalised, not lifting is penalised.
        # Sim log confirmed: FL doing 0.102s hops (8 events), all penalised.
        # RL escaped by random long swing (0.251s) → only leg earning positive.
        #
        # FIX: relu(air - 0.20) — removes negative, keeps positive.
        # Short hop (0.05s): earns 0 (neutral, free to explore)
        # Proper swing (0.25s): earns +0.75 per touch (rewarded)
        # Still blocks ALL exploits via trot_penalty:
        #   Static all-4 → vel_gate=0 → 0 air, -0.40 penalty = -0.40/step
        #   1-leg cycling → 0.045 air, -0.40 penalty = -0.355/step (negative)
        #   1-leg ∞ swing → max 0.15 air, -0.40 = -0.25/step (always negative)
        #   2-leg trot → 0.12 air, 0 penalty = +0.12/step (ONLY positive)
        # No upper cap needed: multi-leg touchdown frequency mathematically dominates.
        sensor      = self.scene.sensors["contact_sensor"]
        last_air    = sensor.data.last_air_time[:, :4]   # [N,4] non-zero at touchdown

        # Velocity gate for all gait rewards
        vel_gate_gait = torch.clamp(
            torch.norm(lin_vel[:, :2], dim=1) / 0.2, 0.0, 1.0)

        first_touch = (feet_contact - self._prev_feet_contact).clamp(min=0.0)

        # v11: Air-time target from gait frequency command
        # target_swing = 1 / (2 × f_cmd)
        #   f_cmd=1Hz → 0.500s  f_cmd=2Hz → 0.250s  f_cmd=4Hz → 0.125s
        # Policy learns to sustain swings that match the commanded frequency.
        # At deploy: obs[45]=(2.0-1.0)/3.0=0.333 → f_cmd=2.0 → target=0.250s
        # Hardware timing: policy explicitly targets 0.250s → gait ≈2Hz
        target_swing = (1.0 / (2.0 * self._episode_f_cmd)).unsqueeze(1)  # [N,1]

        r_air_time  = 15.0 * torch.sum(
            torch.relu(last_air - target_swing) * first_touch, dim=1) * vel_gate_gait
        # relu: no penalty for swings shorter than target (catch-22 removed v10b)
        # target is now per-episode per-env (not fixed 0.20s)

        # Trot diagonal bias + N≥3 penalty (no stance component)
        # r_stance_time REMOVED: it always incentivises max-feet-in-stance,
        # creating gradient opposing trot. Without it:
        #   1-leg cycling: r_air - 0.40 penalty = net negative (must cycle while moving)
        #   2-leg trot: r_air - 0 = net positive
        diag1          = (feet_contact[:, 0] + feet_contact[:, 3]) * 0.5
        diag2          = (feet_contact[:, 1] + feet_contact[:, 2]) * 0.5
        r_trot_bias    = 0.2 * torch.abs(diag1 - diag2)
        r_trot_penalty = -0.4 * (feet_contact.sum(dim=1) >= 3).float()
        r_trot = r_air_time + r_trot_bias + r_trot_penalty

        # Update contact history (needed for first_touch next step)
        self._prev_feet_contact[:] = feet_contact.detach()

        # ── Foot clearance and drag ────────────────────────────────────────
        r_foot_clear = torch.zeros(self.num_envs, device=self.device)
        r_foot_drag = torch.zeros(self.num_envs, device=self.device)

        if self._foot_body_ids is not None:
            foot_pos = self._robot.data.body_pos_w[:, self._foot_body_ids, :]
            foot_vel = self._robot.data.body_vel_w[:, self._foot_body_ids, :]
            swing_mask = 1.0 - feet_contact
            ground_z = self.scene.env_origins[:, 2].unsqueeze(1)
            foot_z = foot_pos[:, :, 2]
            foot_z_rel = foot_z - ground_z

            # ── Clearance: asymmetric weights (from real-hw fix) ──────────
            # Front legs FL/FR: 3.0 weight — historically dragged on hw
            # Rear  legs RL/RR: 1.5 weight — already lifting well
            # Deadband REMOVED for rough terrain:
            #   On flat: 3cm deadband prevents passive hang (ok)
            #   On rough: terrain bumps bring ground_z up under foot → deadband
            #             turns clearance negative → policy pushed foot DOWN
            clearance_w = torch.tensor(
                [3.0, 3.0, 1.5, 1.5], device=self.device)
            # No deadband — let any upward foot position earn reward on terrain
            clearance = torch.clamp(foot_z_rel, 0.0, 0.12)
            r_foot_clear = torch.sum(
                swing_mask * clearance
                * clearance_w.unsqueeze(0), dim=1) * vel_gate_gait

            # ── Drag: contact-shuffle ONLY — NO swing-drag penalty ────────
            # Swing-drag (-4.0) was causing foot_drag=-241:
            #   On rough terrain, gravel/wave bumps are near foot during swing
            #   → false drag fires constantly → policy learned to keep feet low
            #   to avoid trigger → made FR_kn mean worse (-0.003→-0.111)
            #
            # Literature (Rudin 2022, ETH ANYmal, Walk These Ways):
            #   No swing-drag penalty for rough terrain — handled by air_time
            #
            # Keep only: penalise fast feet when IN CONTACT (shuffle prevention)
            foot_speed_xy = torch.norm(foot_vel[:, :, :2], dim=-1)
            r_foot_drag = -1.5 * torch.sum(
                feet_contact * foot_speed_xy, dim=1)
            # Weight increased 1.0→1.5 to compensate for removing swing_drag

        # ── FR_th binding proximity — REAL HW FIX: onset 0.800→0.780 ──
        # Real hardware: FR_th>0.800 on 68% of steps with cap=0.800
        # The policy was NEVER penalised for approaching 0.800 — only at it.
        # With onset at 0.780 the gradient starts 20ms earlier, teaching
        # the policy to AVOID the binding zone rather than get clipped at it.
        # Penalty ramp: 0 at 0.780, full at 0.850 (70mrad window)
        fr_th_q  = self._robot.data.joint_pos[:, 5]
        fr_prox = torch.clamp((fr_th_q - 0.790) / 0.070, 0.0, 1.0)
        wz_scale = torch.clamp(self._episode_wz.abs() / self._WZ_MAX, 0.0, 1.0)
        r_fr_binding = -(2.0 + 1.5 * wz_scale) * fr_prox ** 2

        # ── Episode sum tracking ──────────────────────────────────────────
        for k, v in zip(
            ["lin_vel", "ang_vel", "ang_vel_xy", "lin_vel_z", "torques",
             "action_rate", "action_jerk", "upright", "trot", "alive",
             "fall", "hip_reg", "hip_sat", "foot_clear", "foot_drag",
             "lat_vel", "fr_binding", "wz_excess"],
            [r_lin_vel, r_ang_vel, r_ang_vel_xy, r_lin_vel_z, r_torques,
             r_action_rate, r_action_jerk, r_upright, r_trot, r_alive,
             r_fall, r_hip_reg, r_hip_sat, r_foot_clear, r_foot_drag,
             r_lat_vel, r_fr_binding, r_wz_excess]
        ):
            self._ep_sums[k] += v

        # ── Total reward ──────────────────────────────────────────────────
        return self.step_dt * (
            r_lin_vel     + r_ang_vel     + r_ang_vel_xy
            + r_lin_vel_z + r_torques     + r_action_rate  + r_action_jerk
            + r_upright   + r_trot        + r_alive
            + r_hip_reg   + r_hip_sat
            + r_foot_clear + r_foot_drag
            + r_lat_vel   + r_fr_binding  + r_wz_excess    # ← v15: wz_excess added
        ) + r_fall

    # =========================================================================
    # _get_dones — terrain-relative height check
    # =========================================================================
    def _get_dones(self):
        g      = self._robot.data.projected_gravity_b
        height = self._robot.data.root_pos_w[:, 2]
        tilt   = torch.sqrt(g[:, 0]**2 + g[:, 1]**2)
        # Height ABOVE terrain origin — works for both flat (terrain_z=0) and rough
        height_rel = height - self._terrain_z
        terminated = (tilt > 0.8) | (height_rel < 0.22) | (g[:, 2] > 0.3)
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # =========================================================================
    # _reset_idx — v6: RL_th physics DR (τf + d per episode, correlated)
    # =========================================================================
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        if self._global_step % 200 == 0 and self._global_step > 0:
            print(f"\n=== Rewards @ step {self._global_step} ===")
            for k, v in self._ep_sums.items():
                print(f"  {k:14s}: {v[env_ids].mean().item():+.3f}")
            d = self._env_delays.float()
            print(f"  [Delay DR] mean={d.mean():.2f}  max={d.max():.0f}")
            print("=" * 40)

        n        = len(env_ids)
        actuator = self._robot.actuators["legs"]

        super()._reset_idx(env_ids)
        self.command_manager.reset(env_ids)

        # ── v15: Mixed wz assignment — 50/50, ±0.30, _episode_wz ────────
        # v14: 70/30 split, ±0.20 — policy biased toward straight, weak turns
        # v15: 50/50 split, ±0.30 — equal training on straight and turning
        #
        # _episode_wz: separate buffer from _wz_cmd, set here at reset.
        # _get_observations reads _episode_wz, NOT _wz_cmd.
        # This ensures wz is continuous and constant throughout the full episode
        # even if command_manager resamples vx mid-episode.
        # r_ang_vel and r_fr_binding both use _episode_wz in _get_rewards.
        _n_reset    = len(env_ids)
        _is_turning = torch.rand(_n_reset, device=self.device) >= self._WZ_STRAIGHT_PROB
        _wz_turn    = (torch.rand(_n_reset, device=self.device) * 2.0 - 1.0) * self._WZ_MAX
        _wz_assign  = torch.where(
            _is_turning, _wz_turn, torch.zeros(_n_reset, device=self.device))
        self._wz_cmd[env_ids]     = _wz_assign   # keep for debug compatibility
        self._episode_wz[env_ids] = _wz_assign   # persistent per episode

        # Debug: print wz distribution every ~50k env-resets
        self._wz_reset_count += _n_reset
        if self._wz_reset_count % 50000 < _n_reset:
            _n_turn  = _is_turning.sum().item()
            _all_std = self._episode_wz.std().item()
            _all_mean= self._episode_wz.mean().item()
            expected_std = self._WZ_MAX * (1.0 - self._WZ_STRAIGHT_PROB)**0.5 / 3.0**0.5
            print(f"  [WZ MIX v15] reset#{self._wz_reset_count//1000}k  "
                  f"turning={_n_turn}/{_n_reset}({100*_n_turn/_n_reset:.0f}%)  "
                  f"all_envs: mean={_all_mean:.4f}  std={_all_std:.4f}  "
                  f"target={expected_std:.4f}  deploy_safe=±{2*expected_std:.3f}")

        # Re-zero PhysX drive damping
        self._robot.write_joint_damping_to_sim(
            torch.zeros(n, 12, device=self.device), env_ids=env_ids)

        # ── KP / KD DR ───────────────────────────────────────────────────
        kp_scale        = torch.ones(n, 12, device=self.device)
        kp_scale[:, :8] = torch.empty(n, 8, device=self.device).uniform_(
            self._kp_dr_hip_th_lo, self._kp_dr_hip_th_hi)
        kp_scale[:, 8:] = torch.empty(n, 4, device=self.device).uniform_(
            self._kp_dr_kn_lo, self._kp_dr_kn_hi)
        new_kp = self._kp_nominal.unsqueeze(0) * kp_scale
        actuator.stiffness[env_ids] = new_kp
        self._kp_live[env_ids]      = new_kp

        kd_scale        = torch.ones(n, 12, device=self.device)
        kd_scale[:, :8] = torch.empty(n, 8, device=self.device).uniform_(
            self._kd_dr_hip_th_lo, self._kd_dr_hip_th_hi)
        kd_scale[:, 8:] = torch.empty(n, 4, device=self.device).uniform_(
            self._kd_dr_kn_lo, self._kd_dr_kn_hi)
        actuator.damping[env_ids] = self._kd_nominal.unsqueeze(0) * kd_scale

        # ── RL_th Physics DR (v6 — replaces target-scale DR) ─────────────
        # Per-episode fault level: 0=healthy, 1=full PACE fault.
        # Correlated τf and d give physically consistent episodes:
        #   fault=0: RL_th healthy → policy learns to CYCLE delta with gait phase
        #   fault=1: RL_th full fault → policy learns hardware compensation
        # Scale of 1000 envs at any time:
        #   ~333 envs: fault_level < 0.33 → RL_th mostly healthy (good gradient)
        #   ~334 envs: fault_level 0.33–0.67 → partial fault (interpolated)
        #   ~333 envs: fault_level > 0.67 → mostly faulted (hardware-like)
        rl_fault_level = torch.rand(n, device=self.device)   # U[0,1]
        self._rl_fault_level[env_ids] = rl_fault_level

        # d (viscous damping): always works via actuator tensor
        # super() restored cfg baseline (0.048 for RL_th); override per episode
        d_healthy  = self._d_nominal[6].item()     # 3.459 Nm·s/rad (fault upper bound)
        d_nominal_healthy = 0.048                  # healthy thigh value
        rl_d_values = d_nominal_healthy + rl_fault_level * (d_healthy - d_nominal_healthy)
        if hasattr(actuator, 'viscous_friction'):
            # First restore all joints to nominal (super may have reset them)
            actuator.viscous_friction[env_ids] = (
                self._d_nominal.unsqueeze(0).expand(n, -1))
            # Then override RL_th with per-episode value
            actuator.viscous_friction[env_ids, 6] = rl_d_values

        # τf (Coulomb friction): per-episode if API supports env_ids
        # super() does NOT reset τf (it's applied via PhysX, persists across reset)
        # so we only need to apply to reset envs
        tau_f_healthy_rl = 0.007                        # healthy thigh τf
        tau_f_fault_rl   = self._tau_f_nominal[6].item()  # 4.944 Nm
        rl_tau_f_values  = tau_f_healthy_rl + rl_fault_level * (
            tau_f_fault_rl - tau_f_healthy_rl)

        if self._friction_env_ids_ok and self._tau_fn is not None:
            tau_f_reset = self._tau_f_nominal.unsqueeze(0).expand(n, -1).clone()
            tau_f_reset[:, 6] = rl_tau_f_values
            getattr(self._robot, self._tau_fn)(tau_f_reset, env_ids=env_ids)
        # If env_ids not supported: τf stays at init value (4.944 for RL_th).
        # d-only DR still provides meaningful spectrum of viscous resistance.

        # Re-apply encoder bias (super restores cfg 0.0)
        if self._bias_attr is not None:
            getattr(actuator, self._bias_attr)[env_ids] = (
                self._bias_values.unsqueeze(0).expand(n, -1))

        # ── v12: Gait frequency command — U[1.5, 3.0] Hz per episode ────
        # Narrower range than v11 (was U[1,4]) → 2× ratio vs 4×.
        # obs[45] = (f_cmd-1.0)/3.0 → [0.167, 0.667] in this range.
        # Policy learns: obs[45]=0.167→slow(1.5Hz), obs[45]=0.333→trot(2Hz).
        self._episode_f_cmd[env_ids] = (
            self._f_cmd_lo
            + torch.rand(n, device=self.device) * (self._f_cmd_hi - self._f_cmd_lo)
        )

        # ── Delay DR ──────────────────────────────────────────────────────
        # Bug fix: this block was EMPTY in rough terrain version — code was
        # accidentally deleted during terrain additions. Restored here.
        # Phase 1: delay=0 (no lag) for first _DELAY_PHASE1_END steps
        # Phase 2: delay=U[0,8] steps = U[0,16ms] — matches real hardware lag
        if self._global_step < _DELAY_PHASE1_END:
            self._env_delays[env_ids] = 0
        else:
            if self._global_step == _DELAY_PHASE1_END and len(env_ids) > 0:
                print(f"\n  [Delay DR] Phase 2 NOW ACTIVE at step "
                      f"{self._global_step} — delay U[0,{_DELAY_MAX}] steps")
            self._env_delays[env_ids] = torch.randint(
                0, _DELAY_MAX + 1, (n,),
                device=self.device, dtype=torch.long)
        # MUST run BEFORE robot placement below so env_origins are correct
        # when root_state[:, :3] = self.scene.env_origins[env_ids] executes.
        # Previous bug: curriculum ran AFTER placement → old origins used →
        # robots teleported to new terrain next reset → visible stacking.
        #
        # Logic: survived full episode → move up 1 row (harder terrain)
        #        fell in <50 steps    → move down 1 row (easier terrain)
        # Flat training: scene.terrain has no terrain_levels → silent no-op
        try:
            terrain = self.scene.terrain
            if hasattr(terrain, 'terrain_levels') and hasattr(terrain, 'update_terrain_levels'):
                ep_lens   = self.episode_length_buf[env_ids]
                succeeded = ep_lens >= (self.max_episode_length - 5)
                fell_fast = ep_lens < 50
                move = torch.zeros(n, dtype=torch.long, device=self.device)
                move[succeeded] =  1
                move[fell_fast] = -1
                terrain.update_terrain_levels(env_ids, move)
                # Refresh terrain height offsets BEFORE robot placement
                self._terrain_z[env_ids] = self.scene.env_origins[env_ids, 2]
        except Exception:
            pass   # flat terrain: no-op

        # ── Debug print every 200 steps ───────────────────────────────────
        if self._global_step % 200 == 0 and self._global_step > 0:
            diff = (abs(actuator.stiffness[0]-actuator.stiffness[1]).max().item()
                    if self.num_envs > 1 else 0.0)
            print(f"  [KP DR]    diff={diff:.1f} {'✓' if diff>2 else '⚠'}")

            fl_mean = self._rl_fault_level.mean().item()
            fl_low  = (self._rl_fault_level < 0.33).sum().item()
            fl_high = (self._rl_fault_level > 0.67).sum().item()
            print(f"  [RL_th DR] fault_level mean={fl_mean:.2f}  "
                  f"healthy(<0.33)={fl_low}  fault(>0.67)={fl_high} envs")

            if hasattr(actuator, 'viscous_friction'):
                vf6_vals = actuator.viscous_friction[:, 6]
                print(f"  [d RL_th]  mean={vf6_vals.mean():.3f}  "
                      f"min={vf6_vals.min():.3f}  max={vf6_vals.max():.3f}")

            fr_th_cap = (self._robot.data.default_joint_pos[env_ids[0], 5].item()
                         + self._fr_th_max_delta)
            print(f"  [FR_th]    cap={fr_th_cap:.3f}  "
                  f"proximity penalty starts at 0.800")

        # ── Reset state ───────────────────────────────────────────────────
        _mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5
        _half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        _rand = (_mid + _half * self._prev_act_init_scale
                 * (torch.rand(n, 12, device=self.device) * 2 - 1))
        self._actions[env_ids]           = _rand
        self._prev_actions[env_ids]      = _rand
        self._prev_prev_actions[env_ids] = _rand
        self._target_pos[env_ids] = self._robot.data.default_joint_pos[env_ids]
        # FIX 1: Reset contact history — prevents spurious switch reward
        # on first step of new episode when feet go from "previous" to actual state.
        self._prev_feet_contact[env_ids] = 0.0

        for i in range(_DELAY_BUF_SIZE):
            self._delay_buf[i, env_ids] = (
                self._robot.data.default_joint_pos[env_ids])
        for k in self._ep_sums:
            self._ep_sums[k][env_ids] = 0.0

        joint_pos  = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += (torch.rand_like(joint_pos) - 0.5) * 0.05
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3]   = self.scene.env_origins[env_ids]
        root_state[:, 2]   += 0.35
        root_state[:, 3:7]  = torch.tensor([1., 0., 0., 0.], device=self.device)
        root_state[:, 7:13] = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), None, env_ids)

    # =========================================================================
    # step — v7: show trot switch count in debug
    # =========================================================================
    def step(self, action):
        self._global_step += 1

        if self._global_step % 500 == 0:
            idx  = 0
            act  = self._robot.actuators["legs"]
            ph   = ("Phase 2 DR" if self._global_step >= _DELAY_PHASE1_END
                    else f"Phase 1 (DR at step {_DELAY_PHASE1_END})")
            print(f"\n{'='*80}")
            print(f"[DEBUG] step {self._global_step} | {ph} | "
                  f"cmd_vx={self.command_manager.command[idx,0]:.2f}  "
                  f"episode_wz={self._episode_wz[idx]:.3f}  "
                  f"({'TURNING' if abs(self._episode_wz[idx].item())>0.02 else 'STRAIGHT'})")

            kp0  = act.stiffness[0].cpu().numpy().round(1)
            kp1  = act.stiffness[1].cpu().numpy().round(1) if self.num_envs > 1 else kp0
            diff = (abs(act.stiffness[0]-act.stiffness[1]).max().item()
                    if self.num_envs > 1 else 0.)
            print(f"  KP env0: {kp0}  {'✓ DR' if diff>2 else '⚠'}")

            d_delays = self._env_delays.float()
            if self._global_step >= _DELAY_PHASE1_END:
                print(f"  Delay:   mean={d_delays.mean():.1f} "
                      f"({d_delays.mean().item()*2:.0f}ms)")
            else:
                print(f"  Delay:   0ms  [Phase 1]")

            # RL_th physics DR status
            if hasattr(act, 'viscous_friction'):
                vf = act.viscous_friction[:, 6]
                print(f"  RL_th d: mean={vf.mean():.3f}  "
                      f"min={vf.min():.3f}  max={vf.max():.3f}  "
                      f"healthy(<0.1)={(vf<0.1).sum().item()} envs  "
                      f"fault(>3.0)={(vf>3.0).sum().item()} envs")
            fl = self._rl_fault_level
            print(f"  RL_th fault_level: mean={fl.mean():.2f}  "
                  f"healthy(<0.33)={(fl<0.33).sum().item()}  "
                  f"fault(>0.67)={(fl>0.67).sum().item()} envs")

            if self._bias_attr and hasattr(act, self._bias_attr):
                b = getattr(act, self._bias_attr)[0].cpu().numpy()
                print(f"  q̃b env0: RL_th={b[6]:+.3f}  FR_th={b[5]:+.3f}")

            if self.last_obs is not None:
                o      = self.last_obs
                gv     = o[30:33]
                tilt_d = float(torch.sqrt(gv[0]**2+gv[1]**2).item())*57.3
                print(f"  proj_grav: {gv.cpu().numpy()}  tilt≈{tilt_d:.1f}°")

            tgt = self._target_pos[idx].cpu().numpy()
            aq  = self._robot.data.joint_pos[idx].cpu().numpy()
            te  = np.abs(aq - tgt)
            JN  = ['FL_h','FR_h','RL_h','RR_h',
                   'FL_th','FR_th','RL_th','RR_th',
                   'FL_kn','FR_kn','RL_kn','RR_kn']
            print(f"  track_err: mean={te.mean():.4f} max={te.max():.4f} "
                  f"({JN[te.argmax()]})")
            da_rl = float((self._actions[idx,6]-self._prev_actions[idx,6]).abs().item())
            da_fr = float((self._actions[idx,5]-self._prev_actions[idx,5]).abs().item())
            print(f"  RL_th |Δdelta|={da_rl:.4f}  "
                  f"FR_th |Δdelta|={da_fr:.4f}  "
                  f"(RL: expect higher than v5; FR: expect lower)")

            lv = self._robot.data.root_lin_vel_b[idx,0].item()
            vg = min(max(lv/0.2, 0.0), 1.0)
            print(f"  lin_vel:   {lv:.3f} m/s  vel_gate={vg:.2f}  "
                  f"r_alive≈{0.3*vg:.3f}/step")

            fr_th_now = self._target_pos[idx, 5].item()
            fr_th_act = self._robot.data.joint_pos[idx, 5].item()
            fr_th_cap = (self._robot.data.default_joint_pos[idx, 5].item()
                         + self._fr_th_max_delta)
            fr_prox   = max(0.0, min(1.0, (fr_th_act - 0.800) / 0.070))
            print(f"  FR_th: tgt={fr_th_now:.3f}  act={fr_th_act:.3f}  "
                  f"cap={fr_th_cap:.3f}  proximity={fr_prox:.3f}  "
                  f"binding_penalty={-2.0*fr_prox**2:.4f}/step")

            rl_th_tgt = self._target_pos[idx, 6].item()
            rl_th_act = self._robot.data.joint_pos[idx, 6].item()
            print(f"  RL_th: tgt={rl_th_tgt:.3f}  act={rl_th_act:.3f}  "
                  f"fault_level(env0)={self._rl_fault_level[idx].item():.3f}")

            try:
                sensor_dbg = self.scene.sensors["contact_sensor"]
                cf  = sensor_dbg.data.net_forces_w[idx, :, 2]
                fc  = (cf > 1.0).cpu().numpy()
                lat = sensor_dbg.data.last_air_time[idx, :4].cpu().numpy()
                f0  = self._episode_f_cmd[idx].item()
                tgt = 1.0 / (2.0 * f0)
                print(f"  contact:   FL={fc[0]} FR={fc[1]} RL={fc[2]} RR={fc[3]}")
                print(f"  last_air:  FL={lat[0]:.3f} FR={lat[1]:.3f} "
                      f"RL={lat[2]:.3f} RR={lat[3]:.3f} s")
                print(f"  f_cmd={f0:.2f}Hz  target_swing={tgt:.3f}s  "
                      f"obs[45]={((f0-self._f_cmd_lo)/(self._f_cmd_hi-self._f_cmd_lo)):.3f}")
            except Exception:
                pass

        result        = super().step(action)
        self.last_obs = result[0]["policy"][0].clone()

        if self._slog_active and self._slog_step < self._sim_log_maxsteps:
            s    = self._slog_step
            obs0 = result[0]["policy"][0]
            g    = self._robot.data.projected_gravity_b[0]
            tilt = float(torch.sqrt(g[0]**2+g[1]**2).item())
            try:
                fz  = self.scene.sensors["contact_sensor"].data.net_forces_w[0,:,2]
                cnp = fz.cpu().numpy()
            except Exception:
                cnp = np.zeros(4, np.float32)
            self._slog["obs_raw"][s]    = obs0.cpu().numpy()
            self._slog["tanh_delta"][s] = self._actions[0].cpu().numpy()
            self._slog["target_q"][s]   = self._target_pos[0].cpu().numpy()
            self._slog["actual_q"][s]   = self._robot.data.joint_pos[0].cpu().numpy()
            self._slog["actual_qd"][s]  = self._robot.data.joint_vel[0].cpu().numpy()
            self._slog["proj_grav"][s]  = g.cpu().numpy()
            self._slog["ang_vel"][s]    = self._robot.data.root_ang_vel_b[0].cpu().numpy()
            self._slog["lin_vel"][s]    = self._robot.data.root_lin_vel_b[0].cpu().numpy()
            self._slog["cmd"][s]        = self.command_manager.command[0,:3].cpu().numpy()
            self._slog["contact"][s]    = cnp
            self._slog["tilt_deg"][s]   = tilt*(180./3.14159265)
            self._slog["reward"][s]     = float(result[1][0].item())
            self._slog_step += 1

        return result