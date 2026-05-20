# go1_env_sparse.py — Emergent gait experiment
#
# HYPOTHESIS: The trot in model_35000.pt is a shaped local optimum, not the
# physically optimal gait for this hardware. Test by fine-tuning from the
# trained checkpoint using ONLY displacement reward — no gait prescription.
#
# RESEARCH CONTEXT:
#   Heess et al. 2017 (DeepMind): trotting/bounding emerged from scratch
#   with only move_forward + don't_fall on rich terrain.
#   Ng et al. 1999: non-potential-based shaping creates spurious optima.
#   This experiment: start FROM shaped policy, strip rewards, let physics select.
#
# WHAT IS KEPT (unchanged from go1_env.py v14):
#   PACE: Ia, τf DR, d DR, encoder bias, KP/KD DR, delay FIFO
#   Observation: full 46D (policy sees same inputs — checkpoint loads cleanly)
#   Action: tanh squash, FR_th cap, RL_th physics DR
#   Dones: same tilt/height/gravity termination
#
# WHAT IS REMOVED (all gait-prescribing terms):
#   r_trot (air_time + trot_bias + trot_penalty)
#   r_ang_vel_xy, r_lin_vel_z, r_hip_sat, r_hip_reg
#   r_foot_clear, r_foot_drag, r_lat_vel, r_alive
#
# WHAT IS KEPT AS MINIMAL SAFETY:
#   r_fall:      -10 × (height < 0.25)    — prevent face-plant exploit
#   r_energy:    -1e-5 × Στ²              — prevent thermal runaway on real HW
#   r_upright:   -1.0 × tilt²            — prevent catastrophic pitch only
#   r_fr_binding: hardware safety (FR_BINDING_WEIGHT=0 to disable)
#
# REWARD: forward displacement from last step position
#   r_forward = clamp(Δx_world / step_dt, -2, 2)
#
# ═══════════════════════════════════════════════════════════════════════════
# DEPLOYMENT:
#   python train.py --task Isaac-Go1-Sparse-Direct-v0 \
#     --num_envs 1000 \
#     --checkpoint /path/to/model_35000.pt
# ═══════════════════════════════════════════════════════════════════════════

import torch
import numpy as np
import os
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand

_DELAY_PHASE1_END = 50_000
_DELAY_MAX        = 8
_DELAY_BUF_SIZE   = 10

# ── Experiment knobs ──────────────────────────────────────────────────────────
SPARSE_MODE         = False   # True = reward only at episode end (very hard)
FR_BINDING_WEIGHT   = -2.0    # set 0.0 to disable hardware safety term
UPRIGHT_WEIGHT      = -1.0    # reduced from v14 -3.5; minimal, not gait shaping
ENERGY_WEIGHT       = -1e-5   # prevent thermal runaway; set 0.0 for pure test


class Go1EnvSparse(DirectRLEnv):
    """
    Sparse displacement reward variant of Go1Env for emergent gait experiment.
    Load from trained checkpoint via --checkpoint in train.py.
    All PACE physics, DR, observation, and action structure from v14 unchanged.
    Only _get_rewards is fundamentally different.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "="*72)
        print("Go1EnvSparse | EMERGENT GAIT EXPERIMENT")
        print("  Reward: displacement_x only + minimal survival constraints")
        print(f"  SPARSE_MODE={SPARSE_MODE}  FR_BINDING={FR_BINDING_WEIGHT}  "
              f"UPRIGHT={UPRIGHT_WEIGHT}  ENERGY={ENERGY_WEIGHT}")
        print("="*72)

        _n    = self.num_envs
        _j    = 12
        _zero = torch.zeros(_n, _j, device=self.device)

        # ── PACE: Ia (armature) — identical to v9 fix ─────────────────────
        self._robot.write_joint_armature_to_sim(_zero)
        Ia = torch.tensor([
            0.0026, 0.0037, 0.0029, 0.0031,
            0.0052, 0.0045, 0.1343, 0.0064,
            0.0070, 0.0080, 0.0065, 0.0070,
        ], device=self.device).unsqueeze(0).expand(_n, -1)
        self._robot.write_joint_armature_to_sim(Ia)

        self._robot.write_joint_damping_to_sim(_zero)

        # ── PACE: τf Coulomb friction ─────────────────────────────────────
        self._tau_f_nominal = torch.tensor([
            0.026, 0.028, 0.029, 0.031,
            0.052, 0.044, 4.944, 0.045,
            0.070, 0.071, 0.078, 0.079,
        ], device=self.device)
        tau_f_init = self._tau_f_nominal.unsqueeze(0).expand(_n, -1)
        self._tau_fn = None
        for _fn in ("write_joint_friction_coefficient_to_sim",
                    "write_joint_friction_to_sim"):
            if hasattr(self._robot, _fn):
                getattr(self._robot, _fn)(tau_f_init)
                self._tau_fn = _fn
                break

        self._friction_env_ids_ok = False
        if self._tau_fn is not None:
            try:
                _t = torch.zeros(1, 12, device=self.device)
                _t[0] = self._tau_f_nominal
                getattr(self._robot, self._tau_fn)(
                    _t, env_ids=torch.tensor([0], device=self.device))
                getattr(self._robot, self._tau_fn)(tau_f_init)
                self._friction_env_ids_ok = True
                print("  [API] τf per-episode DR: SUPPORTED ✓")
            except TypeError:
                print("  [API] τf per-episode DR: not supported — d-only DR active")

        # ── PACE: d (viscous damping) ─────────────────────────────────────
        self._d_nominal = torch.tensor([
            0.039, 0.040, 0.040, 0.042,
            0.045, 0.050, 3.459, 0.050,
            0.124, 0.092, 0.108, 0.108,
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

        # Rate weights identical to v14 — action space unchanged
        self._rate_weights = torch.tensor([
            1.500, 1.500, 1.500, 1.500,
            0.900, 1.800, 0.150, 0.900,
            0.564, 0.564, 0.564, 0.564,
        ], device=self.device)

        self._delta_soft_lo = torch.tensor(
            [-0.08, -0.08, -0.08, -0.08,
             -0.35, -0.35, -0.35, -0.35,
             -0.35, -0.35, -0.35, -0.35], device=self.device)
        self._delta_soft_hi = torch.tensor(
            [ 0.08,  0.08,  0.08,  0.08,
              0.35,  0.35,  0.35,  0.35,
              0.35,  0.35,  0.35,  0.35], device=self.device)

        self._rl_fault_level = torch.ones(_n, device=self.device)

        # ── Displacement tracking ─────────────────────────────────────────
        self._episode_start_pos = torch.zeros(_n, 2, device=self.device)
        self._prev_pos_x        = torch.zeros(_n, device=self.device)
        self._episode_total_displacement = torch.zeros(_n, device=self.device)

        # ── f_cmd fixed at 2.0Hz — CRITICAL for obs[45] to match checkpoint ─
        # obs[45] = (2.0 - 1.5) / 1.5 = 0.333 — same value model_35000.pt trained on
        self._episode_f_cmd = torch.full((_n,), 2.0, device=self.device)
        self._f_cmd_lo = 1.5
        self._f_cmd_hi = 3.0

        # ── wz command — keep 70/30 split from v14 ────────────────────────
        self._wz_cmd           = torch.zeros(_n, device=self.device)
        self._WZ_STRAIGHT_PROB = 0.70
        self._WZ_MAX           = 0.20
        self._wz_reset_count   = 0

        # ── Episode reward tracking ───────────────────────────────────────
        _rk = ["forward_disp", "fall", "energy", "upright",
               "fr_binding", "action_rate", "action_jerk"]
        self._ep_sums = {k: torch.zeros(_n, device=self.device) for k in _rk}

        # ── Gait analysis logging ─────────────────────────────────────────
        self._gait_log = {
            "n_contact_hist":  torch.zeros(_n, 5, device=self.device),
            "air_time_sum":    torch.zeros(_n, 4, device=self.device),
            "air_time_count":  torch.zeros(_n, 4, device=self.device),
            "step_count":      torch.zeros(_n, device=self.device),
        }
        self._prev_feet_contact = torch.zeros(_n, 4, device=self.device)

        self._global_step = (_DELAY_PHASE1_END
                             if os.environ.get("GO1_EVAL_PHASE2") == "1" else 0)

        # Obs noise identical to v14 (46D)
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
            *([0.0] * 12),
            0.0,
        ], device=self.device)

        self._obs_noise_enabled   = True
        self._prev_act_init_scale = 0.3
        self.last_obs             = None

        try:
            _fl, _ = self._robot.find_bodies("FL_foot")
            _fr, _ = self._robot.find_bodies("FR_foot")
            _rl, _ = self._robot.find_bodies("RL_foot")
            _rr, _ = self._robot.find_bodies("RR_foot")
            self._foot_body_ids = [_fl[0], _fr[0], _rl[0], _rr[0]]
        except Exception:
            self._foot_body_ids = None

        N = 2000
        self._sim_log_maxsteps = N
        self._slog = {
            k: np.zeros((N, s) if s > 1 else N, np.float32)
            for k, s in [("obs_raw",46),("tanh_delta",12),
                         ("target_q",12),("actual_q",12),("actual_qd",12),
                         ("proj_grav",3),("ang_vel",3),("lin_vel",3),
                         ("cmd",3),("contact",4),("tilt_deg",1),("reward",1),
                         ("displacement_x",1)]
        }
        self._slog_step   = 0
        self._slog_active = False

        # ── CHECKPOINT VERIFICATION CHECKLIST ────────────────────────────
        obs45_val = (2.0 - self._f_cmd_lo) / (self._f_cmd_hi - self._f_cmd_lo)
        print("\n" + "─"*60)
        print("CHECKPOINT VERIFICATION — what to look for in console:")
        print("  Before 'Learning iteration 1' you must see ALL of:")
        print("  [WARM-START] Loading checkpoint: ...")
        print("  [LOAD] ✓ Normalizer restored from ...")
        print("  [LOAD] ✓ Policy weights restored from ...")
        print("  [WARM-START] Actor noise std = 0.03xx  (must be <<0.10)")
        print()
        print(f"  obs[45] fixed = {obs45_val:.3f}  (f_cmd=2.0Hz — matches v14 training)")
        print()
        print("  PASS signals at step 500 [SPARSE DEBUG] print:")
        print("    lin_vel_x > 0.1 m/s        → WALKING ✓  (checkpoint loaded)")
        print("    2-foot% > 35%               → trot pattern from v14 preserved")
        print("    forward_disp ep_sum > 0     → reward signal working")
        print()
        print("  FAIL signals (checkpoint NOT loaded):")
        print("    noise std = 0.1000 exactly  → fresh random policy")
        print("    lin_vel_x ≈ 0 at step 500  → robot standing / random actions")
        print("    4-foot% > 70%               → same as original bad run")
        print("─"*60 + "\n")

    # =========================================================================
    # _setup_scene
    # =========================================================================
    def _setup_scene(self):
        self._robot = self.scene["robot"]

    # =========================================================================
    # _pre_physics_step — identical to v14
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

        fr_th_max = (self._robot.data.default_joint_pos[:, 5]
                     + self._fr_th_max_delta)
        self._target_pos[:, 5] = torch.minimum(self._target_pos[:, 5], fr_th_max)

        self._delay_buf[self._delay_ptr] = self._target_pos.clone()
        read_ptrs = (self._delay_ptr - self._env_delays) % _DELAY_BUF_SIZE
        idx       = read_ptrs.view(1, self.num_envs, 1).expand(1, self.num_envs, 12)
        delayed   = self._delay_buf.gather(0, idx).squeeze(0)
        no_delay  = (self._env_delays == 0).unsqueeze(-1)
        cmd_to_send = torch.where(no_delay, self._target_pos, delayed)
        self._delay_ptr = (self._delay_ptr + 1) % _DELAY_BUF_SIZE
        self._robot.set_joint_position_target(cmd_to_send)

    def _apply_action(self):
        pass

    # =========================================================================
    # _get_observations — identical to v14 (46D, wz override)
    # CRITICAL: must match model_35000.pt training exactly
    # =========================================================================
    def _get_observations(self) -> dict:
        f_cmd_norm = ((self._episode_f_cmd - self._f_cmd_lo)
                      / (self._f_cmd_hi - self._f_cmd_lo)).unsqueeze(1)

        cmd_obs = self.command_manager.command[:, :3].clone()
        cmd_obs[:, 2] = self._wz_cmd

        obs = torch.cat([
            cmd_obs,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            torch.clamp(self._robot.data.joint_vel,      -5.0, 5.0),
            torch.clamp(self._robot.data.root_ang_vel_b, -5.0, 5.0),
            self._robot.data.projected_gravity_b,
            self._prev_actions,
            f_cmd_norm,
        ], dim=-1)
        if self._obs_noise_enabled:
            obs = obs + torch.randn_like(obs) * self._obs_noise_std
        return {"policy": obs}

    # =========================================================================
    # _get_rewards — SPARSE DISPLACEMENT ONLY
    # =========================================================================
    def _get_rewards(self):
        gravity = self._robot.data.projected_gravity_b
        height  = self._robot.data.root_pos_w[:, 2]
        pos_x   = self._robot.data.root_pos_w[:, 0]

        # ── Core: step-wise forward displacement ─────────────────────────
        delta_x   = pos_x - self._prev_pos_x
        r_forward = torch.clamp(delta_x / self.step_dt, -2.0, 2.0)
        self._prev_pos_x[:] = pos_x.detach()

        if SPARSE_MODE:
            self._episode_total_displacement += delta_x.detach()
            r_forward = torch.zeros_like(r_forward)

        # ── Safety: fall penalty ──────────────────────────────────────────
        r_fall = -10.0 * (height < 0.25).float()

        # ── Minimal stability ─────────────────────────────────────────────
        r_upright = UPRIGHT_WEIGHT * (gravity[:, 0]**2 + gravity[:, 1]**2)

        # ── Torque energy ─────────────────────────────────────────────────
        r_energy = ENERGY_WEIGHT * torch.sum(
            self._robot.data.applied_torque**2, dim=1)

        # ── Action smoothness (reduced weight vs v14) ─────────────────────
        d1 = self._actions - self._prev_actions
        d2 = self._actions - 2*self._prev_actions + self._prev_prev_actions
        r_action_rate = -0.3 * torch.sum(self._rate_weights * d1**2, dim=1)
        r_action_jerk = -0.1 * torch.sum(self._rate_weights * d2**2, dim=1)

        # ── FR_th binding proximity ───────────────────────────────────────
        fr_th_q      = self._robot.data.joint_pos[:, 5]
        fr_prox      = torch.clamp((fr_th_q - 0.800) / 0.070, 0.0, 1.0)
        r_fr_binding = FR_BINDING_WEIGHT * fr_prox ** 2

        # ── Gait analysis — logging only, NOT reward ──────────────────────
        contact_fz   = self.scene.sensors["contact_sensor"].data.net_forces_w[:, :, 2]
        feet_contact = (contact_fz > 1.0).float()
        n_contact    = feet_contact.sum(dim=1).long().clamp(0, 4)

        for nc in range(5):
            mask = (n_contact == nc)
            self._gait_log["n_contact_hist"][mask, nc] += 1.0

        sensor      = self.scene.sensors["contact_sensor"]
        last_air    = sensor.data.last_air_time[:, :4]
        first_touch = (feet_contact - self._prev_feet_contact).clamp(min=0.0)
        self._gait_log["air_time_sum"]   += last_air * first_touch
        self._gait_log["air_time_count"] += first_touch
        self._gait_log["step_count"]     += 1.0
        self._prev_feet_contact[:] = feet_contact.detach()

        # ── Episode sums ──────────────────────────────────────────────────
        self._ep_sums["forward_disp"] += r_forward
        self._ep_sums["fall"]         += r_fall
        self._ep_sums["energy"]       += r_energy
        self._ep_sums["upright"]      += r_upright
        self._ep_sums["fr_binding"]   += r_fr_binding
        self._ep_sums["action_rate"]  += r_action_rate
        self._ep_sums["action_jerk"]  += r_action_jerk

        return self.step_dt * (
            r_forward + r_upright + r_energy
            + r_action_rate + r_action_jerk + r_fr_binding
        ) + r_fall

    # =========================================================================
    # _get_dones — identical to v14
    # =========================================================================
    def _get_dones(self):
        g      = self._robot.data.projected_gravity_b
        height = self._robot.data.root_pos_w[:, 2]
        tilt   = torch.sqrt(g[:, 0]**2 + g[:, 1]**2)
        terminated = (tilt > 0.8) | (height < 0.25) | (g[:, 2] > 0.3)
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # =========================================================================
    # _reset_idx
    # =========================================================================
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        if self._global_step % 200 == 0 and self._global_step > 0:
            print(f"\n=== Sparse rewards @ step {self._global_step} ===")
            for k, v in self._ep_sums.items():
                print(f"  {k:16s}: {v[env_ids].mean().item():+.3f}")

            hist  = self._gait_log["n_contact_hist"][env_ids]
            total = hist.sum(dim=1, keepdim=True).clamp(min=1)
            frac  = (hist / total).mean(dim=0)
            air_sum   = self._gait_log["air_time_sum"][env_ids]
            air_count = self._gait_log["air_time_count"][env_ids].clamp(min=1)
            mean_air  = (air_sum / air_count).mean(dim=0)

            print("\n  ── EMERGENT GAIT ANALYSIS ──")
            labels = ["0-foot", "1-foot", "2-foot", "3-foot", "4-foot"]
            for nc, (f, lbl) in enumerate(zip(frac, labels)):
                bar    = "█" * int(f.item() * 30)
                marker = " ← TROT ✓" if nc == 2 and f.item() > 0.35 else (
                         " ← SHUFFLE ⚠" if nc == 4 and f.item() > 0.60 else "")
                print(f"    {lbl}: {bar:<30s} {f.item()*100:5.1f}%{marker}")

            print(f"  Mean air time [FL FR RL RR]: "
                  f"{mean_air[0]:.3f}  {mean_air[1]:.3f}  "
                  f"{mean_air[2]:.3f}  {mean_air[3]:.3f} s")

            avg_air = mean_air.mean().item()
            if avg_air > 0.01:
                inferred_hz = 1.0 / (2.0 * avg_air)
                delta_hz    = inferred_hz - 2.0
                trend = ("↑ faster than v14" if delta_hz > 0.3 else
                         "↓ slower than v14" if delta_hz < -0.3 else
                         "≈ same as v14 trot")
                print(f"  Inferred gait freq ≈ {inferred_hz:.2f} Hz  ({trend})")
            else:
                print(f"  Air time near zero — robot standing still")
                print(f"  ⚠ Verify checkpoint loaded: see [SPARSE DEBUG] at step 500")

            fwd = self._ep_sums["forward_disp"][env_ids].mean().item()
            print(f"\n  Health: forward_disp={fwd:+.3f}  "
                  f"{'✓ moving forward' if fwd > 0.5 else '⚠ not moving — checkpoint load issue?'}")
            print("  ─────────────────────────────")

        n        = len(env_ids)
        actuator = self._robot.actuators["legs"]

        super()._reset_idx(env_ids)
        self.command_manager.reset(env_ids)

        if SPARSE_MODE:
            self._episode_total_displacement[env_ids] = 0.0

        self._gait_log["n_contact_hist"][env_ids]  = 0.0
        self._gait_log["air_time_sum"][env_ids]    = 0.0
        self._gait_log["air_time_count"][env_ids]  = 0.0
        self._gait_log["step_count"][env_ids]      = 0.0
        self._prev_feet_contact[env_ids]           = 0.0

        _n_reset    = len(env_ids)
        _is_turning = torch.rand(_n_reset, device=self.device) >= self._WZ_STRAIGHT_PROB
        _wz_turn    = (torch.rand(_n_reset, device=self.device) * 2.0 - 1.0) * self._WZ_MAX
        self._wz_cmd[env_ids] = torch.where(
            _is_turning, _wz_turn, torch.zeros(_n_reset, device=self.device))

        self._robot.write_joint_damping_to_sim(
            torch.zeros(n, 12, device=self.device), env_ids=env_ids)

        # ── KP / KD DR ────────────────────────────────────────────────────
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

        # ── RL_th Physics DR ──────────────────────────────────────────────
        rl_fault_level = torch.rand(n, device=self.device)
        self._rl_fault_level[env_ids] = rl_fault_level
        d_healthy         = self._d_nominal[6].item()
        d_nominal_healthy = 0.048
        rl_d_values = d_nominal_healthy + rl_fault_level * (d_healthy - d_nominal_healthy)
        if hasattr(actuator, 'viscous_friction'):
            actuator.viscous_friction[env_ids] = (
                self._d_nominal.unsqueeze(0).expand(n, -1))
            actuator.viscous_friction[env_ids, 6] = rl_d_values

        tau_f_healthy_rl = 0.007
        tau_f_fault_rl   = self._tau_f_nominal[6].item()
        rl_tau_f_values  = tau_f_healthy_rl + rl_fault_level * (
            tau_f_fault_rl - tau_f_healthy_rl)
        if self._friction_env_ids_ok and self._tau_fn is not None:
            tau_f_reset = self._tau_f_nominal.unsqueeze(0).expand(n, -1).clone()
            tau_f_reset[:, 6] = rl_tau_f_values
            getattr(self._robot, self._tau_fn)(tau_f_reset, env_ids=env_ids)

        if self._bias_attr is not None:
            getattr(actuator, self._bias_attr)[env_ids] = (
                self._bias_values.unsqueeze(0).expand(n, -1))

        self._episode_f_cmd[env_ids] = 2.0  # fixed — obs[45]=0.333 always

        if self._global_step < _DELAY_PHASE1_END:
            self._env_delays[env_ids] = 0
        else:
            self._env_delays[env_ids] = torch.randint(
                0, _DELAY_MAX + 1, (n,), device=self.device)

        _mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5
        _half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        _rand = (_mid + _half * self._prev_act_init_scale
                 * (torch.rand(n, 12, device=self.device) * 2 - 1))
        self._actions[env_ids]           = _rand
        self._prev_actions[env_ids]      = _rand
        self._prev_prev_actions[env_ids] = _rand
        self._target_pos[env_ids] = self._robot.data.default_joint_pos[env_ids]

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

        # CRITICAL: reset prev_pos AFTER write_root_pose
        self._prev_pos_x[env_ids]        = self._robot.data.root_pos_w[env_ids, 0]
        self._episode_start_pos[env_ids] = self._robot.data.root_pos_w[env_ids, :2]

    # =========================================================================
    # step — with checkpoint verification diagnostic at step 500
    # =========================================================================
    def step(self, action):
        self._global_step += 1

        if self._global_step % 500 == 0:
            idx = 0
            lv  = self._robot.data.root_lin_vel_b[idx, 0].item()
            pos_x_now   = self._robot.data.root_pos_w[idx, 0].item()
            net_disp    = pos_x_now - self._episode_start_pos[idx, 0].item()

            try:
                fz  = self.scene.sensors["contact_sensor"].data.net_forces_w[idx, :, 2]
                fc  = (fz > 1.0).cpu().numpy()
                nc  = int(fc.sum())
                pat = f"FL={int(fc[0])} FR={int(fc[1])} RL={int(fc[2])} RR={int(fc[3])}"
            except Exception:
                nc, pat = 0, "N/A"

            obs45 = ((self._episode_f_cmd[idx] - self._f_cmd_lo)
                     / (self._f_cmd_hi - self._f_cmd_lo)).item()

            print(f"\n{'='*60}")
            print(f"[SPARSE DEBUG] step={self._global_step}")
            print(f"  lin_vel_x  = {lv:+.4f} m/s  "
                  f"{'→ WALKING ✓' if lv > 0.10 else '← backward' if lv < -0.05 else '◦ standing'}")
            print(f"  net_disp_x = {net_disp:+.3f} m from episode start")
            print(f"  contact    = {pat}  ({nc} feet down)")
            print(f"  obs[45]    = {obs45:.4f}  (target=0.3333 for f_cmd=2Hz)")
            print(f"  ep fwd_sum = {self._ep_sums['forward_disp'][idx].item():+.3f}  "
                  f"ar_sum = {self._ep_sums['action_rate'][idx].item():+.3f}")

            if self._global_step == 500:
                print(f"\n  ── CHECKPOINT LOAD VERDICT ──")
                if lv > 0.05:
                    print(f"  ✓ lin_vel_x={lv:.3f} > 0.05 → v14 policy walking → LOADED ✓")
                else:
                    print(f"  ⚠ lin_vel_x={lv:.3f} ≈ 0 → robot NOT walking")
                    print(f"    Check train.py output for '[WARM-START] Loading checkpoint'")
                    print(f"    Check '[WARM-START] Actor noise std' — must be <<0.10")
                    print(f"    If std=0.1000 exactly → checkpoint was NOT loaded")
            print(f"{'='*60}\n")

        result        = super().step(action)
        self.last_obs = result[0]["policy"][0].clone()

        if SPARSE_MODE:
            terminated, truncated = result[2], result[3]
            done_mask = terminated | truncated
            if done_mask.any():
                sparse_bonus = self._episode_total_displacement.clone() * done_mask.float()
                result = (result[0], result[1] + sparse_bonus,
                          result[2], result[3], result[4])

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
            pos_x = self._robot.data.root_pos_w[0, 0].item()
            self._slog["obs_raw"][s]        = obs0.cpu().numpy()
            self._slog["tanh_delta"][s]     = self._actions[0].cpu().numpy()
            self._slog["target_q"][s]       = self._target_pos[0].cpu().numpy()
            self._slog["actual_q"][s]       = self._robot.data.joint_pos[0].cpu().numpy()
            self._slog["actual_qd"][s]      = self._robot.data.joint_vel[0].cpu().numpy()
            self._slog["proj_grav"][s]      = g.cpu().numpy()
            self._slog["ang_vel"][s]        = self._robot.data.root_ang_vel_b[0].cpu().numpy()
            self._slog["lin_vel"][s]        = self._robot.data.root_lin_vel_b[0].cpu().numpy()
            self._slog["cmd"][s]            = self.command_manager.command[0,:3].cpu().numpy()
            self._slog["contact"][s]        = cnp
            self._slog["tilt_deg"][s]       = tilt*(180./3.14159265)
            self._slog["reward"][s]         = float(result[1][0].item())
            self._slog["displacement_x"][s] = pos_x - self._episode_start_pos[0, 0].item()
            self._slog_step += 1

        return result