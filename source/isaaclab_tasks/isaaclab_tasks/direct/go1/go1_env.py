# go1_env.py — Final: FR_th Position Cap + Full PACE + Confirmed Rewards
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CALIBRATION STATUS (from PACE run 26_04_02_12-07-29 + manual tests)    │
# │                                                                         │
# │ RL_th stiction: τf=4.944 Nm — detected by PACE AND Test 4 (confirmed) │
# │   Fix: Kim masking p=0.10 in training, jdelta_offset in deploy         │
# │                                                                         │
# │ FR_th mechanical binding at >0.9 rad — NOT detected by PACE/tests      │
# │   Why missed: tests were in-air, high-freq chirp — binding requires    │
# │   sustained ground contact reaction force (walking stance phase only)  │
# │   PACE Ia=0.0045 (lowest thigh) may encode binding zone compliance     │
# │   Fix: hard position cap in training sim at 0.87 rad max               │
# │   → Policy learns gait not relying on deep FR_th flexion               │
# │   → Deploy: same cap in go1_deploy.py as hardware safety               │
# │                                                                         │
# │ FR_th cap vs Kim masking — NOT the same:                                │
# │   Kim masking: KP=0 → joint free, policy learns compensation           │
# │   FR_th cap: target_q clamped → policy command limited but PD active   │
# │   Cap is correct because FR_th motor is functional up to 0.87 rad     │
# │                                                                         │
# │ REWARD BALANCE (A/B/C framework confirmed):                             │
# │   A/B = 0.55 (standing earns 55% of walking — no exploit)             │
# │   B/C = 0.84 (partial walking well rewarded)                           │
# │   Velocity-gated alive: gate=clamp(vx/0.2, 0, 1)                      │
# │   _DELAY_PHASE1_END=50000 (2083 iters Phase 1 before delay DR)        │
# └─────────────────────────────────────────────────────────────────────────┘

import torch
import numpy as np
import os
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand

_DELAY_PHASE1_END = 50_000   # steps — 2083 iters Phase 1
_DELAY_MAX        = 8
_DELAY_BUF_SIZE   = 10


class Go1Env(DirectRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "="*72)
        print("Go1Env | Final: FR_th Cap + Full PACE + Velocity-Gated Alive")
        print("  RL_th: Kim masking p=0.10  τf=4.944 Nm  d=3.459 Nm·s/rad")
        print("  FR_th: position cap max=default+0.07=0.87 rad (binding fault)")
        print("  r_alive: 0.3 × clamp(vx/0.2,0,1)  [A/B=0.55 no exploit]")
        print(f"  Delay: P1 0ms → P2 U[0,8] at step {_DELAY_PHASE1_END} (~iter 2083)")
        print("="*72 + "\n")

        _n    = self.num_envs
        _j    = 12
        _zero = torch.zeros(_n, _j, device=self.device)

        # ── PACE: Ia (armature) ───────────────────────────────────────────
        # Note: FR_th Ia=0.0045 is lowest — may encode binding zone compliance.
        # We keep PACE value as identified and fix the hardware issue via cap.
        self._robot.write_joint_armature_to_sim(_zero)
        Ia = torch.tensor([
            0.0026, 0.0037, 0.0029, 0.0031,   # hips
            0.0052, 0.0045, 0.1343, 0.0064,   # thighs (FR=0.0045, RL fault=0.1343)
            0.0141, 0.0159, 0.0130, 0.0140,   # calves
        ], device=self.device).unsqueeze(0).expand(_n, -1)
        self._robot.write_joint_armature_to_sim(Ia)

        # ── PACE: zero PhysX drive damping ────────────────────────────────
        self._robot.write_joint_damping_to_sim(_zero)

        # ── PACE: τf Coulomb friction ──────────────────────────────────────
        # FR_th τf=0.044 (healthy — no elevated friction detected by PACE)
        # RL_th τf=4.944 (stiction fault — PACE AND Test 4 confirmed)
        tau_f = torch.tensor([
            0.026, 0.028, 0.029, 0.031,   # hips
            0.052, 0.044, 4.944, 0.045,   # thighs: FR healthy, RL stiction
            0.070, 0.071, 0.078, 0.079,   # calves
        ], device=self.device).unsqueeze(0).expand(_n, -1)
        for _fn in ("write_joint_friction_coefficient_to_sim",
                    "write_joint_friction_to_sim"):
            if hasattr(self._robot, _fn):
                getattr(self._robot, _fn)(tau_f)
                break

        # ── PACE: d (viscous damping) ─────────────────────────────────────
        self._d_values = torch.tensor([
            0.039, 0.040, 0.040, 0.042,   # hips
            0.045, 0.050, 3.459, 0.050,   # thighs: RL high viscous damping
            0.124, 0.092, 0.108, 0.108,   # calves
        ], device=self.device)
        actuator = self._robot.actuators["legs"]
        if hasattr(actuator, 'viscous_friction'):
            actuator.viscous_friction[:] = (
                self._d_values.unsqueeze(0).expand(_n, -1))

        # ── PACE: q̃b (encoder bias) ───────────────────────────────────────
        self._bias_values = torch.tensor([
            -0.006, -0.002, -0.005,  0.004,   # hips
            -0.018, -0.022,  0.063, -0.020,   # thighs
            -0.001, -0.011,  0.012, -0.021,   # calves
        ], device=self.device)
        self._bias_attr = None
        for _attr in ("encoder_bias", "position_offset", "bias"):
            if hasattr(actuator, _attr):
                getattr(actuator, _attr)[:] = (
                    self._bias_values.unsqueeze(0).expand(_n, -1))
                self._bias_attr = _attr
                break

        # ── FR_th position cap ────────────────────────────────────────────
        # FR_th mechanical binding fault: joint loses back-torque capability
        # past ~0.90 rad during loaded stance. Not detectable by in-air PACE
        # or manual chirp tests (requires sustained ground contact reaction).
        # Cap at 0.07 rad above default (0.800) = 0.870 rad max target.
        # This is a POSITION LIMIT on the commanded target, not KP masking.
        # PD controller remains active — joint is functional, just range-limited.
        self._fr_th_max_delta = 0.02   # rad above default → max target = 0.870
        # FR_th is Isaac index 5
        # In _pre_physics_step: cmd_to_send[:,5] clamped to ≤ default+0.07

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

        # Hip rate weight 1.5 — suppresses 5-8Hz hip oscillation confirmed in real logs
        self._rate_weights = torch.tensor([
            1.500, 1.500, 1.500, 1.500,   # hips   BW=3.1Hz
            0.620, 0.620, 0.620, 0.620,   # thighs BW=5.0Hz
            0.564, 0.564, 0.564, 0.564,   # calves BW=5.5Hz
        ], device=self.device)

        self._delta_soft_lo = torch.tensor(
            [-0.20, -0.20, -0.25, -0.20,
             -0.35, -0.35, -0.35, -0.35,
             -0.35, -0.35, -0.35, -0.35], device=self.device)
        self._delta_soft_hi = torch.tensor(
            [ 0.20,  0.20,  0.25,  0.20,
              0.35,  0.35,  0.35,  0.35,
              0.35,  0.35,  0.35,  0.35], device=self.device)

        _rk = ["lin_vel", "ang_vel", "ang_vel_xy", "lin_vel_z", "torques",
               "action_rate", "action_jerk", "upright", "trot", "alive", "fall"]
        self._ep_sums     = {k: torch.zeros(_n, device=self.device) for k in _rk}

        # Eval mode: set _global_step to Phase 2 if env var set (play.py --phase2)
        self._global_step = (_DELAY_PHASE1_END
                             if os.environ.get("GO1_EVAL_PHASE2") == "1" else 0)

        self._obs_noise_std = torch.tensor([
            0.0, 0.0, 0.0,
            0.000132, 0.001016, 0.000132, 0.000132,   # hips (FR elevated)
            0.000061, 0.000061, 0.000061, 0.000061,   # thighs
            0.000070, 0.000070, 0.000070, 0.000070,   # calves
            0.008715, 0.008715, 0.008715, 0.008715,
            0.008554, 0.008554, 0.008554, 0.008554,
            0.005621, 0.005621, 0.005621, 0.005621,
            0.01689,  0.00606,  0.01315,
            0.03086,  0.06381,  0.04405,
            *([0.0] * 12),
        ], device=self.device)

        self._rl_th_mask_prob     = 0.10   # Kim masking: RL_th KP=0 with p=0.10
        self._prev_act_init_scale = 0.3
        self._obs_noise_enabled   = True
        self.last_obs = None

        N = 2000
        self._sim_log_maxsteps = N
        self._slog = {
            k: np.zeros((N, s) if s > 1 else N, np.float32)
            for k, s in [("obs_raw",45),("tanh_delta",12),("raw_net",12),
                         ("target_q",12),("actual_q",12),("actual_qd",12),
                         ("proj_grav",3),("ang_vel",3),("lin_vel",3),
                         ("cmd",3),("contact",4),("tilt_deg",1),("reward",1)]
        }
        self._slog_step   = 0
        self._slog_active = False

        print(f"  [PACE]  RL_th: Ia={Ia[0,6].item():.4f}  "
              f"τf={tau_f[0,6].item():.3f}  d={self._d_values[6].item():.3f}")
        print(f"  [PACE]  FR_th: Ia={Ia[0,5].item():.4f}  "
              f"τf={tau_f[0,5].item():.3f}  (healthy — fault is mechanical)")
        print(f"  [FR_th CAP] max delta={self._fr_th_max_delta:.3f} "
              f"→ max target={0.800+self._fr_th_max_delta:.3f} rad")
        if self._bias_attr:
            print(f"  [q̃b]    via actuator.{self._bias_attr}  "
                  f"RL_th={self._bias_values[6].item():+.3f}")

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        print("─"*60 + "\nACTUATOR VERIFICATION\n" + "─"*60)
        for name, act in (getattr(self._robot, "_actuators", None) or {}).items():
            kp = getattr(act, "stiffness",       None)
            kd = getattr(act, "damping",          None)
            vf = getattr(act, "viscous_friction", None)
            print(f"  [{name}]  KP={kp[0,0].item() if kp is not None else 'N/A':5.1f}"
                  f"  KD={kd[0,0].item() if kd is not None else 'N/A':4.1f}"
                  f"  d[6]={vf[0,6].item() if vf is not None else 'N/A':6.3f}")
        print("─"*60 + "\n")

    def _pre_physics_step(self, actions: torch.Tensor):
        a     = actions.clone()
        _mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5
        _half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        a = _mid + _half * torch.tanh(a)
        self._prev_prev_actions[:] = self._prev_actions
        self._prev_actions[:]      = self._actions
        self._actions[:]           = a
        self._target_pos = self._actions + self._robot.data.default_joint_pos

        # ── FR_th position cap ────────────────────────────────────────────
        # Hard limit: FR_th (Isaac index 5) cannot be commanded above 0.870 rad.
        # This prevents the policy from ever pushing FR_th into the mechanical
        # binding zone (>0.90 rad) which was found to cause falls on the real robot.
        # The policy is trained to walk within this constraint → learns a gait
        # that doesn't require deep FR_th flexion.
        fr_th_max = (self._robot.data.default_joint_pos[:, 5]
                     + self._fr_th_max_delta)   # [N] = 0.870 rad for all envs
        self._target_pos[:, 5] = torch.minimum(self._target_pos[:, 5], fr_th_max)

        # Delay DR ring buffer (vectorised gather — no Python loop)
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

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self.command_manager.command[:, :3],
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            torch.clamp(self._robot.data.joint_vel,      -5.0, 5.0),
            torch.clamp(self._robot.data.root_ang_vel_b, -5.0, 5.0),
            self._robot.data.projected_gravity_b,
            self._prev_actions,
        ], dim=-1)
        if self._obs_noise_enabled:
            obs = obs + torch.randn_like(obs) * self._obs_noise_std
        return {"policy": obs}

    def _get_rewards(self):
        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        gravity = self._robot.data.projected_gravity_b
        height  = self._robot.data.root_pos_w[:, 2]
        cmd     = self.command_manager.command
        tilt    = torch.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)

        # ── Velocity tracking (σ²=0.25 — gradual gradient) ───────────────
        r_lin_vel = 1.5 * torch.exp(-(lin_vel[:, 0] - cmd[:, 0])**2 / 0.25)
        r_ang_vel = 0.5 * torch.exp(-(ang_vel[:, 2] - cmd[:, 2])**2 / 0.25)

        # ── Velocity-gated alive (A/B=0.55, no standing exploit) ─────────
        # Standing: gate=0 → r_alive=0. Walking@0.2m/s: gate=1 → r_alive=0.3
        vel_gate = torch.clamp(lin_vel[:, 0] / 0.2, 0.0, 1.0)
        r_alive  = 0.3 * vel_gate * ((height > 0.28) & (tilt < 0.3)).float()

        # ── Penalties ─────────────────────────────────────────────────────
        r_ang_vel_xy = -0.05 * (ang_vel[:, 0]**2 + ang_vel[:, 1]**2)
        r_lin_vel_z  = -4.0  * lin_vel[:, 2]**2   # bounce suppression (-2→-4)
        r_torques    = -1e-5 * torch.sum(
            self._robot.data.applied_torque**2, dim=1)
        r_upright    = -2.0  * (gravity[:, 0]**2 + gravity[:, 1]**2)
        r_fall       = -10.0 * (height < 0.25).float()

        d1 = self._actions - self._prev_actions
        d2 = self._actions - 2*self._prev_actions + self._prev_prev_actions
        r_action_rate = -1.0 * torch.sum(self._rate_weights * d1**2, dim=1)
        r_action_jerk = -0.5 * torch.sum(self._rate_weights * d2**2, dim=1)
        r_trot        = torch.zeros(self.num_envs, device=self.device)

        for k, v in zip(
            ["lin_vel","ang_vel","ang_vel_xy","lin_vel_z","torques",
             "action_rate","action_jerk","upright","trot","alive","fall"],
            [r_lin_vel,r_ang_vel,r_ang_vel_xy,r_lin_vel_z,r_torques,
             r_action_rate,r_action_jerk,r_upright,r_trot,r_alive,r_fall]
        ):
            self._ep_sums[k] += v

        return self.step_dt * (
            r_lin_vel + r_ang_vel + r_ang_vel_xy + r_lin_vel_z
            + r_torques + r_action_rate + r_action_jerk
            + r_upright + r_trot + r_alive
        ) + r_fall

    def _get_dones(self):
        g      = self._robot.data.projected_gravity_b
        height = self._robot.data.root_pos_w[:, 2]
        tilt   = torch.sqrt(g[:, 0]**2 + g[:, 1]**2)
        terminated = (tilt > 0.8) | (height < 0.25) | (g[:, 2] > 0.3)
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        if self._global_step % 200 == 0 and self._global_step > 0:
            print(f"\n=== Rewards @ step {self._global_step} ===")
            for k, v in self._ep_sums.items():
                print(f"  {k:14s}: {v[env_ids].mean().item():+.3f}")
            d = self._env_delays.float()
            print(f"  [Delay DR] mean={d.mean():.2f}  "
                  f"max={d.max():.0f}  zero={(d==0).sum().item()}")
            print("=" * 40)

        n        = len(env_ids)
        actuator = self._robot.actuators["legs"]

        super()._reset_idx(env_ids)
        self.command_manager.reset(env_ids)

        # Re-zero PhysX drive damping (super restores cfg baseline)
        self._robot.write_joint_damping_to_sim(
            torch.zeros(n, 12, device=self.device), env_ids=env_ids)

        # KP DR ±25% hips/thighs, ±20% knees
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

        # ── Kim masking: RL_th p=0.10 AFTER KP DR ────────────────────────
        # Models the RL_th stiction fault (τf=4.944 Nm) behaviourally.
        # p=0.10: 10% of training envs have RL_th completely free (KP=0).
        # Applied AFTER KP DR so the masking overrides the DR value.
        mask = torch.rand(n, device=self.device) < self._rl_th_mask_prob
        if mask.any():
            masked = env_ids[mask]
            actuator.stiffness[masked, 6] = 0.0   # RL_th = Isaac index 6
            self._kp_live[masked, 6]      = 0.0

        # Re-apply RL_th viscous damping (super restores cfg baseline)
        if hasattr(actuator, 'viscous_friction'):
            actuator.viscous_friction[env_ids, 6] = 3.459

        # Re-apply encoder bias (super restores cfg 0.0)
        if self._bias_attr is not None:
            getattr(actuator, self._bias_attr)[env_ids] = (
                self._bias_values.unsqueeze(0).expand(n, -1))

        # Per-episode delay DR
        if self._global_step < _DELAY_PHASE1_END:
            self._env_delays[env_ids] = 0
        else:
            if self._global_step == _DELAY_PHASE1_END and env_ids[0] == 0:
                print(f"\n  [Delay DR] Phase 2 at step {self._global_step} "
                      f"(iter ~{self._global_step//24})")
            self._env_delays[env_ids] = torch.randint(
                0, _DELAY_MAX + 1, (n,), device=self.device)

        # Debug print
        if self._global_step % 200 == 0 and self._global_step > 0:
            diff = (abs(actuator.stiffness[0]-actuator.stiffness[1]).max().item()
                    if self.num_envs > 1 else 0.0)
            print(f"  [KP DR]   diff={diff:.1f} {'✓' if diff>2 else '⚠'}")
            if mask.any():
                print(f"  [Kim RL_th] {mask.sum().item()} masked of {n}")
            if hasattr(actuator, 'viscous_friction'):
                vf6 = actuator.viscous_friction[env_ids[0], 6].item()
                print(f"  [d RL_th] {vf6:.3f} ({'✓' if abs(vf6-3.459)<0.01 else '⚠'})")
            # FR_th cap check
            fr_th_max_pos = self._robot.data.default_joint_pos[env_ids[0], 5].item() + self._fr_th_max_delta
            print(f"  [FR_th cap] max_target={fr_th_max_pos:.3f} rad ✓")

        # Reset state
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

    def step(self, action):
        self._global_step += 1

        if self._global_step % 500 == 0:
            idx  = 0
            act  = self._robot.actuators["legs"]
            ph   = ("Phase 2 DR" if self._global_step >= _DELAY_PHASE1_END
                    else f"Phase 1 (DR at step {_DELAY_PHASE1_END})")
            print(f"\n{'='*80}")
            print(f"[DEBUG] step {self._global_step} | {ph} | "
                  f"cmd_vx={self.command_manager.command[idx,0]:.2f}")

            kp0  = act.stiffness[0].cpu().numpy().round(1)
            kp1  = act.stiffness[1].cpu().numpy().round(1) if self.num_envs > 1 else kp0
            diff = abs(act.stiffness[0]-act.stiffness[1]).max().item() if self.num_envs > 1 else 0.
            print(f"  KP env0: {kp0}  {'✓ DR' if diff>2 else '⚠'}")
            print(f"  KP env1: {kp1}  max_diff={diff:.1f}")

            d = self._env_delays.float()
            if self._global_step >= _DELAY_PHASE1_END:
                print(f"  Delay:   mean={d.mean():.1f} ({d.mean().item()*2:.0f}ms)  "
                      f"env0={self._env_delays[0].item()} steps")
            else:
                print(f"  Delay:   0ms  [Phase 1]")

            if hasattr(act, 'viscous_friction'):
                vf = act.viscous_friction[0].cpu().numpy()
                print(f"  d env0:  RL_th={vf[6]:.3f}  hip={vf[:4].mean():.3f}  "
                      f"calf={vf[8:].mean():.3f}")
            if self._bias_attr and hasattr(act, self._bias_attr):
                b = getattr(act, self._bias_attr)[0].cpu().numpy()
                print(f"  q̃b env0: RL_th={b[6]:+.3f}  FR_th={b[5]:+.3f}  "
                      f"max_abs={np.abs(b).max():.3f}")

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
            print(f"  track_err: mean={te.mean():.4f} max={te.max():.4f} ({JN[te.argmax()]})")
            da = (self._actions[idx]-self._prev_actions[idx]).abs().mean().item()
            print(f"  |Δdelta|:  {da:.4f}  ({'✓' if da<0.030 else '*** jerky'})")
            print(f"  root_z:    {self._robot.data.root_pos_w[idx,2]:.3f}")
            lv = self._robot.data.root_lin_vel_b[idx,0].item()
            vg = min(max(lv/0.2, 0.0), 1.0)
            print(f"  lin_vel:   {lv:.3f} m/s  vel_gate={vg:.2f}  "
                  f"r_alive≈{0.3*vg:.3f}/step")
            masked = (self._kp_live[:,6]==0).sum().item()
            print(f"  RL_th_masked: {masked} envs  "
                  f"({'✓' if masked>self.num_envs*0.05 else '⚠ low (normal <20 envs)'})")
            # FR_th cap verification
            fr_th_now = self._target_pos[idx, 5].item()
            fr_th_max = self._robot.data.default_joint_pos[idx, 5].item() + self._fr_th_max_delta
            print(f"  FR_th target: {fr_th_now:.3f}  cap={fr_th_max:.3f}  "
                  f"{'✓ within cap' if fr_th_now <= fr_th_max+0.001 else '⚠ ABOVE CAP'}")

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