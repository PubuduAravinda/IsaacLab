# go2_env.py — Go2 EDU Flat Terrain RL Environment
# Based on working v3 from previous session
# ADDED: full --log --log_steps compatibility with play.py
#
# KEY SPECS (from working training run):
#   OBS:    45D  (no f_cmd — Go2 uses fixed gait frequency)
#   KP=60   KD=5  (uniform all joints — average Go2 values)
#   Hip:    ±0.30 rad
#   Thigh:  ±0.35 rad
#   Knee:   ±0.35 rad
#   Delay:  U[0,5] steps at 50Hz (~0-10ms, Go2 SDK 500Hz)
#   No PACE, no fault masking — nominal clean Go2 platform
#
# PLAY.PY COMPATIBILITY (--log --log_steps N):
#   _slog dict with all 13 keys matching play.py lines 294-313
#   _slog_active, _slog_step, _sim_log_maxsteps
#   _obs_noise_enabled, _global_step, _env_delays, last_obs
#   _delta_soft_lo, _delta_soft_hi (action range print)

import os
import torch
import numpy as np
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand

_DELAY_PHASE1_END = 50_000
_DELAY_MAX        = 5
_DELAY_BUF_SIZE   = 8


class Go2Env(DirectRLEnv):
    """
    Go2 flat terrain locomotion — nominal platform, no PACE calibration.
    45D obs, KP=60 KD=5 uniform, hip ±0.30, delay U[0,5].
    Compatible with play.py --log flag.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "="*72)
        print("Go2Env | Flat | No PACE | KP=60 KD=5 | 45D obs")
        print("  Hip ±0.30 rad | Delay U[0,5] steps")
        print("  play.py --log compatible ✓")
        print("="*72 + "\n")

        _n = self.num_envs
        _j = 12

        # ── Zero PhysX drive damping (prevent double KD) ──────────────────
        _zero = torch.zeros(_n, _j, device=self.device)
        self._robot.write_joint_armature_to_sim(_zero)
        self._robot.write_joint_damping_to_sim(_zero)

        # ── Command manager ────────────────────────────────────────────────
        self.command_manager = UniformVelocityCommand(
            cfg=self.cfg.commands, env=self)

        # ── Action buffers ─────────────────────────────────────────────────
        self._actions           = torch.zeros(_n, _j, device=self.device)
        self._prev_actions      = torch.zeros(_n, _j, device=self.device)
        self._prev_prev_actions = torch.zeros(_n, _j, device=self.device)
        self._target_pos        = torch.zeros(_n, _j, device=self.device)

        # ── Action limits — Go2 standard ──────────────────────────────────
        self._delta_soft_lo = torch.tensor(
            [-0.30, -0.30, -0.30, -0.30,
             -0.35, -0.35, -0.35, -0.35,
             -0.35, -0.35, -0.35, -0.35],
            device=self.device)
        self._delta_soft_hi = torch.tensor(
            [ 0.30,  0.30,  0.30,  0.30,
              0.35,  0.35,  0.35,  0.35,
              0.35,  0.35,  0.35,  0.35],
            device=self.device)

        # ── KP/KD — average Go2 values, uniform all joints ────────────────
        # KP=60 KD=5 chosen from working training run
        # Matching what Go2 SDK deployment script uses
        self._kp_nominal = torch.full((_j,), 60.0, device=self.device)
        self._kd_nominal = torch.full((_j,),  5.0, device=self.device)
        self._kp_live    = self._kp_nominal.unsqueeze(0).expand(_n,-1).clone()

        # KP/KD DR bounds — same ±25%/±20% philosophy as Go1
        self._kp_dr_hi_th = 1.25;  self._kp_dr_lo_th = 0.75
        self._kp_dr_hi_kn = 1.20;  self._kp_dr_lo_kn = 0.80
        self._kd_dr_hi_th = 1.25;  self._kd_dr_lo_th = 0.75
        self._kd_dr_hi_kn = 1.20;  self._kd_dr_lo_kn = 0.80

        # ── Delay FIFO ─────────────────────────────────────────────────────
        self._delay_buf  = torch.zeros(
            _DELAY_BUF_SIZE, _n, _j, device=self.device)
        self._delay_ptr  = 0
        self._env_delays = torch.zeros(
            _n, dtype=torch.long, device=self.device)

        # ── Rate weights — uniform (healthy Go2, no fault) ────────────────
        self._rate_weights = torch.ones(_j, device=self.device)

        # ── WZ command mix ─────────────────────────────────────────────────
        self._episode_wz       = torch.zeros(_n, device=self.device)
        self._WZ_STRAIGHT_PROB = 0.50
        self._WZ_MAX           = 0.30

        # ── Gait frequency (fixed 2Hz for Go2 — no f_cmd in obs) ──────────
        self._episode_f_cmd = torch.full((_n,), 2.0, device=self.device)

        # ── Terrain height ─────────────────────────────────────────────────
        self._terrain_z = self.scene.env_origins[:, 2].clone()

        # ── Contact tracking ───────────────────────────────────────────────
        self._prev_feet_contact = torch.zeros(_n, 4, device=self.device)

        # ── Foot body IDs ──────────────────────────────────────────────────
        try:
            _fl, _ = self._robot.find_bodies("FL_foot")
            _fr, _ = self._robot.find_bodies("FR_foot")
            _rl, _ = self._robot.find_bodies("RL_foot")
            _rr, _ = self._robot.find_bodies("RR_foot")
            self._foot_body_ids = [_fl[0], _fr[0], _rl[0], _rr[0]]
            print(f"  [FOOT IDs] FL={_fl[0]} FR={_fr[0]} "
                  f"RL={_rl[0]} RR={_rr[0]} ✓")
        except Exception as e:
            self._foot_body_ids = None
            print(f"  [FOOT IDs] WARNING: {e}")

        # ── Global step ────────────────────────────────────────────────────
        self._global_step = (
            _DELAY_PHASE1_END
            if os.environ.get("GO2_EVAL_PHASE2") == "1" else 0)

        # ── Observation noise ──────────────────────────────────────────────
        # 45D: cmd(3)+jpos(12)+jvel(12)+angvel(3)+grav(3)+prevact(12)
        self._obs_noise_std = torch.tensor([
            0.0,  0.0,  0.0,           # [0:3]   cmd
            *([0.0001]*12),             # [3:15]  jpos — small
            *([0.008 ]*12),             # [15:27] jvel
            0.01689, 0.00606, 0.01315,  # [27:30] gyro (Go1 Phase1 proxy)
            0.031,   0.064,   0.044,    # [30:33] proj_grav
            *([0.0]  *12),              # [33:45] prev_actions
        ], device=self.device)
        assert self._obs_noise_std.shape[0] == 45, \
            f"noise std dim {self._obs_noise_std.shape[0]} != 45"
        self._obs_noise_enabled = True

        # ── Episode reward sums ────────────────────────────────────────────
        _rk = ["lin_vel","ang_vel","ang_vel_xy","lin_vel_z",
               "alive","upright","lat_vel","air_time",
               "action_rate","action_jerk","hip_reg","torque","fall"]
        self._ep_sums = {
            k: torch.zeros(_n, device=self.device) for k in _rk}

        # ── last_obs ───────────────────────────────────────────────────────
        self.last_obs = None

        # ── Sim log — play.py --log compatible ────────────────────────────
        # 45D obs_raw (no f_cmd) matching training obs dim
        # All 13 keys match play.py lines 294-313 exactly
        _N = 2000
        self._sim_log_maxsteps = _N
        self._slog_step        = 0
        self._slog_active      = False
        self._slog = {
            k: np.zeros((_N, s) if s > 1 else _N, np.float32)
            for k, s in [
                ("obs_raw",    45),  # 45D — no f_cmd
                ("raw_net",    12),  # filled by play.py hook
                ("tanh_delta", 12),
                ("target_q",   12),
                ("actual_q",   12),
                ("actual_qd",  12),
                ("proj_grav",   3),
                ("ang_vel",     3),
                ("lin_vel",     3),
                ("cmd",         3),
                ("contact",     4),
                ("tilt_deg",    1),
                ("reward",      1),
            ]
        }
        print(f"  [SLOG] 13 keys pre-allocated ({_N} steps) "
              f"— play.py --log ready ✓")
        print(f"  [OBS]  45D | KP={self._kp_nominal[0]:.0f} "
              f"KD={self._kd_nominal[0]:.0f}")
        print(f"  [DELAY] Phase1: 0ms | "
              f"Phase2 (>{_DELAY_PHASE1_END}): "
              f"U[0,{_DELAY_MAX}] steps")
        print("="*72 + "\n")

    # =========================================================================
    def _setup_scene(self):
        self._robot = self.scene["robot"]
        self.scene.filter_collisions(global_prim_paths=[])

    # =========================================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        _mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5
        _half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        a = _mid + _half * torch.tanh(actions)

        self._prev_prev_actions[:] = self._prev_actions
        self._prev_actions[:]      = self._actions
        self._actions[:]           = a

        # Delay FIFO
        if self._global_step < _DELAY_PHASE1_END:
            delayed = a
        else:
            self._delay_buf[self._delay_ptr] = a
            self._delay_ptr = (self._delay_ptr + 1) % _DELAY_BUF_SIZE
            idx = (self._delay_ptr - self._env_delays - 1) % _DELAY_BUF_SIZE
            delayed = self._delay_buf[
                idx,
                torch.arange(self.num_envs, device=self.device)]

        self._target_pos = delayed + self._robot.data.default_joint_pos
        self._robot.set_joint_position_target(self._target_pos)

    def _apply_action(self):
        pass

    # =========================================================================
    def _get_observations(self) -> dict:
        cmd_obs       = self.command_manager.command[:, :3].clone()
        cmd_obs[:, 2] = self._episode_wz

        # 45D — no f_cmd (matches training obs dim)
        obs = torch.cat([
            cmd_obs,                                           # [0:3]
            self._robot.data.joint_pos
                - self._robot.data.default_joint_pos,         # [3:15]
            torch.clamp(self._robot.data.joint_vel, -5., 5.), # [15:27]
            torch.clamp(
                self._robot.data.root_ang_vel_b, -5., 5.),    # [27:30]
            self._robot.data.projected_gravity_b,              # [30:33]
            self._prev_actions,                                # [33:45]
        ], dim=-1)  # total = 3+12+12+3+3+12 = 45

        if self._obs_noise_enabled:
            obs = obs + torch.randn_like(obs) * self._obs_noise_std

        self.last_obs = obs[0].clone()
        return {"policy": obs}

    # =========================================================================
    def _get_rewards(self):
        lin_vel  = self._robot.data.root_lin_vel_b
        ang_vel  = self._robot.data.root_ang_vel_b
        gravity  = self._robot.data.projected_gravity_b
        height   = self._robot.data.root_pos_w[:, 2]
        cmd      = self.command_manager.command
        tilt     = torch.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)
        ht_rel   = height - self._terrain_z

        sensor       = self.scene.sensors["contact_sensor"]
        contact_fz   = sensor.data.net_forces_w[:, :, 2]
        feet_contact = (contact_fz > 1.0).float()
        n_contact    = feet_contact.sum(dim=1)
        gait_q       = torch.clamp(
            1.0 - torch.abs(n_contact - 2.0) / 2.0, 0.0, 1.0)
        gait_gate    = 0.30 + 0.70 * gait_q
        vel_gate     = torch.clamp(lin_vel[:, 0] / 0.2, 0.0, 1.0)

        r_lin_vel    = 1.5 * torch.exp(
            -(lin_vel[:, 0] - cmd[:, 0])**2 / 0.25) * gait_gate
        r_ang_vel    = 1.0 * torch.exp(
            -(ang_vel[:, 2] - self._episode_wz)**2 / 0.25)
        r_alive      = 0.3 * vel_gate * (
            (ht_rel > 0.28) & (tilt < 0.3)).float()
        r_ang_vel_xy = -0.08 * (ang_vel[:,0]**2 + ang_vel[:,1]**2)
        r_lin_vel_z  = -4.0  * lin_vel[:,2]**2
        r_upright    = -3.5  * (gravity[:,0]**2 + gravity[:,1]**2)
        r_lat_vel    = -3.5  * lin_vel[:,1]**2
        r_torques    = -1e-5 * torch.sum(
            self._robot.data.applied_torque**2, dim=1)
        r_fall       = -10.0 * (ht_rel < 0.18).float()
        r_hip_reg    = -1.5  * torch.sum(
            self._actions[:, :4]**2, dim=1)

        d1 = self._actions - self._prev_actions
        d2 = self._actions - 2*self._prev_actions + self._prev_prev_actions
        r_action_rate = -1.0 * torch.sum(self._rate_weights * d1**2, dim=1)
        r_action_jerk = -0.5 * torch.sum(self._rate_weights * d2**2, dim=1)

        last_air    = sensor.data.last_air_time[:, :4]
        vel_gate2   = torch.clamp(
            torch.norm(lin_vel[:,:2], dim=1) / 0.2, 0.0, 1.0)
        first_touch = (feet_contact - self._prev_feet_contact).clamp(min=0.)
        t_swing     = torch.full(
            (self.num_envs, 1), 0.25, device=self.device)  # 2Hz fixed
        r_air_time  = 20.0 * torch.sum(
            torch.relu(last_air - t_swing) * first_touch,
            dim=1) * vel_gate2

        self._prev_feet_contact[:] = feet_contact.detach()

        total = (r_lin_vel + r_ang_vel + r_alive
                 + r_ang_vel_xy + r_lin_vel_z + r_upright
                 + r_lat_vel + r_torques + r_fall
                 + r_hip_reg + r_action_rate + r_action_jerk
                 + r_air_time)

        for k, v in [
            ("lin_vel",    r_lin_vel),
            ("ang_vel",    r_ang_vel),
            ("ang_vel_xy", r_ang_vel_xy),
            ("lin_vel_z",  r_lin_vel_z),
            ("alive",      r_alive),
            ("upright",    r_upright),
            ("lat_vel",    r_lat_vel),
            ("air_time",   r_air_time),
            ("action_rate",r_action_rate),
            ("action_jerk",r_action_jerk),
            ("hip_reg",    r_hip_reg),
            ("torque",     r_torques),
            ("fall",       r_fall),
        ]:
            self._ep_sums[k] += v

        return total * self.step_dt

    # =========================================================================
    def _get_dones(self):
        ht_rel   = self._robot.data.root_pos_w[:, 2] - self._terrain_z
        gravity  = self._robot.data.projected_gravity_b
        tilt     = torch.sqrt(gravity[:,0]**2 + gravity[:,1]**2)
        terminated = (ht_rel < 0.18) | (tilt > 0.8)
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # =========================================================================
    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)
        _n_r = len(env_ids)

        self.command_manager.reset(env_ids)

        # WZ mix
        straight = torch.rand(_n_r, device=self.device) < self._WZ_STRAIGHT_PROB
        wz_turn  = (torch.rand(_n_r, device=self.device)*2-1) * self._WZ_MAX
        self._episode_wz[env_ids] = torch.where(
            straight, torch.zeros_like(wz_turn), wz_turn)

        # KP/KD DR
        kp_s = torch.ones(_n_r, 12, device=self.device)
        kp_s[:, :8] = (torch.rand(_n_r, 8, device=self.device)
                       * (self._kp_dr_hi_th - self._kp_dr_lo_th)
                       + self._kp_dr_lo_th)
        kp_s[:, 8:] = (torch.rand(_n_r, 4, device=self.device)
                       * (self._kp_dr_hi_kn - self._kp_dr_lo_kn)
                       + self._kp_dr_lo_kn)
        kp_new = self._kp_nominal.unsqueeze(0) * kp_s
        self._kp_live[env_ids] = kp_new
        act = self._robot.actuators["legs"]
        if hasattr(act, 'stiffness'):
            act.stiffness[env_ids] = kp_new

        # Delay DR — Phase 2 only
        if self._global_step >= _DELAY_PHASE1_END:
            self._env_delays[env_ids] = torch.randint(
                0, _DELAY_MAX+1, (_n_r,),
                device=self.device, dtype=torch.long)

        self._prev_feet_contact[env_ids] = 0.0

        joint_pos  = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += (torch.rand_like(joint_pos) - 0.5) * 0.05
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3]  = self.scene.env_origins[env_ids]
        root_state[:,  2] += 0.35
        root_state[:, 3:7] = torch.tensor(
            [1., 0., 0., 0.], device=self.device)
        root_state[:, 7:]  = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), None, env_ids)

    # =========================================================================
    def step(self, action):
        self._global_step += 1

        # ── Debug every 500 steps ──────────────────────────────────────────
        if self._global_step % 500 == 0:
            idx   = 0
            lv    = self._robot.data.root_lin_vel_b[idx, 0].item()
            cv    = self.command_manager.command[idx, 0].item()
            grav  = self._robot.data.projected_gravity_b[idx]
            tilt  = float(torch.sqrt(grav[0]**2+grav[1]**2).item())*57.3
            dly   = self._env_delays[idx].item()
            phase = (f"Phase2 U[0,{_DELAY_MAX}]"
                     if self._global_step >= _DELAY_PHASE1_END
                     else f"Phase1 (>={_DELAY_PHASE1_END})")

            print(f"\n{'='*70}")
            print(f"[Go2 DEBUG] step {self._global_step} | "
                  f"cmd={cv:.2f} wz={self._episode_wz[idx]:.3f} "
                  f"delay={phase} env0={dly}steps")

            te = (self._robot.data.joint_pos[idx]
                  - self._target_pos[idx]).abs().cpu().numpy()
            JN = ['FLh','FRh','RLh','RRh',
                  'FLt','FRt','RLt','RRt',
                  'FLk','FRk','RLk','RRk']
            print(f"  tilt={tilt:.1f}°  lin_vel={lv:.3f} m/s")
            print(f"  track_err: mean={te.mean():.4f} "
                  f"max={te.max():.4f} ({JN[te.argmax()]})")

            print(f"=== Go2 Rewards @ step {self._global_step} ===")
            for k, v in self._ep_sums.items():
                print(f"  {k:<14}: {v[idx].item():+.3f}")
            print(f"  [Delay] mean={self._env_delays.float().mean():.2f} "
                  f"max={self._env_delays.max().item()}  "
                  f"[Alpha] mean="
                  f"{(0.02/(0.02+self._env_delays.float()*0.002)).mean():.3f}")
            print(f"{'='*70}")

        # ── Parent step ────────────────────────────────────────────────────
        result        = super().step(action)
        self.last_obs = result[0]["policy"][0].clone()

        # ── Slog — play.py --log compatible ───────────────────────────────
        if self._slog_active and self._slog_step < self._sim_log_maxsteps:
            s    = self._slog_step
            obs0 = result[0]["policy"][0]
            g    = self._robot.data.projected_gravity_b[0]
            tilt = float(torch.sqrt(g[0]**2 + g[1]**2).item())

            try:
                fz  = self.scene.sensors["contact_sensor"]\
                          .data.net_forces_w[0, :, 2]
                cnp = fz.cpu().numpy()
            except Exception:
                cnp = np.zeros(4, np.float32)

            # raw_net filled by play.py forward hook
            self._slog["obs_raw"][s]    = obs0.cpu().numpy()
            self._slog["tanh_delta"][s] = self._actions[0].cpu().numpy()
            self._slog["target_q"][s]   = self._target_pos[0].cpu().numpy()
            self._slog["actual_q"][s]   = (
                self._robot.data.joint_pos[0].cpu().numpy())
            self._slog["actual_qd"][s]  = (
                self._robot.data.joint_vel[0].cpu().numpy())
            self._slog["proj_grav"][s]  = g.cpu().numpy()
            self._slog["ang_vel"][s]    = (
                self._robot.data.root_ang_vel_b[0].cpu().numpy())
            self._slog["lin_vel"][s]    = (
                self._robot.data.root_lin_vel_b[0].cpu().numpy())
            self._slog["cmd"][s]        = (
                self.command_manager.command[0, :3].cpu().numpy())
            self._slog["contact"][s]    = cnp
            self._slog["tilt_deg"][s]   = tilt * (180.0 / 3.14159265)
            self._slog["reward"][s]     = float(result[1][0].item())
            self._slog_step += 1

        return result