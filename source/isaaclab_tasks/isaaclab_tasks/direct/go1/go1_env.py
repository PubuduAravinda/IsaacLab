# go1_env.py  — v4 (robust state-based rewards)
#
# Core insight: reward the physical state you want, not action proxies.
#
# Reward philosophy change from v3:
#   REMOVED r_hip_reg = -K * sum(hip_actions^2)
#     → Penalised the tool (hip motion) not the problem (lean)
#     → Sim leans right, real leans left — proxy creates directional bias
#     → Policy fights itself: gait needs hip swing, penalty suppresses it
#
#   ADDED r_upright = -5.0 * (grav_x^2 + grav_y^2)
#     → Penalises ACTUAL lean, any direction equally (symmetric)
#     → Quadratic: gentle near-level, strong when tilting — natural curriculum
#     → At 6°: -4% forward. At 15°: -22%. At 20°: -49%.
#
#   ADDED r_trot = diagonal contact pair reward
#     → Directly rewards FL+RR / FR+RL alternating foot contact
#     → Based on actual foot state — if it trots, it gets rewarded
#
#   ADDED r_alive = +0.5 per step while standing (height>0.28, tilt<0.3)
#     → Dense survival signal — prevents training collapse
#
#   KP knee floor raised: 0.60 → 0.80
#     → Real knee tracking err = 0.34x sim at 0.60 — sim over-lags
#     → At 0.80: min knee KP = 64 Nm, closer to real hardware

import torch
import numpy as np
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand


class Go1Env(DirectRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "=" * 70)
        print("Go1Env v4 | state-based rewards | r_upright + r_trot + r_alive")
        print("=" * 70 + "\n")

        self.command_manager = UniformVelocityCommand(cfg=self.cfg.commands, env=self)

        self._actions           = torch.zeros(self.num_envs, 12, device=self.device)
        self._prev_actions      = torch.zeros_like(self._actions)
        self._prev_prev_actions = torch.zeros_like(self._actions)
        self._target_pos        = torch.zeros_like(self._actions)
        self._action_scale      = torch.ones(12, device=self.device)

        self._kp_nominal = torch.tensor(
            [35,35,35,35, 65,65,65,65, 80,80,80,80],
            dtype=torch.float32, device=self.device)
        self._kd_nominal = torch.tensor(
            [4.0,4.0,4.0,4.0, 4.5,4.5,4.5,4.5, 5.0,5.0,5.0,5.0],
            dtype=torch.float32, device=self.device)
        self._kp_live = self._kp_nominal.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._kd_live = self._kd_nominal.unsqueeze(0).expand(self.num_envs, -1).clone()

        self._delta_soft_lo = torch.tensor(
            [-0.20,-0.20,-0.25,-0.20, -0.35,-0.35,-0.35,-0.35, -0.35,-0.35,-0.35,-0.35],
            device=self.device)
        self._delta_soft_hi = torch.tensor(
            [ 0.20, 0.20, 0.25, 0.20,  0.35, 0.35, 0.35, 0.35,  0.35, 0.35, 0.35, 0.35],
            device=self.device)

        self._ep_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in [
            "lin_vel","ang_vel","ang_vel_xy","lin_vel_z",
            "torques","action_rate","action_jerk",
            "upright","trot","alive","fall",
        ]}

        self._global_step = 0

        # FIX 1: Actuator lag
        self._lag_alpha = 0.6
        self._lag_pos   = None

        # FIX 2: Obs noise
        self._obs_noise_std = torch.tensor([
            0.0, 0.0, 0.0,
            *([0.005]*12), *([0.050]*12),
            0.02, 0.02, 0.02,
            0.01, 0.01, 0.01,
            *([0.0]*12),
        ], device=self.device)

        # FIX 3: Per-joint KP DR
        # Hips/thighs: [0.40,1.20] — covers noisy FR motor
        # Knees: [0.80,1.20] — was 0.60, sim over-lagged vs real by 3x
        self._kp_hip_thigh_lo = 0.40;  self._kp_hip_thigh_hi = 1.20
        self._kp_knee_lo      = 0.80;  self._kp_knee_hi      = 1.20
        self._kd_hip_thigh_lo = 0.50;  self._kd_hip_thigh_hi = 1.30
        self._kd_knee_lo      = 0.70;  self._kd_knee_hi      = 1.30

        # FIX 4: prev_actions init
        self._prev_act_init_scale = 0.3

        self._obs_noise_enabled = True
        self.last_obs = None

        N = 2000; self._sim_log_maxsteps = N
        self._slog = {
            k: np.zeros((N,s) if s>1 else N, np.float32)
            for k,s in [("obs_raw",45),("tanh_delta",12),("raw_net",12),
                        ("target_q",12),("actual_q",12),("actual_qd",12),
                        ("proj_grav",3),("ang_vel",3),("lin_vel",3),
                        ("cmd",3),("contact",4),("tilt_deg",1),("reward",1)]
        }
        self._slog_step = 0; self._slog_active = False

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        print("\n" + "─"*60 + "\nACTUATOR GAIN VERIFICATION\n" + "─"*60)
        for act_name, act in (getattr(self._robot,"_actuators",None) or {}).items():
            kp = getattr(act,"stiffness",None)
            print(f"  [{act_name}]  KP={kp[0].cpu().numpy() if kp is not None else 'N/A'}")
        print("─"*60 + "\n")

    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clone()
        _mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5
        _half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        a = _mid + _half * torch.tanh(a)
        self._prev_prev_actions[:] = self._prev_actions
        self._prev_actions[:]      = self._actions
        self._actions[:]           = a
        self._target_pos = self._actions + self._robot.data.default_joint_pos
        if self._lag_pos is None:
            self._lag_pos = self._robot.data.joint_pos.clone()
        self._lag_pos = (self._lag_alpha * self._target_pos
                         + (1.0 - self._lag_alpha) * self._lag_pos)
        self._robot.set_joint_position_target(self._lag_pos)

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
        tilt    = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)

        # 1. Forward velocity tracking
        r_lin_vel = 1.5 * torch.exp(-(lin_vel[:, 0] - cmd[:, 0]) ** 2 / 0.25)

        # 2. Yaw tracking
        r_ang_vel = 0.5 * torch.exp(-(ang_vel[:, 2] - cmd[:, 2]) ** 2 / 0.25)

        # 3. Body roll/pitch rate
        r_ang_vel_xy = -0.05 * (ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2)

        # 4. No bouncing
        r_lin_vel_z = -2.0 * lin_vel[:, 2] ** 2

        # 5. Torques
        r_torques = -2e-4 * torch.sum(self._robot.data.applied_torque ** 2, dim=1)

        # 6. Smoothness (calibrated: sim≈real=0.030 ✓)
        r_action_rate = -0.5 * torch.sum(
            (self._actions - self._prev_actions) ** 2, dim=1)
        r_action_jerk = -0.3 * torch.sum(
            (self._actions - 2*self._prev_actions + self._prev_prev_actions) ** 2, dim=1)

        # 7. UPRIGHT — state-based lean penalty (replaces r_hip_reg)
        # Direct: penalises grav_x^2 + grav_y^2 regardless of direction.
        # Symmetric: same penalty left/right/forward/backward.
        # Quadratic growth ensures it dominates at large tilts.
        r_upright = -5.0 * (gravity[:, 0] ** 2 + gravity[:, 1] ** 2)

        # 8. TROT — direct diagonal contact reward
        # Rewards the actual gait we want, not a joint proxy.
        # diag1 = FL+RR in contact simultaneously
        # diag2 = FR+RL in contact simultaneously
        # Anti-phase bonus when only one diagonal is down (clean trot)
        foot_fz = self.scene.sensors["contact_sensor"].data.net_forces_w[:, :, 2]
        in_c    = (foot_fz > 8.0).float()
        diag1   = in_c[:, 0] * in_c[:, 3]   # FL + RR
        diag2   = in_c[:, 1] * in_c[:, 2]   # FR + RL

        all_down = (in_c.sum(dim=1) >= 3).float()  # 3+ feet = standing
        r_trot = (0.6 * torch.abs(diag1 - diag2)  # reward only alternation
                  - 0.4 * all_down)


        # 9. ALIVE — dense survival signal
        # +0.5 per step while robot is standing and not severely tilted.
        # Without this, episodes where robot falls immediately only see r_fall
        # and get no gradient signal for staying upright.
        r_alive = 0.5 * ((height > 0.28) & (tilt < 0.3)).float()

        # 10. Fall
        r_fall = -10.0 * (height < 0.25).float()

        keys = ["lin_vel","ang_vel","ang_vel_xy","lin_vel_z","torques",
                "action_rate","action_jerk","upright","trot","alive","fall"]
        vals = [r_lin_vel,r_ang_vel,r_ang_vel_xy,r_lin_vel_z,r_torques,
                r_action_rate,r_action_jerk,r_upright,r_trot,r_alive,r_fall]
        for k,v in zip(keys,vals):
            self._ep_sums[k] += v

        return self.step_dt * (
            r_lin_vel + r_ang_vel + r_ang_vel_xy + r_lin_vel_z + r_torques +
            r_action_rate + r_action_jerk + r_upright + r_trot + r_alive
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
            for k,v in self._ep_sums.items():
                print(f"  {k:14s}: {v[env_ids].mean().item():+.3f}")
            print("=" * 40)

        n = len(env_ids)
        hip_th_kp = torch.empty(n,8,device=self.device).uniform_(self._kp_hip_thigh_lo,self._kp_hip_thigh_hi)
        knee_kp   = torch.empty(n,4,device=self.device).uniform_(self._kp_knee_lo,self._kp_knee_hi)
        hip_th_kd = torch.empty(n,8,device=self.device).uniform_(self._kd_hip_thigh_lo,self._kd_hip_thigh_hi)
        knee_kd   = torch.empty(n,4,device=self.device).uniform_(self._kd_knee_lo,self._kd_knee_hi)
        self._kp_live[env_ids] = self._kp_nominal * torch.cat([hip_th_kp,knee_kp],dim=1)
        self._kd_live[env_ids] = self._kd_nominal * torch.cat([hip_th_kd,knee_kd],dim=1)
        self._robot.write_joint_stiffness_to_sim(self._kp_live[env_ids], env_ids=env_ids)
        self._robot.write_joint_damping_to_sim(  self._kd_live[env_ids], env_ids=env_ids)

        if self._global_step < 50 and len(env_ids) >= 2:
            kp0=self._kp_live[env_ids[0]].cpu().numpy().round(1)
            kp1=self._kp_live[env_ids[1]].cpu().numpy().round(1)
            d=abs(self._kp_live[env_ids[0]]-self._kp_live[env_ids[1]]).max().item()
            print(f"  [KP rand] env{env_ids[0].item()}: knees={kp0[8:]}  {'✓' if d>0.5 else '⚠'}")
            print(f"  [KP rand] env{env_ids[1].item()}: knees={kp1[8:]}")

        super()._reset_idx(env_ids)
        self.command_manager.reset(env_ids)

        _mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5
        _half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        _rand = (_mid + _half * self._prev_act_init_scale
                 * (torch.rand(len(env_ids),12,device=self.device)*2-1))
        self._actions[env_ids]           = _rand
        self._prev_actions[env_ids]      = _rand
        self._prev_prev_actions[env_ids] = _rand
        self._target_pos[env_ids]        = self._robot.data.default_joint_pos[env_ids]
        if self._lag_pos is not None:
            self._lag_pos[env_ids] = self._robot.data.joint_pos[env_ids].clone()
        for k in self._ep_sums:
            self._ep_sums[k][env_ids] = 0.0

        joint_pos  = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += (torch.rand_like(joint_pos)-0.5)*0.05
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:,:3]   = self.scene.env_origins[env_ids]
        root_state[:,2]   += 0.35
        root_state[:,3:7]  = torch.tensor([1.,0.,0.,0.],device=self.device)
        root_state[:,7:13] = 0.0
        self._robot.write_root_pose_to_sim(root_state[:,:7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:,7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), None, env_ids)

    def step(self, action):
        self._global_step += 1

        if self._global_step % 500 == 0:
            idx = 0
            print(f"\n{'='*80}")
            print(f"[DEBUG] step {self._global_step} | env0 | cmd_vx={self.command_manager.command[idx,0]:.2f}")
            kp0=self._kp_live[0].cpu().numpy().round(1)
            kp1=self._kp_live[1].cpu().numpy().round(1) if self.num_envs>1 else kp0
            d=abs(self._kp_live[0]-self._kp_live[1]).max().item() if self.num_envs>1 else 0
            print(f"  KP env0: {kp0}  {'✓' if d>0.5 else '⚠'}")
            print(f"  KP env1: {kp1}")
            if self.last_obs is not None:
                o=self.last_obs; i=0
                print(f"  cmd:       {o[i:i+3].cpu().numpy()}"); i+=3
                print(f"  jpos_delta:{o[i:i+12].cpu().numpy()}"); i+=12
                i+=15  # skip jvel+ang_vel
                gv=o[i:i+3]; tilt_d=float(torch.sqrt(gv[0]**2+gv[1]**2).item())*57.3
                print(f"  proj_grav: {gv.cpu().numpy()}  tilt≈{tilt_d:.1f}°")
            if self._lag_pos is not None:
                import numpy as _np
                tgt=self._target_pos[idx].cpu().numpy()
                act=self._robot.data.joint_pos[idx].cpu().numpy()
                te=_np.abs(act-tgt)
                JNAMES=['FL_hip','FR_hip','RL_hip','RR_hip','FL_th','FR_th',
                        'RL_th','RR_th','FL_kn','FR_kn','RL_kn','RR_kn']
                print(f"  total_err mean={te.mean():.4f}  max={te.max():.4f} ({JNAMES[te.argmax()]})")
            da=(self._actions[idx]-self._prev_actions[idx]).abs().mean().item()
            print(f"  |Δdelta|: {da:.4f}  (target <0.025  {'✓' if da<0.030 else '*** jerky'})")
            print(f"  root_z:   {self._robot.data.root_pos_w[idx,2]:.3f}")
            print(f"  lin_vel:  {self._robot.data.root_lin_vel_b[idx,:2].norm():.3f}")
            print("  rewards:")
            for k in self._ep_sums:
                v=self._ep_sums[k][idx].item()
                flag="  ***" if k in ["upright","trot","alive"] else ""
                print(f"    {k:14s}: {v:+.3f}{flag}")

        result = super().step(action)
        self.last_obs = result[0]["policy"][0].clone()

        if self._slog_active and self._slog_step < self._sim_log_maxsteps:
            s=self._slog_step; obs0=result[0]["policy"][0]
            g=self._robot.data.projected_gravity_b[0]
            tilt=float(torch.sqrt(g[0]**2+g[1]**2).item())
            try:
                fz=self.scene.sensors["contact_sensor"].data.net_forces_w[0,:,2]
                cnp=fz.cpu().numpy()
            except Exception:
                cnp=np.zeros(4,np.float32)
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