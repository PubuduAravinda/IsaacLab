# go1_env.py
# 45-D observation (no foot contacts — not available on real Go1).
# Hard hip clamp in _pre_physics_step instead of r_limits reward.
# Clean reward set — every term has a measurable effect.
# Obs normalisation saved/loaded with checkpoint for sim-to-real.

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand


class Go1Env(DirectRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "=" * 70)
        print("Go1Env  |  45-D obs  |  50 Hz  |  hard hip clamp  |  clean rewards")
        print("=" * 70 + "\n")

        self.command_manager = UniformVelocityCommand(cfg=self.cfg.commands, env=self)

        # ── Action buffers ────────────────────────────────────────────────────
        self._actions          = torch.zeros(self.num_envs, 12, device=self.device)
        self._prev_actions     = torch.zeros_like(self._actions)
        self._prev_prev_actions= torch.zeros_like(self._actions)
        self._target_pos       = torch.zeros_like(self._actions)

        # ── Reward tracking ───────────────────────────────────────────────────
        self._ep_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in [
            "forward", "upright",
            "smooth",  "rate",    "torques",
            "yaw_err", "even",    "base_z_vel",
            "fall",
        ]}

        self._global_step = 0

        # Obs normalisation is handled by actor_obs_normalizer inside RSL-RL network
        # (actor_obs_normalization=True in PPO cfg). Do NOT double-normalise here.

        self.last_obs = None

    # ── Scene ─────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self._robot = self.scene["robot"]

    # ── Control ───────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clone()

        # Hard joint-range clamps — prevents wild postures, forces thigh+knee gait
        a[:, 0:4]  = torch.clamp(a[:, 0:4],  -0.2, 0.2)   # hip  (tight — small lateral only)
        a[:, 4:8]  = torch.clamp(a[:, 4:8],  -1.2, 1.2)   # thigh (main propulsion)
        a[:, 8:12] = torch.clamp(a[:, 8:12], -1.5, 1.5)   # knee

        self._prev_prev_actions[:] = self._prev_actions
        self._prev_actions[:]      = self._actions
        self._actions[:]           = a

        # Delta from default standing pose → absolute joint target
        self._target_pos = self._actions + self._robot.data.default_joint_pos
        self._robot.set_joint_position_target(self._target_pos)

    def _apply_action(self):
        """No-op — targets already set in _pre_physics_step."""
        pass

    # ── Observations (45-D) ───────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        Return raw scaled observations. RSL-RL's actor_obs_normalizer (RunningMeanStd
        inside the network) handles normalisation — do NOT normalise here too.

        Layout (45-D, all readable from real Go1 hardware):
            [0:3]   velocity commands (vx, vy, wz)          — bounded by cmd ranges
            [3:15]  joint pos delta from default             — encoder, ~[-π, π]
            [15:27] joint velocity                           — encoder, clip ±10 rad/s
            [27:30] base angular velocity                    — IMU gyro, clip ±5 rad/s
            [30:33] projected gravity                        — IMU, unit vector [-1,1]
            [33:45] previous actions                         — policy output [-1,1]
        """
        obs = torch.cat([
            self.command_manager.command[:, :3],                              # 3
            self._robot.data.joint_pos - self._robot.data.default_joint_pos, # 12
            torch.clamp(self._robot.data.joint_vel,      -10.0, 10.0),       # 12
            torch.clamp(self._robot.data.root_ang_vel_b,  -5.0,  5.0),       # 3
            self._robot.data.projected_gravity_b,                             # 3
            self._prev_actions,                                               # 12
        ], dim=-1)  # = 45

        return {"policy": obs}

    # ── Rewards ───────────────────────────────────────────────────────────────
    def _get_rewards(self):
        lin_vel  = self._robot.data.root_lin_vel_b
        ang_vel  = self._robot.data.root_ang_vel_b
        gravity  = self._robot.data.projected_gravity_b
        height   = self._robot.data.root_pos_w[:, 2]
        cmd      = self.command_manager.command

        # ── 1. Forward velocity tracking (dominant positive signal) ──────────
        # Gaussian centred on commanded vx. Width 0.3 gives gradient even when
        # tracking is imperfect early in training (wider = easier to learn from).
        r_forward = 4.0 * torch.exp(
            -((lin_vel[:, 0] - cmd[:, 0]) ** 2) / 0.25 ** 2
        )

        # ── 2. Flat orientation (AnymalC: flat_orientation_l2) ───────────────
        # Penalise tilt in roll/pitch via projected gravity x/y components.
        # When upright: gravity = [0, 0, -1], so grav_xy ≈ 0.
        # Weight reduced from -2.0 — too strong was fighting forward motion.
        r_upright = -1.0 * torch.sum(gravity[:, :2] ** 2, dim=1)

        # NOTE: r_height REMOVED — base height emerges naturally from upright
        # and contact rewards without needing explicit height tracking.

        # ── 3. Action smoothness (AnymalC: action_rate_l2) ───────────────────
        # 1st-order: penalise large step-to-step action changes
        r_smooth = -0.1 * torch.sum((self._actions - self._prev_actions) ** 2, dim=1)
        # 2nd-order: penalise jerk (acceleration of actions)
        r_rate   = -0.05 * torch.sum(
            (self._actions - 2 * self._prev_actions + self._prev_prev_actions) ** 2, dim=1
        )

        # ── 4. Joint torques (AnymalC: dof_torques_l2) ───────────────────────
        # REPLACES r_low_vel. Penalising dof_vel² punishes the joint MOTION
        # needed to walk. Penalising torques² penalises WASTED ENERGY instead.
        # Formula: torque ≈ kp*(q_target - q) + kd*(0 - dq)
        # With kp=50, kd=6 this approximates the real actuator torque.
        # On real Go1: sdk provides motor torque estimates — same concept applies.
        q_err    = self._target_pos - self._robot.data.joint_pos
        dq       = self._robot.data.joint_vel
        torques  = 50.0 * q_err - 6.0 * dq   # implicit actuator torque estimate
        r_torques = -2e-4 * torch.sum(torques ** 2, dim=1)

        # ── 5. Yaw tracking error (not punish ALL yaw) ────────────────────────
        # Penalise deviation from COMMANDED yaw rate (cmd[:,2]).
        # With ang_vel_z=0 command this reduces spinning without punishing
        # small natural yaw corrections during a trot gait.
        r_yaw_err = -0.3 * (ang_vel[:, 2] - cmd[:, 2]) ** 2

        # ── 6. Feet air time (AnymalC: feet_air_time) ────────────────────────
        # Encourage rhythmic stepping — ~30% contact ratio for a trot.
        foot_fz      = self.scene.sensors["contact_sensor"].data.net_forces_w[:, :, 2]
        in_contact   = (foot_fz > 8.0).float()
        contact_freq = in_contact.mean(dim=1)
        r_even = 0.1 * torch.exp(-10.0 * (contact_freq - 0.30) ** 2)

        # ── 7. Vertical body velocity (AnymalC: lin_vel_z_l2) ────────────────
        # Penalise bouncing. Increased weight — was too weak before.
        r_base_z_vel = -4.0 * (lin_vel[:, 2] ** 2)  # stronger — penalise diving hard

        # ── 8. Fall termination penalty ───────────────────────────────────────
        # No alive bonus — it caused stand-still exploitation.
        # Falling is discouraged implicitly by losing forward+upright rewards,
        # plus this explicit one-time penalty at termination height.
        r_fall = -5.0 * (height < 0.25).float()  # matches termination height

        # Scale by step_dt — standard in all IsaacLab locomotion envs (AnymalC, Spot, Go1).
        # step_dt = decimation * sim_dt = 10 * 0.002 = 0.02s
        # This keeps returns in a learnable range for the value function.
        # Without this, cumulative returns reach thousands → VF loss explodes.
        total = self.step_dt * (
            r_forward + r_upright +
            r_smooth  + r_rate    + r_torques +
            r_yaw_err + r_even    + r_base_z_vel
        ) + r_fall  # fall is a one-time penalty, not scaled by dt

        keys = ["forward","upright","smooth","rate","torques",
                "yaw_err","even","base_z_vel","fall"]
        vals = [r_forward,r_upright,r_smooth,r_rate,r_torques,
                r_yaw_err,r_even,r_base_z_vel,r_fall]
        for k, v in zip(keys, vals):
            self._ep_sums[k] += v

        return total

    # ── Termination ───────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        g      = self._robot.data.projected_gravity_b
        height = self._robot.data.root_pos_w[:, 2]
        tilt   = torch.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)

        terminated = (tilt > 0.8) | (height < 0.25) | (g[:, 2] > 0.3)  # tighter — stop before dive
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        # Print reward breakdown every 200 resets
        if self._global_step % 200 == 0 and self._global_step > 0:
            print(f"\n=== Rewards @ step {self._global_step} ===")
            for k, v in self._ep_sums.items():
                print(f"  {k:12s}: {v[env_ids].mean().item():+.3f}")
            print("=" * 40)

        super()._reset_idx(env_ids)
        self.command_manager.reset(env_ids)

        # Clear buffers
        self._actions[env_ids]          = 0.0
        self._prev_actions[env_ids]     = 0.0
        self._prev_prev_actions[env_ids]= 0.0
        self._target_pos[env_ids]       = self._robot.data.default_joint_pos[env_ids]
        for k in self._ep_sums:
            self._ep_sums[k][env_ids]   = 0.0

        # Gentle init with small noise
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += (torch.rand_like(joint_pos) - 0.5) * 0.05
        joint_vel  = torch.zeros_like(joint_pos)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        # write_root_pose_to_sim expects WORLD-FRAME coordinates.
        # default_root_state pos is (0,0,0.35) — local to env origin.
        # Without env_origins, ALL robots write to world (0,0,0.35) = same spot.
        root_state[:, :3]   = self.scene.env_origins[env_ids]  # world x,y + terrain z
        root_state[:, 2]   += 0.35                              # height above ground
        root_state[:, 3:7]  = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        root_state[:, 7:10] = 0.0   # zero linear vel
        root_state[:, 10:13]= 0.0   # zero angular vel

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ── Step (with periodic debug) ────────────────────────────────────────────
    def step(self, action):
        self._global_step += 1

        if self._global_step % 500 == 0:
            idx = 0
            print(f"\n{'='*80}")
            print(f"[DEBUG] step {self._global_step} | env 0 | cmd_vx={self.command_manager.command[idx,0]:.2f}")

            if self.last_obs is not None:
                o = self.last_obs
                i = 0
                print(f"  cmd:        {o[i:i+3].cpu().numpy()}");  i+=3
                print(f"  jpos_delta: {o[i:i+12].cpu().numpy()}"); i+=12
                print(f"  jvel:       {o[i:i+12].cpu().numpy()}"); i+=12
                print(f"  ang_vel:    {o[i:i+3].cpu().numpy()}");  i+=3
                print(f"  proj_grav:  {o[i:i+3].cpu().numpy()}");  i+=3
                print(f"  prev_act:   {o[i:i+12].cpu().numpy()}")
                # Sanity checks on raw obs (no normaliser applied in env)
                grav_mag = o[30:33].norm().item()
                print(f"  grav_mag:   {grav_mag:.3f} (should be ~1.0 if upright)")

            print(f"  action out: {action[idx].cpu().numpy()}")
            print(f"  root_z:     {self._robot.data.root_pos_w[idx,2]:.3f}")
            print(f"  lin_vel_xy: {self._robot.data.root_lin_vel_b[idx,:2].norm():.3f}")
            print("  rewards:")
            for k in self._ep_sums:
                print(f"    {k:12s}: {self._ep_sums[k][idx]:+.3f}")

        result = super().step(action)
        self.last_obs = result[0]["policy"][0].clone()
        return result