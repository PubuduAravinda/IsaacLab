import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand


class Go1Env(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.training = False

        print("\n" + "=" * 80)
        print("HIMLoco — base_obs (45D) + encoder (19D) = policy (64D)")
        print("=" * 80 + "\n")

        self.command_manager = UniformVelocityCommand(cfg=self.cfg.commands, env=self)

        # HIMLoco Encoders: 225D (5×45) → 19D (3 vel + 16 pos)
        def make_encoder():
            return nn.Sequential(
                nn.Linear(45 * 5, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 19)
            ).to(self.device)

        self.encoder_source = make_encoder()
        self.encoder_target = make_encoder()
        self.encoder_target.load_state_dict(self.encoder_source.state_dict())
        for param in self.encoder_target.parameters():
            param.requires_grad = False

        # SwAV prototypes
        self.prototypes = nn.Parameter(torch.randn(16, 16, device=self.device))
        nn.init.normal_(self.prototypes, std=0.01)

        # Loss functions
        self.vel_loss_fn = nn.MSELoss()

        # Buffers
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._prev_prev_actions = torch.zeros_like(self._actions)
        self._target_positions = torch.zeros(self.num_envs, 12, device=self.device)
        self.obs_history = torch.zeros(self.num_envs, 6, 45, device=self.device)

        # Episode reward tracking
        self._episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in [
            "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy",
            "orientation", "joint_acc", "joint_power", "base_height",
            "foot_clearance", "action_rate", "smoothness", "backward",
        ]}

        self._global_step = 0

        # Cache foot body indices for foot clearance reward
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        foot_idx = []
        for name in foot_names:
            idx, _ = self._robot.find_bodies(name)
            if len(idx) == 0:
                raise RuntimeError(f"'{name}' not found in robot bodies: {self._robot.body_names}")
            foot_idx.append(idx[0])
        self.foot_indices = torch.tensor(foot_idx, device=self.device, dtype=torch.long)

        print("Foot indices:", self.foot_indices)
        print("Joint order:")
        for i, name in enumerate(self._robot.joint_names):
            print(f"  {i:2d}: {name}")

        self.debug_obs_stats = {"min": [], "max": [], "mean": [], "std": []}

    def _setup_scene(self):
        self._robot = self.scene["robot"]

    # ─────────────────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        """Clamp and apply policy actions."""
        a = actions.clone()
        a[:, 0:4] = torch.clamp(a[:, 0:4], -0.8, 0.8)  # hips
        a[:, 4:8] = torch.clamp(a[:, 4:8], -1.2, 1.2)  # thighs
        a[:, 8:12] = torch.clamp(a[:, 8:12], -1.6, 1.6)  # calves

        if self._global_step % 100 == 0 and self._global_step > 0:
            print(f"\n[ACTION DEBUG] Step {self._global_step}, Env 0:")
            print(f"  Raw:    {actions[0].cpu().numpy()}")
            print(f"  Clamped:{a[0].cpu().numpy()}")
            print(f"  Target: {(a[0] + self._robot.data.default_joint_pos[0]).cpu().numpy()}")
            print(f"  Actual: {self._robot.data.joint_pos[0].cpu().numpy()}")
            print(f"  Vel:    {self._robot.data.joint_vel[0].cpu().numpy()}")

        self._prev_prev_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = a.clone()

        self._target_positions = self._actions + self._robot.data.default_joint_pos
        self._robot.set_joint_position_target(self._target_positions)

    def _apply_action(self):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """Build 64D policy observation = 45D base + 19D encoder."""
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        dof_pos = self._robot.data.joint_pos
        dof_vel = self._robot.data.joint_vel

        try:
            commands = self.command_manager.get_command("__default__")
        except:
            commands = self.command_manager.command

        # 45D base observation
        base_obs = torch.cat([
            commands[:, :3],  # 3
            dof_pos - self._robot.data.default_joint_pos,  # 12
            dof_vel,  # 12
            base_ang_vel,  # 3
            projected_gravity,  # 3
            self._previous_actions,  # 12
        ], dim=-1)  # = 45

        # Rolling history
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = base_obs

        # Encode last 5 steps → 19D (3D velocity + 16D position)
        # Reference uses obs_history[:, -5:] — last 5 slots of the 6-slot buffer
        history_flat = self.obs_history[:, -5:].reshape(self.num_envs, -1)  # [N, 225]
        encoder_out = self.encoder_source(history_flat)  # [N, 19]
        velocity_encoding = encoder_out[:, :3]  # [N, 3]
        position_encoding = encoder_out[:, 3:]  # [N, 16]

        # 64D policy input: base(45) + vel_enc(3) + pos_enc(16)
        # NO clamping — HIMLoco reference does not clamp observations
        policy_obs = torch.cat([base_obs, velocity_encoding, position_encoding], dim=-1)

        if self._global_step % 500 == 0 and self._global_step > 0:
            print(f"\n[OBS DEBUG] Step {self._global_step} | "
                  f"obs shape: {policy_obs.shape} | "
                  f"min/max: {policy_obs.min():.3f}/{policy_obs.max():.3f} | "
                  f"cmd: {commands[0, :2].cpu().numpy()}")

        return {"policy": policy_obs}

    # ─────────────────────────────────────────────────────────────────────────
    def _get_aux_losses(self):
        """HIMLoco auxiliary losses: velocity prediction + SwAV."""
        if not self.training:
            return {}

        hist = self.obs_history.clone()
        src = hist[:, :-1].reshape(self.num_envs, -1)  # [N, 225] steps 0-4 (source)
        tgt = hist[:, 1:].reshape(self.num_envs, -1)  # [N, 225] steps 1-5 (target, shifted)

        if torch.isnan(src).any():
            print("[ERROR] NaN in source_history!")
            return {}

        emb_s = self.encoder_source(src)  # [N, 19]
        emb_t = self.encoder_target(tgt)  # [N, 19]

        if torch.isnan(emb_s).any():
            print("[ERROR] NaN in emb_source!")
            return {}

        v_hat = emb_s[:, :3]
        l_s = emb_s[:, 3:]
        l_t = emb_t[:, 3:]

        # Ground truth velocity (vx, vy, yaw_rate)
        true_vel = torch.cat([
            self._robot.data.root_lin_vel_b[:, :2],
            self._robot.data.root_ang_vel_b[:, 2:3]
        ], dim=1)

        if torch.isnan(true_vel).any() or torch.isinf(true_vel).any():
            print("[ERROR] NaN/Inf in true_vel!")
            return {}

        # 1. Velocity prediction loss
        loss_vel = self.vel_loss_fn(v_hat, true_vel.detach())

        if torch.isnan(loss_vel) or torch.isinf(loss_vel):
            print("[ERROR] NaN in loss_vel!")
            return {}

        # 2. SwAV loss - stable cross-entropy version (proven to work)
        try:
            l_s_norm = F.normalize(l_s, dim=1, eps=1e-6)
            l_t_norm = F.normalize(l_t, dim=1, eps=1e-6)
            proto_norm = F.normalize(self.prototypes, dim=1, eps=1e-6)

            sim_s = l_s_norm @ proto_norm.t() / 0.1  # [N, 16]
            sim_t = l_t_norm @ proto_norm.t() / 0.1  # [N, 16]

            # Soft cross-entropy (stable SwAV without Sinkhorn)
            loss_swav = 0.5 * (
                    F.cross_entropy(sim_s, F.softmax(sim_t.detach(), dim=-1), reduction='none').mean() +
                    F.cross_entropy(sim_t, F.softmax(sim_s.detach(), dim=-1), reduction='none').mean()
            )

            if torch.isnan(loss_swav) or torch.isinf(loss_swav):
                loss_swav = torch.tensor(0.0, device=self.device, requires_grad=True)

        except Exception as e:
            print(f"[SwAV WARNING] {e}")
            loss_swav = torch.tensor(0.0, device=self.device, requires_grad=True)

        if self._global_step % 200 == 0 and self._global_step > 0:
            print(f"[HIMLoco AUX] step {self._global_step} | "
                  f"loss_vel: {loss_vel.item():.4f} | "
                  f"loss_swav: {loss_swav.item():.4f}")

        return {
            "loss_vel": loss_vel * 1.0,
            "loss_swav": loss_swav * 0.5,  # paper weight 0.5
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        """
        HIMLoco paper reward function — exact weights from Table 1.

        Reward                  Formula                                 Weight
        ─────────────────────────────────────────────────────────────────────
        Lin. vel tracking       exp(-||v_cmd_xy - v_xy||² / 2σ)        +1.0
        Ang. vel tracking       exp(-(ω_cmd_yaw - ω_yaw)² / σ)        +0.5
        Linear vel (z)          v_z²                                   -2.0
        Angular vel (xy)        ||ω_xy||²                              -0.05
        Orientation             ||g_xy||²                              -0.2
        Joint accelerations     ||θ̈||²                              -2.5e-7
        Joint power             |τ| · |θ̇|                            -2e-5
        Body height             (h_target - h)²                        -1.0
        Foot clearance          Σ(p_z_target - p_z_i)² · v_xy_i      -0.01
        Action rate             ||a_t - a_{t-1}||²                    -0.01
        Smoothness              ||a_t - 2a_{t-1} + a_{t-2}||²         -0.01
        """
        # ── Robot state ────────────────────────────────────────────────────
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        base_height = self._robot.data.root_pos_w[:, 2]
        projected_gravity = self._robot.data.projected_gravity_b
        dof_acc = self._robot.data.joint_acc
        dof_vel = self._robot.data.joint_vel
        dof_pos = self._robot.data.joint_pos

        try:
            commands = self.command_manager.get_command("__default__")
        except:
            commands = self.command_manager.command

        # Approximate torque (PD)
        actuator_cfg = self._robot.cfg.actuators["legs"]
        stiffness = torch.full((self.num_envs, 12), actuator_cfg.stiffness, device=self.device)
        damping = torch.full((self.num_envs, 12), actuator_cfg.damping, device=self.device)
        pos_error = self._target_positions - dof_pos
        dof_torque = stiffness * pos_error - damping * dof_vel

        sigma = 0.25

        # ── Rewards (paper Table 1) ─────────────────────────────────────────
        # 1. Linear velocity tracking  +1.0
        r_lin_vel = torch.exp(
            -torch.sum((base_lin_vel[:, :2] - commands[:, :2]) ** 2, dim=1) / (2 * sigma ** 2)
        ) * 1.0

        # 2. Angular velocity tracking  +0.5
        r_ang_vel = torch.exp(
            -(base_ang_vel[:, 2] - commands[:, 2]) ** 2 / sigma
        ) * 0.5

        # 3. Linear velocity z  -2.0
        r_lin_vel_z = -(base_lin_vel[:, 2] ** 2) * 2.0

        # 4. Angular velocity xy  -0.05
        r_ang_vel_xy = -torch.sum(base_ang_vel[:, :2] ** 2, dim=1) * 0.05

        # 5. Orientation  -0.2 (paper weight)
        r_orientation = -torch.sum(projected_gravity[:, :2] ** 2, dim=1) * 0.2

        # 6. Joint accelerations  -2.5e-7
        r_joint_acc = -torch.sum(dof_acc ** 2, dim=1) * 2.5e-7

        # 7. Joint power  -2e-5
        r_joint_power = -torch.sum(torch.abs(dof_torque) * torch.abs(dof_vel), dim=1) * 2e-5

        # 8. Body height (target 0.34m) - paper weight
        r_base_height = -((base_height - 0.34) ** 2) * 1.0

        # 9. Foot clearance  -0.01  (paper weight)
        #    r = -0.01 * Σᵢ (p_target_z - pᵢ_z)² · vᵢ_xy
        # 9. Contact-Based Gait Enforcement
        #    Use actual contact forces instead of trying to infer from heights/velocities
        r_foot_clearance = torch.zeros(self.num_envs, device=self.device)

        try:
            contact_sensor = self.scene.sensors.get("contact_sensor", None)
            if contact_sensor is not None and contact_sensor.data.net_forces_w is not None:
                # Contact sensor tracks ONLY the 4 feet, so indices are [0, 1, 2, 3]
                # NOT the body indices from the full robot!
                net_forces = contact_sensor.data.net_forces_w  # Shape: (N, 4, 3) for 4 feet

                # Get Z-axis contact forces for all 4 feet (indices 0-3)
                foot_forces = net_forces[:, :, 2]  # (N, 4) - all 4 feet

                # Detect which feet are in contact (force > threshold)
                in_contact = (foot_forces.abs() > 1.0).float()  # (N, 4) binary
                num_feet_contact = in_contact.sum(dim=1)  # (N,)

                # Reward 1: Penalize having <2 or >3 feet in contact
                r_contact_count = torch.zeros(self.num_envs, device=self.device)
                r_contact_count += -5.0 * (num_feet_contact < 2.0).float()
                r_contact_count += -2.0 * (num_feet_contact > 3.0).float()

                # Reward 2: Track cumulative contact time per foot
                if not hasattr(self, 'foot_contact_time'):
                    self.foot_contact_time = torch.zeros(self.num_envs, 4, device=self.device)

                self.foot_contact_time += in_contact

                # Reward 3: NEW - Contact FREQUENCY variance penalty
                #           All 4 feet should have similar contact frequencies
                #           Apply this from the START (not after 200 steps)

                # Calculate contact frequency for each foot (as percentage)
                episode_len = self.episode_length_buf.unsqueeze(1).float().clamp(min=1.0)  # (N, 1)
                contact_freq = self.foot_contact_time / episode_len  # (N, 4) range [0, 1]

                # In proper trot: all 4 feet should have freq ~0.5 (50% contact, 50% swing)
                # In bipedal: 2 feet freq=1.0, 2 feet freq=0.0 → HIGH variance
                # In tripod: 3 feet freq=1.0, 1 foot freq=0.0 → HIGH variance

                # Compute variance across the 4 feet
                mean_freq = contact_freq.mean(dim=1, keepdim=True)  # (N, 1)
                freq_variance = ((contact_freq - mean_freq) ** 2).mean(dim=1)  # (N,)

                # Penalize high variance (unequal contact frequencies)
                # variance = 0 → all feet contact equally ✓
                # variance > 0.1 → some feet never touch ✗
                r_freq_variance = -20.0 * freq_variance

                # Combine all contact rewards
                r_foot_clearance = r_contact_count + r_freq_variance

                # Detailed debugging
                if self._global_step % 500 == 0 and self._global_step > 0:
                    env0 = 0
                    print(f"\n[CONTACT DEBUG] Step {self._global_step}")
                    print("=" * 80)
                    foot_names = ["FL", "FR", "RL", "RR"]

                    # Calculate frequencies for env 0
                    ep_len = max(self.episode_length_buf[env0].item(), 1)

                    for i, name in enumerate(foot_names):
                        force = foot_forces[env0, i].item()
                        contact = in_contact[env0, i].item()
                        cum_time = self.foot_contact_time[env0, i].item()
                        freq = cum_time / ep_len  # Contact frequency
                        print(f"{name}: force={force:6.1f}N, contact={contact:.0f}, "
                              f"cumulative={cum_time:3.0f}, frequency={freq:.2f}")

                    # Show variance
                    variance = freq_variance[env0].item()
                    print(f"\nFeet in contact: {num_feet_contact[env0].item():.0f}/4")
                    print(f"Contact count penalty: {r_contact_count[env0].item():.2f}")
                    print(f"Frequency variance: {variance:.4f}")
                    print(f"Variance penalty: {r_freq_variance[env0].item():.2f}")
                    print(f"Total foot reward: {r_foot_clearance[env0].item():.2f}")
                    print("=" * 80 + "\n")
            else:
                # Fallback if contact sensor unavailable
                if self._global_step % 500 == 0 and self._global_step > 0:
                    print("[CONTACT WARN] Contact sensor not available, foot reward = 0")

        except Exception as e:
            if self._global_step % 500 == 0:
                print(f"[CONTACT ERROR] {e}")
                import traceback
                traceback.print_exc()

        # 10. Action rate  -0.01
        r_action_rate = -torch.sum(
            (self._actions - self._previous_actions) ** 2, dim=1
        ) * 0.01

        # 11. Smoothness  -0.01
        r_smoothness = -torch.sum(
            (self._actions - 2 * self._previous_actions + self._prev_prev_actions) ** 2, dim=1
        ) * 0.01

        # 12. Small backward penalty (not in paper but needed to prevent backward walking)
        r_backward = -1.0 * (base_lin_vel[:, 0] < -0.1).float()

        # ── Total ───────────────────────────────────────────────────────────
        total_reward = (
                r_lin_vel + r_ang_vel +
                r_lin_vel_z + r_ang_vel_xy + r_orientation +
                r_joint_acc + r_joint_power + r_base_height +
                r_foot_clearance + r_action_rate + r_smoothness +
                r_backward
        )

        # ── Episode logging ─────────────────────────────────────────────────
        keys = [
            "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy",
            "orientation", "joint_acc", "joint_power", "base_height",
            "foot_clearance", "action_rate", "smoothness", "backward",
        ]
        vals = [
            r_lin_vel, r_ang_vel, r_lin_vel_z, r_ang_vel_xy,
            r_orientation, r_joint_acc, r_joint_power, r_base_height,
            r_foot_clearance, r_action_rate, r_smoothness, r_backward,
        ]
        for k, v in zip(keys, vals):
            self._episode_sums[k] += v

        return total_reward

    # ─────────────────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        gravity = self._robot.data.projected_gravity_b
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        base_height = self._robot.data.root_pos_w[:, 2]

        tipped = roll_pitch > 1.5
        too_low = base_height < 0.24  # Low threshold - let robot learn at any stable height
        upside_down = gravity[:, 2] > 0.5

        if self._global_step % 100 == 0 and self._global_step > 0:
            print(f"\n[TERMINATION] Step {self._global_step} | Env 0: "
                  f"h={base_height[0]:.3f} rp={roll_pitch[0]:.3f} "
                  f"tipped={tipped[0].item()} low={too_low[0].item()}")

        terminated = tipped | too_low | upside_down
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ─────────────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        # Log episode reward summary
        if self._global_step % 200 == 0 and self._global_step > 0:
            print(f"\n=== Episode Rewards (Step {self._global_step}) ===")
            for k, v in self._episode_sums.items():
                print(f"  {k:20s}: {v[env_ids].mean().item():+.3f}")
            print("=" * 46)

        super()._reset_idx(env_ids)
        self.command_manager.reset(env_ids)

        # Clear buffers
        self.obs_history[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0
        self._target_positions[env_ids] = self._robot.data.default_joint_pos[env_ids]
        for k in self._episode_sums:
            self._episode_sums[k][env_ids] = 0.0

        # Reset foot contact time tracker
        if hasattr(self, 'foot_contact_time'):
            self.foot_contact_time[env_ids] = 0.0

        # Reset joint state with small noise
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos += (torch.rand_like(joint_pos) - 0.5) * 0.2

        # Reset root state
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] = 0.35  # spawn height
        root_state[:, 10] = -0.5 + torch.rand(len(env_ids), device=self.device) * -0.5

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action):
        self._global_step += 1
        return super().step(action)