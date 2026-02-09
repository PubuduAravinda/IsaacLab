import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand


class SwAVLoss(nn.Module):
    def __init__(self, K=16, tau=0.1, sinkhorn_epsilon=0.05, sinkhorn_iterations=3):
        super().__init__()
        self.K = K
        self.tau = tau
        self.epsilon = sinkhorn_epsilon
        self.iters = sinkhorn_iterations

    def sinkhorn(self, scores):
        Q = torch.exp(scores / self.epsilon)
        Q /= Q.sum(dim=-1, keepdim=True)
        for _ in range(self.iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=-1, keepdim=True)
        return Q

    def forward(self, z_s, z_t, prototypes):
        z_s = F.normalize(z_s, dim=1)
        z_t = F.normalize(z_t, dim=1)
        P = F.normalize(prototypes, dim=0)  # normalize prototypes (columns)

        scores_s = z_s @ P.t() / self.tau
        scores_t = z_t @ P.t() / self.tau

        Q_s = self.sinkhorn(scores_s.detach())
        Q_t = self.sinkhorn(scores_t.detach())

        P_s = F.softmax(scores_s, dim=-1)
        P_t = F.softmax(scores_t, dim=-1)

        loss = -0.5 * (
            (Q_s * torch.log(P_t.detach() + 1e-8)).sum(-1).mean() +
            (Q_t * torch.log(P_s.detach() + 1e-8)).sum(-1).mean()
        )
        return loss


class Go1Env(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "="*80)
        print("HIMLoco Replication — Ready for RSL_RL (Aux losses active, target unfrozen)")
        print("="*80 + "\n")

        self.command_manager = UniformVelocityCommand(cfg=self.cfg.commands, env=self)

        # HIM Encoders (both trainable — unfrozen target for aux optimization)
        def make_encoder():
            return nn.Sequential(
                nn.Linear(45 * 5, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 19)
            ).to(self.device)

        self.encoder_source = make_encoder()
        self.encoder_target = make_encoder()  # UNFROZEN — will be updated via aux losses in RSL_RL

        self.prototypes = nn.Parameter(torch.randn(16, 16))
        nn.init.normal_(self.prototypes, std=0.01)

        self.vel_loss_fn = nn.MSELoss()
        self.swav_loss = SwAVLoss(K=16, tau=0.1).to(self.device)

        # Buffers
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._prev_prev_actions = torch.zeros_like(self._actions)
        self._target_positions = torch.zeros(self.num_envs, 12, device=self.device)
        self.obs_history = torch.zeros(self.num_envs, 6, 45, device=self.device)  # H=5 +1 for shift

        self._episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in [
            "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy",
            "orientation", "joint_acc", "joint_power", "base_height",
            "foot_clearance", "action_rate", "smoothness", "r_lateral_vel", "r_alive", "r_no_movement", "r_backward_penalty"
        ]}

        self._global_step = 0

        # Cache foot indices safely
        self.foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        foot_indices_list = []
        for name in self.foot_names:
            body_indices, matched_names = self._robot.find_bodies(name)
            if len(body_indices) == 0:
                raise RuntimeError(f"Foot body '{name}' not found! Available bodies: {self._robot.body_names}")
            foot_indices_list.append(body_indices[0])  # body_indices[0] is an int

        self.foot_indices = torch.tensor(foot_indices_list, device=self.device, dtype=torch.long)

        # Debug: Print to confirm valid indices (e.g., tensor([13,14,15,16]))
        print("Cached foot indices:", self.foot_indices)

        print("Joint names and indices (from robot data):")
        for i, name in enumerate(self._robot.joint_names):
            print(f"  {i:2d}: {name}")

        # For debug prints
        self.debug_joint_history = []   # list of (joint_delta, action) tuples
        self.debug_obs_stats = {"min": [], "max": [], "mean": [], "std": []}

    def _setup_scene(self):
        self._robot = self.scene["robot"]

    # def _pre_physics_step(self, actions: torch.Tensor):
    #     # Per-joint-group hard clamping — prevents extremes while allowing expressive range
    #     # Order: FL/FR/RL/RR hip → FL/FR/RL/RR thigh → FL/FR/RL/RR calf
    #     # Hips:   ±0.8 rad (abduction/adduction — limited by mechanics)
    #     # Thighs: ±1.2 rad (hip flexion/extension — more range)
    #     # Calves: ±1.6 rad (knee flexion/extension — largest needed for propulsion)
    #
    #     clamped_actions = actions.clone()
    #
    #     # Hips (indices 0–3)
    #     clamped_actions[:, 0:4] = torch.clamp(clamped_actions[:, 0:4], -0.8, 0.8)
    #
    #     # Thighs (indices 4–7)
    #     clamped_actions[:, 4:8] = torch.clamp(clamped_actions[:, 4:8], -1.2, 1.2)
    #
    #     # Calves (indices 8–11)
    #     clamped_actions[:, 8:12] = torch.clamp(clamped_actions[:, 8:12], -1.6, 1.6)
    #
    #     # Optional: small smoothing / momentum (uncomment if jerky)
    #     # clamped_actions = 0.85 * clamped_actions + 0.15 * self._previous_actions
    #
    #     # History shift
    #     self._prev_prev_actions = self._previous_actions.clone()
    #     self._previous_actions = self._actions.clone()
    #     self._actions = clamped_actions.clone()
    #
    #     # Targets: direct delta from default (your successful style)
    #     self._target_positions = clamped_actions + self._robot.data.default_joint_pos
    #

    # def _pre_physics_step(self, actions: torch.Tensor):
    #     """
    #     Curriculum on per-group clamp ranges based on mean episode reward
    #     """
    #     # Compute mean episode reward (total sum across all terms, averaged over envs)
    #     total_reward_sum = torch.zeros(self.num_envs, device=self.device)
    #     for key in self._episode_sums:
    #         total_reward_sum += self._episode_sums[key]
    #
    #     mean_reward_per_env = total_reward_sum / len(self._episode_sums) if len(
    #         self._episode_sums) > 0 else torch.zeros(self.num_envs, device=self.device)
    #     mean_episode_reward = mean_reward_per_env.mean().item()
    #
    #     # Thresholds (tune these!)
    #     start_ramp_reward = 15.0
    #     full_ramp_reward = 30.0
    #
    #     ramp_progress = 0.0
    #     if mean_episode_reward > start_ramp_reward:
    #         ramp_progress = min(1.0, (mean_episode_reward - start_ramp_reward) / (full_ramp_reward - start_ramp_reward))
    #
    #     hip_max = 0.8
    #     thigh_max = 0.8 + ramp_progress * (1.2 - 0.8)
    #     calf_max = 0.8 + ramp_progress * (1.6 - 0.8)
    #
    #     if ramp_progress < 1e-3:
    #         mode = "uniform ±0.8 (stable)"
    #     elif ramp_progress < 1.0:
    #         mode = f"ramping (progress {ramp_progress:.2f})"
    #     else:
    #         mode = "full per-group (±0.8/1.2/1.6)"
    #
    #     # ALWAYS PRINT EVERY 50 STEPS — no conditions
    #     if self._global_step % 150 == 0 and self._global_step > 0:
    #         print(f"[CURRICULUM] Step {self._global_step:>8d} | "
    #               f"Mean reward: {mean_episode_reward:+.1f} | "
    #               f"Ramp progress: {ramp_progress:.2f} | "
    #               f"Ranges: hip±{hip_max:.2f} thigh±{thigh_max:.2f} calf±{calf_max:.2f} | "
    #               f"Mode: {mode}")
    #         import sys
    #         sys.stdout.flush()  # force output now
    #
    #     # Apply clamping
    #     clamped_actions = actions.clone()
    #     clamped_actions[:, 0:4] = torch.clamp(clamped_actions[:, 0:4], -hip_max, hip_max)
    #     clamped_actions[:, 4:8] = torch.clamp(clamped_actions[:, 4:8], -thigh_max, thigh_max)
    #     clamped_actions[:, 8:12] = torch.clamp(clamped_actions[:, 8:12], -calf_max, calf_max)
    #
    #     # Smoothing during transition
    #     actions_smoothed = clamped_actions #0.92 * clamped_actions + 0.08 * self._previous_actions
    #
    #     # History + targets
    #     self._prev_prev_actions = self._previous_actions.clone()
    #     self._previous_actions = self._actions.clone()
    #     self._actions = actions_smoothed.clone()
    #     self._target_positions = actions_smoothed + self._robot.data.default_joint_pos


    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.clamp(actions, -1.6, 1.6)
        self._prev_prev_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()

        self._target_positions = self.cfg.action_scale * actions + self._robot.data.default_joint_pos


    # def _pre_physics_step(self, actions: torch.Tensor):
    #
    #     self.action_scales = torch.tensor([1.0] * 12, device=self.device)  # Mid: uniform medium
    #     actions_scaled = self.action_scales * torch.tanh(actions)
    #
    #     #actions = torch.clamp(actions, -1.0, 1.0)
    #     self._prev_prev_actions = self._previous_actions.clone()
    #     self._previous_actions = self._actions.clone()
    #     self._actions = actions_scaled.clone()
    #
    #     self._target_positions = actions_scaled + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._target_positions)
        self.scene.write_data_to_sim()

    def _get_observations(self):
        commands = self.command_manager.command

        base_obs = torch.cat([
            commands[:, :3],
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._previous_actions,
        ], dim=-1)  # 45D

        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = base_obs

        history_flat = self.obs_history[:, -5:].reshape(self.num_envs, -1)

        emb_source = self.encoder_source(history_flat)

        policy_obs = torch.cat([base_obs, emb_source], dim=1)  # 64D

        # Add this line to define base_height
        base_height = self._robot.data.root_pos_w[:, 2]



        # ── Collect stats for sim2real matching ───────────────────────
        self.debug_obs_stats["min"].append(policy_obs.min(dim=0).values)
        self.debug_obs_stats["max"].append(policy_obs.max(dim=0).values)
        self.debug_obs_stats["mean"].append(policy_obs.mean(dim=0))
        self.debug_obs_stats["std"].append(policy_obs.std(dim=0))

        if len(self.debug_obs_stats["min"]) > 200:
            for k in self.debug_obs_stats:
                self.debug_obs_stats[k].pop(0)

        # ── Detailed debug print every 100 steps ──────────────────────
        if self._global_step % 100 == 0 and self._global_step > 0:
            print("\n" + "="*100)
            print(f"Global step: {self._global_step:>8d} | Mean forward vel: {self._robot.data.root_lin_vel_b[:,0].mean().item():.3f} m/s")

            contact_sensor = self.scene.sensors.get("contact_sensor", None)
            if contact_sensor is None:
                print("Warning: contact_sensor not found!")
                mean_force_z = 0.0
            else:
                net_forces = contact_sensor.data.net_forces_w
                if net_forces is None or net_forces.numel() == 0:
                    mean_force_z = 0.0
                else:
                    # Sensor is only on 4 feet → all bodies are feet → use all
                    foot_net_z = net_forces[:, :, 2]  # (num_envs, 4) Z forces
                    mean_force_z = foot_net_z.abs().mean().item()

            print(f"=============>>>> Step {self._global_step} | Mean foot Z force: {mean_force_z:.2f} N")
            # Debug: Joint order, default, current pos, action, next (target) state, error (env 0 only)
            joint_names = self._robot.joint_names
            default_pos = self._robot.data.default_joint_pos[0]
            current_pos = self._robot.data.joint_pos[0]
            action_pos = self._actions[0]  # final clamped actions
            target_pos = self._target_positions[0]  # default + action
            error = target_pos - current_pos

            print(f"Avg base height: {base_height.mean().item():.3f} m (std {base_height.std().item():.3f})")

            print("Joint order, default/current pos, policy action, next state (target), error (env 0):")
            print("  idx | name            | default | current | action  | next    | error   ")
            for i, name in enumerate(joint_names):
                print(f"  {i:3d} | {name:15} | {default_pos[i]:+6.3f} | {current_pos[i]:+6.3f} | {action_pos[i]:+6.3f} | {target_pos[i]:+6.3f} | {error[i]:+6.3f}")

            # Last 5 raw cycles (joint_delta | action)
            print("Last 5 raw cycles (joint_delta | action):")
            for i, (jdelta, act) in enumerate(self.debug_joint_history[-5:]):
                jstr = " ".join([f"{x:+.3f}" for x in jdelta[0]])
                astr = " ".join([f"{x:+.3f}" for x in act[0]])
                print(f"  t-{4-i:1d} jdelta: {jstr}")
                print(f"        action: {astr}")

            # Current state (env 0)
            cmd = commands[0]
            grav = self._robot.data.projected_gravity_b[0]
            linvel = self._robot.data.root_lin_vel_b[0]
            print("Current state (env 0):")
            print(f"  command: vx = {cmd[0]:+5.3f} vy = {cmd[1]:+5.3f} yaw_rate = {cmd[2]:+5.3f}")
            print(f"  gravity: x = {grav[0]:+5.3f} y = {grav[1]:+5.3f} z = {grav[2]:+5.3f}")
            print(f"  lin_vel_b: x = {linvel[0]:+5.3f} y = {linvel[1]:+5.3f} z = {linvel[2]:+5.3f}")

            # Observation statistics (64D)
            if len(self.debug_obs_stats["min"]) > 0:
                obs_min  = torch.stack(self.debug_obs_stats["min"]).min(dim=0).values
                obs_max  = torch.stack(self.debug_obs_stats["max"]).max(dim=0).values
                obs_mean = torch.stack(self.debug_obs_stats["mean"]).mean(dim=0)
                obs_std  = torch.stack(self.debug_obs_stats["std"]).mean(dim=0)

                print("\nPolicy observation stats (last ~200 env steps):")
                print(" idx | min max mean std")
                for i in range(0, 64, 8):
                    print(f"  {i:2d}-{i+7:2d} | {obs_min[i]:+6.2f} {obs_max[i]:+6.2f} {obs_mean[i]:+6.2f} {obs_std[i]:5.3f}")

            # Foot contact forces (if sensor exists)
            if hasattr(self.scene.sensors, "contact_sensor"):
                contact_sensor = self.scene.sensors["contact_sensor"]
                if contact_sensor.data.net_forces_w is not None:
                    foot_z = contact_sensor.data.net_forces_w[:, self.foot_indices, 2]
                    print(f"Mean | max foot Z-force: {foot_z.abs().mean().item():.2f} | {foot_z.abs().max().item():.2f} N")

            print("="*100 + "\n")


        self._global_step += 1
        return {"policy": policy_obs, "critic": policy_obs}

    def _get_aux_losses(self):
        if not self.training:
            return {}

        # Shifted views
        source_history = self.obs_history[:, :-1].reshape(self.num_envs, -1)  # Last 5 for source
        target_history = self.obs_history[:, 1:].reshape(self.num_envs, -1)  # Shifted 5 for target

        emb_source = self.encoder_source(source_history)
        emb_target = self.encoder_target(target_history)

        v_hat = emb_source[:, :3]
        l_s = emb_source[:, 3:]
        l_t = emb_target[:, 3:]

        true_vel = torch.cat([
            self._robot.data.root_lin_vel_b[:, :2],
            self._robot.data.root_ang_vel_b[:, 2:3]
        ], dim=1)

        loss_vel = self.vel_loss_fn(v_hat, true_vel)
        loss_swav = self.swav_loss(l_s, l_t, self.prototypes)

        return {
            "loss_vel": loss_vel * 1.0,
            "loss_swav": loss_swav * 1.0
        }

    def _get_rewards(self) -> torch.Tensor:
        # Base states
        base_lin_vel = self._robot.data.root_lin_vel_b  # (num_envs, 3)
        base_ang_vel = self._robot.data.root_ang_vel_b  # (num_envs, 3)
        base_height = self._robot.data.root_pos_w[:, 2]  # (num_envs,)
        projected_gravity = self._robot.data.projected_gravity_b  # (num_envs, 3)

        # Joint states
        dof_acc = self._robot.data.joint_acc  # (num_envs, 12)
        dof_pos = self._robot.data.joint_pos  # (num_envs, 12)
        dof_vel = self._robot.data.joint_vel  # (num_envs, 12)

        # Commands
        try:
            commands = self.command_manager.get_command("__default__")
        except:
            commands = self.command_manager.command  # (num_envs, 4): vx, vy, vz_cmd (unused), yaw_rate

        # Approximate torque via PD control (for power reward)
        actuator_cfg = self._robot.cfg.actuators["legs"]
        stiffness = torch.full((self.num_envs, 12), actuator_cfg.stiffness, device=self.device)
        damping = torch.full((self.num_envs, 12), actuator_cfg.damping, device=self.device)
        pos_error = self._target_positions - dof_pos
        dof_torque_approx = stiffness * pos_error - damping * dof_vel

        sigma = 0.25

        # === Individual rewards (exactly as in HIMLoco paper) ===
        r_lin_vel = torch.exp(-torch.sum((base_lin_vel[:, :2] - commands[:, :2]) ** 2, dim=1) / (2 * sigma ** 2)) * 10.0  # *8.0 (was 2.0) — huge incentive
        r_ang_vel = torch.exp(-(base_ang_vel[:, 2] - commands[:, 2]) ** 2 / sigma) * 1.0  # keep your fix
        # r_ang_vel = torch.exp(-(base_ang_vel[:, 2] - commands[:, 2]) ** 2 / (sigma ** 2)) * 2.0

        r_lin_vel_z = - (base_lin_vel[:, 2] ** 2) * 2.0
        r_ang_vel_xy = - (torch.sum(base_ang_vel[:, :2] ** 2, dim=1)) * 0.05
        r_orientation = - (torch.sum(projected_gravity[:, :2] ** 2, dim=1)) * 0.2
        r_joint_acc = - torch.sum(dof_acc ** 2, dim=1) * 2.5e-7
        r_joint_power = - torch.sum(torch.abs(dof_torque_approx) * torch.abs(dof_vel), dim=1) * 2e-5
        r_base_height = - ((base_height - 0.34) ** 2) * 1.0

        r_lateral_vel = - (base_lin_vel[:, 1] ** 2) * 1.5  # Penalize abs(lateral vel); scale 0.5–1.0
        r_alive = torch.ones(self.num_envs, device=self.device) * 0.25  # +3 per step alive — strongly encourages long episodes
        # After computing base_lin_vel
        vel_norm_xy = torch.norm(base_lin_vel[:, :2], dim=1)  # forward + lateral speed
        r_no_movement = -3.0 * (vel_norm_xy < 0.3).float()  # -3 if speed < 0.3 m/s
        # or more aggressive: -5.0 * torch.relu(0.4 - vel_norm_xy)  # linear penalty below 0.4 m/s

        # Temporarily disable foot clearance to avoid body indexing issues
        r_foot_clearance = torch.zeros(self.num_envs, device=self.device)

        r_action_rate = - torch.sum((self._actions - self._previous_actions) ** 2, dim=1) * 0.005
        r_smoothness = - torch.sum((self._actions - 2 * self._previous_actions + self._prev_prev_actions) ** 2, dim=1) * 0.005

        r_backward_penalty = -1.0 * (base_lin_vel[:, 0] < 0.0).float()

        # === Total reward ===
        total_reward = (
                r_lin_vel + r_ang_vel + r_lin_vel_z + r_ang_vel_xy +
                r_orientation + r_joint_acc + r_joint_power + r_base_height +
                r_foot_clearance + r_action_rate + r_smoothness + r_lateral_vel + r_alive + r_no_movement + r_backward_penalty
        )

        # === Accumulate per-episode sums for curriculum / logging ===
        reward_terms = [
            r_lin_vel, r_ang_vel, r_lin_vel_z, r_ang_vel_xy,
            r_orientation, r_joint_acc, r_joint_power, r_base_height,
            r_foot_clearance, r_action_rate, r_smoothness, r_lateral_vel, r_alive, r_no_movement, r_backward_penalty
        ]
        for key, value in zip(self._episode_sums.keys(), reward_terms):
            self._episode_sums[key] += value

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        gravity = self._robot.data.projected_gravity_b
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        base_height = self._robot.data.root_pos_w[:, 2]

        tipped = roll_pitch > 1.2  # was 0.9 → more tolerant
        too_low = base_height < 0.10  # was 0.18 → give more room
        upside_down = gravity[:, 2] > -0.2
        almost_inverted = gravity[:, 2] > 0.3
        inverted = upside_down | almost_inverted

        poor_tracking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._global_step > 500000:
            steps = self.episode_length_buf.float() + 1e-6
            poor_tracking = (self._episode_sums[
                                 "tracking_lin_vel"] / steps) < 0.8  # Enabled with paper threshold 0.8 (was 0.5)

        terminated = tipped | too_low | poor_tracking | inverted
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # if self._global_step % 10 == 0:
        #     print(
        #         f"Step {self._global_step} | Tipped: {tipped.mean().item():.2f} | Too low: {too_low.mean().item():.2f} | Poor tracking: {poor_tracking.mean().item():.2f}")

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        # Print reward stats for resetting envs (per-episode summary)
        if len(env_ids) > 0 and self._global_step % 200 == 0:  # print occasionally
            print(f"\n=== Episode End Reward Summary (Step {self._global_step}) ===")
            for key, value in self._episode_sums.items():
                # Mean over resetting envs only
                avg = value[env_ids].mean().item() if len(env_ids) > 0 else 0.0
                print(f"{key:20}: {avg:.3f}")
            print("=====================================\n")

        super()._reset_idx(env_ids)

        self.command_manager.reset(env_ids)

        # Zero buffers
        self.obs_history[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0
        self._target_positions[env_ids] = self._robot.data.default_joint_pos[env_ids]

        # Reset episode sums
        for k in self._episode_sums:
            self._episode_sums[k][env_ids] = 0.0

        # Initial joint state with small randomization
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos += (torch.rand_like(joint_pos) - 0.5) * 0.2  # ±0.1 rad noise

        # Root state
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        # Lower spawn height + randomization
        # root_state[:, 2] = 0.25 + torch.rand(len(env_ids), device=self.device) * 0.05  # 0.25–0.30m
        root_state[:, 2] = 0.35

        # Add downward velocity to force quick drop
        root_state[:, 10] = -0.5 + torch.rand(len(env_ids), device=self.device) * -0.5  # -0.5 to -1.0 m/s (Z velocity)

        # print(f"Reset env_ids {env_ids[0]} | Spawn height: {root_state[0, 2]:.3f}m")

        # Write to sim
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)