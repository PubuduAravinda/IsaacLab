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
            "foot_clearance", "action_rate", "smoothness"
        ]}

        self._global_step = 0

        # Foot indices for clearance reward
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        # self.foot_indices = torch.tensor(
        #     [self._robot.find_bodies(name)[0] for name in foot_names],
        #     device=self.device, dtype=torch.long
        # )

    def _setup_scene(self):
        self._robot = self.scene["robot"]

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.clamp(actions, -1.0, 1.0)
        self._prev_prev_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()

        self._target_positions = self.cfg.action_scale * actions + self._robot.data.default_joint_pos

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

        if self._global_step % 100 == 0:
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

            mean_vel_x = self._robot.data.root_lin_vel_b[:, 0].mean().item()
            print(f"Step {self._global_step} | Mean forward vel: {mean_vel_x:.2f} m/s")


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
        # === Same 11 rewards as before (only relevant parts shown) ===
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        base_height = self._robot.data.root_pos_w[:, 2]
        projected_gravity = self._robot.data.projected_gravity_b
        dof_acc = self._robot.data.joint_acc
        dof_pos = self._robot.data.joint_pos
        dof_vel = self._robot.data.joint_vel

        try:
            commands = self.command_manager.get_command("__default__")
        except:
            commands = self.command_manager.command

        # Torque approximation (PD)
        actuator_cfg = self._robot.cfg.actuators["legs"]
        stiffness = torch.full((self.num_envs, 12), actuator_cfg.stiffness, device=self.device)
        damping = torch.full((self.num_envs, 12), actuator_cfg.damping, device=self.device)
        pos_error = self._target_positions - dof_pos
        dof_torque_approx = stiffness * pos_error - damping * dof_vel

        sigma = 0.25

        # 11 rewards exactly as paper
        r_lin_vel = torch.exp(-torch.sum((base_lin_vel[:, :2] - commands[:, :2])**2, dim=1) / (2 * sigma**2)) * 1.0
        r_ang_vel = torch.exp(-(base_ang_vel[:, 2] - commands[:, 2])**2 / sigma) * 0.5
        r_lin_vel_z = -(base_lin_vel[:, 2]**2) * 2.0
        r_ang_vel_xy = -(torch.sum(base_ang_vel[:, :2]**2, dim=1) / 2) * 0.05
        r_orientation = -(torch.sum(projected_gravity[:, :2]**2, dim=1) / 2) * 0.2
        r_joint_acc = -torch.sum(dof_acc**2, dim=1) * 2.5e-7
        r_joint_power = -torch.sum(torch.abs(dof_torque_approx) * torch.abs(dof_vel), dim=1) * 2e-5
        r_base_height = -((base_height - 0.40)**2) * 1.0
        # Note: foot_clearance removed or set to 0 since no contacts in obs
        r_foot_clearance = torch.zeros(self.num_envs, device=self.device)
        r_action_rate = -(torch.sum((self._actions - self._previous_actions)**2, dim=1) / 2) * 0.01
        r_smoothness = -(torch.sum((self._actions - 2*self._previous_actions + self._prev_prev_actions)**2, dim=1) / 2) * 0.01

        total_reward = (r_lin_vel + r_ang_vel + r_lin_vel_z + r_ang_vel_xy +
                        r_orientation + r_joint_acc + r_joint_power + r_base_height +
                        r_foot_clearance + r_action_rate + r_smoothness)

        # Accumulate
        for k, v in zip(self._episode_sums.keys(), [
            r_lin_vel, r_ang_vel, r_lin_vel_z, r_ang_vel_xy,
            r_orientation, r_joint_acc, r_joint_power, r_base_height,
            r_foot_clearance, r_action_rate, r_smoothness
        ]):
            self._episode_sums[k] += v

        # Add early termination for poor tracking (curriculum)
        steps = self.episode_length_buf.float() + 1e-6  # avoid div0
        poor_tracking = (self._episode_sums["tracking_lin_vel"] / steps) < 0.8

        self._global_step += 1
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        gravity = self._robot.data.projected_gravity_b
        roll_pitch = torch.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)
        base_height = self._robot.data.root_pos_w[:, 2]

        tipped = roll_pitch > 0.9
        too_low = base_height < 0.18

        # Temporarily disable poor_tracking to allow longer episodes and contact learning
        poor_tracking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Optional: re-enable after some steps
        # if self._global_step > 200000:
        #     steps = self.episode_length_buf.float() + 1e-6
        #     poor_tracking = (self._episode_sums["tracking_lin_vel"] / steps) < 0.5

        terminated = tipped | too_low | poor_tracking
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        if self._global_step % 10 == 0:
            print(f"Step {self._global_step} | Tipped: {tipped.mean().item():.2f} | Too low: {too_low.mean().item():.2f} | Poor tracking: {poor_tracking.mean().item():.2f}")

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

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
        root_state[:, 2] = 0.25 + torch.rand(len(env_ids), device=self.device) * 0.05  # 0.25–0.30m

        # Add downward velocity to force quick drop
        root_state[:, 10] = -0.5 + torch.rand(len(env_ids), device=self.device) * -0.5  # -0.5 to -1.0 m/s (Z velocity)

        # print(f"Reset env_ids {env_ids[0]} | Spawn height: {root_state[0, 2]:.3f}m")

        # Write to sim
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)