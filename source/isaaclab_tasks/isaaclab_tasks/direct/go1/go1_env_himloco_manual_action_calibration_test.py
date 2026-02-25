import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        """Sinkhorn-Knopp algorithm with numerical stability."""
        Q = torch.exp(scores / self.epsilon)
        # Add small epsilon to prevent division by zero
        Q = Q / (Q.sum(dim=-1, keepdim=True) + 1e-8)

        for _ in range(self.iters):
            # Row normalization
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
            # Column normalization
            Q = Q / (Q.sum(dim=-1, keepdim=True) + 1e-8)

        return Q

    def forward(self, z_s, z_t, prototypes):
        # Normalize embeddings
        z_s = F.normalize(z_s, dim=1, eps=1e-8)
        z_t = F.normalize(z_t, dim=1, eps=1e-8)

        # Normalize prototypes (only rows, not columns as before)
        P = F.normalize(prototypes, dim=1, eps=1e-8)  # Each prototype is unit norm

        # Compute similarity scores
        scores_s = z_s @ P.t() / self.tau  # [batch, K]
        scores_t = z_t @ P.t() / self.tau  # [batch, K]

        # Get soft assignments via Sinkhorn-Knopp (detached for stability)
        with torch.no_grad():
            Q_s = self.sinkhorn(scores_s)
            Q_t = self.sinkhorn(scores_t)

        # Compute log probabilities
        log_P_s = F.log_softmax(scores_s, dim=-1)
        log_P_t = F.log_softmax(scores_t, dim=-1)

        # SwAV loss: cross-entropy between assignments
        loss = -0.5 * (
                torch.mean(torch.sum(Q_t * log_P_s, dim=-1)) +
                torch.mean(torch.sum(Q_s * log_P_t, dim=-1))
        )

        return loss


class Go1Env(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # CRITICAL: Set training to False initially (CustomPPO will set to True)
        self.training = False

        print("\n" + "=" * 80)
        print("HIMLoco Replication — Policy gets base_obs (45D) + encoder_outs (19D) = 64D")
        print("=" * 80 + "\n")

        self.command_manager = UniformVelocityCommand(cfg=self.cfg.commands, env=self)

        # HIM Encoders: 225D (5*45 history) → 19D (3 vel + 16 pos)
        def make_encoder():
            return nn.Sequential(
                nn.Linear(45 * 5, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 19)
            ).to(self.device)

        self.encoder_source = make_encoder()
        self.encoder_target = make_encoder()

        # Initialize target as copy of source
        self.encoder_target.load_state_dict(self.encoder_source.state_dict())

        # CRITICAL: Freeze target encoder (only updated via EMA in CustomPPO)
        for param in self.encoder_target.parameters():
            param.requires_grad = False

        # SwAV prototypes
        self.prototypes = nn.Parameter(torch.randn(16, 16, device=self.device))
        nn.init.normal_(self.prototypes, std=0.01)

        # Loss functions
        self.vel_loss_fn = nn.MSELoss()
        self.swav_loss = SwAVLoss(K=16, tau=0.1).to(self.device)

        # Action buffers
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._prev_prev_actions = torch.zeros_like(self._actions)
        self._target_positions = torch.zeros(self.num_envs, 12, device=self.device)

        # Observation history: 6 slots (H=5 + 1 for rolling)
        self.obs_history = torch.zeros(self.num_envs, 6, 45, device=self.device)

        # Episode reward tracking
        self._episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in [
            "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy",
            "orientation", "joint_acc", "joint_power", "base_height",
            "foot_clearance", "action_rate", "smoothness", "r_lateral_vel",
            "r_alive", "r_no_movement", "r_backward_penalty"
        ]}

        # Global step counter
        self._global_step = 0

        # CALIBRATION MODE: Hardcoded sequence from real robot
        # Set CALIBRATION_MODE = True to test, False for normal training
        self.CALIBRATION_MODE = True  # CHANGE TO False FOR NORMAL TRAINING!

        if self.CALIBRATION_MODE:
            print("\n" + "=" * 80)
            print("⚠️  CALIBRATION MODE ACTIVE - POLICY ACTIONS IGNORED!")
            print("Using hardcoded real robot sequence for FL leg")
            print("🚀 Robot held in air (no gravity), no termination")
            print("=" * 80 + "\n")

            # Stand pose in Isaac Sim order
            # Sim order: [FL_hip, FR_hip, RL_hip, RR_hip, FL_th, FR_th, RL_th, RR_th, FL_cal, FR_cal, RL_cal, RR_cal]
            self.stand_q = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.77, 0.77, 0.77, 0.77, -1.5, -1.5, -1.5, -1.5],
                                        device=self.device)

            # Your crawl sequence - ALREADY in Isaac Sim order!
            # [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
            crawl_seq = [
                # Stabilize (balance adjustment: FR_calf -0.15, RL_calf +0.10, RR_calf -0.10)
                *[[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.15, 0.10, -0.10] for _ in range(5)],
                # Lift FL gradually (FL_thigh negative, FL_calf negative)
                [0.00, 0.00, 0.00, 0.00, -0.06, 0.00, 0.00, 0.00, -0.10, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.12, 0.00, 0.00, 0.00, -0.20, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.18, 0.00, 0.00, 0.00, -0.30, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.24, 0.00, 0.00, 0.00, -0.40, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00, -0.50, -0.15, 0.10, -0.10],
                # Hold lifted
                *[[0.00, 0.00, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00, -0.50, -0.15, 0.10, -0.10] for _ in range(10)],
                # Bend knee more (FL_calf more negative)
                [0.00, 0.00, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00, -0.62, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00, -0.74, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00, -0.86, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00, -0.98, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10],
                # Hold bent
                *[[0.00, 0.00, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10] for _ in range(10)],
                # Swing forward (FL_thigh more negative)
                [0.00, 0.00, 0.00, 0.00, -0.46, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.62, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.78, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.94, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -1.10, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10],
                # Hold forward
                *[[0.00, 0.00, 0.00, 0.00, -1.10, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10] for _ in range(10)],
                # Retract
                [0.00, 0.00, 0.00, 0.00, -0.94, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.78, 0.00, 0.00, 0.00, -1.10, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, -0.30, 0.00, 0.00, 0.00, -0.50, -0.15, 0.10, -0.10],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.15, 0.10, -0.10],
                # Final stabilize
                *[[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.15, 0.10, -0.10] for _ in range(5)],
            ]

            # Use directly - already in sim order!
            self.crawl_sequence = torch.tensor(crawl_seq, device=self.device, dtype=torch.float32)
            self.sequence_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            print(f"Loaded sequence with {len(crawl_seq)} steps (~{len(crawl_seq) * 0.02:.1f}s per loop)")
            print(f"Sequence already in Isaac Sim order - no conversion needed!")

        # Cache foot indices
        self.foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        foot_indices_list = []
        for name in self.foot_names:
            body_indices, matched_names = self._robot.find_bodies(name)
            if len(body_indices) == 0:
                raise RuntimeError(f"Foot body '{name}' not found! Available: {self._robot.body_names}")
            foot_indices_list.append(body_indices[0])
        self.foot_indices = torch.tensor(foot_indices_list, device=self.device, dtype=torch.long)

        print("Cached foot indices:", self.foot_indices)
        print("Joint names and indices:")
        for i, name in enumerate(self._robot.joint_names):
            print(f"  {i:2d}: {name}")

        # YOUR DEBUG BUFFERS
        self.debug_joint_history = []
        self.debug_obs_stats = {"min": [], "max": [], "mean": [], "std": []}

    def _setup_scene(self):
        self._robot = self.scene["robot"]

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions with per-joint group clamping OR use calibration sequence."""

        # CALIBRATION MODE: Override policy actions with hardcoded sequence
        if self.CALIBRATION_MODE:
            # Get current sequence step delta for all envs
            delta = self.crawl_sequence[self.sequence_step]  # [num_envs, 12]

            # Target = stand + delta
            target_positions = self.stand_q.unsqueeze(0) + delta

            # Clamp to joint limits (Go1 real limits: hip ±0.8, thigh ±3.0, calf ±3.0)
            # Apply per-group clamping to match real robot capabilities
            clamped_target = target_positions.clone()
            clamped_target[:, 0:4] = torch.clamp(clamped_target[:, 0:4], -0.8, 0.8)  # Hips
            clamped_target[:, 4:8] = torch.clamp(clamped_target[:, 4:8], -3.0, 3.0)  # Thighs
            clamped_target[:, 8:12] = torch.clamp(clamped_target[:, 8:12], -3.0, 3.0)  # Calves
            self._target_positions = clamped_target

            # Apply to robot
            self._robot.set_joint_position_target(self._target_positions)

            # DETAILED MONITORING: Print state of multiple robots
            if self._global_step % 10 == 0:  # Print every 10 steps (0.2s)
                step_num = self.sequence_step[0].item()
                print(f"\n{'=' * 100}")
                print(f"[CALIBRATION] Step {self._global_step}, Sequence {step_num}/{len(self.crawl_sequence)}")
                print(f"{'=' * 100}")

                # Monitor robots 0, 5, 10 (if they exist)
                monitor_envs = [i for i in [0, 5, 10] if i < self.num_envs]

                for env_idx in monitor_envs:
                    print(f"\n--- Robot {env_idx} ---")

                    # Current sequence delta
                    curr_delta = delta[env_idx].cpu().numpy()
                    print(
                        f"Sequence delta: FL_hip={curr_delta[0]:.3f}, FL_thigh={curr_delta[4]:.3f}, FL_calf={curr_delta[8]:.3f}")

                    # Target vs Current joint positions
                    target = self._target_positions[env_idx].cpu().numpy()
                    current = self._robot.data.joint_pos[env_idx].cpu().numpy()
                    error = target - current

                    print(f"\nJoint Positions (Target / Current / Error):")
                    joint_names = ["FL_hip", "FR_hip", "RL_hip", "RR_hip",
                                   "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
                                   "FL_calf", "FR_calf", "RL_calf", "RR_calf"]

                    # Print in groups for readability
                    print("  Hips:")
                    for i in range(4):
                        print(f"    {joint_names[i]:8s}: {target[i]:+7.3f} / {current[i]:+7.3f} / {error[i]:+7.3f}")

                    print("  Thighs:")
                    for i in range(4, 8):
                        print(f"    {joint_names[i]:8s}: {target[i]:+7.3f} / {current[i]:+7.3f} / {error[i]:+7.3f}")

                    print("  Calves:")
                    for i in range(8, 12):
                        print(f"    {joint_names[i]:8s}: {target[i]:+7.3f} / {current[i]:+7.3f} / {error[i]:+7.3f}")

                    # Joint velocities
                    velocities = self._robot.data.joint_vel[env_idx].cpu().numpy()
                    print(f"\nJoint Velocities (rad/s):")
                    print(f"  FL: hip={velocities[0]:+6.2f}, thigh={velocities[4]:+6.2f}, calf={velocities[8]:+6.2f}")
                    print(f"  FR: hip={velocities[1]:+6.2f}, thigh={velocities[5]:+6.2f}, calf={velocities[9]:+6.2f}")
                    print(f"  RL: hip={velocities[2]:+6.2f}, thigh={velocities[6]:+6.2f}, calf={velocities[10]:+6.2f}")
                    print(f"  RR: hip={velocities[3]:+6.2f}, thigh={velocities[7]:+6.2f}, calf={velocities[11]:+6.2f}")

                    # Body pose
                    base_pos = self._robot.data.root_pos_w[env_idx].cpu().numpy()
                    base_quat = self._robot.data.root_quat_w[env_idx].cpu().numpy()
                    base_lin_vel = self._robot.data.root_lin_vel_w[env_idx].cpu().numpy()
                    base_ang_vel = self._robot.data.root_ang_vel_w[env_idx].cpu().numpy()

                    print(f"\nBody State:")
                    print(f"  Position: x={base_pos[0]:+6.3f}, y={base_pos[1]:+6.3f}, z={base_pos[2]:+6.3f}")
                    print(f"  Orientation (quat): w={base_quat[0]:+6.3f}, x={base_quat[1]:+6.3f}, "
                          f"y={base_quat[2]:+6.3f}, z={base_quat[3]:+6.3f}")
                    print(f"  Lin Vel: x={base_lin_vel[0]:+6.3f}, y={base_lin_vel[1]:+6.3f}, z={base_lin_vel[2]:+6.3f}")
                    print(f"  Ang Vel: x={base_ang_vel[0]:+6.3f}, y={base_ang_vel[1]:+6.3f}, z={base_ang_vel[2]:+6.3f}")

                    # Foot contact forces (if available)
                    if hasattr(self._robot.data, 'contact_forces'):
                        contact = self._robot.data.contact_forces[env_idx].cpu().numpy()
                        print(f"\nContact Forces (N): FL={contact[0]:.1f}, FR={contact[1]:.1f}, "
                              f"RL={contact[2]:.1f}, RR={contact[3]:.1f}")

                print(f"\n{'=' * 100}\n")

            # Advance sequence for all envs
            self.sequence_step += 1
            # Loop when reaching end
            self.sequence_step = torch.where(
                self.sequence_step >= len(self.crawl_sequence),
                torch.zeros_like(self.sequence_step),
                self.sequence_step
            )

            # Update action history with dummy values for compatibility
            self._prev_prev_actions = self._previous_actions.clone()
            self._previous_actions = self._actions.clone()
            self._actions = torch.zeros_like(actions)  # Dummy

            return

        # NORMAL MODE: Use policy actions
        clamped_actions = actions.clone()

        # Per-group clamping
        clamped_actions[:, 0:4] = torch.clamp(clamped_actions[:, 0:4], -0.8, 0.8)
        clamped_actions[:, 4:8] = torch.clamp(clamped_actions[:, 4:8], -1.2, 1.2)
        clamped_actions[:, 8:12] = torch.clamp(clamped_actions[:, 8:12], -1.6, 1.6)

        # Debug actions for env 0
        if self._global_step % 100 == 0 and self._global_step > 0:
            print(f"\n[ACTION DEBUG] Step {self._global_step}, Env 0:")
            print(f"  Raw policy output: {actions[0, :].cpu().numpy()}")
            print(f"  After clamp: {clamped_actions[0, :].cpu().numpy()}")
            print(f"  Default joint pos: {self._robot.data.default_joint_pos[0, :].cpu().numpy()}")
            target_pos = clamped_actions[0, :] + self._robot.data.default_joint_pos[0, :]
            print(f"  Target positions: {target_pos.cpu().numpy()}")
            print(f"  Current joint pos: {self._robot.data.joint_pos[0, :].cpu().numpy()}")
            print(f"  Joint velocities: {self._robot.data.joint_vel[0, :].cpu().numpy()}")

        # Update action history
        self._prev_prev_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = clamped_actions.clone()

        # Compute and apply target positions
        self._target_positions = self._actions + self._robot.data.default_joint_pos
        self._robot.set_joint_position_target(self._target_positions)

    def _apply_action(self):
        """Actions already applied in _pre_physics_step."""
        pass

    def _get_observations(self) -> dict:
        """
        Build observation for policy: base_obs (45D) + encoder_outs (19D) = 64D

        THIS IS THE CORRECT HIMLOCO APPROACH:
        - base_obs contains raw proprioceptive info (45D)
        - encoder processes history and outputs learned representations (19D)
        - policy gets BOTH: base + learned features = 64D total
        """
        # Get robot state
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        dof_pos = self._robot.data.joint_pos
        dof_vel = self._robot.data.joint_vel

        # Get commands
        try:
            commands = self.command_manager.get_command("__default__")
        except:
            commands = self.command_manager.command

        # Build 45D base observation (matching official HIMLoco)
        base_obs = torch.cat([
            commands[:, :3],  # 3: vx, vy, yaw_rate
            dof_pos - self._robot.data.default_joint_pos,  # 12: joint pos error
            dof_vel,  # 12: joint velocities
            base_ang_vel,  # 3: angular velocity
            projected_gravity,  # 3: gravity vector
            self._previous_actions,  # 12: previous actions
        ], dim=-1)  # Total: 45 dims

        # Update history
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = base_obs

        # Encode history to get learned representations (19D: 3 vel + 16 pos)
        source_history = self.obs_history[:, :-1].reshape(self.num_envs, -1)  # [N, 225]

        # Encode WITH gradients so encoder can be trained via aux losses
        encoder_out = self.encoder_source(source_history)  # [N, 19]

        velocity_encoding = encoder_out[:, :3]  # [N, 3]
        position_encoding = encoder_out[:, 3:]  # [N, 16]

        # Concatenate: base_obs (45D) + velocity (3D) + position (16D) = 64D
        policy_obs = torch.cat([
            base_obs,  # 45D
            velocity_encoding,  # 3D
            position_encoding,  # 16D
        ], dim=-1)  # Total: 64D

        # Clip observations to prevent explosion (critical for PPO stability!)
        policy_obs = torch.clamp(policy_obs, -10.0, 10.0)

        # YOUR DEBUG PRINTS
        if self._global_step % 500 == 0 and self._global_step > 0:
            print(f"\n[OBS DEBUG] Step {self._global_step}")
            print(f"  base_obs shape: {base_obs.shape} (should be [N, 45])")
            print(f"  encoder_out shape: {encoder_out.shape} (should be [N, 19])")
            print(f"  policy_obs shape: {policy_obs.shape} (should be [N, 64])")
            print(f"  obs min/max: {policy_obs.min().item():.3f} / {policy_obs.max().item():.3f}")
            print(f"  obs mean/std: {policy_obs.mean().item():.3f} / {policy_obs.std().item():.3f}")
            print(f"  commands: {commands[0, :2]}")
            self.debug_obs_stats["min"].append(policy_obs.min().item())
            self.debug_obs_stats["max"].append(policy_obs.max().item())
            self.debug_obs_stats["mean"].append(policy_obs.mean().item())
            self.debug_obs_stats["std"].append(policy_obs.std().item())

        # Return 64D observation to policy
        return {"policy": policy_obs}

    def _get_aux_losses(self):
        """Compute HIMLoco auxiliary losses."""
        if not self.training:
            return {}

        # Clone obs_history to remove inference mode
        obs_history_clone = self.obs_history.clone()

        # Encode history WITH gradients
        source_history = obs_history_clone[:, :-1].reshape(self.num_envs, -1)  # [N, 225]
        target_history = obs_history_clone[:, 1:].reshape(self.num_envs, -1)  # [N, 225]

        # Check for NaN in input
        if torch.isnan(source_history).any():
            print("[ERROR] NaN detected in source_history!")
            return {}

        emb_source = self.encoder_source(source_history)  # [N, 19]
        emb_target = self.encoder_target(target_history)  # [N, 19]

        # Check embeddings
        if torch.isnan(emb_source).any():
            print("[ERROR] NaN in emb_source after encoder!")
            return {}

        # Split embeddings
        v_hat = emb_source[:, :3]
        l_s = emb_source[:, 3:]
        l_t = emb_target[:, 3:]

        # Ground truth velocity
        true_vel = torch.cat([
            self._robot.data.root_lin_vel_b[:, :2],
            self._robot.data.root_ang_vel_b[:, 2:3]
        ], dim=1)

        # Check true_vel
        if torch.isnan(true_vel).any() or torch.isinf(true_vel).any():
            print(f"[ERROR] NaN/Inf in true_vel!")
            return {}

        # 1. Velocity loss
        loss_vel = self.vel_loss_fn(v_hat, true_vel)

        if torch.isnan(loss_vel) or torch.isinf(loss_vel):
            print(f"[ERROR] NaN/Inf in loss_vel!")
            return {}

        # 2. SwAV loss (simplified and stable version)
        try:
            # Normalize embeddings
            l_s_norm = F.normalize(l_s, dim=1, eps=1e-6)
            l_t_norm = F.normalize(l_t, dim=1, eps=1e-6)

            # Normalize prototypes
            prototypes_norm = F.normalize(self.prototypes, dim=1, eps=1e-6)

            # Compute cosine similarities
            sim_s = l_s_norm @ prototypes_norm.t()  # [N, 16]
            sim_t = l_t_norm @ prototypes_norm.t()  # [N, 16]

            # Temperature scaling
            tau = 0.1
            logits_s = sim_s / tau
            logits_t = sim_t / tau

            # Soft cross-entropy (simplified SwAV without Sinkhorn)
            # This is more stable and still provides contrastive learning
            loss_swav = 0.5 * (
                    F.cross_entropy(logits_s, F.softmax(logits_t.detach(), dim=-1), reduction='none').mean() +
                    F.cross_entropy(logits_t, F.softmax(logits_s.detach(), dim=-1), reduction='none').mean()
            )

            if torch.isnan(loss_swav) or torch.isinf(loss_swav):
                print(f"[WARNING] NaN in SwAV, using zero")
                loss_swav = torch.tensor(0.0, device=self.device, requires_grad=True)

        except Exception as e:
            print(f"[WARNING] SwAV error: {e}, using zero")
            loss_swav = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Log
        if self._global_step % 150 == 0 and self._global_step > 0:
            print(
                f"[AUX SUMMARY] step {self._global_step} | loss_vel: {loss_vel.item():.4f} | loss_swav: {loss_swav.item():.4f}")

        return {
            "loss_vel": loss_vel * 1.0,
            "loss_swav": loss_swav * 0.5,  # Weight 0.5 as in paper
        }

    def _get_rewards(self) -> torch.Tensor:
        """Compute dense reward with survival incentive."""
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

        actuator_cfg = self._robot.cfg.actuators["legs"]
        stiffness = torch.full((self.num_envs, 12), actuator_cfg.stiffness, device=self.device)
        damping = torch.full((self.num_envs, 12), actuator_cfg.damping, device=self.device)
        pos_error = self._target_positions - dof_pos
        dof_torque_approx = stiffness * pos_error - damping * dof_vel

        sigma = 0.25

        # Tracking rewards (reduced from 15.0)
        r_lin_vel = torch.exp(-torch.sum((base_lin_vel[:, :2] - commands[:, :2]) ** 2, dim=1) / (2 * sigma ** 2)) * 2.0
        r_ang_vel = torch.exp(-(base_ang_vel[:, 2] - commands[:, 2]) ** 2 / sigma) * 1.0

        # Physics penalties (light)
        r_lin_vel_z = -(base_lin_vel[:, 2] ** 2) * 2.0
        r_ang_vel_xy = -(torch.sum(base_ang_vel[:, :2] ** 2, dim=1)) * 0.05
        r_orientation = -(torch.sum(projected_gravity[:, :2] ** 2, dim=1)) * 1.0  # Increased from 0.2
        r_joint_acc = -torch.sum(dof_acc ** 2, dim=1) * 2.5e-7
        r_joint_power = -torch.sum(torch.abs(dof_torque_approx) * torch.abs(dof_vel), dim=1) * 2e-5

        # Base height reward (strongly encourage correct height)
        r_base_height = -((base_height - 0.30) ** 2) * 10.0  # Target 0.30m, weight 10.0

        r_lateral_vel = -(base_lin_vel[:, 1] ** 2) * 2.0  # Increased from 1.5
        r_alive = torch.ones(self.num_envs, device=self.device) * 0.5

        # Movement rewards/penalties
        vel_norm_xy = torch.norm(base_lin_vel[:, :2], dim=1)
        r_no_movement = -5.0 * (vel_norm_xy < 0.1).float()

        r_foot_clearance = torch.zeros(self.num_envs, device=self.device)
        r_action_rate = -torch.sum((self._actions - self._previous_actions) ** 2, dim=1) * 0.01  # Increased from 0.005
        r_smoothness = -torch.sum((self._actions - 2 * self._previous_actions + self._prev_prev_actions) ** 2,
                                  dim=1) * 0.01  # Increased
        r_backward_penalty = -2.0 * (base_lin_vel[:, 0] < -0.1).float()

        total_reward = (
                r_lin_vel + r_ang_vel + r_lin_vel_z + r_ang_vel_xy +
                r_orientation + r_joint_acc + r_joint_power + r_base_height +
                r_foot_clearance + r_action_rate + r_smoothness + r_lateral_vel +
                r_alive + r_no_movement + r_backward_penalty
        )

        reward_terms = [
            r_lin_vel, r_ang_vel, r_lin_vel_z, r_ang_vel_xy,
            r_orientation, r_joint_acc, r_joint_power, r_base_height,
            r_foot_clearance, r_action_rate, r_smoothness, r_lateral_vel,
            r_alive, r_no_movement, r_backward_penalty
        ]
        reward_keys = [
            "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy",
            "orientation", "joint_acc", "joint_power", "base_height",
            "foot_clearance", "action_rate", "smoothness", "r_lateral_vel",
            "r_alive", "r_no_movement", "r_backward_penalty"
        ]
        for key, value in zip(reward_keys, reward_terms):
            self._episode_sums[key] += value

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination with debugging."""

        # CALIBRATION MODE: Never terminate, let sequence loop
        if self.CALIBRATION_MODE:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return terminated, truncated

        # NORMAL MODE: Standard termination conditions
        gravity = self._robot.data.projected_gravity_b
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        base_height = self._robot.data.root_pos_w[:, 2]

        # Termination conditions
        tipped = roll_pitch > 1.5  # ~86 degrees
        too_low = base_height < 0.15  # INCREASED from 0.08 - force standing!
        upside_down = gravity[:, 2] > 0.5  # Truly upside down

        # Debug env 0 every 100 steps
        if self._global_step % 100 == 0 and self._global_step > 0:
            print(f"\n[TERMINATION DEBUG] Step {self._global_step}, Env 0:")
            print(f"  Base height: {base_height[0].item():.3f} (min 0.15)")
            print(f"  Gravity z: {gravity[0, 2].item():.3f} (should be ~-1.0, upside_down if > 0.5)")
            print(f"  Roll/pitch magnitude: {roll_pitch[0].item():.3f} (terminate if > 1.5)")
            print(f"  Episode length: {self.episode_length_buf[0].item()}")
            print(f"  Terminated flags: tipped={tipped[0]}, too_low={too_low[0]}, upside_down={upside_down[0]}")

        poor_tracking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._global_step > 500000:
            steps = self.episode_length_buf.float() + 1e-6
            poor_tracking = (self._episode_sums["tracking_lin_vel"] / steps) < 0.8

        terminated = tipped | too_low | poor_tracking | upside_down
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments (your original)."""
        if len(env_ids) == 0:
            return

        if len(env_ids) > 0 and self._global_step % 200 == 0:
            print(f"\n=== Episode End Reward Summary (Step {self._global_step}) ===")
            for key, value in self._episode_sums.items():
                avg = value[env_ids].mean().item() if len(env_ids) > 0 else 0.0
                print(f"{key:20}: {avg:.3f}")
            print("=" * 50 + "\n")

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

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos += (torch.rand_like(joint_pos) - 0.5) * 0.2

        # Get terrain origins (check if _terrain exists)
        if hasattr(self, '_terrain'):
            env_origins = self._terrain.env_origins[env_ids]
        else:
            env_origins = torch.zeros(len(env_ids), 3, device=self.device)

        # CALIBRATION MODE: Hold robot in air at 0.5m height, zero velocity
        if self.CALIBRATION_MODE:
            root_state = self._robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += env_origins
            root_state[:, 2] = 0.5  # Hold at 0.5m height
            root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # No rotation
            root_state[:, 7:] = 0.0  # Zero velocity
            self._robot.write_root_state_to_sim(root_state, env_ids)

            # Reset sequence counter
            if hasattr(self, 'sequence_step'):
                self.sequence_step[env_ids] = 0
        else:
            # NORMAL MODE: Standard reset on ground
            root_state = self._robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += env_origins
            self._robot.write_root_state_to_sim(root_state, env_ids)
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] = 0.35
        root_state[:, 10] = -0.5 + torch.rand(len(env_ids), device=self.device) * -0.5

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def step(self, action):
        """Override to increment global counter."""
        self._global_step += 1
        return super().step(action)