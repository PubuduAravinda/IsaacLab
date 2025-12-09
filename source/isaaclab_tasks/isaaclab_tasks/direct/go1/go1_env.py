# go1_env.py - HIMLoco port for Isaac Lab 2.2.1 (FIXED)
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply
from .go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg


class Go1Env(DirectRLEnv):
    """
    HIMLoco-style Go1 environment for Isaac Lab 2.2.1

    Implements proprioceptive-only walking with:
    - 147D observations (3-step history of 49D base obs)
    - 12D actions (joint position offsets)
    - Reward terms from HIMLoco paper
    """
    cfg: Go1FlatEnvCfg | Go1RoughEnvCfg

    def __init__(self, cfg: Go1FlatEnvCfg | Go1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "=" * 80)
        print("🤖 Go1 HIMLoco Environment - Isaac Lab 2.2.1")
        print("=" * 80)
        print(f"📊 Observation space: {self.cfg.observation_space.shape[0]}D (history: {self.cfg.history_length})")
        print(f"🎯 Action space: {self.cfg.action_space.shape[0]}D")
        print(f"🏃 Policy frequency: {int(1 / (self.cfg.sim.dt * self.cfg.decimation))}Hz")
        print(f"⚙️  Simulation frequency: {int(1 / self.cfg.sim.dt)}Hz")
        print(f"🌍 Environments: {self.num_envs}")
        print("=" * 80 + "\n")

        # Action buffers
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._target_positions = torch.zeros(self.num_envs, 12, device=self.device)

        # Command manager - moved here before observation history
        from isaaclab.envs.mdp.commands import UniformVelocityCommand
        self.command_manager = UniformVelocityCommand(
            cfg=self.cfg.commands,
            env=self
        )

        # Observation history (49D base obs * history_length)
        self.history_length = self.cfg.history_length
        self.base_obs_dim = 49
        self.obs_history = torch.zeros(
            self.num_envs, self.history_length, self.base_obs_dim, device=self.device
        )

        # Reward tracking
        self._episode_sums = {
            "tracking_lin_vel": torch.zeros(self.num_envs, device=self.device),
            "tracking_ang_vel": torch.zeros(self.num_envs, device=self.device),
            "lin_vel_z": torch.zeros(self.num_envs, device=self.device),
            "ang_vel_xy": torch.zeros(self.num_envs, device=self.device),
            "orientation": torch.zeros(self.num_envs, device=self.device),
            "base_height": torch.zeros(self.num_envs, device=self.device),
            "action_rate": torch.zeros(self.num_envs, device=self.device),
            "foot_clearance": torch.zeros(self.num_envs, device=self.device),
            "dof_acc": torch.zeros(self.num_envs, device=self.device),
        }

        self._global_step = 0
        self.feet_indices = None  # Will be set after physics is initialized

    def _setup_scene(self):
        """Initialize scene components"""
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step"""
        # Clip and store actions
        actions = torch.clamp(actions, -1.0, 1.0)
        self._actions = actions.clone()

        # Convert to target joint positions (position control)
        self._target_positions = (
                self.cfg.action_scale * actions + self._robot.data.default_joint_pos
        )

    def _apply_action(self):
        """Apply position targets to robot"""
        self._robot.set_joint_position_target(self._target_positions)
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:
        """
        Construct HIMLoco observation:
        Base obs (49D): cmd(3) + joint_pos(12) + joint_vel(12) +
                        ang_vel(3) + gravity(3) + prev_actions(12) + contacts(4)
        History: Stack 3 most recent base obs → 147D
        """
        # Store previous actions for next step
        self._previous_actions = self._actions.clone()

        # Get proprioceptive data
        gravity = self._robot.data.projected_gravity_b
        angular_vel = self._robot.data.root_ang_vel_b
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        foot_contacts = self._get_foot_contact_binary()

        # Get velocity commands (handle both possible APIs)
        try:
            commands = self.command_manager.get_command("__default__")
        except:
            # Fallback for older command manager API
            commands = self.command_manager.command

        # Construct base observation (49D)
        base_obs = torch.cat([
            commands[:, :3],  # lin_vel_x, lin_vel_y, ang_vel_z (3)
            joint_pos,  # Joint positions (12)
            joint_vel,  # Joint velocities (12)
            angular_vel,  # Angular velocity (3)
            gravity,  # Projected gravity (3)
            self._previous_actions,  # Previous actions (12)
            foot_contacts,  # Binary foot contacts (4)
        ], dim=-1)

        # Update history buffer (FIFO)
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = base_obs

        # Flatten history for policy input (147D)
        obs = self.obs_history.reshape(self.num_envs, -1)

        # Return both policy and critic obs (symmetric - same for both)
        return {"policy": obs, "critic": obs}

    def _get_foot_contact_binary(self) -> torch.Tensor:
        """Binary foot contact detection"""
        # Initialize feet indices on first call (after physics is ready)
        if self.feet_indices is None:
            self.feet_indices = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])[0]
            print(f"✓ Foot indices initialized: {self.feet_indices}")

        force_threshold = 5.0

        # Get contact forces for feet
        foot_forces = torch.norm(
            self._contact_sensor.data.net_forces_w[:, self.feet_indices],
            dim=-1
        )

        # Binary contact (1 if force > threshold)
        binary_contact = (foot_forces > force_threshold).float()

        return binary_contact

    def _get_rewards(self) -> torch.Tensor:
        """
        HIMLoco reward terms (from paper Appendix)
        All terms use exponential/quadratic penalties
        """
        # Initialize feet indices on first call (after physics is ready)
        if self.feet_indices is None:
            self.feet_indices = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])[0]
            print(f"✓ Foot indices initialized: {self.feet_indices}")

        # Get current state
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        base_height = self._robot.data.root_pos_w[:, 2]
        projected_gravity = self._robot.data.projected_gravity_b
        dof_acc = self._robot.data.joint_acc

        # Foot positions and velocities
        foot_pos = self._robot.data.body_pos_w[:, self.feet_indices, 2] - base_height.unsqueeze(1)
        foot_vel_xy = torch.norm(
            self._robot.data.body_lin_vel_w[:, self.feet_indices, :2],
            dim=-1
        )

        # Get velocity commands (handle both possible APIs)
        try:
            commands = self.command_manager.get_command("__default__")
        except:
            commands = self.command_manager.command

        # === HIMLoco Reward Terms (weights from paper) ===

        # 1. Linear velocity tracking (weight: 1.0)
        tracking_lin_vel = torch.exp(
            -torch.sum((base_lin_vel[:, :2] - commands[:, :2]) ** 2, dim=1) / (2 * 0.25 ** 2)
        ) * 1.0

        # 2. Angular velocity tracking (weight: 0.5)
        tracking_ang_vel = torch.exp(
            -(base_ang_vel[:, 2] - commands[:, 2]) ** 2 / (2 * 0.25 ** 2)
        ) * 0.5

        # 3. Vertical velocity penalty (weight: 2.0)
        lin_vel_z = -(base_lin_vel[:, 2] ** 2) * 2.0

        # 4. Angular velocity XY penalty (weight: 0.05)
        ang_vel_xy = -torch.sum(base_ang_vel[:, :2] ** 2, dim=1) * 0.05

        # 5. Orientation penalty (weight: 0.2)
        orientation = -torch.sum(projected_gravity[:, :2] ** 2, dim=1) * 0.2

        # 6. Base height penalty (weight: 1.0, target: 0.32m)
        base_height_reward = -((base_height - 0.32) ** 2) * 1.0

        # 7. Foot clearance penalty (weight: 0.01)
        foot_clearance = -torch.sum(
            (0.05 - foot_pos) ** 2 * foot_vel_xy,
            dim=1
        ) * 0.01

        # 8. Joint acceleration penalty (weight: 2.5e-7)
        dof_acc_penalty = -torch.sum(dof_acc ** 2, dim=1) * 2.5e-7

        # 9. Action rate penalty (weight: 0.01)
        action_rate = -torch.sum(
            (self._actions - self._previous_actions) ** 2,
            dim=1
        ) * 0.01

        # Total reward
        total_reward = (
                tracking_lin_vel +
                tracking_ang_vel +
                lin_vel_z +
                ang_vel_xy +
                orientation +
                base_height_reward +
                foot_clearance +
                dof_acc_penalty +
                action_rate
        )

        # Accumulate episode statistics
        self._episode_sums["tracking_lin_vel"] += tracking_lin_vel
        self._episode_sums["tracking_ang_vel"] += tracking_ang_vel
        self._episode_sums["lin_vel_z"] += lin_vel_z
        self._episode_sums["ang_vel_xy"] += ang_vel_xy
        self._episode_sums["orientation"] += orientation
        self._episode_sums["base_height"] += base_height_reward
        self._episode_sums["action_rate"] += action_rate
        self._episode_sums["foot_clearance"] += foot_clearance
        self._episode_sums["dof_acc"] += dof_acc_penalty

        # Debug logging (every 500 steps)
        if self._global_step % 500 == 0:
            env_idx = 0
            print(f"\n{'=' * 80}")
            print(f"Step {self._global_step} | Env {env_idx}")
            print(f"{'=' * 80}")
            print(f"State:")
            print(f"  Height: {base_height[env_idx]:.3f}m (target: 0.32m)")
            print(f"  Velocity: [{base_lin_vel[env_idx, 0]:.2f}, {base_lin_vel[env_idx, 1]:.2f}]")
            print(f"  Command:  [{commands[env_idx, 0]:.2f}, {commands[env_idx, 1]:.2f}]")
            print(f"  Contacts: {self._get_foot_contact_binary()[env_idx].cpu().numpy()}")
            print(f"\nRewards:")
            print(f"  Lin vel track: {tracking_lin_vel[env_idx]:.3f}")
            print(f"  Ang vel track: {tracking_ang_vel[env_idx]:.3f}")
            print(f"  Orientation:   {orientation[env_idx]:.3f}")
            print(f"  Base height:   {base_height_reward[env_idx]:.3f}")
            print(f"  TOTAL:         {total_reward[env_idx]:.3f}")
            print(f"{'=' * 80}\n")

        self._global_step += 1
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions"""
        # Timeout
        timeout = self.episode_length_buf >= self.max_episode_length - 1

        # Failure conditions
        gravity = self._robot.data.projected_gravity_b
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)

        # Terminate if tipped over or too low
        low_height = self._robot.data.root_pos_w[:, 2] < 0.18
        tipped = roll_pitch > 0.9

        died = tipped | low_height

        return died, timeout

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments"""
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # Resample velocity commands
        self.command_manager.reset(env_ids)

        # Reset observation history
        self.obs_history[env_ids] = 0.0

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._target_positions[env_ids] = self._robot.data.default_joint_pos[env_ids]

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] = 0.35  # Start slightly elevated

        # Write state to simulation
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset episode sums
        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0