# go1_env.py - FIXED HIMLoco port for Isaac Lab 2.2.1 (with Obs Debug Prints)
# All 11 rewards with torque approximation for joint power
import torch
from isaaclab.envs import DirectRLEnv
from .go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg


class Go1Env(DirectRLEnv):
    """
    Exact HIMLoco port for Isaac Lab 2.2.1
    Paper: https://openreview.net/pdf?id=93LoCyww8o
    → ALL 11 reward terms from Appendix A.1 Table 5 (with implicit torque approx)
    """
    cfg: Go1FlatEnvCfg | Go1RoughEnvCfg

    def __init__(self, cfg: Go1FlatEnvCfg | Go1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "=" * 80)
        print("Go1 HIMLoco EXACT 11-Reward Port - Isaac Lab 2.2.1 (with Obs Debug)")
        print("=" * 80)
        print(f"Observation: {self.cfg.observation_space.shape[0]}D (49×3 history)")
        print(f"Action: {self.cfg.action_space.shape[0]}D (joint position offsets)")
        print(f"Rewards: ALL 11 terms from HIMLoco Table 5 (implicit torque approx)")
        print("=" * 80 + "\n")

        # Action history buffers (for action_rate and smoothness)
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._prev_prev_actions = torch.zeros_like(self._actions)  # for smoothness

        # Target joint positions
        self._target_positions = torch.zeros(self.num_envs, 12, device=self.device)

        # Command manager
        from isaaclab.envs.mdp.commands import UniformVelocityCommand
        self.command_manager = UniformVelocityCommand(cfg=self.cfg.commands, env=self)

        # Observation history: 49D base × 3 steps → 147D (update to 5 for paper match)
        self.history_length = self.cfg.history_length
        self.base_obs_dim = 49
        self.obs_history = torch.zeros(
            self.num_envs, self.history_length, self.base_obs_dim, device=self.device
        )

        # Episode reward tracking (11 terms)
        self._episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in [
            "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy",
            "orientation", "joint_acc", "joint_power", "base_height",
            "foot_clearance", "action_rate", "smoothness"
        ]}

        self._global_step = 0
        self.feet_indices = None
        self._printed_structure = False

    def _setup_scene(self):
        """Initialize scene components"""
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.clamp(actions, -1.0, 1.0)
        # Shift action history
        self._prev_prev_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()

        self._target_positions = self.cfg.action_scale * actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._target_positions)
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:
        gravity = self._robot.data.projected_gravity_b
        angular_vel = self._robot.data.root_ang_vel_b
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        foot_contacts = self._get_foot_contact_binary()

        try:
            commands = self.command_manager.get_command("__default__")
        except:
            commands = self.command_manager.command

        base_obs = torch.cat([
            commands[:, :3],           # 3: commands
            joint_pos,                 # 12: joint pos
            joint_vel,                 # 12: joint vel
            angular_vel,               # 3: ang vel
            gravity,                   # 3: gravity
            self._previous_actions,    # 12: prev actions
            foot_contacts,             # 4: contacts
        ], dim=-1)  # → 49D current frame

        # Update FIFO history buffer
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = base_obs

        obs = self.obs_history.reshape(self.num_envs, -1)  # Flattened history (e.g., 147D)

        # Debug prints for obs input to PPO (every 500 steps, env 0)
        if self._global_step % 500 == 0:
            env_idx = 0
            print(f"\n{'-'*60}")
            print(f"Obs Debug | Step {self._global_step} | Env {env_idx}")
            print(f"{'-'*60}")
            print(f"Full Obs Shape: {obs.shape} (num_envs x flattened history)")
            print(f"Base Obs (Current Frame) Sample: {base_obs[env_idx, :10].cpu().numpy()}... (first 10 elems)")
            print(f"History Snippet: {self.obs_history[env_idx, :, :5].cpu().numpy()}... (first 5 elems per frame)")
            print(f"{'-'*60}\n")

        return {"policy": obs, "critic": obs}

    def _get_foot_contact_binary(self) -> torch.Tensor:
        if self.feet_indices is None:
            self.feet_indices = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])[0]
            if not self._printed_structure:
                print(f"Foot indices: {self.feet_indices} (FL, FR, RL, RR)")
                self._printed_structure = True

        foot_forces = torch.norm(
            self._contact_sensor.data.net_forces_w[:, self.feet_indices], dim=-1
        )
        return (foot_forces > 5.0).float()

    def _get_rewards(self) -> torch.Tensor:
        if self.feet_indices is None:
            self.feet_indices = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])[0]

        # === State variables ===
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        base_height = self._robot.data.root_pos_w[:, 2]
        projected_gravity = self._robot.data.projected_gravity_b
        dof_acc = self._robot.data.joint_acc
        dof_pos = self._robot.data.joint_pos  # Current joint positions
        dof_vel = self._robot.data.joint_vel  # Current joint velocities

        # Foot data
        foot_pos_z_rel = self._robot.data.body_pos_w[:, self.feet_indices, 2] - base_height.unsqueeze(1)
        foot_vel_xy = torch.norm(self._robot.data.body_lin_vel_w[:, self.feet_indices, :2], dim=-1)

        # Commands
        try:
            commands = self.command_manager.get_command("__default__")
        except:
            commands = self.command_manager.command

        sigma = 0.25

        # === APPROXIMATE TORQUES for Implicit Actuators (from robot cfg) ===
        # Access actuator params from cfg (scalars)
        actuator_cfg = self._robot.cfg.actuators["legs"]
        stiffness = torch.full((self.num_envs, 12), actuator_cfg.stiffness, device=self.device)
        damping = torch.full((self.num_envs, 12), actuator_cfg.damping, device=self.device)

        # τ ≈ stiffness * (target_pos - current_pos) - damping * current_vel
        pos_error = self._target_positions - dof_pos
        dof_torque_approx = stiffness * pos_error - damping * dof_vel

        # === ALL 11 REWARDS – EXACTLY AS IN HIMLoco TABLE 5 ===
        r_lin_vel = torch.exp(-torch.sum((base_lin_vel[:, :2] - commands[:, :2])**2, dim=1) / (2 * sigma**2)) * 1.0
        r_ang_vel = torch.exp(-(base_ang_vel[:, 2] - commands[:, 2])**2 / sigma) * 0.5
        r_lin_vel_z = -(base_lin_vel[:, 2]**2) * 2.0
        r_ang_vel_xy = -(torch.sum(base_ang_vel[:, :2]**2, dim=1) / 2) * 0.05
        r_orientation = -(torch.sum(projected_gravity[:, :2]**2, dim=1) / 2) * 0.2
        r_joint_acc = -torch.sum(dof_acc**2, dim=1) * 2.5e-7
        r_joint_power = -torch.sum(torch.abs(dof_torque_approx) * torch.abs(dof_vel), dim=1) * 2e-5
        r_base_height = -((base_height - 0.40)**2) * 1.0          # Go1 standing height ≈ 0.40 m
        r_foot_clearance = -torch.sum((0.05 - foot_pos_z_rel)**2 * foot_vel_xy, dim=1) * 0.01
        r_action_rate = -(torch.sum((self._actions - self._previous_actions)**2, dim=1) / 2) * 0.01
        r_smoothness = -(torch.sum((self._actions - 2*self._previous_actions + self._prev_prev_actions)**2, dim=1) / 2) * 0.01

        total_reward = (r_lin_vel + r_ang_vel + r_lin_vel_z + r_ang_vel_xy +
                        r_orientation + r_joint_acc + r_joint_power + r_base_height +
                        r_foot_clearance + r_action_rate + r_smoothness)

        # Accumulate for logging
        rewards_dict = {
            "tracking_lin_vel": r_lin_vel,
            "tracking_ang_vel": r_ang_vel,
            "lin_vel_z": r_lin_vel_z,
            "ang_vel_xy": r_ang_vel_xy,
            "orientation": r_orientation,
            "joint_acc": r_joint_acc,
            "joint_power": r_joint_power,
            "base_height": r_base_height,
            "foot_clearance": r_foot_clearance,
            "action_rate": r_action_rate,
            "smoothness": r_smoothness,
        }
        for k, v in rewards_dict.items():
            self._episode_sums[k] += v

        # Debug print every 500 steps (rewards, for completeness)
        if self._global_step % 500 == 0:
            env_idx = 0
            print(f"\n{'='*80}")
            print(f"Step {self._global_step} | Env {env_idx}")
            print(f"{'='*80}")
            print(f"Height: {base_height[env_idx]:.3f}m (target 0.40m)")
            print(f"Vel XY: [{base_lin_vel[env_idx,0]:.2f}, {base_lin_vel[env_idx,1]:.2f}]  Cmd: [{commands[env_idx,0]:.2f}, {commands[env_idx,1]:.2f}]")
            print(f"LinVel reward: {r_lin_vel[env_idx]:.3f}  |  Total: {total_reward[env_idx]:.3f}")
            print(f"{'='*80}\n")

        self._global_step += 1
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        gravity = self._robot.data.projected_gravity_b
        roll_pitch = torch.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)
        base_height = self._robot.data.root_pos_w[:, 2]

        tipped = roll_pitch > 0.9
        too_low = base_height < 0.18
        died = tipped | too_low
        return died, timeout

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        self.command_manager.reset(env_ids)

        # Reset buffers
        self.obs_history[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0
        self._target_positions[env_ids] = self._robot.data.default_joint_pos[env_ids]

        # Reset pose
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] = 0.42

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset episode sums
        for k in self._episode_sums:
            self._episode_sums[k][env_ids] = 0.0