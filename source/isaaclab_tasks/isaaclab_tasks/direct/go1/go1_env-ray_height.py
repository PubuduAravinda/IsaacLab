import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse
from .go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg


class Go1Env(DirectRLEnv):
    cfg: Go1FlatEnvCfg | Go1RoughEnvCfg

    def __init__(self, cfg: Go1FlatEnvCfg | Go1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # actions and commands
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # logging - reward tracking
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "leg_contact_reward",
                "imu_upright_reward",
                "height_reward",
            ]
        }

        # Initialize body mappings
        self._base_id = None
        self._feet_ids = None
        self._undesired_contact_body_ids = None

        # Real Go1 sensor data
        self._foot_contact_binary = torch.zeros(self.num_envs, 4, device=self.device)

        # Height tracking
        self._target_height = 0.32  # Real Go1 standing height
        self._camera_offset = 0.1  # Camera is 10cm below trunk

    def _setup_scene(self):
        """Setup scene without camera complexity"""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Only add height scanner if it's a rough terrain config
        if isinstance(self.cfg, Go1RoughEnvCfg) and hasattr(self.cfg, 'height_scanner'):
            from isaaclab.sensors import RayCaster
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        print(f"✅ Scene setup complete for {self.num_envs} environments")

    def _get_height_estimate(self):
        """Get reliable height estimate using robot position + camera offset"""
        # Use robot trunk position as base (most reliable)
        trunk_height = self._robot.data.root_pos_w[:, 2]

        # Account for robot tilt using gravity projection
        gravity = self._robot.data.projected_gravity_b
        tilt_factor = torch.abs(gravity[:, 2])  # How upright the robot is (0=tilted, 1=upright)

        # When robot is upright, height is accurate
        # When tilted, effective height is slightly reduced
        effective_height = trunk_height * (0.95 + 0.05 * tilt_factor)

        # Simulate camera measurement: trunk height - camera offset
        camera_height_estimate = effective_height - self._camera_offset

        return torch.clamp(camera_height_estimate, min=0.05, max=1.0), trunk_height

    def _get_foot_contact_binary(self) -> torch.Tensor:
        """Get binary foot contact state (1=contact, 0=no contact)."""
        force_threshold = 5.0
        all_bodies, all_names = self._contact_sensor.find_bodies(".*")

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        feet_sensor_indices = []

        for foot_name in foot_names:
            for i, body_name in enumerate(all_names):
                if foot_name.lower() in body_name.lower():
                    feet_sensor_indices.append(i)
                    break

        binary_contact = torch.zeros(self.num_envs, 4, device=self.device)

        if feet_sensor_indices and len(feet_sensor_indices) == 4:
            foot_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, feet_sensor_indices], dim=-1)
            binary_contact = (foot_forces > force_threshold).float()

        return binary_contact

    def _get_imu_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get IMU data from robot."""
        # Gravity projection tells us robot orientation
        gravity = self._robot.data.projected_gravity_b

        # Angular velocity from base (gyroscope)
        angular_vel = self._robot.data.root_ang_vel_b

        return gravity, angular_vel

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards using reliable sensors only"""
        # 1. LEG CONTACT REWARD - Encourage stable stance
        contact_binary = self._get_foot_contact_binary()
        num_feet_contact = torch.sum(contact_binary, dim=1)
        leg_contact_reward = (num_feet_contact / 4.0) * 2.0

        # 2. IMU UPRIGHT REWARD - Keep robot upright
        gravity, angular_vel = self._get_imu_data()
        roll_pitch_error = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        imu_upright_reward = torch.exp(-roll_pitch_error * 5.0)

        # 3. HEIGHT REWARD - Maintain target standing height
        height_estimate, trunk_height = self._get_height_estimate()
        height_error = torch.abs(height_estimate - self._target_height)
        height_reward = torch.exp(-height_error * 4.0)

        # Training progress monitoring (minimal, informative)
        if self.episode_length_buf[0] % 200 == 0:
            env_idx = 0
            print(f"\n📊 TRAINING PROGRESS - Step {self.episode_length_buf[env_idx].item()}:")
            print(f"  🦵 Feet contact: {contact_binary[env_idx].tolist()}")
            print(f"  📏 Height: {trunk_height[env_idx].item():.3f}m (est: {height_estimate[env_idx].item():.3f}m)")
            print(f"  🎯 Target: {self._target_height}m")
            print(f"  📱 IMU upright: {imu_upright_reward[env_idx].item():.3f}")
            print(f"  💎 Rewards - Leg: {leg_contact_reward[env_idx].item():.3f}, "
                  f"Height: {height_reward[env_idx].item():.3f}")

        rewards = {
            "leg_contact_reward": leg_contact_reward * 1.0,
            "imu_upright_reward": imu_upright_reward * 2.0,
            "height_reward": height_reward * 2.0,
        }

        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Update episode sums
        for key, value in rewards.items():
            if not torch.isnan(value).any():
                self._episode_sums[key] += value

        return total_reward

    def _initialize_body_mappings(self):
        """Initialize body mappings after scene is set up"""
        try:
            all_bodies, all_names = self._contact_sensor.find_bodies(".*")

            self._base_id, base_names = self._contact_sensor.find_bodies("trunk")
            if self._base_id:
                print(f"✅ Base: {base_names}")

            self._feet_ids, feet_names = self._contact_sensor.find_bodies(".*_foot")
            if self._feet_ids and len(self._feet_ids) >= 4:
                print(f"✅ Feet ({len(self._feet_ids)}): {feet_names}")

            self._undesired_contact_body_ids, undesired_names = self._contact_sensor.find_bodies(".*_thigh")
            if self._undesired_contact_body_ids:
                print(f"✅ Undesired bodies ({len(self._undesired_contact_body_ids)}): {undesired_names}")

        except Exception as e:
            print(f"[BODY MAPPING ERROR] {e}")

    def _get_observations(self) -> dict:
        """Get observations from real Go1 sensors"""
        self._previous_actions = self._actions.clone()

        gravity, angular_vel = self._get_imu_data()
        foot_contacts = self._get_foot_contact_binary()

        # Joint state
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        # Height observation
        height_estimate, _ = self._get_height_estimate()
        height_obs = height_estimate.unsqueeze(1)

        # Observation components
        obs_components = [
            gravity,  # IMU accelerometer (3)
            angular_vel,  # IMU gyroscope (3)
            foot_contacts,  # Binary leg contacts (4)
            joint_pos,  # Joint positions (12)
            joint_vel,  # Joint velocities (12)
            height_obs,  # Height estimate (1)
            self._actions,  # Action history (12)
        ]

        obs = torch.cat(obs_components, dim=-1)

        return {"policy": obs}

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions to the robot"""
        action_scale = 0.5
        self._actions = actions.clone() * action_scale
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        """Send actions to simulation"""
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination conditions"""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        gravity, _ = self._get_imu_data()
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        died = roll_pitch > 0.8  # More than ~53 degrees tilt

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments"""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        if torch.any(env_ids >= self.num_envs) or torch.any(env_ids < 0):
            env_ids = env_ids[(env_ids < self.num_envs) & (env_ids >= 0)]
            if len(env_ids) == 0:
                return

        print(f"🔄 Resetting environments: {env_ids.cpu().tolist()}")

        try:
            super()._reset_idx(env_ids)
            self._robot.reset(env_ids)

            if len(env_ids) == self.num_envs:
                self.episode_length_buf[:] = torch.randint_like(
                    self.episode_length_buf, high=int(self.max_episode_length)
                )

            self._actions[env_ids] = 0.0
            self._previous_actions[env_ids] = 0.0
            self._foot_contact_binary[env_ids] = 0.0

            self._commands[env_ids] = torch.zeros_like(self._commands[env_ids])

            joint_pos = self._robot.data.default_joint_pos[env_ids]
            joint_vel = self._robot.data.default_joint_vel[env_ids]
            default_root_state = self._robot.data.default_root_state[env_ids]

            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            default_root_state[:, 2] = 0.35

            self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

            for key in self._episode_sums.keys():
                self._episode_sums[key][env_ids] = 0.0

            if self._feet_ids is None:
                self._initialize_body_mappings()

            print(f"✅ Successfully reset {len(env_ids)} environments")

        except Exception as e:
            print(f"❌ Reset failed: {e}")
            import traceback
            traceback.print_exc()

    def _post_physics_step(self):
        """Called after physics step"""
        super()._post_physics_step()

        # Periodic training summary
        if self.episode_length_buf[0] % 500 == 0:
            env_idx = 0
            print(f"\n" + "=" * 60)
            print(f"🏆 TRAINING SUMMARY - Step {self.episode_length_buf[env_idx].item()}")
            print("=" * 60)

            gravity, angular_vel = self._get_imu_data()
            height_estimate, trunk_height = self._get_height_estimate()

            print(f"🤖 ROBOT STATUS:")
            print(f"   Height: {trunk_height[env_idx].item():.3f}m (est: {height_estimate[env_idx].item():.3f}m)")
            print(f"   Gravity: [{gravity[env_idx, 0]:.3f}, {gravity[env_idx, 1]:.3f}, {gravity[env_idx, 2]:.3f}]")
            print(
                f"   Angular Vel: [{angular_vel[env_idx, 0]:.3f}, {angular_vel[env_idx, 1]:.3f}, {angular_vel[env_idx, 2]:.3f}]")

            print(f"\n📈 CUMULATIVE REWARDS:")
            for key in self._episode_sums.keys():
                avg_val = self._episode_sums[key][env_idx].item() / max(1, self.episode_length_buf[env_idx].item())
                print(f"   {key:20}: {avg_val:.3f}")