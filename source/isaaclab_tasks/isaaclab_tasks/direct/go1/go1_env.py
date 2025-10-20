import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Camera
from isaaclab.utils.math import quat_apply, quat_mul
from .go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg


class Go1Env(DirectRLEnv):
    cfg: Go1FlatEnvCfg | Go1RoughEnvCfg

    def __init__(self, cfg: Go1FlatEnvCfg | Go1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print("🤖 Go1 Environment - Using Automatic Camera Tracking")

        # Initialize action buffers
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)

        # Episode sums for rewards
        self._episode_sums = {
            "leg_contact_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "imu_upright_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "height_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        # Robot parameters
        self._target_height = 0.32
        self._global_step = 0

        # Camera parameters for manual pose computation
        self._cam_local_pos = torch.tensor([0.25, 0.0, 0.12], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self._cam_local_rot = torch.tensor([0.5, -0.5, 0.5, -0.5], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # Already converted in cfg

    def _setup_scene(self):
        """Setup scene - camera will automatically follow robot as child prim"""
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]
        self._camera = self.scene["camera"]

        print("🔧 Scene Setup Complete:")
        print(f"   Robot prim_path: {self._robot.cfg.prim_path}")
        print(f"   Camera prim_path: {self._camera.cfg.prim_path}")
        print("✅ Camera is child of robot trunk - will follow automatically")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step"""
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        """Apply actions - NO manual camera tracking needed"""
        self._robot.set_joint_position_target(self._processed_actions)
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:
        """Get observations for policy"""
        # Store previous actions
        self._previous_actions = self._actions.clone()

        # Get IMU data
        gravity, angular_vel = self._get_imu_data()

        # Get foot contact data
        foot_contacts = self._get_foot_contact_binary()

        # Get joint data (normalized)
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        # Get robot height
        height = self._robot.data.root_pos_w[:, 2:3]

        # Combine all state observations (47 dims)
        state = torch.cat([
            gravity,  # 3 dimensions
            angular_vel,  # 3 dimensions
            foot_contacts,  # 4 dimensions (4 feet)
            joint_pos,  # 12 dimensions
            joint_vel,  # 12 dimensions
            height,  # 1 dimension
            self._actions  # 12 dimensions
        ], dim=-1)

        # Get camera RGB data (num_envs, H, W, 3) and normalize to [0,1]
        rgb = self._camera.data.output["rgb"].float() / 255.0

        # Flatten RGB to (num_envs, H*W*3)
        rgb_flat = rgb.flatten(start_dim=1)

        # Combine state and flattened RGB
        obs = torch.cat([state, rgb_flat], dim=-1)

        # Manually update camera data poses for debug (since XFormPrimView may not update correctly for child prims)
        cam_offset_world = quat_apply(self._robot.data.root_quat_w, self._cam_local_pos)
        self._camera.data.pos_w = self._robot.data.root_pos_w + cam_offset_world
        self._camera.data.quat_w_world = quat_mul(self._robot.data.root_quat_w, self._cam_local_rot)

        # Debug camera tracking every 200 steps
        if self._global_step % 200 == 0:
            self._debug_camera_tracking()

        return {"policy": obs}

    def _debug_camera_tracking(self):
        """Debug camera tracking - verify parent-child relationship"""
        try:
            if (hasattr(self._robot, 'data') and hasattr(self._camera, 'data') and
                    self._robot.data.root_pos_w is not None and self._camera.data.pos_w is not None):

                env_idx = 0
                robot_pos = self._robot.data.root_pos_w[env_idx]
                camera_pos = self._camera.data.pos_w[env_idx]
                robot_rot = self._robot.data.root_quat_w[env_idx]

                # Calculate expected position based on offset
                cam_offset_local = torch.tensor([0.25, 0.0, 0.12], device=self.device)
                cam_offset_world = quat_apply(robot_rot.unsqueeze(0), cam_offset_local.unsqueeze(0)).squeeze(0)
                expected_pos = robot_pos + cam_offset_world

                # Calculate errors
                current_error = torch.norm(camera_pos - expected_pos).item()
                distance = torch.norm(robot_pos - camera_pos).item()

                print(f"\n📊 Step {self._global_step} - Camera Tracking (AUTO):")
                print(f"   Update latest camera pose: {self._camera.cfg.update_latest_camera_pose}")
                print(f"   Camera frame count: {self._camera._frame}")
                print(f"   PRIM PATHS:")
                print(f"   🤖 Robot:  {self._robot.cfg.prim_path}")
                print(f"   📷 Camera: {self._camera.cfg.prim_path}")
                print(f"   POSITIONS:")
                print(f"   🤖 Robot:    [{robot_pos[0]:6.2f}, {robot_pos[1]:6.2f}, {robot_pos[2]:6.3f}]")
                print(f"   📷 Camera (data):   [{camera_pos[0]:6.2f}, {camera_pos[1]:6.2f}, {camera_pos[2]:6.3f}]")
                print(f"   🎯 Expected: [{expected_pos[0]:6.2f}, {expected_pos[1]:6.2f}, {expected_pos[2]:6.3f}]")
                print(f"   METRICS:")
                print(f"   📏 Robot-Camera Distance: {distance:6.3f}m")
                print(f"   ❌ Tracking Error:        {current_error:6.3f}m")

                # Quality assessment
                if current_error < 0.01:
                    print("   ✅ EXCELLENT: Auto tracking working perfectly!")
                elif current_error < 0.05:
                    print("   ✅ GOOD: Auto tracking with minimal error")
                elif current_error < 0.1:
                    print("   ⚠️  ACCEPTABLE: Auto tracking with some error")
                else:
                    print("   ❌ POOR: Auto tracking not working")
                    print("   🔧 TROUBLESHOOTING: Check camera parent-child relationship in USD")

        except Exception as e:
            print(f"📷 Debug error: {e}")

    def _get_foot_contact_binary(self) -> torch.Tensor:
        """Get foot contact state as binary tensor"""
        try:
            if not hasattr(self._contact_sensor.data, 'net_forces_w'):
                return torch.zeros(self.num_envs, 4, device=self.device)

            contact_forces = torch.norm(self._contact_sensor.data.net_forces_w, dim=-1)
            binary_contact = (contact_forces > 1.0).float()

            if binary_contact.shape[1] >= 4:
                return binary_contact[:, :4]
            else:
                return torch.zeros(self.num_envs, 4, device=self.device)

        except Exception:
            return torch.zeros(self.num_envs, 4, device=self.device)

    def _get_imu_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get IMU data (gravity vector and angular velocity)"""
        return self._robot.data.projected_gravity_b, self._robot.data.root_ang_vel_b

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards for the current step"""
        contacts = self._get_foot_contact_binary()
        contact_reward = (torch.sum(contacts, dim=1) / 4.0) * 2.0

        gravity, _ = self._get_imu_data()
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        upright_reward = torch.exp(-roll_pitch * 5.0) * 2.0

        height = self._robot.data.root_pos_w[:, 2]
        height_error = torch.abs(height - self._target_height)
        height_reward = torch.exp(-height_error * 4.0) * 2.0

        total_reward = contact_reward + upright_reward + height_reward

        self._episode_sums["leg_contact_reward"] += contact_reward
        self._episode_sums["imu_upright_reward"] += upright_reward
        self._episode_sums["height_reward"] += height_reward

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination conditions"""
        timeout = self.episode_length_buf >= self.max_episode_length - 1

        gravity, _ = self._get_imu_data()
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        height = self._robot.data.root_pos_w[:, 2]

        died = torch.logical_or(roll_pitch > 0.8, height < 0.1)

        return died, timeout

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments"""
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root_state = self._robot.data.default_root_state[env_ids].clone()

        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] = 0.35

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._robot.reset(env_ids)

        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0

        self.scene.write_data_to_sim()

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment with camera tracking"""
        result = super().step(action)
        self._global_step += 1
        return result