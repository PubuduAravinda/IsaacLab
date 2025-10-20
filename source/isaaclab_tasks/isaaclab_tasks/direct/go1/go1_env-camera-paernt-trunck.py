import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Camera
from .go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg
import omni.usd
from pxr import UsdGeom


class Go1Env(DirectRLEnv):
    cfg: Go1FlatEnvCfg | Go1RoughEnvCfg

    def __init__(self, cfg: Go1FlatEnvCfg | Go1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "leg_contact_reward",
                "imu_upright_reward",
                "height_reward",
            ]
        }
        self._base_id = None
        self._feet_ids = None
        self._target_height = 0.32

    def _setup_scene(self):
        """Setup scene with sensors"""
        # Create robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Create contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Setup terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 🔥 CRITICAL: Clone environments FIRST
        # This creates the USD hierarchy (/World/envs/env_0/Robot/trunk)
        # that the camera needs to attach to
        self.scene.clone_environments(copy_from_source=False)

        # 🔥 CRITICAL: Create camera AFTER cloning
        # Now the parent prim exists in the USD stage
        self._height_camera = Camera(self.cfg.height_camera)
        self.scene.sensors["height_camera"] = self._height_camera
        print(f"✅ Camera created at: {self.cfg.height_camera.prim_path}")

        # Filter collisions for CPU mode
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        print(f"✅ Scene setup complete for {self.num_envs} environments")
        print(f"   Available sensors: {list(self.scene.sensors.keys())}")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions"""
        action_scale = 0.5
        self._actions = actions.clone() * action_scale
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        """Send actions to simulation"""
        self._robot.set_joint_position_target(self._processed_actions)

    def _post_physics_step(self):
        """Post physics step - update camera"""
        super()._post_physics_step()

        # 🔥 CRITICAL: Update camera every step
        if hasattr(self, '_height_camera') and self._height_camera is not None:
            self._height_camera.update(dt=self.physics_dt)

        # Debug logging
        step = self.episode_length_buf[0].item()
        if step > 0 and step % 50 == 0:
            self._debug_camera_state(step)

    def _debug_camera_state(self, step: int):
        """Debug camera attachment and data"""
        try:
            env_idx = 0
            robot_pos = self._robot.data.root_pos_w[env_idx]
            cam_pos = self._height_camera.data.pos_w[env_idx]
            rel_pos = cam_pos - robot_pos
            expected_rel = torch.tensor([0.4, 0.0, 0.0], device=self.device)

            print(f"\n📷 CAMERA DEBUG - Step {step}")
            print("=" * 60)
            print(f"🤖 Robot trunk: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
            print(f"📷 Camera pos:  [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")
            print(f"🔗 Relative:    [{rel_pos[0]:.3f}, {rel_pos[1]:.3f}, {rel_pos[2]:.3f}]")
            print(f"   Expected:    [{expected_rel[0]:.3f}, {expected_rel[1]:.3f}, {expected_rel[2]:.3f}]")

            attached = torch.allclose(rel_pos, expected_rel, atol=0.05)
            print(f"   Status: {'✅ ATTACHED' if attached else '❌ DETACHED'}")

            # Check depth data
            output = self._height_camera.data.output
            if 'distance_to_image_plane' in output:
                depth = output['distance_to_image_plane']
                if depth.dim() == 4:
                    depth = depth.squeeze(-1)
                valid = (depth > 0.01) & (depth < 1.5) & ~torch.isinf(depth)
                print(f"📊 Depth: {valid[env_idx].sum().item()}/{depth[env_idx].numel()} valid pixels")
                if valid[env_idx].sum() > 0:
                    print(
                        f"   Range: {depth[env_idx][valid[env_idx]].min():.3f} - {depth[env_idx][valid[env_idx]].max():.3f}m")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Camera debug error: {e}")

    def _get_height_estimate(self):
        """Get height from camera depth data"""
        step = self.episode_length_buf[0].item()

        if hasattr(self, '_height_camera') and self._height_camera is not None:
            try:
                output = self._height_camera.data.output

                if 'distance_to_image_plane' in output:
                    depth_data = output['distance_to_image_plane']

                    # Fix shape if needed
                    if depth_data.dim() == 4:
                        depth_data = depth_data.squeeze(-1)

                    # Crop to center 48x48
                    crop_size = 48
                    start = (64 - crop_size) // 2
                    depth_data = depth_data[:, start:start + crop_size, start:start + crop_size]

                    if depth_data.shape[0] == self.num_envs:
                        camera_heights = []
                        for env_idx in range(self.num_envs):
                            env_depth = depth_data[env_idx]
                            depth_flat = env_depth.flatten()

                            # Filter valid depths
                            valid_mask = (
                                    (depth_flat > 0.05) &  # Min range
                                    (depth_flat < 1.0) &  # Max range
                                    ~torch.isnan(depth_flat) &
                                    ~torch.isinf(depth_flat)
                            )
                            valid_depths = depth_flat[valid_mask]

                            # Use median if enough valid pixels
                            if len(valid_depths) > 100:  # ~4% of 48x48
                                median_depth = torch.median(valid_depths)
                                camera_heights.append(median_depth)
                            else:
                                # Fallback to robot Z position
                                fallback = self._robot.data.root_pos_w[env_idx, 2]
                                camera_heights.append(fallback)
                                if step % 200 == 0 and env_idx == 0:
                                    print(
                                        f"⚠️ Using fallback height: {fallback:.3f}m ({len(valid_depths)} valid pixels)")

                        camera_heights_tensor = torch.stack(camera_heights)

                        # Apply EMA smoothing
                        if not hasattr(self, '_camera_height_ema'):
                            self._camera_height_ema = camera_heights_tensor.clone()
                        else:
                            alpha = 0.3
                            self._camera_height_ema = alpha * camera_heights_tensor + (
                                        1 - alpha) * self._camera_height_ema

                        return self._camera_height_ema

            except Exception as e:
                if step % 200 == 0:
                    print(f"❌ Camera processing error: {e}")

        # Final fallback
        return self._robot.data.root_pos_w[:, 2]

    def _get_foot_contact_binary(self) -> torch.Tensor:
        """Get binary foot contact state"""
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
        """Get IMU data"""
        gravity = self._robot.data.projected_gravity_b
        angular_vel = self._robot.data.root_ang_vel_b
        return gravity, angular_vel

    def _get_rewards(self):
        """Calculate rewards"""
        step = self.episode_length_buf[0].item()

        # Leg contact reward
        contact_binary = self._get_foot_contact_binary()
        num_feet_contact = torch.sum(contact_binary, dim=1)
        leg_contact_reward = (num_feet_contact / 4.0) * 2.0

        # IMU upright reward
        gravity, angular_vel = self._get_imu_data()
        roll_pitch_error = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        imu_upright_reward = torch.exp(-roll_pitch_error * 5.0)

        # Height reward
        height_estimate = self._get_height_estimate()
        height_error = torch.abs(height_estimate - self._target_height)
        height_reward = torch.exp(-height_error * 4.0)

        # Debug every 200 steps
        if step % 200 == 0:
            env_idx = 0
            print(f"\n💎 REWARD BREAKDOWN - Step {step}")
            print(f"{'=' * 50}")
            print(f"Leg Contact: {leg_contact_reward[env_idx].item():5.3f}")
            print(
                f"IMU Upright: {imu_upright_reward[env_idx].item():5.3f} ×2.0 = {(imu_upright_reward[env_idx] * 2.0):5.3f}")
            print(f"Height:      {height_reward[env_idx].item():5.3f} ×2.0 = {(height_reward[env_idx] * 2.0):5.3f}")
            print(f"{'=' * 50}")

        # Accumulate rewards
        rewards = {
            "leg_contact_reward": leg_contact_reward * 1.0,
            "imu_upright_reward": imu_upright_reward * 2.0,
            "height_reward": height_reward * 2.0,
        }
        total_reward = sum(rewards.values())
        for key, value in rewards.items():
            if not torch.isnan(value).any():
                self._episode_sums[key] += value

        return total_reward

    def _initialize_body_mappings(self):
        """Initialize body mappings"""
        try:
            all_bodies, all_names = self._contact_sensor.find_bodies(".*")
            self._base_id, base_names = self._contact_sensor.find_bodies("trunk")
            self._feet_ids, feet_names = self._contact_sensor.find_bodies(".*_foot")
            print(f"✅ Body mappings: Base={base_names}, Feet={feet_names}")
        except Exception as e:
            print(f"❌ Body mapping error: {e}")

    def _get_observations(self) -> dict:
        """Get observations"""
        self._previous_actions = self._actions.clone()
        gravity, angular_vel = self._get_imu_data()
        foot_contacts = self._get_foot_contact_binary()
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        height_estimate = self._get_height_estimate()
        height_obs = height_estimate.unsqueeze(1)

        obs_components = [
            gravity,  # 3
            angular_vel,  # 3
            foot_contacts,  # 4
            joint_pos,  # 12
            joint_vel,  # 12
            height_obs,  # 1
            self._actions,  # 12
        ]
        obs = torch.cat(obs_components, dim=-1)
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination conditions"""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        gravity, _ = self._get_imu_data()
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        died = roll_pitch > 0.8
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments"""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        if torch.any(env_ids >= self.num_envs) or torch.any(env_ids < 0):
            env_ids = env_ids[(env_ids < self.num_envs) & (env_ids >= 0)]
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)
        self._robot.reset(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
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