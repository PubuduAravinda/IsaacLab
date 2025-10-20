import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Camera, CameraCfg
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

        # Camera setup
        self._height_camera = None
        self._camera_initialized = False

        # Height tracking
        self._target_height = 0.32  # Real Go1 standing height
        self._camera_offset = 0.1  # Camera is 10cm below trunk

    def _setup_scene(self):
        """Setup scene with camera that will be properly attached to robot"""
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

        # 🎯 Setup camera BEFORE cloning
        self._setup_height_camera()

        # Clone environments AFTER camera setup
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        print(f"✅ Scene setup complete for {self.num_envs} environments")

    def _update_camera_transforms(self):
        """Manually update camera transforms to follow robot trunk"""
        if self._height_camera is None or not self._camera_initialized:
            return

        try:
            # Get robot trunk positions and orientations
            trunk_positions = self._robot.data.root_pos_w
            trunk_orientations = self._robot.data.root_quat_w

            # Update camera transforms for all environments
            for env_idx in range(self.num_envs):
                # Calculate camera position relative to trunk
                trunk_pos = trunk_positions[env_idx]
                trunk_rot = trunk_orientations[env_idx]

                # Camera offset is 10cm below trunk in local frame
                # We need to rotate this offset by the trunk's orientation
                offset_local = torch.tensor([0.0, 0.0, -0.1], device=self.device)

                # Rotate offset to world frame
                offset_world = quat_rotate_inverse(trunk_rot, offset_local)

                # Calculate final camera position
                camera_pos = trunk_pos + offset_world

                # Camera orientation: always face downward regardless of robot orientation
                camera_rot = torch.tensor([0.707, 0.0, 0.0, 0.707], device=self.device)  # facing down

                # 🎯 TODO: We need to set the camera transform here
                # This requires accessing the USD prim and setting its transform
                # The exact method depends on your Isaac Lab version

        except Exception as e:
            print(f"⚠️ Camera transform update failed: {e}")

    def _debug_camera_movement(self):
        """Debug to verify camera is moving with robot"""
        if self._height_camera is None:
            return

        env_idx = 0
        if hasattr(self._height_camera.data, 'pos_w'):
            robot_pos = self._robot.data.root_pos_w[env_idx]
            camera_pos = self._height_camera.data.pos_w[env_idx]
            distance = torch.norm(robot_pos - camera_pos).item()

            print(f"\n🎯 CAMERA MOVEMENT CHECK:")
            print(f"   🤖 Robot: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.3f}]m")
            print(f"   📷 Camera: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.3f}]m")
            print(f"   📏 Distance: {distance:.3f}m")

            if distance < 0.2:
                print("   ✅ Camera is properly attached to robot")
            else:
                print("   ❌ Camera is NOT attached to robot")

    def _setup_height_camera(self):
        """Create height camera with USD prim access for manual updates"""
        print(f"\n🎥 Setting up height camera for {self.num_envs} environments")

        try:
            camera_cfg = self.cfg.height_camera

            print(f"   Camera prim_path: {camera_cfg.prim_path}")
            print(f"   Camera offset: {camera_cfg.offset.pos}")

            # Create the camera
            self._height_camera = Camera(camera_cfg)
            self.scene.sensors["height_camera"] = self._height_camera
            self._camera_initialized = True

            print(f"✅ Camera created successfully")

            # 🎯 Store camera prim paths for manual updates
            self._camera_prim_paths = []
            for env_idx in range(self.num_envs):
                camera_path = f"/World/envs/env_{env_idx}/Robot/trunk/height_camera"
                self._camera_prim_paths.append(camera_path)

            print(f"   Camera prim paths stored for {self.num_envs} environments")

        except Exception as e:
            print(f"❌ Camera setup failed: {e}")
            self._height_camera = None
            self._camera_initialized = False

    def _update_camera_transforms_usd(self):
        """Update camera transforms using USD prim access"""
        if not hasattr(self, '_camera_prim_paths') or not self._camera_prim_paths:
            return

        try:
            import omni.usd
            from pxr import UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()

            for env_idx in range(self.num_envs):
                camera_path = self._camera_prim_paths[env_idx]
                camera_prim = stage.GetPrimAtPath(camera_path)

                if camera_prim and camera_prim.IsValid():
                    # Get robot trunk transform
                    trunk_pos = self._robot.data.root_pos_w[env_idx].cpu().numpy()
                    trunk_rot = self._robot.data.root_quat_w[env_idx].cpu().numpy()

                    # Calculate camera position (10cm below trunk)
                    camera_pos = Gf.Vec3f(trunk_pos[0], trunk_pos[1], trunk_pos[2] - 0.1)

                    # Camera always faces downward
                    camera_rot = Gf.Quatf(0.707, 0.0, 0.0, 0.707)

                    # Set camera transform
                    xform = UsdGeom.Xformable(camera_prim)
                    ops = xform.GetOrderedXformOps()

                    # Clear existing transforms
                    for op in ops:
                        xform.ClearXformOpOrder()

                    # Add new transforms
                    translate_op = xform.AddTranslateOp()
                    rotate_op = xform.AddOrientOp()

                    translate_op.Set(camera_pos)
                    rotate_op.Set(camera_rot)

        except Exception as e:
            if self.episode_length_buf[0] % 500 == 0:  # Don't spam errors
                print(f"⚠️ USD camera transform update failed: {e}")

    def _debug_camera_status(self):
        """Comprehensive camera debugging"""
        print(f"\n🔍 CAMERA DEBUG - Step {self.episode_length_buf[0].item()}:")

        # Basic camera object check
        if self._height_camera is None:
            print("❌ Camera: _height_camera is None")
            return

        if not self._camera_initialized:
            print("❌ Camera: _camera_initialized is False")
            return

        print("✅ Camera object exists and is initialized")

        # Check camera data structure
        if not hasattr(self._height_camera, 'data'):
            print("❌ Camera: No 'data' attribute")
            return

        camera_data = self._height_camera.data
        if camera_data is None:
            print("❌ Camera: data is None")
            return

        print("✅ Camera data object exists")

        # Check camera position
        env_idx = 0
        robot_pos = self._robot.data.root_pos_w[env_idx]

        if hasattr(camera_data, 'pos_w'):
            camera_pos = camera_data.pos_w[env_idx]
            distance = torch.norm(robot_pos - camera_pos).item()
            print(f"   🤖 Robot: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.3f}]m")
            print(f"   📷 Camera: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.3f}]m")
            print(f"   📏 Distance: {distance:.3f}m")

            if distance < 0.2:
                print("   ✅ Camera is properly attached to robot")
            else:
                print("   ⚠️  Camera may not be properly attached")
        else:
            print("❌ Camera: No position data (pos_w)")

        # Check camera output
        if not hasattr(camera_data, 'output'):
            print("❌ Camera: No 'output' attribute")
            return

        output = camera_data.output
        if output is None:
            print("❌ Camera: output is None")
            return

        print("✅ Camera output exists")
        print(f"   Output keys: {list(output.keys())}")

        # Check for depth data
        if 'distance_to_image_plane' not in output:
            print("❌ Camera: 'distance_to_image_plane' not in output")
            return

        depth_data = output['distance_to_image_plane']
        if depth_data is None:
            print("❌ Camera: depth_data is None")
            return

        print(f"✅ Depth data available, shape: {depth_data.shape}")

        # Analyze depth data
        try:
            depth_flat = depth_data[env_idx].view(-1)
            valid_depths = depth_flat[(depth_flat > 0.01) & (depth_flat < 10.0)]

            if len(valid_depths) > 0:
                min_depth = valid_depths.min().item()
                max_depth = valid_depths.max().item()
                median_depth = valid_depths.median().item()
                print(f"   📊 Depth Analysis:")
                print(f"      Valid pixels: {len(valid_depths)}/{len(depth_flat)}")
                print(f"      Min: {min_depth:.3f}m, Max: {max_depth:.3f}m, Median: {median_depth:.3f}m")

                # 🎯 ADD THIS: Height validation check
                expected_height = median_depth + 0.1  # Camera offset
                actual_robot_height = self._robot.data.root_pos_w[env_idx, 2].item()

                print(f"   🎯 HEIGHT VALIDATION:")
                print(f"      Median Depth: {median_depth:.3f}m")
                print(f"      Expected Height: {expected_height:.3f}m (depth + 0.1m offset)")
                print(f"      Actual Robot Height: {actual_robot_height:.3f}m")
                print(f"      Difference: {abs(expected_height - actual_robot_height):.3f}m")

                if abs(expected_height - actual_robot_height) > 0.1:
                    print("   ⚠️  Height estimate may be inaccurate")
                else:
                    print("   ✅ Height estimate looks accurate")
            else:
                print("   📊 Depth Data: No valid measurements")

        except Exception as e:
            print(f"❌ Depth analysis failed: {e}")


    def _get_height_from_camera(self):
        """Get height from camera with robust reference checking"""
        # Multiple ways to access camera
        camera = None

        # Method 1: Use stored reference
        if self._height_camera is not None:
            camera = self._height_camera
        # Method 2: Get from scene sensors
        elif "height_camera" in self.scene.sensors:
            camera = self.scene.sensors["height_camera"]
            self._height_camera = camera  # Recover reference
            self._camera_initialized = True
            print("🔄 Retrieved camera from scene sensors")
        else:
            return None, "Camera not found in any location"

        if not self._camera_initialized:
            return None, "Camera marked as not initialized"

        try:
            # Wait a few steps for camera to initialize
            if self.episode_length_buf[0] < 5:
                return None, "Camera initializing (early steps)"

            # Check camera data structure
            if not hasattr(camera, 'data'):
                return None, "No camera data attribute"

            camera_data = camera.data
            if camera_data is None:
                return None, "Camera data is None"

            # Check output
            if not hasattr(camera_data, 'output'):
                return None, "No camera output attribute"

            output = camera_data.output
            if output is None:
                return None, "Camera output is None"

            # Check for depth data
            if 'distance_to_image_plane' not in output:
                return None, "No distance_to_image_plane in output"

            depth_data = output['distance_to_image_plane']
            if depth_data is None:
                return None, "Depth data is None"

            # Process depth data for all environments
            camera_heights = []
            valid_count = 0

            for env_idx in range(self.num_envs):
                env_depth = depth_data[env_idx]
                if env_depth is None:
                    camera_heights.append(torch.tensor(0.0, device=self.device))
                    continue

                depth_flat = env_depth.view(-1)

                # Filter valid depth values
                valid_mask = (depth_flat > 0.01) & (depth_flat < 2.0)
                valid_mask = valid_mask & ~torch.isnan(depth_flat) & ~torch.isinf(depth_flat)
                valid_depths = depth_flat[valid_mask]

                if len(valid_depths) > 0:
                    median_depth = torch.median(valid_depths)
                    trunk_height = median_depth + 0.1  # Camera offset
                    camera_heights.append(trunk_height)
                    valid_count += 1
                else:
                    camera_heights.append(torch.tensor(0.0, device=self.device))

            if valid_count == 0:
                return None, "No valid depth measurements"

            camera_heights_tensor = torch.stack(camera_heights)
            return camera_heights_tensor, f"Camera success ({valid_count}/{self.num_envs} envs)"

        except Exception as e:
            return None, f"Camera processing error: {str(e)}"

    def _check_camera_health(self):
        """Periodically check camera health and recover if needed"""
        if self.episode_length_buf[0] % 100 == 0:  # Check every 100 steps
            camera_healthy = True

            # Check primary reference
            if self._height_camera is None:
                print("⚠️  Camera health: Primary reference is None")
                camera_healthy = False

            # Check scene reference
            if "height_camera" not in self.scene.sensors:
                print("⚠️  Camera health: Not in scene sensors")
                camera_healthy = False

            # Try to recover if unhealthy
            if not camera_healthy:
                self._recover_camera_reference()

            return camera_healthy
        return True

    def _get_height_estimate(self):
        """Get height estimate - try camera first, then fallback to robot position"""
        # Try to get height from actual camera
        camera_height, camera_status = self._get_height_from_camera()

        # Get robot trunk height (this is what we use for rewards)
        trunk_height = self._robot.data.root_pos_w[:, 2]

        if camera_height is not None:
            # Use camera-based height for rewards
            return camera_height, trunk_height, camera_status
        else:
            # Fallback: use robot position with offset approximation
            camera_height_estimate = trunk_height - 0.1  # Approximate camera offset

            return torch.clamp(camera_height_estimate, min=0.05, max=1.0), trunk_height, f"Fallback - {camera_status}"

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
        gravity = self._robot.data.projected_gravity_b
        angular_vel = self._robot.data.root_ang_vel_b
        return gravity, angular_vel

    def _get_rewards(self):
        """Calculate rewards with comprehensive debugging"""
        # 1. LEG CONTACT REWARD
        contact_binary = self._get_foot_contact_binary()
        num_feet_contact = torch.sum(contact_binary, dim=1)
        leg_contact_reward = (num_feet_contact / 4.0) * 2.0

        # 2. IMU UPRIGHT REWARD
        gravity, angular_vel = self._get_imu_data()
        roll_pitch_error = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        imu_upright_reward = torch.exp(-roll_pitch_error * 5.0)

        # 3. HEIGHT REWARD - This uses the camera_height from _get_height_estimate()
        height_estimate, trunk_height, height_source = self._get_height_estimate()
        height_error = torch.abs(height_estimate - self._target_height)
        height_reward = torch.exp(-height_error * 4.0)

        # Progress monitoring
        if self.episode_length_buf[0] % 200 == 0:
            env_idx = 0
            print(f"\n📊 TRAINING - Step {self.episode_length_buf[env_idx].item()}:")
            print(f"  🦵 Feet: {contact_binary[env_idx].tolist()}")
            print(f"  📏 {height_source}: {height_estimate[env_idx].item():.3f}m")
            print(f"  🤖 Robot Height: {trunk_height[env_idx].item():.3f}m")
            print(f"  🎯 Target: {self._target_height}m")
            print(f"  💎 Rewards - Leg: {leg_contact_reward[env_idx].item():.3f}, "
                  f"Height: {height_reward[env_idx].item():.3f}")

        # Comprehensive camera debugging every 500 steps
        if self.episode_length_buf[0] % 500 == 0:
            self._debug_camera_status()

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
        """Get observations from sensors"""
        self._previous_actions = self._actions.clone()

        gravity, angular_vel = self._get_imu_data()
        foot_contacts = self._get_foot_contact_binary()

        # Joint state
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        # Height observation - IMPORTANT: This uses camera_height for observation
        height_estimate, trunk_height, _ = self._get_height_estimate()
        height_obs = height_estimate.unsqueeze(1)

        # Observation components (compatible with real Go1)
        obs_components = [
            gravity,  # IMU accelerometer (3) - real Go1 has IMU
            angular_vel,  # IMU gyroscope (3) - real Go1 has IMU
            foot_contacts,  # Binary leg contacts (4) - real Go1 has foot contact sensors
            joint_pos,  # Joint positions (12) - real Go1 has joint encoders
            joint_vel,  # Joint velocities (12) - real Go1 can estimate joint velocity
            height_obs,  # Height estimate (1) - THIS IS WHAT THE POLICY SEES
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
        died = roll_pitch > 0.8

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments - camera will automatically follow due to USD hierarchy"""
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

            # Camera is automatically attached via USD hierarchy - no manual repositioning needed
            print(f"✅ Camera automatically attached via USD hierarchy")

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

    def _verify_camera_attachment(self):
        """Verify camera is properly attached to robot trunk"""
        if self._height_camera is None:
            return

        env_idx = 0
        print(f"\n🔧 VERIFYING CAMERA ATTACHMENT:")

        # Check if camera has the correct prim path
        if hasattr(self._height_camera, '_cfg'):
            cfg = self._height_camera._cfg
            print(f"   Camera prim_path: {cfg.prim_path}")
            print(f"   Camera offset: {cfg.offset.pos}")

        # Check camera and robot positions over time
        robot_pos = self._robot.data.root_pos_w[env_idx]

        if hasattr(self._height_camera.data, 'pos_w'):
            camera_pos = self._height_camera.data.pos_w[env_idx]
            distance = torch.norm(robot_pos - camera_pos).item()

            print(f"   🤖 Robot: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.3f}]m")
            print(f"   📷 Camera: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.3f}]m")
            print(f"   📏 Distance: {distance:.3f}m")

            if distance > 0.5:
                print("   ❌ CAMERA NOT ATTACHED: Distance too large")
                print("   💡 Solution: Check camera prim_path in configuration")
            else:
                print("   ✅ Camera properly attached")

    def _post_physics_step(self):
        """Called after physics step - update camera data"""
        super()._post_physics_step()

        if not self._camera_initialized or self._height_camera is None:
            self._recover_camera_reference()

        # 🎯 Update camera transforms to follow robot
        self._update_camera_transforms_usd()

        # 🎯 Debug camera movement periodically
        if self.episode_length_buf[0] % 300 == 0:
            self._debug_camera_movement()

        # Update camera if it exists
        if self._height_camera is not None and self._camera_initialized:
            try:
                self._height_camera.update(dt=0.0)
            except Exception as e:
                print(f"⚠️ Camera update failed: {e}")

        # Periodic training summary
        if self.episode_length_buf[0] % 1000 == 0:
            env_idx = 0
            height_estimate, trunk_height, height_source = self._get_height_estimate()

            print(f"\n" + "=" * 60)
            print(f"🏆 TRAINING SUMMARY - Step {self.episode_length_buf[env_idx].item()}")
            print("=" * 60)
            print(f"🤖 ROBOT STATUS:")
            print(f"   Height Source: {height_source}")
            print(f"   Estimated Height: {height_estimate[env_idx].item():.3f}m")
            print(f"   Robot Trunk Height: {trunk_height[env_idx].item():.3f}m")
            print(f"   Target Height: {self._target_height}m")

            print(f"\n📈 CUMULATIVE REWARDS:")
            for key in self._episode_sums.keys():
                avg_val = self._episode_sums[key][env_idx].item() / max(1, self.episode_length_buf[env_idx].item())
                print(f"   {key:20}: {avg_val:.3f}")

    def _recover_camera_reference(self):
        """Try multiple methods to recover camera reference"""
        # Method 1: Check scene sensors
        if "height_camera" in self.scene.sensors and self.scene.sensors["height_camera"] is not None:
            self._height_camera = self.scene.sensors["height_camera"]
            self._camera_initialized = True
            print("🔄 Recovered camera from scene sensors")
            return True

        # Method 2: Try to recreate camera if it doesn't exist
        if self._height_camera is None and not self._camera_initialized:
            try:
                camera_cfg = self.cfg.height_camera
                self._height_camera = Camera(camera_cfg)
                self.scene.sensors["height_camera"] = self._height_camera
                self._camera_initialized = True
                print("🔄 Recreated camera successfully")
                return True
            except Exception as e:
                print(f"❌ Camera recreation failed: {e}")

        return False