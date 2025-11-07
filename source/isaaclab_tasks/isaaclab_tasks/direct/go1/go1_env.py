import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Camera
from isaaclab.utils.math import quat_apply, quat_mul
from torchvision.utils import save_image
import omni.usd
from pxr import Usd, UsdShade, Sdf, Gf, UsdGeom
import os
from .go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2


class Go1Env(DirectRLEnv):
    cfg: Go1FlatEnvCfg | Go1RoughEnvCfg

    def __init__(self, cfg: Go1FlatEnvCfg | Go1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print("🤖 Go1 Environment - Using Automatic Camera Tracking")

        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)

        self._episode_sums = {
            "leg_contact_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "imu_upright_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "height_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "torque_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "action_smoothness_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        self._target_height = 0.32
        self._global_step = 0

        # DEBUG MODE: Set to True to freeze robot in standing pose for camera testing
        self._debug_camera_mode = True
        self._debug_steps = 0

        self._trunk_idx = self._robot.find_bodies("trunk")[0]

        self._camera_offset = 0.10

        # Camera setup
        cam_pos_offset = [0.05, 0.0, -0.01]
        self._cam_local_pos = torch.tensor(cam_pos_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        pitch_angle_deg = 180
        roll_angle_deg = 90
        r_pitch = R.from_euler('x', pitch_angle_deg, degrees=True)
        r_roll = R.from_euler('z', roll_angle_deg, degrees=True)
        r_combined = r_roll * r_pitch
        quat_xyzw = r_combined.as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        self._cam_local_rot = torch.tensor(quat_wxyz, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        print(f"🎯 Camera Setup:")
        print(f"   Local position: {cam_pos_offset}")
        print(f"   Pitch: {pitch_angle_deg}°, Roll: {roll_angle_deg}°")

        # MiDaS setup
        print("[INFO] Loading MiDaS-small model...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        # Height tracking
        self._raycaster_height = torch.zeros(self.num_envs, device=self.device)
        self._vision_height = torch.zeros(self.num_envs, device=self.device)
        self._height_errors = []

    def step(self, actions: torch.Tensor):
        self._global_step += 1
        return super().step(actions)

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]
        self._camera = self.scene["camera"]
        self._raycaster = self.scene["raycaster"]
        self._apply_carpet_material()
        print("🔧 Scene Setup Complete")

    def _get_raycaster_height(self) -> torch.Tensor:
        """Get trunk-to-ground distance using raycaster"""
        try:
            ray_origins = self._raycaster.data.pos_w
            ray_hits = self._raycaster.data.ray_hits_w
            heights = torch.zeros(self.num_envs, device=self.device)

            for env_idx in range(self.num_envs):
                env_origins = ray_origins[env_idx]
                env_hits = ray_hits[env_idx]
                distances = torch.norm(env_hits - env_origins, dim=-1)
                valid_mask = distances > 0.01
                valid_distances = distances[valid_mask]

                if len(valid_distances) > 0:
                    heights[env_idx] = torch.min(valid_distances)
                else:
                    heights[env_idx] = self._robot.data.root_pos_w[env_idx, 2] - 0.05

            return heights

        except Exception as e:
            print(f"📏 Raycaster height error: {e}")
            return self._robot.data.root_pos_w[:, 2] - 0.05

    def _apply_carpet_material(self):
        stage = omni.usd.get_context().get_stage()
        ground_prim = stage.GetPrimAtPath("/World/ground")
        if not ground_prim.IsValid():
            print("❌ Error: Ground prim at '/World/ground' not found")
            return

        material_path = "/World/Materials/BerberCarpet"
        try:
            existing_material = stage.GetPrimAtPath(material_path)
            if existing_material.IsValid():
                stage.RemovePrim(material_path)

            material_prim = stage.DefinePrim(material_path, "Material")
            shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")

            texture_path = "/home/sripu715/Downloads/catpet_al_lab.jpg"
            if os.path.exists(texture_path):
                print(f"✅ Found carpet texture: {texture_path}")
                uv_reader = UsdShade.Shader.Define(stage, f"{material_path}/UVReader")
                uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
                uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                transform_2d = UsdShade.Shader.Define(stage, f"{material_path}/Transform2d")
                transform_2d.CreateIdAttr("UsdTransform2d")
                transform_2d.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(500.0, 500.0))
                transform_2d.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(),
                                                                                          "result")
                texture_shader = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
                texture_shader.CreateIdAttr("UsdUVTexture")
                texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
                texture_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                texture_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    transform_2d.ConnectableAPI(), "result")
                texture_shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((1.5, 1.5, 1.5, 1.0))
                diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                diffuse_input.ConnectToSource(texture_shader.ConnectableAPI(), "rgb")
            else:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.2, 0.8))

            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material = UsdShade.Material(material_prim)
            surface_output = material.CreateSurfaceOutput()
            surface_output.ConnectToSource(shader.ConnectableAPI(), "surface")

            if ground_prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(ground_prim)
                st_primvar = mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
                st_primvar.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

            binding_api = UsdShade.MaterialBindingAPI.Apply(ground_prim)
            binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
            print("✅ Applied carpet material")

        except Exception as e:
            print(f"❌ Error creating carpet material: {e}")

    def _pre_physics_step(self, actions: torch.Tensor):
        if self._debug_camera_mode and self._debug_steps < 100:
            self._debug_steps += 1
            self._actions = torch.zeros_like(actions)
            self._processed_actions = self._robot.data.default_joint_pos
            if self._debug_steps <= 10 or self._debug_steps % 20 == 0:
                print(f"🔍 DEBUG MODE: Robot frozen in standing pose (step {self._debug_steps}/100)")
        else:
            if self._debug_camera_mode and self._debug_steps == 200:
                print("✅ DEBUG MODE: Complete! Set self._debug_camera_mode = False to resume training")
                self._debug_steps += 1
            self._actions = actions.clone()
            self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        self.scene.write_data_to_sim()

    def _get_vision_height(self, depth_maps: np.ndarray) -> torch.Tensor:
        h, w = depth_maps.shape[1], depth_maps.shape[2]

        # Define sampling region
        y_center = int(h * 0.5)
        x_center = int(w * 0.5)
        region_size = 20
        y1 = max(0, y_center - region_size // 2)
        y2 = min(h, y_center + region_size // 2)
        x1 = max(0, x_center - region_size // 2)
        x2 = min(w, x_center + region_size // 2)

        # ——— DELAY CALIBRATION UNTIL ROBOT IS STABLE ———
        if not hasattr(self, "MIDAS_TO_M"):
            # Wait until robot is stable (after debug mode or minimal movement)
            if self._debug_camera_mode and self._debug_steps < 100:
                # During debug mode, just use raycaster height directly
                return self._raycaster_height

            # Start calibration only after debug mode ends
            if not hasattr(self, "calib_vals"):
                self.calib_vals = []
                self.calib_raycaster_heights = []
                self.calib_step = 0
                print("🎯 STARTING CALIBRATION (robot should be stable now)")

            if self.calib_step < 30:  # More samples for better calibration
                # Check if robot is reasonably stable (not falling)
                root_height = self._robot.data.root_pos_w[0, 2].item()
                gravity, _ = self._get_imu_data()
                tilt = torch.sqrt(gravity[0, 0] ** 2 + gravity[0, 1] ** 2).item()

                # Only calibrate when robot is upright and not falling
                if root_height > 0.25 and tilt < 0.3:
                    val = np.mean(depth_maps[0, y1:y2, x1:x2])
                    self.calib_vals.append(val)
                    raycaster_h = self._raycaster_height[0].item()
                    self.calib_raycaster_heights.append(raycaster_h)
                    self.calib_step += 1

                    if self.calib_step % 10 == 0:
                        print(f"  [CALIB] {self.calib_step}/30 → MiDaS {val:.1f}, Raycaster {raycaster_h:.3f}m")

                # During calibration, use simple linear estimate
                if len(self.calib_vals) > 5:
                    current_midas = np.mean(depth_maps[0, y1:y2, x1:x2])
                    current_raycaster = self._raycaster_height[0].item()
                    temp_ratio = current_midas / max(current_raycaster, 0.01)
                    ground_vals = [np.mean(depth_maps[i, y1:y2, x1:x2]) for i in range(self.num_envs)]
                    ground_depth = torch.tensor(ground_vals, device=self.device)
                    return torch.clamp(ground_depth / temp_ratio, 0.05, 1.0)

                return self._raycaster_height

            # CALIBRATION COMPLETE
            if len(self.calib_vals) > 10:
                median_midas = np.median(self.calib_vals)
                median_raycaster = np.median(self.calib_raycaster_heights)
                self.MIDAS_TO_M = median_midas / median_raycaster

                print(f"\n🎯 CALIBRATION COMPLETE! MIDAS_TO_M = {self.MIDAS_TO_M:.1f}")
                print(f"   Based on {len(self.calib_vals)} stable samples")
                print(f"   MiDaS: {median_midas:.1f}, Raycaster: {median_raycaster:.3f}m")
                print(f"   Vision system ready! 🚀\n")
            else:
                self.MIDAS_TO_M = 1000.0
                print("⚠️ Insufficient calibration data, using fallback")

        # ——— NORMAL OPERATION ———
        ground_vals = [np.mean(depth_maps[i, y1:y2, x1:x2]) for i in range(self.num_envs)]
        ground_depth = torch.tensor(ground_vals, device=self.device)
        height_m = ground_depth / self.MIDAS_TO_M

        return torch.clamp(height_m, 0.05, 1.0)

    def _get_height_estimate(self, depth_maps: np.ndarray) -> torch.Tensor:
        try:
            return self._get_vision_height(depth_maps)
        except Exception as e:
            print(f"[VISION FALLBACK] {e}")
            trunk_h = self._robot.data.root_pos_w[:, 2]
            gravity = self._robot.data.projected_gravity_b
            tilt = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
            h = trunk_h * (1 - tilt * 0.2) - getattr(self, "_camera_offset", 0.10)
            return torch.clamp(h, 0.05, 1.0)

    def _debug_system_status(self):
        """Consolidated debug function for robot, camera, and height status"""
        env_idx = 0

        # Robot state
        root_pos = self._robot.data.root_pos_w[env_idx]
        root_rot = self._robot.data.root_quat_w[env_idx]

        # Euler angles from quaternion
        w, x, y, z = root_rot
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin(2 * (w * y - z * x))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        roll_deg = roll.item() * 180 / np.pi
        pitch_deg = pitch.item() * 180 / np.pi

        # Joint position check
        joint_pos = self._robot.data.joint_pos[env_idx]
        default_joint_pos = self._robot.data.default_joint_pos[env_idx]
        joint_diff = torch.norm(joint_pos - default_joint_pos).item()

        # Camera state - use the actual data we already have
        camera_pos = self._camera.data.pos_w[env_idx]
        camera_rot = self._camera.data.quat_w_world[env_idx]

        # Calculate what the camera position SHOULD be based on robot pose
        trunk_pos = self._robot.data.body_pos_w[env_idx, self._trunk_idx]
        trunk_quat = self._robot.data.body_quat_w[env_idx, self._trunk_idx]

        # Use the same calculation as in _get_observations but for single env
        expected_cam_pos = trunk_pos + quat_apply(trunk_quat.unsqueeze(0),
                                                  self._cam_local_pos[env_idx].unsqueeze(0)).squeeze(0)

        # Camera attachment error (just position for now - simpler)
        cam_pos_error = torch.norm(camera_pos - expected_cam_pos).item()

        # Camera forward vector
        forward_local = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
        forward_world = quat_apply(camera_rot.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0)

        # Height comparison
        vision_h = self._vision_height[env_idx].item()
        raycaster_h = self._raycaster_height[env_idx].item()
        error = abs(vision_h - raycaster_h)

        print(f"\n📊 SYSTEM STATUS (env {env_idx}, step {self._global_step}):")
        print(f"🤖 Robot: Pos[{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}] "
              f"Rot[Roll={roll_deg:5.1f}°, Pitch={pitch_deg:5.1f}°] "
              f"Joints[{joint_diff:5.3f}]")
        print(f"📷 Camera: Pos[{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}] "
              f"Fwd[{forward_world[0]:5.2f}, {forward_world[1]:5.2f}, {forward_world[2]:5.2f}]")
        print(f"🔗 Attachment Error: {cam_pos_error:6.4f}m "
              f"{'✅' if cam_pos_error < 0.01 else '⚠️'}")
        print(f"📏 Heights: Vision={vision_h:5.3f}m, Raycaster={raycaster_h:5.3f}m, "
              f"Error={error:5.3f}m ({error / raycaster_h * 100:4.1f}%) "
              f"{'✅' if error < 0.05 else '⚠️'}")

        # Analysis of vision vs raycaster
        if error > 0.1:
            print(f"💡 Analysis: Large vision error - MiDaS struggling with perspective/calibration")
        elif error > 0.05:
            print(f"💡 Analysis: Moderate vision error - typical for non-downward camera")
        else:
            print(f"💡 Analysis: Good agreement - vision and raycaster aligned")

        # Track errors for statistics
        self._height_errors.append(error)
        if len(self._height_errors) >= 5 and len(self._height_errors) % 5 == 0:
            avg_error = np.mean(self._height_errors[-5:])
            max_error = np.max(self._height_errors[-5:])
            print(f"📈 Height Stats (last 5): Avg={avg_error:.3f}m, Max={max_error:.3f}m")


    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        gravity, angular_vel = self._get_imu_data()
        foot_contacts = self._get_foot_contact_binary()
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        # Camera pose update
        trunk_pos = self._robot.data.body_pos_w[:, self._trunk_idx]
        trunk_quat = self._robot.data.body_quat_w[:, self._trunk_idx]
        if trunk_pos.dim() == 3: trunk_pos = trunk_pos.squeeze(1)
        if trunk_quat.dim() == 3: trunk_quat = trunk_quat.squeeze(1)

        cam_pos = trunk_pos + quat_apply(trunk_quat, self._cam_local_pos)
        cam_quat = quat_mul(trunk_quat, self._cam_local_rot)

        self._camera.set_world_poses(cam_pos, cam_quat, convention="ros")
        self._camera._update_poses(list(range(self.num_envs)))
        self._camera.update(dt=self.physics_dt)

        # RGB processing
        rgb = self._camera.data.output["rgb"].float() / 255.0
        rgb_enhanced = torch.clamp(rgb * 1.3, 0.0, 1.0)

        kernel = torch.tensor([[[[0, -0.2, 0], [-0.2, 2.0, -0.2], [0, -0.2, 0]]]], device=self.device)
        kernel = kernel.repeat(3, 1, 1, 1)
        rgb_permuted = rgb_enhanced.permute(0, 3, 1, 2)
        rgb_sharpened = torch.nn.functional.conv2d(rgb_permuted, kernel, padding=1, groups=3)
        rgb_sharpened = rgb_sharpened.permute(0, 2, 3, 1)
        rgb_input = torch.clamp(rgb_sharpened, 0.0, 1.0)

        rgb_flipped = torch.flip(rgb_input, dims=[1])
        rgb_upright = torch.rot90(rgb_flipped, k=2, dims=[1, 2])

        # MiDaS depth estimation
        with torch.no_grad():
            inputs = []
            for img in rgb_upright:
                img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                tensor = self.midas_transform(img_np)
                inputs.append(tensor)
            batch = torch.cat(inputs, dim=0).to(self.device)
            prediction = self.midas(batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(rgb_input.shape[1], rgb_input.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            depth_maps = prediction.cpu().numpy()
        self.last_depth_maps = depth_maps

        # Height estimation
        height_estimate = self._get_height_estimate(self.last_depth_maps)
        if height_estimate.dim() == 1:
            height_estimate = height_estimate.unsqueeze(1)

        raycaster_height = self._get_raycaster_height()
        self._raycaster_height = raycaster_height
        self._vision_height = height_estimate.squeeze(1)

        # Consolidated debug
        if (self._debug_camera_mode and (self._debug_steps <= 10 or self._debug_steps % 20 == 0)) or (
                self._global_step % 100 == 0):
            self._debug_system_status()

        # Save images periodically
        if self._global_step % 500 == 0:
            os.makedirs("camera_images", exist_ok=True)
            for env_idx in range(min(3, self.num_envs)):
                img_np = (rgb_upright[env_idx].cpu().numpy() * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"camera_images/env_{env_idx}_step_{self._global_step:06d}.png", img_bgr)

                depth = depth_maps[env_idx]
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
                cv2.imwrite(f"camera_images/env_{env_idx}_step_{self._global_step:06d}_depth.png", depth_color)

            print(f"📸 Saved images at step {self._global_step}")

        # Final observation
        obs = torch.cat([
            gravity,  # (num_envs, 3)
            angular_vel,  # (num_envs, 3)
            foot_contacts,  # (num_envs, 4)
            joint_pos,  # (num_envs, 12)
            joint_vel,  # (num_envs, 12)
            height_estimate,  # (num_envs, 1)
            self._actions  # (num_envs, 12)
        ], dim=-1)

        return {"policy": obs}

    def _get_foot_contact_binary(self) -> torch.Tensor:
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
        return self._robot.data.projected_gravity_b, self._robot.data.root_ang_vel_b

    def _get_rewards(self) -> torch.Tensor:
        contact_binary = self._get_foot_contact_binary()
        leg_contact_reward = (torch.sum(contact_binary, dim=1) / 4.0) * 2.0

        gravity, angular_vel = self._get_imu_data()
        roll_pitch_error = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        imu_upright_reward = torch.exp(-roll_pitch_error * 5.0)

        height_estimate = self._get_height_estimate(self.last_depth_maps)
        height_error = torch.abs(height_estimate - 0.32)
        height_reward = torch.exp(-height_error * 6.0) * 2.0

        h, w = self.last_depth_maps.shape[1], self.last_depth_maps.shape[2]
        y1 = int(h * 0.88)
        left = np.mean(self.last_depth_maps[:, y1:, :w // 3], axis=(1, 2))
        right = np.mean(self.last_depth_maps[:, y1:, 2 * w // 3:], axis=(1, 2))
        pitch_tilt = torch.tensor(left - right, device=self.device) / 100.0
        pitch_reward = torch.exp(-pitch_tilt.abs() * 12.0) * 0.8

        crash_penalty = torch.clamp(0.18 - height_estimate, 0.0, 0.2) * 40.0
        torques = self._robot.data.applied_torque
        torque_penalty = -0.0001 * torch.sum(torques ** 2, dim=1)
        action_diff = self._actions - self._previous_actions
        action_smoothness_penalty = -0.5 * torch.sum(action_diff ** 2, dim=1)

        total_reward = (
                leg_contact_reward * 1.0 +
                imu_upright_reward * 2.0 +
                height_reward +
                pitch_reward -
                crash_penalty +
                torque_penalty +
                action_smoothness_penalty
        )

        if self._global_step % 200 == 0:
            print("\n" + "═" * 70)
            print(f" TRAINING PROGRESS - Step {self.episode_length_buf[0].item():,}")
            print(f"  Feet: {contact_binary[0].tolist()}")
            print(f"  Height: {height_estimate[0].item():.3f}m (target 0.32m)")
            print(f"  Rewards: Leg {leg_contact_reward[0].item():.2f} | "
                  f"IMU {imu_upright_reward[0].item():.2f} | "
                  f"Height {height_reward[0].item():.2f} | "
                  f"Pitch {pitch_reward[0].item():.2f}")
            print(f"  Penalties: Torque {torque_penalty[0].item():.3f} | "
                  f"Smooth {action_smoothness_penalty[0].item():.3f} | "
                  f"Crash -{crash_penalty[0].item():.1f}")
            print(f"  TOTAL: {total_reward[0].item():.3f}")
            print("═" * 70)

        rewards_dict = {
            "leg_contact_reward": leg_contact_reward,
            "imu_upright_reward": imu_upright_reward,
            "height_reward": height_reward,
            "pitch_reward": pitch_reward,
            "crash_penalty": -crash_penalty,
            "torque_penalty": torque_penalty,
            "action_smoothness_penalty": action_smoothness_penalty
        }
        for key, value in rewards_dict.items():
            self._episode_sums[key] = self._episode_sums.get(key, 0) + value

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        gravity, _ = self._get_imu_data()
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        height = self._robot.data.root_pos_w[:, 2]
        died = torch.logical_or(roll_pitch > 0.8, height < 0.1)
        return died, timeout

    def _reset_idx(self, env_ids: torch.Tensor):
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