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
from scipy.spatial.transform import Rotation as R  # Added for adjustable rotation
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
        }

        self._target_height = 0.32
        self._global_step = 0

        # DEBUG MODE: Set to True to freeze robot in standing pose for camera testing
        self._debug_camera_mode = True
        self._debug_steps = 0

        self._trunk_idx = self._robot.find_bodies("trunk")[0]

        self._camera_offset = 0.10

        # ADJUST THESE VALUES TO TEST DIFFERENT CAMERA POSITIONS/ANGLES:
        # Start with old-style higher pos + slight forward for legs/shadows
        cam_pos_offset = [0.05, 0.0, -0.01]  # Forward 5cm, below 1cm (old height for peripheral view)
        self._cam_local_pos = torch.tensor(cam_pos_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Adjustable rotation (from new version): Pitch down + Roll
        pitch_angle_deg = 180  # As per your working value
        roll_angle_deg = 90  # As per your working value

        # Compute combined rotation
        r_pitch = R.from_euler('x', pitch_angle_deg, degrees=True)
        r_roll = R.from_euler('z', roll_angle_deg, degrees=True)
        r_combined = r_roll * r_pitch
        quat_xyzw = r_combined.as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        self._cam_local_rot = torch.tensor(quat_wxyz, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        print(f"🎯 Camera Setup (Old-Style View with Adjustable Rot):")
        print(f"   Local position: {cam_pos_offset} (forward/higher for legs/shadows)")
        print(f"   Pitch: {pitch_angle_deg}° (your working value)")
        print(f"   Roll: {roll_angle_deg}° (your working value)")
        print(f"   Expected: Clear carpet + legs/shadows visible")

        print("[INFO] Loading MiDaS-small model...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        self.midas.to(self.device)
        self.midas.eval()

        self.midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        self.region_size = 40
        self.region_points = {
            "tl": (0.1, 0.1), "tr": (0.9, 0.1),
            "bl": (0.1, 0.9), "br": (0.9, 0.9),
            "center": (0.5, 0.5)
        }

    def _debug_camera_transforms(self):
        try:
            env_idx = 0
            robot_pos = self._robot.data.root_pos_w[env_idx]
            robot_rot = self._robot.data.root_quat_w[env_idx]
            camera_pos = self._camera.data.pos_w[env_idx]
            camera_rot = self._camera.data.quat_w_world[env_idx]

            print(f"\n🔧 COMPLETE TRANSFORM DEBUG:")
            print(f"   Robot Position: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
            print(f"   Robot Rotation: [{robot_rot[0]:.3f}, {robot_rot[1]:.3f}, {robot_rot[2]:.3f}, {robot_rot[3]:.3f}]")
            print(f"   Camera Position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
            print(f"   Camera Rotation: [{camera_rot[0]:.3f}, {camera_rot[1]:.3f}, {camera_rot[2]:.3f}, {camera_rot[3]:.3f}]")

            test_identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
            test_forward = quat_apply(test_identity.unsqueeze(0), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).unsqueeze(0)).squeeze(0)  # Fixed dtype
            print(f"   Identity Forward: [{test_forward[0]:.3f}, {test_forward[1]:.3f}, {test_forward[2]:.3f}]")

            local_quat = self._cam_local_rot[env_idx].unsqueeze(0).float()  # Ensure float32
            local_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).unsqueeze(0)
            local_forward = quat_apply(local_quat, local_vec).squeeze(0)
            print(f"   Local Rot Forward: [{local_forward[0]:.3f}, {local_forward[1]:.3f}, {local_forward[2]:.3f}]")

        except Exception as e:
            print(f"📷 Transform debug error: {e}")

    def step(self, actions: torch.Tensor):
        self._global_step += 1
        return super().step(actions)

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]
        self._camera = self.scene["camera"]
        self._apply_carpet_material()
        print("🔧 Scene Setup Complete:")
        print(f"   Robot prim_path: {self._robot.cfg.prim_path}")
        print(f"   Camera prim_path: {self._camera.cfg.prim_path}")
        print("✅ Camera is child of robot trunk - will follow automatically")
        print("✅ Berber carpet material applied to ground plane")

    def _debug_camera_direction(self):
        try:
            env_idx = 0
            robot_pos = self._robot.data.root_pos_w[env_idx]
            camera_pos = self._camera.data.pos_w[env_idx]
            camera_rot = self._camera.data.quat_w_world[env_idx]

            forward_local = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)  # +Z for ROS, dtype fix
            forward_world = quat_apply(camera_rot.unsqueeze(0).float(), forward_local.unsqueeze(0)).squeeze(0)  # Ensure float32

            # Angle from vertical down
            straight_down = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)  # World down, dtype fix
            dot_down = torch.dot(forward_world, straight_down).item()
            angle_down = torch.acos(torch.clamp(torch.tensor(dot_down, dtype=torch.float32), -1.0, 1.0)).item() * 180 / np.pi

            print(f"\n🎯 Camera Direction Debug:")
            print(f"   Camera Height: {camera_pos[2]:.3f}m (should be lower than robot)")
            print(f"   Robot Height: {robot_pos[2]:.3f}m")
            print(f"   Forward Vector: [{forward_world[0]:.3f}, {forward_world[1]:.3f}, {forward_world[2]:.3f}]")
            print(f"   Angle from down: {angle_down:.1f}°")
            print(f"   Applied Local Rot: [{self._cam_local_rot[env_idx][0]:.3f}, {self._cam_local_rot[env_idx][1]:.3f}, {self._cam_local_rot[env_idx][2]:.3f}, {self._cam_local_rot[env_idx][3]:.3f}]")

        except Exception as e:
            print(f"📷 Camera direction debug error: {e}")

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

            print("🔧 Creating carpet material with small pattern...")
            material_prim = stage.DefinePrim(material_path, "Material")
            shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")

            # texture_path = "/home/sripu715/Downloads/carpet.jpg"
            texture_path = "/home/sripu715/Downloads/catpet_al_lab.jpg"
            if os.path.exists(texture_path):
                print(f"✅ Found carpet texture: {texture_path}")
                uv_reader = UsdShade.Shader.Define(stage, f"{material_path}/UVReader")
                uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
                uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                transform_2d = UsdShade.Shader.Define(stage, f"{material_path}/Transform2d")
                transform_2d.CreateIdAttr("UsdTransform2d")
                transform_2d.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(500.0, 500.0))
                transform_2d.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
                texture_shader = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
                texture_shader.CreateIdAttr("UsdUVTexture")
                texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
                texture_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                texture_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(transform_2d.ConnectableAPI(), "result")
                texture_shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((1.5, 1.5, 1.5, 1.0))
                diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                diffuse_input.ConnectToSource(texture_shader.ConnectableAPI(), "rgb")
                print("✅ Applied carpet texture with 500x scaling and contrast boost")
            else:
                print("⚠️ Using solid color fallback")
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
            print("✅ Applied carpet with proper scaling")

        except Exception as e:
            print(f"❌ Error creating carpet material: {e}")
            import traceback
            traceback.print_exc()

    def _pre_physics_step(self, actions: torch.Tensor):
        # DEBUG MODE: Override actions to keep robot standing
        if self._debug_camera_mode and self._debug_steps < 100:
            self._debug_steps += 1
            # Use default joint positions (standing pose)
            self._actions = torch.zeros_like(actions)
            self._processed_actions = self._robot.data.default_joint_pos
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

        # ——— AUTO-CALIBRATE ONCE (first 20 steps) ———
        if not hasattr(self, "MIDAS_TO_M"):
            # First time only
            if not hasattr(self, "calib_vals"):
                self.calib_vals = []
                self.calib_step = 0
                print("AUTO-CALIB STARTED (20 steps)")

            if self.calib_step < 20:
                # Sample pure ground from env 0
                y1 = int(h * 0.88)
                x1, x2 = int(w * 0.38), int(w * 0.62)
                val = np.mean(depth_maps[0, y1:, x1:x2])
                self.calib_vals.append(val)
                self.calib_step += 1
                print(f"  [CALIB] step {self.calib_step}/20 → MiDaS {val:.1f}")
                # FALLBACK: use temporary ratio
                temp_ratio = 1570.0
                ground_vals = [np.mean(depth_maps[i, y1:, x1:x2]) for i in range(self.num_envs)]
                ground_depth = torch.tensor(ground_vals, device=self.device)
                return torch.clamp(ground_depth / temp_ratio, 0.05, 1.0)

            # CALIBRATION DONE
            avg = np.mean(self.calib_vals)
            self.MIDAS_TO_M = avg / 0.277
            print(f"\nAUTO-CALIBRATED! MIDAS_TO_M = {self.MIDAS_TO_M:.1f}")
            print(f"   Vision locked to 0.277 m when standing\n")

        # ——— NORMAL MODE ———
        y1 = int(h * 0.88)
        x1, x2 = int(w * 0.38), int(w * 0.62)
        ground_vals = [np.mean(depth_maps[i, y1:, x1:x2]) for i in range(self.num_envs)]
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
            tilt = torch.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)
            h = trunk_h * (1 - tilt * 0.2) - getattr(self, "_camera_offset", 0.10)
            return torch.clamp(h, 0.05, 1.0)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        gravity, angular_vel = self._get_imu_data()
        foot_contacts = self._get_foot_contact_binary()
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        # --- Camera pose update ---
        trunk_pos = self._robot.data.body_pos_w[:, self._trunk_idx]
        trunk_quat = self._robot.data.body_quat_w[:, self._trunk_idx]
        if trunk_pos.dim() == 3: trunk_pos = trunk_pos.squeeze(1)
        if trunk_quat.dim() == 3: trunk_quat = trunk_quat.squeeze(1)

        cam_pos = trunk_pos + quat_apply(trunk_quat, self._cam_local_pos)
        cam_quat = quat_mul(trunk_quat, self._cam_local_rot)

        self._camera.set_world_poses(cam_pos, cam_quat, convention="ros")
        self._camera._update_poses(list(range(self.num_envs)))
        self._camera.update(dt=self.physics_dt)

        # --- RGB Preprocessing (match your working version) ---
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

        # --- LIVE MiDaS ---
        with torch.no_grad():
            input_list = [self.midas_transform(img.cpu().numpy()) for img in rgb_upright]
            batch = torch.cat(input_list, dim=0).to(self.device).squeeze(1)

            prediction = self.midas(batch)
            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(1)
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=(rgb_input.shape[1], rgb_input.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        depth_maps = prediction.cpu().numpy()  # (N, H, W)
        self.last_depth_maps = depth_maps

        # --- Height from MiDaS (live calc) ---
        height_estimate = self._get_height_estimate(depth_maps).unsqueeze(1)  # (N,1)

        # --- Full obs ---
        obs = torch.cat([
            gravity, angular_vel, foot_contacts,
            joint_pos, joint_vel, height_estimate,
            self._actions
        ], dim=-1)

        # --- Save images/debug (your %5 logic) ---
        if self._global_step % 5 == 0:
            os.makedirs("camera_images", exist_ok=True)
            os.makedirs("midas_input_debug", exist_ok=True)

            for env_idx in range(min(3, self.num_envs)):
                save_image(rgb_upright[env_idx].permute(2, 0, 1),
                           f"camera_images/env_{env_idx}_step_{self._global_step:06d}.png")

                img_np = (rgb_upright[env_idx].cpu().numpy() * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"midas_input_debug/env_{env_idx}_step_{self._global_step:06d}.png", img_bgr)

            for env_idx in range(self.num_envs):
                depth = depth_maps[env_idx]
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
                cv2.imwrite(f"camera_images/env_{env_idx}_step_{self._global_step:06d}_depth.png", depth_color)

            print(f"Saved RGB, Depth, and MiDaS input at step {self._global_step}")

        # --- Debug print ---
        if self._global_step % 5 == 0:
            robot_pos = self._robot.data.root_pos_w
            cam_pos_actual = self._camera.data.pos_w
            cam_pos_expected = cam_pos
            errors = torch.norm(cam_pos_actual - cam_pos_expected, dim=-1)

            print(f"\nStep {self._global_step} - Camera + Depth Tracking (ALL {self.num_envs} envs):")
            for i in range(self.num_envs):
                r = robot_pos[i]
                c = cam_pos_actual[i]
                e = cam_pos_expected[i]
                err = errors[i].item()
                print(
                    f"  env {i:02d} | "
                    f"Robot: [{r[0]:6.3f}, {r[1]:6.3f}, {r[2]:6.3f}] | "
                    f"Camera: [{c[0]:6.3f}, {c[1]:6.3f}, {c[2]:6.3f}] | "
                    f"Expected: [{e[0]:6.3f}, {e[1]:6.3f}, {e[2]:6.3f}] | "
                    f"Error: {err:6.3f}m"
                )

            self._debug_camera_tracking()
            self._debug_camera_direction()
            self._debug_camera_transforms()

        return {"policy": obs}

    def _debug_camera_tracking(self):
        """
        Debug camera tracking for *all* environments in a single, compact line.
        Example output:
            🤖 Robot: [ 2.97, -1.50, 0.263]   📷 Camera (actual): [ 3.01, -1.51, 0.248]   🎯 Expected: [ 3.01, -1.51, 0.248]   Tracking Error: 0.000m
        """
        try:
            robot_pos = self._robot.data.root_pos_w          # (num_envs, 3)
            robot_quat = self._robot.data.root_quat_w        # (num_envs, 4)
            cam_pos = self._camera.data.pos_w                # (num_envs, 3)
            # Local offset & rotation (already broadcast to all envs)
            local_pos = self._cam_local_pos                  # (num_envs, 3)
            local_rot = self._cam_local_rot                  # (num_envs, 4)
            cam_pos_expected = robot_pos + quat_apply(robot_quat, local_pos)
            cam_quat_expected = quat_mul(robot_quat, local_rot)   # not printed but kept for sanity
            errors = torch.norm(cam_pos - cam_pos_expected, dim=-1)   # (num_envs,)
            print(f"\nStep {self._global_step} - Camera Tracking (ALL {self.num_envs} envs):")
            for i in range(self.num_envs):
                r = robot_pos[i]
                c = cam_pos[i]
                e = cam_pos_expected[i]
                err = errors[i].item()

                line = (
                    f"  env {i:02d} | "
                    f"Robot: [{r[0]:6.3f}, {r[1]:6.3f}, {r[2]:6.3f}] | "
                    f"Camera (actual): [{c[0]:6.3f}, {c[1]:6.3f}, {c[2]:6.3f}] | "
                    f"Expected: [{e[0]:6.3f}, {e[1]:6.3f}, {e[2]:6.3f}] | "
                    f"Tracking Error: {err:6.3f}m"
                )
                print(line)
            if errors[0] < 0.01:
                print("   EXCELLENT: Auto tracking working perfectly!")
            elif errors[0] < 0.05:
                print("   GOOD: Auto tracking with minimal error")
            elif errors[0] < 0.1:
                print("   ACCEPTABLE: Auto tracking with some error")
            else:
                print("   POOR: Auto tracking not working")

        except Exception as e:
            print(f"Debug error: {e}")

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
        return self._robot.data.projected_gravity_b, self._robot.data.root_ang_vel_b

    def _get_rewards(self) -> torch.Tensor:
        # ==============================================================
        # 1. LEG CONTACT
        # ==============================================================
        contact_binary = self._get_foot_contact_binary()
        leg_contact_reward = (torch.sum(contact_binary, dim=1) / 4.0) * 2.0

        # ==============================================================
        # 2. IMU UPRIGHT
        # ==============================================================
        gravity, _ = self._get_imu_data()
        roll_pitch_error = torch.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)
        imu_upright_reward = torch.exp(-roll_pitch_error * 5.0)

        # ==============================================================
        # 3. VISION HEIGHT
        # ==============================================================
        height_estimate = self._get_height_estimate(self.last_depth_maps)
        height_error = torch.abs(height_estimate - 0.32)
        height_reward = torch.exp(-height_error * 6.0) * 2.0

        # ==============================================================
        # 4. PITCH FROM DEPTH (left vs right ground)
        # ==============================================================
        h, w = self.last_depth_maps.shape[1], self.last_depth_maps.shape[2]
        y1 = int(h * 0.88)
        left  = np.mean(self.last_depth_maps[:, y1:, :w//3],      axis=(1,2))
        right = np.mean(self.last_depth_maps[:, y1:, 2*w//3:],    axis=(1,2))
        pitch_tilt = torch.tensor(left - right, device=self.device) / 100.0
        pitch_reward = torch.exp(-pitch_tilt.abs() * 12.0) * 0.8

        # ==============================================================
        # 5. CRASH PENALTY
        # ==============================================================
        crash_penalty = torch.clamp(0.18 - height_estimate, 0.0, 0.2) * 40.0

        # ==============================================================
        # 6. TOTAL REWARD
        # ==============================================================
        total_reward = (
            leg_contact_reward * 1.0 +
            imu_upright_reward * 2.0 +
            height_reward +
            pitch_reward -
            crash_penalty
        )

        # ==============================================================
        # 7. LOUD & CLEAR LOGGING (every 50 steps)
        # ==============================================================
        if self.episode_length_buf[0] % 50 == 0:
            print("\n" + "═" * 70, flush=True)
            print(f" TRAINING PROGRESS - Step {self.episode_length_buf[0].item():,}", flush=True)
            print(f"  Feet     : {contact_binary[0].tolist()}", flush=True)
            print(f"  Height   : {height_estimate[0].item():.3f}m  (target 0.32m)", flush=True)
            print(f"  Pitch    : {pitch_tilt[0].item():+.3f} rad  → Reward {pitch_reward[0].item():.3f}", flush=True)
            print(f"  Rewards  : Leg {leg_contact_reward[0].item():.2f} | "
                  f"IMU {imu_upright_reward[0].item():.2f} | "
                  f"Height {height_reward[0].item():.2f} | "
                  f"Pitch {pitch_reward[0].item():.2f} | "
                  f"Crash -{crash_penalty[0].item():.1f}", flush=True)
            print(f"  TOTAL    : {total_reward[0].item():.3f}", flush=True)
            print("═" * 70, flush=True)

        # ==============================================================
        # 8. EPISODE SUMS (ALL REWARDS INCLUDED)
        # ==============================================================
        rewards_dict = {
            "leg_contact_reward": leg_contact_reward,
            "imu_upright_reward": imu_upright_reward,
            "height_reward":      height_reward,
            "pitch_reward":       pitch_reward,
            "crash_penalty":      -crash_penalty  # negative for logging
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