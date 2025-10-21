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

        self._cam_local_pos = torch.tensor([0.0, 0.0, -0.20], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self._cam_local_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # Identity
        print(f"   Camera Offset: [0.0, 0.0, -0.20] (lowered for better ground view)")
        print(f"   Camera Rotation: [0.0, 0.0, 0.0, 1.0] (identity, adjusted by trunk orientation)")

        self._trunk_idx = self._robot.find_bodies("trunk")[0]

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

            test_identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            test_forward = quat_apply(test_identity.unsqueeze(0), torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0)).squeeze(0)
            print(f"   Identity Forward: [{test_forward[0]:.3f}, {test_forward[1]:.3f}, {test_forward[2]:.3f}]")

            local_forward = quat_apply(self._cam_local_rot[env_idx].unsqueeze(0), torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0)).squeeze(0)
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

            forward_local = torch.tensor([0.0, 0.0, -1.0], device=self.device)  # Downward vector
            forward_world = quat_apply(camera_rot.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0)

            horizontal_angle = torch.atan2(forward_world[2], torch.sqrt(forward_world[0]**2 + forward_world[1]**2))
            angle_degrees = torch.abs(horizontal_angle * 180 / torch.pi).item()

            print(f"\n🎯 Camera Direction Debug:")
            print(f"   Camera Height: {camera_pos[2]:.3f}m (should be lower than robot)")
            print(f"   Robot Height: {robot_pos[2]:.3f}m")
            print(f"   Forward Vector: [{forward_world[0]:.3f}, {forward_world[1]:.3f}, {forward_world[2]:.3f}]")
            print(f"   Angle from horizontal: {angle_degrees:.1f}°")
            print(f"   Applied Local Rot: [{self._cam_local_rot[env_idx][0]:.3f}, {self._cam_local_rot[env_idx][1]:.3f}, {self._cam_local_rot[env_idx][2]:.3f}, {self._cam_local_rot[env_idx][3]:.3f}]")

            if forward_world[2] < -0.8:
                print("   ✅ Camera is pointing STRAIGHT DOWN at ground")
            elif forward_world[2] < -0.5:
                print("   ✅ Camera is pointing DOWNWARD at ground")
            elif forward_world[2] < -0.2:
                print("   ⚠️ Camera is pointing somewhat downward")
            elif forward_world[2] < 0.2:
                print("   ❌ Camera is pointing FORWARD (horizontal)")
            else:
                print("   ❌ Camera is pointing UPWARD, not at ground!")

            if camera_pos[2] < 0.15:
                print("   ✅ Camera is low enough to see ground")
            else:
                print("   ⚠️ Camera might be too high to see ground clearly")

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

            texture_path = "/home/sripu715/Downloads/carpet.jpg"
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
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        gravity, angular_vel = self._get_imu_data()
        foot_contacts = self._get_foot_contact_binary()
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        height = self._robot.data.root_pos_w[:, 2:3]

        # State-only observation (47D)
        state = torch.cat([
            gravity,  # 3
            angular_vel,  # 3
            foot_contacts,  # 4
            joint_pos,  # 12
            joint_vel,  # 12
            height,  # 1
            self._actions  # 12
        ], dim=-1)
        assert state.shape[1] == 47, f"State shape mismatch: expected 47, got {state.shape[1]}"
        print(f"Debug: observation shape = {state.shape}")

        # Manual camera pose update using trunk
        trunk_pos = self._robot.data.body_pos_w[:, self._trunk_idx].squeeze(1)  # Remove extra dimension
        trunk_quat = self._robot.data.body_quat_w[:, self._trunk_idx].squeeze(1)  # Remove extra dimension
        cam_pos = quat_apply(trunk_quat, self._cam_local_pos) + trunk_pos
        cam_quat = quat_mul(trunk_quat, self._cam_local_rot)  # Should now match shapes
        self._camera.set_world_poses(cam_pos, cam_quat, convention="world")  # Apply pose

        # Camera and RGB processing for visualization only (not included in obs)
        rgb = self._camera.data.output["rgb"].float() / 255.0
        if self._global_step % 50 == 0:
            roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
            print(f"Robot roll_pitch: {roll_pitch[0].item():.3f}")
            rgb_enhanced = torch.clamp(rgb * 1.3, 0.0, 1.0)
            kernel = torch.tensor([[[[0, -0.2, 0], [-0.2, 2.0, -0.2], [0, -0.2, 0]]]], device=self.device)
            kernel = kernel.repeat(3, 1, 1, 1)
            rgb_permuted = rgb.permute(0, 3, 1, 2)
            rgb_sharpened = torch.nn.functional.conv2d(rgb_permuted, kernel, padding=1, groups=3)
            rgb_sharpened = rgb_sharpened.permute(0, 2, 3, 1)
            rgb_enhanced = torch.clamp(rgb_sharpened, 0.0, 1.0)
            save_rgb = rgb_enhanced

            # Flip the image vertically to correct orientation
            save_rgb = torch.flip(save_rgb, dims=[1])  # Flip along height dimension

            self._debug_camera_tracking()
            self._debug_camera_direction()
            self._debug_camera_transforms()
            os.makedirs("camera_images", exist_ok=True)
            for env_idx in range(min(3, self.num_envs)):
                img_tensor = save_rgb[env_idx].permute(2, 0, 1)
                save_image(img_tensor, f"camera_images/env_{env_idx}_step_{self._global_step:06d}.png")
                h, w = img_tensor.shape[1], img_tensor.shape[2]
                crop_h, crop_w = h // 2, w // 2
                start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
                cropped = img_tensor[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
                save_image(cropped, f"camera_images/env_{env_idx}_step_{self._global_step:06d}_cropped.png")
            print(f"💾 Saved enhanced belly camera images at step {self._global_step}")

        if self._global_step % 50 == 0:
            test_quat = self._cam_local_rot[0]
            print(f"🔧 APPLIED CAMERA ROTATION: [{test_quat[0]:.4f}, {test_quat[1]:.4f}, {test_quat[2]:.4f}, {test_quat[3]:.4f}]")

        return {"policy": state}

    def _debug_camera_tracking(self):
        try:
            if (hasattr(self._robot, 'data') and hasattr(self._camera, 'data') and
                    self._robot.data.root_pos_w is not None and self._camera.data.pos_w is not None):
                env_idx = 0
                robot_pos = self._robot.data.root_pos_w[env_idx]
                camera_pos = self._camera.data.pos_w[env_idx]
                robot_rot = self._robot.data.root_quat_w[env_idx]
                cam_offset_local = self._cam_local_pos[env_idx]
                cam_offset_world = quat_apply(robot_rot.unsqueeze(0), cam_offset_local.unsqueeze(0)).squeeze(0)
                expected_pos = robot_pos + cam_offset_world
                current_error = torch.norm(camera_pos - expected_pos).item()
                distance = torch.norm(robot_pos - camera_pos).item()

                print(f"\n📊 Step {self._global_step} - Camera Tracking (AUTO):")
                print(f"   Update latest camera pose: {self._camera.cfg.update_latest_camera_pose}")
                print(f"   Camera frame count: {self._camera._frame}")
                print(f"   PRIM PATHS:")
                print(f"   🤖 Robot:  {self._robot.cfg.prim_path}")
                print(f"   📷 Camera: {self._camera.cfg.prim_path}")
                print(f"   CAMERA OFFSET (local): [{cam_offset_local[0]:.2f}, {cam_offset_local[1]:.2f}, {cam_offset_local[2]:.2f}]")
                print(f"   POSITIONS:")
                print(f"   🤖 Robot:    [{robot_pos[0]:6.2f}, {robot_pos[1]:6.2f}, {robot_pos[2]:6.3f}]")
                print(f"   📷 Camera (actual):   [{camera_pos[0]:6.2f}, {camera_pos[1]:6.2f}, {camera_pos[2]:6.3f}]")
                print(f"   🎯 Expected: [{expected_pos[0]:6.2f}, {expected_pos[1]:.2f}, {expected_pos[2]:6.3f}]")
                print(f"   METRICS:")
                print(f"   📏 Robot-Camera Distance: {distance:6.3f}m")
                print(f"   ❌ Tracking Error:        {current_error:6.3f}m")

                if current_error < 0.01:
                    print("   ✅ EXCELLENT: Auto tracking working perfectly!")
                elif current_error < 0.05:
                    print("   ✅ GOOD: Auto tracking with minimal error")
                elif current_error < 0.1:
                    print("   ⚠️ ACCEPTABLE: Auto tracking with some error")
                else:
                    print("   ❌ POOR: Auto tracking not working")
                    print("   🔧 TROUBLESHOOTING: Check camera parent-child relationship in USD")

        except Exception as e:
            print(f"📷 Debug error: {e}")

    def _get_foot_contact_binary(self) -> torch.Tensor:
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
        return self._robot.data.projected_gravity_b, self._robot.data.root_ang_vel_b

    def _get_rewards(self) -> torch.Tensor:
        contacts = self._get_foot_contact_binary()
        contact_reward = (torch.sum(contacts, dim=1) / 4.0) * 2.0
        gravity, _ = self._get_imu_data()
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        upright_reward = torch.exp(-roll_pitch * 10.0) * 2.0
        height = self._robot.data.root_pos_w[:, 2]
        height_error = torch.abs(height - self._target_height)
        height_reward = torch.exp(-height_error * 4.0) * 2.0
        total_reward = contact_reward + upright_reward + height_reward
        self._episode_sums["leg_contact_reward"] += contact_reward
        self._episode_sums["imu_upright_reward"] += upright_reward
        self._episode_sums["height_reward"] += height_reward
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