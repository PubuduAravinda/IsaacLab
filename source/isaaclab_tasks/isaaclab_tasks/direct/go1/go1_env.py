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

        # LOWER CAMERA: Closer to ground for better pattern visibility
        self._cam_local_pos = torch.tensor([0.0, 0.0, -0.18], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self._cam_local_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs,
                                                                                                         1)

        print(f"   Camera Offset: [0.0, 0.0, -0.18] (lowered for better ground view)")
        print(f"   Camera Rotation: [0.0, 0.0, 0.0, 1.0] (straight down)")

    def _debug_camera_transforms(self):
        """Debug the complete camera transform chain"""
        try:
            env_idx = 0
            robot_pos = self._robot.data.root_pos_w[env_idx]
            robot_rot = self._robot.data.root_quat_w[env_idx]
            camera_pos = self._camera.data.pos_w[env_idx]
            camera_rot = self._camera.data.quat_w_world[env_idx]

            print(f"\n🔧 COMPLETE TRANSFORM DEBUG:")
            print(f"   Robot Position: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
            print(
                f"   Robot Rotation: [{robot_rot[0]:.3f}, {robot_rot[1]:.3f}, {robot_rot[2]:.3f}, {robot_rot[3]:.3f}]")
            print(f"   Camera Position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
            print(
                f"   Camera Rotation: [{camera_rot[0]:.3f}, {camera_rot[1]:.3f}, {camera_rot[2]:.3f}, {camera_rot[3]:.3f}]")

            # Test what happens if we set camera to identity rotation
            test_identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            test_forward = quat_apply(test_identity.unsqueeze(0),
                                      torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0)).squeeze(0)
            print(f"   Identity Forward: [{test_forward[0]:.3f}, {test_forward[1]:.3f}, {test_forward[2]:.3f}]")

            # Test what the current local rotation produces
            local_forward = quat_apply(self._cam_local_rot[env_idx].unsqueeze(0),
                                       torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0)).squeeze(0)
            print(f"   Local Rot Forward: [{local_forward[0]:.3f}, {local_forward[1]:.3f}, {local_forward[2]:.3f}]")

        except Exception as e:
            print(f"📷 Transform debug error: {e}")

    def step(self, actions: torch.Tensor):
        """Override step method to increment global step counter"""
        # Call parent step method
        result = super().step(actions)

        # Increment global step counter
        self._global_step += 1

        return result

    def _setup_scene(self):
        """Setup scene - camera will automatically follow robot as child prim"""
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]
        self._camera = self.scene["camera"]

        # Apply carpet material to ground plane - CALL THIS AFTER SCENE IS SETUP
        self._apply_carpet_material()

        print("🔧 Scene Setup Complete:")
        print(f"   Robot prim_path: {self._robot.cfg.prim_path}")
        print(f"   Camera prim_path: {self._camera.cfg.prim_path}")
        print("✅ Camera is child of robot trunk - will follow automatically")
        print("✅ Berber carpet material applied to ground plane")


    def _debug_camera_direction(self):
        """Debug what direction the camera is pointing with more details"""
        try:
            env_idx = 0
            robot_pos = self._robot.data.root_pos_w[env_idx]
            camera_pos = self._camera.data.pos_w[env_idx]
            camera_rot = self._camera.data.quat_w_world[env_idx]

            # Calculate forward vector from camera rotation
            forward_local = torch.tensor([0.0, 0.0, -1.0], device=self.device)
            forward_world = quat_apply(camera_rot.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0)

            # Calculate angle from horizontal
            horizontal_angle = torch.atan2(forward_world[2], torch.sqrt(forward_world[0] ** 2 + forward_world[1] ** 2))
            angle_degrees = torch.abs(horizontal_angle * 180 / torch.pi).item()

            print(f"\n🎯 Camera Direction Debug:")
            print(f"   Camera Height: {camera_pos[2]:.3f}m (should be lower than robot)")
            print(f"   Robot Height: {robot_pos[2]:.3f}m")
            print(f"   Forward Vector: [{forward_world[0]:.3f}, {forward_world[1]:.3f}, {forward_world[2]:.3f}]")
            print(f"   Angle from horizontal: {angle_degrees:.1f}°")
            print(
                f"   Applied Local Rot: [{self._cam_local_rot[env_idx][0]:.3f}, {self._cam_local_rot[env_idx][1]:.3f}, {self._cam_local_rot[env_idx][2]:.3f}, {self._cam_local_rot[env_idx][3]:.3f}]")

            # Check camera direction
            if forward_world[2] < -0.8:
                print("   ✅ Camera is pointing STRAIGHT DOWN at ground")
            elif forward_world[2] < -0.5:
                print("   ✅ Camera is pointing DOWNWARD at ground")
            elif forward_world[2] < -0.2:
                print("   ⚠️  Camera is pointing somewhat downward")
            elif forward_world[2] < 0.2:
                print("   ❌ Camera is pointing FORWARD (horizontal)")
            else:
                print("   ❌ Camera is pointing UPWARD, not at ground!")

            # Check if camera is low enough to see ground
            if camera_pos[2] < 0.15:
                print("   ✅ Camera is low enough to see ground")
            else:
                print("   ⚠️  Camera might be too high to see ground clearly")

        except Exception as e:
            print(f"📷 Camera direction debug error: {e}")


    def _apply_carpet_material(self):
        """Apply carpet texture with much smaller pattern scaling"""
        stage = omni.usd.get_context().get_stage()
        ground_prim = stage.GetPrimAtPath("/World/ground")
        if not ground_prim.IsValid():
            print("❌ Error: Ground prim at '/World/ground' not found")
            return

        material_path = "/World/Materials/BerberCarpet"

        try:
            # Remove existing material
            existing_material = stage.GetPrimAtPath(material_path)
            if existing_material.IsValid():
                stage.RemovePrim(material_path)

            print("🔧 Creating carpet material with small pattern...")

            # Create material
            material_prim = stage.DefinePrim(material_path, "Material")
            shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")

            # Texture path
            # texture_path = "/home/sripu715/Downloads/Carpet_Berber_Gray.jpg"
            texture_path = "/home/sripu715/Downloads/carpet.jpg"

            if os.path.exists(texture_path):
                print(f"✅ Found carpet texture: {texture_path}")

                # Create UV reader
                uv_reader = UsdShade.Shader.Define(stage, f"{material_path}/UVReader")
                uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
                uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

                # MAJOR CHANGE: Much larger scale to make pattern smaller
                # Scale of 10.0 means the texture repeats 10 times, making each instance smaller
                transform_2d = UsdShade.Shader.Define(stage, f"{material_path}/Transform2d")
                transform_2d.CreateIdAttr("UsdTransform2d")
                transform_2d.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
                    Gf.Vec2f(500.0, 500.0))  # Much larger scale = smaller pattern
                transform_2d.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(),
                                                                                          "result")

                # Create texture shader
                texture_shader = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
                texture_shader.CreateIdAttr("UsdUVTexture")
                texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
                texture_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                texture_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    transform_2d.ConnectableAPI(), "result"
                )

                # Connect texture to diffuse color
                diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                diffuse_input.ConnectToSource(texture_shader.ConnectableAPI(), "rgb")

                print("✅ Applied carpet texture with 20x scaling (small pattern)")
            else:
                # Fallback - use a solid color
                print("⚠️ Using solid color fallback")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.2, 0.8))

            # Material properties
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

            # Create material
            material = UsdShade.Material(material_prim)
            surface_output = material.CreateSurfaceOutput()
            surface_output.ConnectToSource(shader.ConnectableAPI(), "surface")

            # Apply proper UV coordinates
            if ground_prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(ground_prim)
                # Set UVs for the entire ground
                st_primvar = mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
                st_primvar.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

            # Bind material
            binding_api = UsdShade.MaterialBindingAPI.Apply(ground_prim)
            binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)

            print("✅ Applied carpet with proper scaling")

        except Exception as e:
            print(f"❌ Error creating carpet material: {e}")
            import traceback
            traceback.print_exc()


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

        # Get camera RGB data and normalize to [0,1]
        rgb = self._camera.data.output["rgb"].float() / 255.0

        # ENHANCED: Apply image processing for clearer carpet pattern
        if self._global_step % 200 == 0:
            # For saved images only: enhance contrast and brightness
            rgb_enhanced = torch.clamp(rgb * 1.4, 0.0, 1.0)  # Increase contrast
            # Add slight sharpening (edge enhancement)
            save_rgb = rgb_enhanced
        else:
            save_rgb = rgb

        # Flatten RGB for observations
        rgb_flat = rgb.flatten(start_dim=1)

        # Combine state and flattened RGB
        obs = torch.cat([state, rgb_flat], dim=-1)

        # Update camera data poses
        cam_offset_world = quat_apply(self._robot.data.root_quat_w, self._cam_local_pos)
        self._camera.data.pos_w = self._robot.data.root_pos_w + cam_offset_world
        self._camera.data.quat_w_world = quat_mul(self._robot.data.root_quat_w, self._cam_local_rot)

        # TEMPORARY: Verify the quaternion is correct
        if self._global_step % 50 == 0:
            test_quat = self._cam_local_rot[0]
            print(
                f"🔧 APPLIED CAMERA ROTATION: [{test_quat[0]:.4f}, {test_quat[1]:.4f}, {test_quat[2]:.4f}, {test_quat[3]:.4f}]")

        # Save high-quality enhanced images every 200 steps
        if self._global_step % 200 == 0:
            self._debug_camera_tracking()
            self._debug_camera_direction()
            self._debug_camera_transforms()

            # Create images directory if it doesn't exist
            os.makedirs("camera_images", exist_ok=True)

            # Save multiple environment images with enhancements
            for env_idx in range(min(3, self.num_envs)):
                img_tensor = save_rgb[env_idx].permute(2, 0, 1)

                # Save both original and enhanced versions
                save_image(img_tensor, f"camera_images/env_{env_idx}_step_{self._global_step:06d}.png")

                # Save a close-up crop if needed (center 50%)
                h, w = img_tensor.shape[1], img_tensor.shape[2]
                crop_h, crop_w = h // 2, w // 2
                start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
                cropped = img_tensor[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
                save_image(cropped, f"camera_images/env_{env_idx}_step_{self._global_step:06d}_cropped.png")

            print(f"💾 Saved enhanced belly camera images at step {self._global_step}")

        return {"policy": obs}



    def _save_camera_debug_image(self, rgb_image, step):
        """Save a debug image with camera info overlay"""
        try:
            # Convert to numpy for processing
            img_np = (rgb_image.cpu().numpy() * 255).astype(np.uint8)

            # Create a simple text overlay (you can use OpenCV if available)
            # For now, just save the raw image
            debug_path = f"camera_images/debug_step_{step:06d}.png"
            save_image(torch.tensor(img_np).permute(2, 0, 1).float() / 255.0, debug_path)

        except Exception as e:
            print(f"Could not create debug image: {e}")


    def _debug_camera_tracking(self):
        """Debug camera tracking - use CURRENT camera offsets"""
        try:
            if (hasattr(self._robot, 'data') and hasattr(self._camera, 'data') and
                    self._robot.data.root_pos_w is not None and self._camera.data.pos_w is not None):

                env_idx = 0
                robot_pos = self._robot.data.root_pos_w[env_idx]
                camera_pos = self._camera.data.pos_w[env_idx]
                robot_rot = self._robot.data.root_quat_w[env_idx]

                # Use CURRENT camera offset from your initialization
                cam_offset_local = self._cam_local_pos[env_idx]  # This is [0.08, 0.0, -0.05]
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
                print(
                    f"   CAMERA OFFSET (local): [{cam_offset_local[0]:.2f}, {cam_offset_local[1]:.2f}, {cam_offset_local[2]:.2f}]")
                print(f"   POSITIONS:")
                print(f"   🤖 Robot:    [{robot_pos[0]:6.2f}, {robot_pos[1]:6.2f}, {robot_pos[2]:6.3f}]")
                print(f"   📷 Camera (actual):   [{camera_pos[0]:6.2f}, {camera_pos[1]:6.2f}, {camera_pos[2]:6.3f}]")
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