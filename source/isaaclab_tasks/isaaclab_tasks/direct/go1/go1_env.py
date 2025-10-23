import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Camera
from isaaclab.utils.math import quat_apply, quat_mul, quat_conjugate
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
        print("🤖 Go1 Environment - Belly Camera Setup")

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

        # Camera tracking setup
        self._trunk_idx = self._robot.find_bodies("trunk")[0]

        # Camera position: 1cm below trunk center in trunk's local frame
        self._cam_local_pos = torch.tensor([0.0, 0.0, -0.01], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Camera rotation: Point straight down + 90° rotation to align image
        # [0, 1, 0, 0] (points down) * [0.7071, 0, 0, 0.7071] (90° around Z)
        # Result: [0, 0.7071, 0.7071, 0] for +90° rotation
        self._cam_local_rot = torch.tensor([0.0, 0.7071, -0.7071, 0.0], device=self.device).unsqueeze(0).repeat(
            self.num_envs, 1)

        print(f"🎯 CAMERA SETUP:")
        print(f"   Position: [0.0, 0.0, -0.01] in trunk local frame (1cm below)")
        print(f"   Rotation: [0, 1, 0, 0] pointing down")
        print(f"   Convention: ROS (optical axis +Z, up +Y, right +X)")

    def step(self, actions: torch.Tensor):
        """Override step to increment global step counter"""
        self._global_step += 1
        return super().step(actions)

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]
        self._camera = self.scene["camera"]
        self._apply_carpet_material()
        print("✅ Scene setup complete")

    def _debug_camera_coordinate_system(self):
        """Debug camera positioning and orientation"""
        try:
            env_idx = 0

            # Get camera data
            camera_rot_w = self._camera.data.quat_w_world[env_idx] if self.num_envs > 1 else \
            self._camera.data.quat_w_world[0]
            camera_pos_w = self._camera.data.pos_w[env_idx] if self.num_envs > 1 else self._camera.data.pos_w[0]

            # Get trunk data
            trunk_pos_w = self._robot.data.body_pos_w[:, self._trunk_idx]
            trunk_rot_w = self._robot.data.body_quat_w[:, self._trunk_idx]

            # Handle dimensions
            if trunk_pos_w.dim() == 3:
                trunk_pos_w = trunk_pos_w.squeeze(1)
            if trunk_rot_w.dim() == 3:
                trunk_rot_w = trunk_rot_w.squeeze(1)

            trunk_pos_w = trunk_pos_w[env_idx] if self.num_envs > 1 else trunk_pos_w[0]
            trunk_rot_w = trunk_rot_w[env_idx] if self.num_envs > 1 else trunk_rot_w[0]

            print(f"\n🎯 Camera Orientation:")

            # Camera forward direction (optical axis +Z in ROS)
            forward_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            forward_world = quat_apply(camera_rot_w.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0)

            # Camera up direction (image top, +Y in ROS)
            up_local = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            up_world = quat_apply(camera_rot_w.unsqueeze(0), up_local.unsqueeze(0)).squeeze(0)

            # Robot forward direction
            robot_forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            robot_forward_world = quat_apply(trunk_rot_w.unsqueeze(0), robot_forward_local.unsqueeze(0)).squeeze(0)

            # Calculate angles
            straight_down = torch.tensor([0.0, 0.0, -1.0], device=self.device)
            dot_down = torch.dot(forward_world, straight_down).item()
            angle_down = torch.acos(torch.clamp(torch.tensor(dot_down), -1.0, 1.0)).item() * 180 / np.pi

            # Image alignment with robot forward (project to XY plane)
            up_xy = up_world[:2] / (torch.norm(up_world[:2]) + 1e-6)
            forward_xy = robot_forward_world[:2] / (torch.norm(robot_forward_world[:2]) + 1e-6)
            dot_align = torch.dot(up_xy, forward_xy).item()
            angle_align = torch.acos(torch.clamp(torch.tensor(dot_align), -1.0, 1.0)).item() * 180 / np.pi

            print(
                f"   Camera Forward (+Z optical): [{forward_world[0].item():.3f}, {forward_world[1].item():.3f}, {forward_world[2].item():.3f}]")
            print(f"   Target (straight down): [0.000, 0.000, -1.000]")
            print(f"   Angle from down: {angle_down:.1f}°")
            print(
                f"   Camera Up (+Y image top): [{up_world[0].item():.3f}, {up_world[1].item():.3f}, {up_world[2].item():.3f}]")
            print(
                f"   Robot Forward: [{robot_forward_world[0].item():.3f}, {robot_forward_world[1].item():.3f}, {robot_forward_world[2].item():.3f}]")
            print(f"   Image/Robot alignment: {angle_align:.1f}°")

            print(f"\n📊 Camera Attachment:")
            print(
                f"   🤖 Trunk (world): [{trunk_pos_w[0].item():.3f}, {trunk_pos_w[1].item():.3f}, {trunk_pos_w[2].item():.3f}]m")
            print(
                f"   📷 Camera (world): [{camera_pos_w[0].item():.3f}, {camera_pos_w[1].item():.3f}, {camera_pos_w[2].item():.3f}]m")

            # Calculate relative position in trunk local frame
            rel_pos_world = camera_pos_w - trunk_pos_w
            trunk_rot_inv = quat_conjugate(trunk_rot_w.unsqueeze(0)).squeeze(0)
            rel_pos_local = quat_apply(trunk_rot_inv.unsqueeze(0), rel_pos_world.unsqueeze(0)).squeeze(0)

            expected_local = self._cam_local_pos[0]
            pos_error = torch.norm(rel_pos_local - expected_local).item()

            print(
                f"   📏 Relative (world): [{rel_pos_world[0].item():.4f}, {rel_pos_world[1].item():.4f}, {rel_pos_world[2].item():.4f}]m")
            print(
                f"   🔄 Relative (trunk local): [{rel_pos_local[0].item():.4f}, {rel_pos_local[1].item():.4f}, {rel_pos_local[2].item():.4f}]m")
            print(
                f"   🎯 Expected (trunk local): [{expected_local[0].item():.4f}, {expected_local[1].item():.4f}, {expected_local[2].item():.4f}]m")
            print(f"   ✓ Error: {pos_error:.4f}m")

            # Status
            if pos_error < 0.005 and angle_down < 15 and angle_align < 20:
                print(f"   ✅ PERFECT: Camera properly attached and oriented!")
            elif pos_error < 0.01 and angle_down < 60:
                print(f"   ✅ GOOD: Camera attached correctly (robot tilted during motion)")
            elif pos_error < 0.02:
                print(f"   ⚠️ OK: Minor positioning error (robot moving)")
            else:
                print(f"   ❌ ISSUE: Check camera attachment")

        except Exception as e:
            print(f"❌ Debug error: {e}")
            import traceback
            traceback.print_exc()

    def _debug_what_camera_sees(self):
        """Analyze camera image content"""
        try:
            env_idx = 0
            rgb = self._camera.data.output["rgb"].float() / 255.0
            image = rgb[env_idx] if self.num_envs > 1 else rgb.squeeze(0)

            height, width, _ = image.shape
            center_h, center_w = height // 2, width // 2
            region_size = min(height, width) // 4

            # Sample regions
            center = image[
                center_h - region_size:center_h + region_size, center_w - region_size:center_w + region_size, :]
            top = image[0:region_size, :, :]
            bottom = image[height - region_size:height, :, :]
            left = image[:, 0:region_size, :]
            right = image[:, width - region_size:width, :]

            print(f"\n🔍 Image Content:")
            print(f"   Center: {torch.mean(center).item():.3f} (ground below)")
            print(f"   Top: {torch.mean(top).item():.3f} (robot forward)")
            print(f"   Bottom: {torch.mean(bottom).item():.3f} (robot back)")
            print(f"   Left: {torch.mean(left).item():.3f}")
            print(f"   Right: {torch.mean(right).item():.3f}")

            # Check for robot parts
            if torch.mean(left).item() < torch.mean(center).item() - 0.3 or torch.mean(right).item() < torch.mean(
                    center).item() - 0.3:
                print(f"   ⚠️ Robot legs visible (dark spots)")
            else:
                print(f"   ✅ Clean ground view")

        except Exception as e:
            print(f"❌ Image analysis error: {e}")

    def _apply_carpet_material(self):
        stage = omni.usd.get_context().get_stage()
        ground_prim = stage.GetPrimAtPath("/World/ground")
        if not ground_prim.IsValid():
            print("⚠️ Ground prim not found")
            return

        material_path = "/World/Materials/BerberCarpet"
        try:
            existing_material = stage.GetPrimAtPath(material_path)
            if existing_material.IsValid():
                stage.RemovePrim(material_path)

            material_prim = stage.DefinePrim(material_path, "Material")
            shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")

            texture_path = "/home/sripu715/Downloads/carpet.jpg"
            if os.path.exists(texture_path):
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
                print("✅ Carpet texture applied")
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

        except Exception as e:
            print(f"❌ Material error: {e}")

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # Get robot state
        gravity, angular_vel = self._get_imu_data()
        foot_contacts = self._get_foot_contact_binary()
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        height = self._robot.data.root_pos_w[:, 2:3]

        # State observation (47D)
        state = torch.cat([
            gravity,  # 3
            angular_vel,  # 3
            foot_contacts,  # 4
            joint_pos,  # 12
            joint_vel,  # 12
            height,  # 1
            self._actions  # 12
        ], dim=-1)

        # Update camera pose to follow trunk - PROPER METHOD
        trunk_pos = self._robot.data.body_pos_w[:, self._trunk_idx]
        trunk_quat = self._robot.data.body_quat_w[:, self._trunk_idx]

        # Handle dimension squeezing
        if trunk_pos.dim() == 3:
            trunk_pos = trunk_pos.squeeze(1)
        if trunk_quat.dim() == 3:
            trunk_quat = trunk_quat.squeeze(1)

        # Calculate camera world pose from trunk pose + local offset
        cam_pos_world = trunk_pos + quat_apply(trunk_quat, self._cam_local_pos)
        cam_quat_world = quat_mul(trunk_quat, self._cam_local_rot)

        # Set camera pose using proper method
        self._camera.set_world_poses(cam_pos_world, cam_quat_world, convention="ros")

        # CRITICAL: Must call _update_poses to actually apply the transform!
        # Without this, the camera stays at initialization position
        env_ids = list(range(self.num_envs))
        self._camera._update_poses(env_ids)

        # Now update camera data with new pose
        self._camera.update(dt=self.physics_dt)

        # Get camera images
        rgb = self._camera.data.output["rgb"].float() / 255.0

        # Periodic debugging and image saving
        if self._global_step % 1 == 0:
            os.makedirs("camera_images", exist_ok=True)
            for env_idx in range(min(3, self.num_envs)):
                img_tensor = rgb[env_idx].squeeze(0).permute(2, 0, 1) if rgb.dim() == 4 else rgb.permute(2, 0, 1)
                save_image(img_tensor, f"camera_images/env_{env_idx}_step_{self._global_step:06d}.png")

            self._debug_camera_coordinate_system()
            self._debug_what_camera_sees()
            print(f"💾 Saved at step {self._global_step}\n")

        return {"policy": state}

    def _get_foot_contact_binary(self) -> torch.Tensor:
        try:
            if not hasattr(self._contact_sensor.data, 'net_forces_w'):
                return torch.zeros(self.num_envs, 4, device=self.device)
            contact_forces = torch.norm(self._contact_sensor.data.net_forces_w, dim=-1)
            binary_contact = (contact_forces > 1.0).float()
            return binary_contact[:, :4] if binary_contact.shape[1] >= 4 else torch.zeros(self.num_envs, 4,
                                                                                          device=self.device)
        except:
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