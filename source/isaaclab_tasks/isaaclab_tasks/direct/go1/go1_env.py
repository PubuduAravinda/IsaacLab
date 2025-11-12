import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_mul
import omni.usd
from pxr import Usd, UsdShade, Sdf, Gf, UsdGeom
import os
from .go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg
import numpy as np
from scipy.spatial.transform import Rotation as R


class Go1Env(DirectRLEnv):
    cfg: Go1FlatEnvCfg | Go1RoughEnvCfg

    def __init__(self, cfg: Go1FlatEnvCfg | Go1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)

        self._episode_sums = {
            "leg_contact_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "imu_upright_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "height_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "velocity_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),  # NEW
            "torque_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "action_smoothness_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        self._target_height = 0.32
        self._global_step = 0

        # DEBUG MODE: Set to True to freeze robot in standing pose for camera testing
        self._debug_camera_mode = False
        self._debug_steps = 0

        self._trunk_idx = self._robot.find_bodies("trunk")[0]

        # Height tracking
        self._raycaster_height = torch.zeros(self.num_envs, device=self.device)
        self._height_errors = []

    def step(self, actions: torch.Tensor):
        self._global_step += 1
        return super().step(actions)

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]
        self._raycaster = self.scene["raycaster"]
        self._apply_carpet_material()

    def _get_raycaster_height(self) -> torch.Tensor:
        """Get trunk-to-ground distance using raycaster"""
        try:
            ray_origins = self._raycaster.data.pos_w
            ray_hits = self._raycaster.data.ray_hits_w
            heights = torch.zeros(self.num_envs, device=self.device)

            for env_idx in range(self.num_envs):
                env_origins = ray_origins[env_idx]
                env_hits = ray_hits[env_idx]
                distances = env_origins[..., 2] - env_hits[..., 2]  # Relative z (like ANYmal – stable for tilt)
                valid_mask = distances > 0.01
                valid_distances = distances[valid_mask]

                if len(valid_distances) > 0:
                    heights[env_idx] = torch.min(valid_distances)  # Min height (ground closest)
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

        except Exception as e:
            pass

    def _pre_physics_step(self, actions: torch.Tensor):
        if self._debug_camera_mode and self._debug_steps < 100:
            self._debug_steps += 1
            self._actions = torch.zeros_like(actions)
            self._processed_actions = self._robot.data.default_joint_pos
        else:
            self._actions = actions.clone()
            self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        self.scene.write_data_to_sim()

    def _get_height_estimate(self) -> torch.Tensor:
        trunk_h = self._robot.data.root_pos_w[:, 2]
        gravity = self._robot.data.projected_gravity_b
        tilt = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        h = trunk_h * (1 - tilt * 0.2) - 0.10
        return torch.clamp(h, 0.05, 1.0)

    def _debug_system_status(self):
        """Consolidated debug for robot and raycaster status"""
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

        # Raycaster state (auto-tracked! No 'quat' needed)
        ray_pos = self._raycaster.data.pos_w[env_idx]  # Reflects trunk + offset
        raycaster_h = self._raycaster_height[env_idx].item()

        print(f"\n📊 SYSTEM STATUS (env {env_idx}, step {self._global_step}):")
        print(f"🤖 Robot: Pos[{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}] "
              f"Rot[Roll={roll_deg:5.1f}°, Pitch={pitch_deg:5.1f}°] "
              f"Joints[{joint_diff:5.3f}]")
        print(f"📏 Raycaster: Pos[{ray_pos[0]:.3f}, {ray_pos[1]:.3f}, {ray_pos[2]:.3f}] "
              f"Height={raycaster_h:5.3f}m")


    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # --- IMU & kinematics ---
        gravity, angular_vel = self._get_imu_data()
        foot_contacts = self._get_foot_contact_binary()
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        # --- HEIGHT from raycaster (auto-attached to trunk — no manual pose update!) ---
        raycaster_height = self._get_raycaster_height()
        self._raycaster_height = raycaster_height
        height_estimate = raycaster_height.unsqueeze(1)  # (N, 1)

        # --- DEBUG SYSTEM STATUS ---
        if (self._debug_camera_mode and (self._debug_steps <= 10 or self._debug_steps % 20 == 0)) or \
                (self._global_step % 100 == 0):
            self._debug_system_status()

        # --- FINAL OBSERVATION ---
        obs = torch.cat([
            gravity,  # (N, 3)
            angular_vel,  # (N, 3)
            foot_contacts,  # (N, 4)
            joint_pos,  # (N, 12)
            joint_vel,  # (N, 12)
            height_estimate,  # (N, 1)
            self._actions  # (N, 12)
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

    # --------------------------------------------------------------
    # 2. NEW _get_rewards() – stable & smooth standing
    # --------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # ----- contacts -------------------------------------------------
        contact_binary = self._get_foot_contact_binary()
        leg_contact_reward = (torch.sum(contact_binary, dim=1) / 4.0) * 2.0

        # ----- orientation (IMU) ----------------------------------------
        gravity, angular_vel = self._get_imu_data()
        roll_pitch_error = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        imu_upright_reward = torch.exp(-roll_pitch_error * 5.0) * 2.0

        # ----- height ---------------------------------------------------
        height_estimate = self._raycaster_height
        height_error = torch.abs(height_estimate - 0.32)
        height_reward = torch.exp(-height_error * 6.0) * 2.0

        # ----- **NEW** low-velocity reward (stillness) -------------------
        base_lin_vel = self._robot.data.root_lin_vel_b          # (N,3)
        base_ang_vel = self._robot.data.root_ang_vel_b          # (N,3)
        # Combine with a small weight on angular velocity
        vel_mag = torch.norm(base_lin_vel, dim=1) + 0.3 * torch.norm(base_ang_vel, dim=1)
        velocity_reward = torch.exp(-vel_mag * 10.0) * 1.0

        # ----- crash ----------------------------------------------------
        crash_penalty = torch.clamp(0.18 - height_estimate, 0.0, 0.2) * 40.0

        # ----- torque ---------------------------------------------------
        torques = self._robot.data.applied_torque
        torque_penalty = -0.0001 * torch.sum(torques ** 2, dim=1)

        # ----- **REDUCED** action-smoothness (was -0.5 → -0.05) ----------
        action_diff = self._actions - self._previous_actions
        action_smoothness_penalty = -0.05 * torch.sum(action_diff ** 2, dim=1)

        # ----- total ----------------------------------------------------
        total_reward = (
            leg_contact_reward * 1.0 +
            imu_upright_reward * 2.0 +
            height_reward +
            velocity_reward +                     # NEW
            -crash_penalty +
            torque_penalty +
            action_smoothness_penalty
        )

        # ----- logging --------------------------------------------------
        if self._global_step % 5000 == 0:
            print("\n" + "═" * 70)
            print(f" TRAINING PROGRESS - Step {self.episode_length_buf[0].item():,}")
            print(f"  Feet: {contact_binary[0].tolist()}")
            print(f"  Height: {height_estimate[0].item():.3f}m (target 0.32m)")
            print(f"  Rewards: Leg {leg_contact_reward[0].item():.2f} | "
                  f"IMU {imu_upright_reward[0].item():.2f} | "
                  f"Height {height_reward[0].item():.2f} | "
                  f"Vel {velocity_reward[0].item():.2f}")          # NEW
            print(f"  Penalties: Torque {torque_penalty[0].item():.3f} | "
                  f"Smooth {action_smoothness_penalty[0].item():.3f} | "
                  f"Crash -{crash_penalty[0].item():.1f}")
            print(f"  TOTAL: {total_reward[0].item():.3f}")
            print("═" * 70)

        # ----- episode sums ---------------------------------------------
        rewards_dict = {
            "leg_contact_reward": leg_contact_reward,
            "imu_upright_reward": imu_upright_reward,
            "height_reward":      height_reward,
            "velocity_reward":    velocity_reward,          # NEW
            "crash_penalty":      -crash_penalty,
            "torque_penalty":     torque_penalty,
            "action_smoothness_penalty": action_smoothness_penalty,
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