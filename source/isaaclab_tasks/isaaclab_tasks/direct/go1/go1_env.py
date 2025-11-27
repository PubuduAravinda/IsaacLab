# go1_env.py - SIMPLIFIED with detailed debugging and removed velocity/torque penalties
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
        print("🤖 Go1 Environment - Simplified Variable Impedance Control")
        print(f"🎯 Action Space: {self.cfg.action_space}D (12 pos + 12 KP + 12 KD)")

        # Now actions are 36D: [positions(12), kp(12), kd(12)]
        self._actions = torch.zeros(self.num_envs, 36, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # Separate tensors for each component
        self._target_positions = torch.zeros(self.num_envs, 12, device=self.device)
        self._kp_gains = torch.zeros(self.num_envs, 12, device=self.device)
        self._kd_gains = torch.zeros(self.num_envs, 12, device=self.device)

        self._episode_sums = {
            "leg_contact_reward": torch.zeros(self.num_envs, device=self.device),
            "imu_upright_reward": torch.zeros(self.num_envs, device=self.device),
            "height_reward": torch.zeros(self.num_envs, device=self.device),
        }

        self._target_height = 0.32
        self._global_step = 0

        # Height curriculum
        self._height_target = torch.full((self.num_envs,), self.cfg.height_target_min, dtype=torch.float,
                                         device=self.device)

        # DEBUG MODE
        self._debug_camera_mode = False
        self._debug_steps = 0

        self._trunk_idx = self._robot.find_bodies("trunk")[0]

        # Height tracking
        self._raycaster_height = torch.zeros(self.num_envs, device=self.device)
        self._height_errors = []

        # Store previous state for debugging
        self._prev_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)
        self._prev_root_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def step(self, actions: torch.Tensor):
        # Store state before applying action
        self._prev_joint_pos = self._robot.data.joint_pos.clone()
        self._prev_root_pos = self._robot.data.root_pos_w.clone()

        self._global_step += 1
        return super().step(actions)

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]
        self._raycaster = self.scene["raycaster"]
        self._apply_carpet_material()

    def _get_raycaster_height(self) -> torch.Tensor:
        """KEEP: Your working raycaster height estimation"""
        try:
            ray_origins = self._raycaster.data.pos_w
            ray_hits = self._raycaster.data.ray_hits_w
            heights = torch.zeros(self.num_envs, device=self.device)

            for env_idx in range(self.num_envs):
                env_origins = ray_origins[env_idx]
                env_hits = ray_hits[env_idx]
                distances = env_origins[..., 2] - env_hits[..., 2]  # Relative z for stability
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
            # Use default positions and medium gains for debug
            self._target_positions = self._robot.data.default_joint_pos
            self._kp_gains = torch.full((self.num_envs, 12), 50.0, device=self.device)
            self._kd_gains = torch.full((self.num_envs, 12), 2.5, device=self.device)
        else:
            # Split the 36D action into components
            actions = torch.clamp(actions, -3.0, 3.0)  # Prevent extreme actions
            self._actions = actions.clone()

            # First 12: position targets (scaled as before)
            pos_actions = actions[:, 0:12]
            self._target_positions = self.cfg.action_scale * pos_actions + self._robot.data.default_joint_pos

            # Next 12: KP gains (scaled from [-1,1] to [kp_min, kp_max] and CLAMPED)
            kp_actions = actions[:, 12:24]
            kp_min, kp_max = self.cfg.kp_range
            self._kp_gains = (kp_actions + 1.0) * 0.5 * (kp_max - kp_min) + kp_min
            self._kp_gains = torch.clamp(self._kp_gains, kp_min, kp_max)

            # Last 12: KD gains (scaled from [-1,1] to [kd_min, kd_max] and CLAMPED)
            kd_actions = actions[:, 24:36]
            kd_min, kd_max = self.cfg.kd_range
            self._kd_gains = (kd_actions + 1.0) * 0.5 * (kd_max - kd_min) + kd_min
            self._kd_gains = torch.clamp(self._kd_gains, kd_min, kd_max)

    def _apply_action(self):
        """Apply actions - IsaacLab handles PD gains internally"""
        self._robot.set_joint_position_target(self._target_positions)
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        gravity, angular_vel = self._get_imu_data()
        foot_contacts = self._get_foot_contact_binary()
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        # Update curriculum height
        if self.cfg.height_curriculum:
            progress = min(self._global_step / self.cfg.height_ramp_steps, 1.0)
            target = self.cfg.height_target_min + progress * (self.cfg.height_target_max - self.cfg.height_target_min)
            self._height_target.fill_(target)

        # Use raycaster height
        raycaster_height = self._get_raycaster_height()
        self._raycaster_height = raycaster_height
        height_estimate = raycaster_height.unsqueeze(1) if raycaster_height.dim() == 1 else raycaster_height

        # Final observation
        obs = torch.cat([
            gravity,  # (num_envs, 3)
            angular_vel,  # (num_envs, 3)
            foot_contacts,  # (num_envs, 4)
            joint_pos,  # (num_envs, 12)
            joint_vel,  # (num_envs, 12)
            height_estimate,  # (num_envs, 1)
            self._actions  # (num_envs, 36) - ALL components now!
        ], dim=-1)

        return {"policy": obs}

    def _get_foot_contact_binary(self) -> torch.Tensor:
        """PROVEN: Your original working contact detection"""
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
        """SIMPLIFIED REWARD STRUCTURE"""
        env_idx = 0  # Focus on first environment for debugging

        # 1. Contact reward
        contact_binary = self._get_foot_contact_binary()
        num_feet_contact = torch.sum(contact_binary, dim=1)
        leg_contact_reward = (num_feet_contact / 4.0) * 2.0  # Max 2.0

        # 2. Upright reward
        roll_pitch_error = torch.sqrt(self._robot.data.projected_gravity_b[:, 0] ** 2 +
                                      self._robot.data.projected_gravity_b[:, 1] ** 2)
        imu_upright_reward = torch.exp(-roll_pitch_error * 5.0)  # Max ~1.0

        # 3. Height reward
        height_error = torch.abs(self._raycaster_height - self._height_target)
        height_reward = torch.exp(-height_error * 4.0)  # Max ~1.0

        # 4. Crash penalty
        crash_penalty = torch.where(self._raycaster_height < 0.20,
                                    5.0 * (0.20 - self._raycaster_height),
                                    torch.zeros_like(self._raycaster_height))

        # 5. Alive bonus
        alive_bonus = 0.1

        # TOTAL REWARD - SIMPLIFIED
        total_reward = (leg_contact_reward * 1.0 +
                        imu_upright_reward * 2.0 +
                        height_reward * 2.0 +
                        alive_bonus -
                        crash_penalty)

        # DETAILED DEBUGGING - Print every 500 steps
        if self._global_step % 500 == 0:
            print("\n" + "═" * 120)
            print(f"🚀 STEP {self._global_step} - ENV {env_idx}")
            print("═" * 120)

            # Current state
            current_height = self._raycaster_height[env_idx].item()
            current_roll_pitch = roll_pitch_error[env_idx].item()
            current_contacts = contact_binary[env_idx].cpu().numpy()

            print(f"📊 CURRENT STATE:")
            print(f"   Height: {current_height:.3f}m (target: {self._height_target[env_idx]:.3f}m)")
            print(f"   Tilt error: {current_roll_pitch:.3f}")
            print(f"   Foot contacts: {current_contacts}")
            print(
                f"   Root position: [{self._prev_root_pos[env_idx, 0]:.2f}, {self._prev_root_pos[env_idx, 1]:.2f}, {self._prev_root_pos[env_idx, 2]:.2f}]")

            print(f"   Prev Joint Positions: {self._prev_joint_pos[env_idx].cpu().numpy().round(3)}")
            # Calculate actual joint position changes (12D vector)
            joint_pos_change = torch.abs(self._robot.data.joint_pos[env_idx] - self._prev_joint_pos[env_idx])

            # Actions (positions, KP, KD) - ALL 12 DIMENSIONS
            print(f"🎯 ACTIONS APPLIED (12D vectors):")
            print(f"   Target Positions: {self._target_positions[env_idx].cpu().numpy().round(3)}")
            print(f"   KP Gains:         {self._kp_gains[env_idx].cpu().numpy().round(2)}")
            print(f"   KD Gains:         {self._kd_gains[env_idx].cpu().numpy().round(2)}")

            tracking_error = torch.abs(self._robot.data.joint_pos[env_idx] - self._target_positions[env_idx])
            print(f"   Tracking Errors: {tracking_error.cpu().numpy().round(3)}")

            # Next state (after action) - Show joint positions
            current_joint_pos = self._robot.data.joint_pos[env_idx].cpu().numpy()
            next_root_pos = self._robot.data.root_pos_w[env_idx].cpu().numpy()

            print(f"📈 NEXT STATE (after action):")
            print(f"   Joint positions:    {current_joint_pos.round(3)}")
            print(f"   Joint pos changes:  {joint_pos_change.cpu().numpy().round(3)}")
            print(f"   Root position: [{next_root_pos[0]:.2f}, {next_root_pos[1]:.2f}, {next_root_pos[2]:.2f}]")

            # Show which joints moved the most
            max_change_idx = torch.argmax(joint_pos_change).item()
            max_change_val = joint_pos_change[max_change_idx].item()
            print(f"   Most movement: Joint {max_change_idx} changed by {max_change_val:.3f} rad")

            # Rewards breakdown
            print(f"💰 REWARDS:")
            print(f"   Contact: {leg_contact_reward[env_idx]:.2f} (feet: {num_feet_contact[env_idx].item()}/4)")
            print(f"   Upright: {imu_upright_reward[env_idx]:.2f} (tilt: {current_roll_pitch:.3f})")
            print(f"   Height:  {height_reward[env_idx]:.2f} (error: {height_error[env_idx]:.3f})")
            print(f"   Alive:   +{alive_bonus:.2f}")
            print(f"   Crash:   -{crash_penalty[env_idx]:.2f}")
            print(f"   TOTAL:   {total_reward[env_idx]:.3f}")
            print("═" * 120)

        # Accumulate for episode stats
        rewards_to_sum = {
            "leg_contact_reward": leg_contact_reward,
            "imu_upright_reward": imu_upright_reward,
            "height_reward": height_reward,
        }
        for k, v in rewards_to_sum.items():
            self._episode_sums[k] += v

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        gravity, _ = self._get_imu_data()
        roll_pitch = torch.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        height = self._robot.data.root_pos_w[:, 2]
        died = torch.logical_or(roll_pitch > 0.8, height < 0.2)
        return died, timeout

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # Reset actions (now 36D)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._target_positions[env_ids] = 0.0
        self._kp_gains[env_ids] = 50.0
        self._kd_gains[env_ids] = 2.5

        # Reset robot to default standing pose
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] = 0.35

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.reset(env_ids)

        # Reset episode sums
        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0

        self.scene.write_data_to_sim()