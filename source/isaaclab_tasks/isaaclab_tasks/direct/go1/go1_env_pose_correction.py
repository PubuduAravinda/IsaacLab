import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg


class Go1Env(DirectRLEnv):
    cfg: Go1FlatEnvCfg | Go1RoughEnvCfg

    def __init__(self, cfg: Go1FlatEnvCfg | Go1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # actions and commands
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # logging - UPDATED with new reward components
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "contact",
                "upright",
                "stability",
                "movement_smoothness",
                "stance_duration",
                "joint_position",
                "standing_bonus",
                "energy_efficiency",
                "leg_configuration",
            ]
        }

        # Base link is called "trunk"
        self._base_id, _ = self._contact_sensor.find_bodies("trunk")

        # Feet names are lowercase: FL_foot, FR_foot, RL_foot, RR_foot
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")

        # Thighs are lowercase too: FL_thigh, FR_thigh, ...
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh")

        # Track foot contact history for contact time rewards
        self._foot_contact_history = torch.zeros(self.num_envs, 4, device=self.device)  # 4 feet
        self._foot_force_history = torch.zeros(self.num_envs, 4, device=self.device)  # 4 feet

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        if isinstance(self.cfg, Go1RoughEnvCfg):
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, Go1RoughEnvCfg):
            height_data = (
                    self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[
                ..., 2] - 0.5
            ).clip(-1.0, 1.0)

        obs = torch.cat(
            [
                tensor
                for tensor in (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                height_data,
                self._actions,
            )
                if tensor is not None
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Binary contact detection (available on real Go1)
        force_threshold = 0.1

        # Get foot contact information
        contact_bodies_ids, contact_bodies_names = self._contact_sensor.find_bodies(".*")
        feet_sensor_indices = []
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        for foot_name in foot_names:
            foot_ids = [body_id for body_id, name in zip(contact_bodies_ids, contact_bodies_names)
                        if foot_name in name]
            if foot_ids:
                sensor_idx = contact_bodies_ids.index(foot_ids[0])
                feet_sensor_indices.append(sensor_idx)

        # Track individual foot contact states (binary - available on real robot)
        foot_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        if feet_sensor_indices:
            foot_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, feet_sensor_indices], dim=-1)
            foot_contacts = foot_forces > force_threshold

        # 1. CONTACT REWARD (Positive - reward more feet in contact)
        num_feet_in_contact = torch.sum(foot_contacts, dim=1)
        contact_reward = num_feet_in_contact.float() / 4.0  # Normalize to 0-1

        # 2. UPRIGHT REWARD (Positive - reward good orientation)
        gravity_vector = self._robot.data.projected_gravity_b
        upright_reward = torch.exp(-torch.sum(torch.square(gravity_vector[:, :2]), dim=1) * 5.0)

        # 3. STABILITY REWARD (Positive - reward low angular velocity)
        angular_velocity = torch.norm(self._robot.data.root_ang_vel_b, dim=1)
        stability_reward = torch.exp(-angular_velocity * 1.0)

        # 4. MOVEMENT SMOOTHNESS (Positive - reward stable contact patterns)
        movement_smoothness = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self, '_prev_foot_contacts'):
            # Count how many feet changed contact state
            contact_changes = torch.sum(foot_contacts != self._prev_foot_contacts, dim=1)
            movement_smoothness = torch.exp(-contact_changes.float() * 2.0)
        else:
            movement_smoothness = torch.ones(self.num_envs, device=self.device)
        self._prev_foot_contacts = foot_contacts.clone()

        # 5. STANCE DURATION (Positive - reward maintaining reasonable contact times)
        if not hasattr(self, '_contact_duration'):
            self._contact_duration = torch.zeros(self.num_envs, 4, device=self.device)

        # Update contact duration
        self._contact_duration = torch.where(foot_contacts,
                                             self._contact_duration + self.step_dt,
                                             torch.zeros_like(self._contact_duration))

        # Reward feet that maintain contact for reasonable durations
        reasonable_stance = (self._contact_duration > 0.05) & (self._contact_duration < 0.4)
        stance_duration_reward = torch.mean(reasonable_stance.float(), dim=1)

        # 6. JOINT POSITION REWARD (Positive - encourage neutral standing pose)
        joint_pos_error = torch.norm(self._robot.data.joint_pos - self._robot.data.default_joint_pos, dim=1)
        joint_position_reward = torch.exp(-joint_pos_error * 2.0)

        # 7. STANDING BONUS (Big positive reward for achieving good standing state)
        standing_bonus = torch.zeros(self.num_envs, device=self.device)
        # Define what "good standing" means
        good_standing = (num_feet_in_contact >= 2) & (upright_reward > 0.8) & (angular_velocity < 3.0)
        standing_bonus = good_standing.float()

        # 8. ENERGY EFFICIENCY (Small positive reward for efficient movement)
        joint_velocities = torch.norm(self._robot.data.joint_vel, dim=1)
        energy_efficiency = torch.exp(-joint_velocities * 0.1)

        # 9. LEG CONFIGURATION REWARD (Proper standing posture)
        leg_configuration_reward = torch.zeros(self.num_envs, device=self.device)

        # For Go1, we need to consider multiple aspects of leg posture:

        # Aspect 1: Hip abduction (sideways angle) - should be close to neutral
        # Typical Go1 joint order: [FL_hip_abduction, FL_hip, FL_knee, FR_hip_abduction, FR_hip, FR_knee, ...]
        hip_abduction_indices = [0, 3, 6, 9]  # Adjust based on your robot

        # Target: small abduction angles for straight-down legs
        target_abduction = torch.tensor([0.1, -0.1, 0.1, -0.1], device=self.device)  # Small outward angles
        current_abduction = self._robot.data.joint_pos[:, hip_abduction_indices]
        abduction_error = torch.mean(torch.abs(current_abduction - target_abduction), dim=1)

        # Aspect 2: Hip and knee flexion - should be in standing configuration
        hip_flexion_indices = [1, 4, 7, 10]  # Hip flexion/extension
        knee_indices = [2, 5, 8, 11]  # Knee joints

        target_hip_flexion = 0.8  # Slightly bent hips for standing
        target_knee_angle = -1.2  # Slightly bent knees for standing

        hip_error = torch.mean(torch.abs(self._robot.data.joint_pos[:, hip_flexion_indices] - target_hip_flexion),
                               dim=1)
        knee_error = torch.mean(torch.abs(self._robot.data.joint_pos[:, knee_indices] - target_knee_angle), dim=1)

        # Aspect 3: Symmetry between left and right sides
        left_hip_abduction = self._robot.data.joint_pos[:, [0, 6]]  # FL and RL abduction
        right_hip_abduction = self._robot.data.joint_pos[:, [3, 9]]  # FR and RR abduction
        abduction_symmetry_error = torch.mean(torch.abs(left_hip_abduction + right_hip_abduction), dim=1)

        # Aspect 4: Front-back symmetry
        front_hip_flexion = self._robot.data.joint_pos[:, [1, 4]]  # FL and FR hips
        rear_hip_flexion = self._robot.data.joint_pos[:, [7, 10]]  # RL and RR hips
        front_rear_symmetry_error = torch.mean(torch.abs(front_hip_flexion - rear_hip_flexion), dim=1)

        # Combined leg configuration error
        leg_config_error = (abduction_error * 0.4 +
                            hip_error * 0.2 +
                            knee_error * 0.2 +
                            abduction_symmetry_error * 0.1 +
                            front_rear_symmetry_error * 0.1)

        leg_configuration_reward = torch.exp(-leg_config_error * 2.0)


        # POSITIVE-ONLY REWARDS (No penalties, only rewards)
        rewards = {
            "contact": contact_reward * 0.8 * self.step_dt,
            "upright": upright_reward * 0.7 * self.step_dt,
            "stability": stability_reward * 0.6 * self.step_dt,
            "movement_smoothness": movement_smoothness * 0.5 * self.step_dt,
            "stance_duration": stance_duration_reward * 0.6 * self.step_dt,
            "joint_position": joint_position_reward * 0.4 * self.step_dt,
            "standing_bonus": standing_bonus * 1.0 * self.step_dt,
            "energy_efficiency": energy_efficiency * 0.3 * self.step_dt,
            "leg_configuration": leg_configuration_reward * 0.6 * self.step_dt,
        }

        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Update episode sums for logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Positive-focused debug output
        if self.episode_length_buf[0] % 120 == 0:
            good_standing_count = torch.sum(good_standing).item()
            avg_contact = torch.mean(contact_reward).item()
            avg_upright = torch.mean(upright_reward).item()
            avg_stability = torch.mean(stability_reward).item()
            avg_abduction = torch.mean(torch.abs(current_abduction)).item()
            avg_hip = torch.mean(torch.abs(self._robot.data.joint_pos[:, hip_flexion_indices])).item()
            avg_knee = torch.mean(torch.abs(self._robot.data.joint_pos[:, knee_indices])).item()

            print(f"[POSITIVE-ONLY] Standing: {good_standing_count}/{self.num_envs}, "
                  f"Contact: {avg_contact:.2f}, Upright: {avg_upright:.2f}, "
                  f"Stability: {avg_stability:.2f}, Reward: {total_reward[0]:.3f}")
            print(
                f"[LEG CONFIG] Abduction: {avg_abduction:.3f}, Hip: {avg_hip:.3f}, Knee: {avg_knee:.3f}, Reward: {torch.mean(leg_configuration_reward).item():.3f}")

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Termination when base hits ground or max episode length reached
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Reset contact history
        self._foot_contact_history[env_ids] = 0.0
        self._foot_force_history[env_ids] = 0.0

        # sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # LOWER THE ROBOT TO ENSURE CONTACT
        default_root_state[:, 2] = 0.3  # Lower from 0.4 to 0.3m to ensure contact

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Simple debug - check if robot is near ground
        if env_ids[0] == 0:  # Only for first environment
            print(f"[DEBUG] Robot root height: {self._robot.data.root_pos_w[0][2].item():.3f}m")
            # Check initial contact forces
            if hasattr(self._contact_sensor.data, 'net_forces_w'):
                initial_forces = torch.norm(self._contact_sensor.data.net_forces_w[0], dim=-1)
                max_force = torch.max(initial_forces).item()
                print(f"[DEBUG] Max initial contact force: {max_force:.4f}N")

        # logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)