# go1_env.py
# 45-D observation (no foot contacts — not available on real Go1).
# Hard hip clamp in _pre_physics_step instead of r_limits reward.
# Clean reward set — every term has a measurable effect.
# Obs normalisation saved/loaded with checkpoint for sim-to-real.
#
# Sim-to-real gap fixes (v2):
#   1. Actuator lag      — first-order filter on joint targets (α=0.6, τ≈13ms)
#                          models CAN-bus delay + motor inductance on real Go1
#   2. Observation noise — jpos/jvel/ang_vel/proj_grav noise matching real IMU/encoder
#   3. KP/KD range       — widened to [0.60,1.20]×nominal (was [0.80,1.20])
#                          real hardware shows up to 40% effective gain drop
#   4. prev_actions init — randomized at reset (was zeros)
#                          prevents policy learning "prev=0 means episode start"

import torch
import numpy as np
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand


class Go1Env(DirectRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "=" * 70)
        print("Go1Env  |  45-D obs  |  50 Hz  |  hard hip clamp  |  clean rewards")
        print("=" * 70 + "\n")

        self.command_manager = UniformVelocityCommand(cfg=self.cfg.commands, env=self)

        # ── Action buffers ────────────────────────────────────────────────────
        self._actions          = torch.zeros(self.num_envs, 12, device=self.device)

        # action_scale removed — tanh squashing in _pre_physics_step maps
        # raw network output directly into per-joint [lo, hi] delta_q limits.
        # Gradient always alive (d(tanh)/dx > 0), policy output is meaningful.
        self._action_scale = torch.ones(12, device=self.device)  # unused, kept for compat

        # Nominal KP/KD per joint (Isaac order: hips×4, thighs×4, knees×4)
        # Domain rand in _reset_idx scales these ±20% independently per env per episode.
        self._kp_nominal = torch.tensor(
            [35,35,35,35, 65,65,65,65, 80,80,80,80],
            dtype=torch.float32, device=self.device
        )
        self._kd_nominal = torch.tensor(
            [4.0,4.0,4.0,4.0, 4.5,4.5,4.5,4.5, 5.0,5.0,5.0,5.0],
            dtype=torch.float32, device=self.device
        )
        # Live per-env KP/KD tensors — shape (num_envs, 12)
        # Written to physics at each reset via set_joint_stiffness/damping
        self._kp_live = self._kp_nominal.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._kd_live = self._kd_nominal.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Soft delta_q limits (action = delta from DEFAULT_JOINT_POS).
        # Derived from joint_calibrate.py phase 2 & 4 measurements on real Go1:
        #   Hip:   swing range ~±0.15 rad, FL_hip hard-limited to +0.13 rad
        #   Thigh: liftoff needs 0.15-0.40 rad, tracking errors 0.10-0.35 rad
        #   Knee:  functional stance ±0.30 rad, passive rest up to -1.3 rad off default
        # Isaac order: [FL_hip, FR_hip, RL_hip, RR_hip,
        #               FL_thigh, FR_thigh, RL_thigh, RR_thigh,
        #               FL_knee, FR_knee, RL_knee, RR_knee]
        self._delta_soft_lo = torch.tensor(
            [-0.15,-0.15,-0.15,-0.15,   # hips:   can't swing too far inward
             -0.35,-0.35,-0.35,-0.35,   # thighs: flex down limit
             -0.35,-0.35,-0.35,-0.35],  # knees:  max bend from default
            device=self.device
        )
        self._delta_soft_hi = torch.tensor(
            [ 0.25, 0.25, 0.25, 0.25,   # hips:   outward swing limit
              0.35, 0.35, 0.35, 0.35,   # thighs: lift limit (FL lifts at 0.15, RR at 0.40)
              0.35, 0.35, 0.35, 0.35],  # knees:  max extend from default
            device=self.device
        )
        self._prev_actions     = torch.zeros_like(self._actions)
        self._prev_prev_actions= torch.zeros_like(self._actions)
        self._target_pos       = torch.zeros_like(self._actions)

        # ── Reward tracking ───────────────────────────────────────────────────
        self._ep_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in [
            "forward", "upright",
            "smooth",  "rate",    "torques",
            "yaw_err", "even",    "base_z_vel",
            "fall",    "hip_center", "action_l2",
        ]}

        self._global_step = 0

        # Obs normalisation is handled by actor_obs_normalizer inside RSL-RL network
        # (actor_obs_normalization=True in PPO cfg). Do NOT double-normalise here.

        # ── Sim-to-real gap fixes ─────────────────────────────────────────────
        #
        # FIX 1 — Actuator lag model
        # Real Go1 motors have ~13ms lag (CAN bus + motor inductance).
        # Model as first-order filter on joint position targets:
        #   lag_pos[t] = α * target[t] + (1-α) * lag_pos[t-1]
        # α=0.6 → τ = step_dt*(1-α)/α = 0.02*0.4/0.6 ≈ 13ms
        # Effect in sim: tracking error +0.04–0.08 rad, jvel std ×1.5–2×
        # matching real observations (tracking err 0.09–0.12 rad, jvel std 0.71)
        self._lag_alpha = 0.6   # tune: 0.5=slower/more lag, 0.8=faster/less lag
        self._lag_pos   = None  # initialised at first reset in _reset_idx

        # FIX 2 — Observation noise
        # Calibrated from real_log data vs sim_log (model_100, same policy):
        #   jpos noise: encoder quantisation + cable flex   → std=0.005 rad
        #   jvel noise: velocity estimation from encoder    → std=0.05  rad/s
        #   ang_vel:    ICM-42688-P gyro datasheet noise    → std=0.02  rad/s
        #   proj_grav:  accelerometer + complementary filter→ std=0.01  (unit vec)
        self._obs_noise_std = torch.tensor([
            0.0,  0.0,  0.0,                     # [0:3]   cmd — no noise (fixed value)
            *([0.005]*12),                        # [3:15]  jpos_delta
            *([0.050]*12),                        # [15:27] jvel
            0.02, 0.02, 0.02,                     # [27:30] ang_vel
            0.01, 0.01, 0.01,                     # [30:33] proj_grav
            *([0.0]*12),                          # [33:45] prev_actions — no noise
        ], device=self.device)                    # shape (45,)

        # FIX 3 — KP/KD randomization range widened (applied in _reset_idx)
        # Real tracking errors 3–5× larger than sim suggest effective KP can be
        # as low as 60% of nominal due to motor compliance, backlash, cable routing.
        # KP range: [0.60, 1.20] × nominal  (was [0.80, 1.20])
        # KD range: [0.70, 1.30] × nominal  (was [0.80, 1.20]) — wider damping spread
        self._kp_rand_lo = 0.60   # tune: lower = more lag/compliance in sim
        self._kp_rand_hi = 1.20
        self._kd_rand_lo = 0.70
        self._kd_rand_hi = 1.30

        # FIX 4 — prev_actions init randomization (applied in _reset_idx)
        # Policy learned "prev_actions=0 → start of episode" in earlier training.
        # On real hardware, deploy script seeds prev_actions from warmup loop output,
        # never zeros. Randomize at reset so policy learns to handle any history.
        # Range: ±30% of tanh limits (mild — don't start in extreme states)
        self._prev_act_init_scale = 0.3  # fraction of [lo, hi] range to randomize

        # Noise gate — True during training, set to False by play.py before inference
        # (DirectRLEnv does not inherit nn.Module so has no .training attribute)
        self._obs_noise_enabled = True

        self.last_obs = None

        # ── Sim logger — captures same channels as real_log_*.npz ────────────
        # Filled in step() for env 0 only. Saved to sim_log_*.npz by play_log.py.
        # raw_net (pre-tanh network output) is injected by play_log.py via hook.
        self._sim_log_maxsteps = 2000          # ~40s at 50Hz — change in play_log.py
        N = self._sim_log_maxsteps
        self._slog = {
            "obs_raw":   np.zeros((N, 45), np.float32),  # pre-normalizer obs (env 0)
            "tanh_delta":np.zeros((N, 12), np.float32),  # self._actions (post-tanh)
            "raw_net":   np.zeros((N, 12), np.float32),  # injected by play_log.py hook
            "target_q":  np.zeros((N, 12), np.float32),  # self._target_pos
            "actual_q":  np.zeros((N, 12), np.float32),  # robot.data.joint_pos
            "actual_qd": np.zeros((N, 12), np.float32),  # robot.data.joint_vel
            "proj_grav": np.zeros((N,  3), np.float32),  # projected_gravity_b
            "ang_vel":   np.zeros((N,  3), np.float32),  # root_ang_vel_b
            "lin_vel":   np.zeros((N,  3), np.float32),  # root_lin_vel_b (sim only)
            "cmd":       np.zeros((N,  3), np.float32),  # velocity command
            "contact":   np.zeros((N,  4), np.float32),  # foot contact forces [FL FR RL RR]
            "tilt_deg":  np.zeros(N,       np.float32),  # scalar tilt angle
            "reward":    np.zeros(N,       np.float32),  # total reward env 0
        }
        self._slog_step = 0       # how many steps written so far
        self._slog_active = False # set True by play_log.py before inference loop

    # ── Scene ─────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self._robot = self.scene["robot"]

        # ── ACTUATOR GAIN VERIFICATION ────────────────────────────────────────
        # Read KP/KD from the actuator config objects attached to the asset.
        # Isaac Lab stores these on the asset's actuator dict keyed by group name.
        # API: self._robot._actuators  (internal dict, stable across versions)
        # Fallback: read directly from cfg if runtime dict unavailable.
        print("\n" + "─"*60)
        print("ACTUATOR GAIN VERIFICATION (startup, env 0)")
        print("─"*60)
        # Try the internal _actuators dict (Isaac Lab ≥ 1.1)
        actuator_dict = getattr(self._robot, "_actuators", None)
        if actuator_dict:
            for act_name, act in actuator_dict.items():
                kp = getattr(act, "stiffness", None)
                kd = getattr(act, "damping",   None)
                jn = getattr(act, "joint_names", [])
                kp_str = f"{kp[0].cpu().numpy()}" if kp is not None else "N/A"
                kd_str = f"{kd[0].cpu().numpy()}" if kd is not None else "N/A"
                print(f"  [{act_name}]  joints={jn}")
                print(f"    KP env0: {kp_str}")
                print(f"    KD env0: {kd_str}")
        else:
            # Fallback: read from cfg directly (always available)
            print("  _actuators dict not found — reading from cfg:")
            robot_cfg = self.cfg.scene.robot
            for act_name, act_cfg in robot_cfg.actuators.items():
                print(f"  [{act_name}]  KP={act_cfg.stiffness}  KD={act_cfg.damping}")
        print("─"*60 + "\n")

    # ── Control ───────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clone()

        # tanh squashing — maps raw network output (-∞,+∞) → [lo, hi] per joint.
        #
        # Why tanh instead of hard clamp or fixed action_scale:
        #   hard clamp:   gradient=0 when saturated → 9/12 joints get no signal
        #   action_scale: policy can't explore beyond ceiling → misses stuck joints
        #   tanh:         gradient always alive: d(tanh)/dx = 1-tanh²(x) > 0 always
        #                 naturally bounded, smooth, policy outputs are meaningful
        #
        # Limits from real Go1 calibration (delta_q from DEFAULT_JOINT_POS):
        #   Hip:   [-0.15, +0.25]  lateral swing, FL_hip hard-limited at +0.13
        #   Thigh: [-0.35, +0.35]  liftoff needs 0.15-0.35 rad flex
        #   Knee:  [-0.35, +0.35]  functional trot range at KP=80, effort_limit=28 Nm
        #
        # Squash: a = mid + half * tanh(raw)
        #   mid  = (hi + lo) / 2   center of range
        #   half = (hi - lo) / 2   half-width
        #   tanh(raw) ∈ (-1,+1) → a ∈ (lo, hi) strictly always
        _mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5
        _half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        a = _mid + _half * torch.tanh(a)

        self._prev_prev_actions[:] = self._prev_actions
        self._prev_actions[:]      = self._actions
        self._actions[:]           = a

        # Delta from default standing pose → absolute joint target
        self._target_pos = self._actions + self._robot.data.default_joint_pos

        # FIX 1: Actuator lag — filter target before sending to physics.
        # Initialised to current joint pos on first call (after first reset).
        if self._lag_pos is None:
            self._lag_pos = self._robot.data.joint_pos.clone()
        self._lag_pos = (self._lag_alpha * self._target_pos
                         + (1.0 - self._lag_alpha) * self._lag_pos)
        self._robot.set_joint_position_target(self._lag_pos)

    def _apply_action(self):
        """No-op — targets already set in _pre_physics_step."""
        pass

    # ── Observations (45-D) ───────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        Return raw scaled observations. RSL-RL's actor_obs_normalizer (RunningMeanStd
        inside the network) handles normalisation — do NOT normalise here too.

        Layout (45-D, all readable from real Go1 hardware):
            [0:3]   velocity commands (vx, vy, wz)          — bounded by cmd ranges
            [3:15]  joint pos delta from default             — encoder, ~[-π, π]
            [15:27] joint velocity                           — encoder, clip ±10 rad/s
            [27:30] base angular velocity                    — IMU gyro, clip ±5 rad/s
            [30:33] projected gravity                        — IMU, unit vector [-1,1]
            [33:45] previous actions                         — policy output [-1,1]
        """
        obs = torch.cat([
            self.command_manager.command[:, :3],                              # 3
            self._robot.data.joint_pos - self._robot.data.default_joint_pos, # 12
            torch.clamp(self._robot.data.joint_vel,       -5.0,  5.0),       # 12 — tighter clip, real Go1 rarely exceeds 5 rad/s
            torch.clamp(self._robot.data.root_ang_vel_b,  -5.0,  5.0),       # 3
            self._robot.data.projected_gravity_b,                             # 3
            self._prev_actions,                                               # 12
        ], dim=-1)  # = 45

        # FIX 2: Observation noise — inject per-channel Gaussian noise.
        # Only active when _obs_noise_enabled=True (training).
        # play.py sets go1_env._obs_noise_enabled=False before inference loop
        # so sim logs record clean obs matching what real hardware would see.
        if self._obs_noise_enabled:
            noise = torch.randn_like(obs) * self._obs_noise_std
            obs   = obs + noise

        return {"policy": obs}

    # ── Rewards ───────────────────────────────────────────────────────────────
    def _get_rewards(self):
        lin_vel  = self._robot.data.root_lin_vel_b
        ang_vel  = self._robot.data.root_ang_vel_b
        gravity  = self._robot.data.projected_gravity_b
        height   = self._robot.data.root_pos_w[:, 2]
        cmd      = self.command_manager.command

        # ── 1. Forward velocity tracking (dominant positive signal) ──────────
        # Reduced from 6.0 → 2.0. At 6.0: smooth+rate = only 4.7% of forward
        # → policy ignored smoothness and learned bang-bang joint slamming.
        # At 2.0: smooth+rate will be ~15-20% of forward signal — policy must
        # balance speed vs smoothness to maximise total reward.
        r_forward = 2.0 * torch.exp(
            -((lin_vel[:, 0] - cmd[:, 0]) ** 2) / 0.25 ** 2
        )

        # ── 2. Flat orientation (AnymalC: flat_orientation_l2) ───────────────
        r_upright = -1.0 * torch.sum(gravity[:, :2] ** 2, dim=1)

        # ── 3. Action smoothness ──────────────────────────────────────────────
        # Calibrated from sim_log_model_24900 actual action changes:
        #   mean sum_sq per step = 0.042  (tanh outputs change ~0.05 rad/joint/step)
        # Target: smooth+rate ≈ 30% of forward peak (2.0) = 0.60 total
        #   smooth at -7.0: mean = -7.0 * 0.042 = -0.29
        #   rate   at -3.5: mean = -3.5 * 0.042 = -0.15
        #   total  ≈ 0.44 → ~22% of forward ✓ (conservative — won't block gait)
        # Previous run: smooth=-101 at end = policy slamming joints every step
        # (sum_sq was ~0.67/step = 16× higher than a smooth trot)
        r_smooth = -7.0 * torch.sum((self._actions - self._prev_actions) ** 2, dim=1)
        r_rate   = -3.5 * torch.sum(
            (self._actions - 2 * self._prev_actions + self._prev_prev_actions) ** 2, dim=1
        )

        # ── 4. Joint torques ──────────────────────────────────────────────────
        torques   = self._robot.data.applied_torque
        r_torques = -1e-4 * torch.sum(torques ** 2, dim=1)

        # ── 5. Yaw tracking ───────────────────────────────────────────────────
        r_yaw_err = -0.3 * (ang_vel[:, 2] - cmd[:, 2]) ** 2

        # ── 6. Feet air time ──────────────────────────────────────────────────
        foot_fz      = self.scene.sensors["contact_sensor"].data.net_forces_w[:, :, 2]
        in_contact   = (foot_fz > 8.0).float()
        contact_freq = in_contact.mean(dim=1)
        r_even = 0.1 * torch.exp(-10.0 * (contact_freq - 0.30) ** 2)

        # ── 7. Vertical body velocity ─────────────────────────────────────────
        r_base_z_vel = -4.0 * (lin_vel[:, 2] ** 2)

        # ── 8. Fall penalty ───────────────────────────────────────────────────
        r_fall = -5.0 * (height < 0.25).float()

        # ── 9. Hip centering (NEW) ────────────────────────────────────────────
        # Previous training: RR_hip 99% at -0.15 limit, FL_hip 76%.
        # Hips slamming to limits → raw_net explodes to ±40 → OOD on real.
        # Penalise large hip delta (actions[:4]) to keep hips near zero.
        # Weight -0.5: strong enough to compete with forward at reduced scale.
        r_hip_center = -0.5 * torch.sum(self._actions[:, :4] ** 2, dim=1)

        # ── 10. Action L2 (NEW) ───────────────────────────────────────────────
        # Penalise large tanh outputs directly. Prevents raw_net → ±40 spiral.
        # At raw_net=40: tanh=1.0, gradient≈0, policy gets no learning signal.
        # This term keeps raw_net in [-4,+4] where tanh has useful gradient.
        # Weight -0.05: soft constraint, not blocking the policy.
        r_action_l2 = -0.05 * torch.sum(self._actions ** 2, dim=1)

        keys = ["forward","upright","smooth","rate","torques",
                "yaw_err","even","base_z_vel","fall","hip_center","action_l2"]
        vals = [r_forward,r_upright,r_smooth,r_rate,r_torques,
                r_yaw_err,r_even,r_base_z_vel,r_fall,r_hip_center,r_action_l2]
        for k, v in zip(keys, vals):
            self._ep_sums[k] += v

        total = self.step_dt * (
            r_forward + r_upright +
            r_smooth  + r_rate    + r_torques +
            r_yaw_err + r_even    + r_base_z_vel +
            r_hip_center + r_action_l2
        ) + r_fall

        return total

    # ── Termination ───────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        g      = self._robot.data.projected_gravity_b
        height = self._robot.data.root_pos_w[:, 2]
        tilt   = torch.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)

        terminated = (tilt > 0.8) | (height < 0.25) | (g[:, 2] > 0.3)  # tighter — stop before dive
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        # Print reward breakdown every 200 resets
        if self._global_step % 200 == 0 and self._global_step > 0:
            print(f"\n=== Rewards @ step {self._global_step} ===")
            for k, v in self._ep_sums.items():
                print(f"  {k:12s}: {v[env_ids].mean().item():+.3f}")
            print("=" * 40)

        # ── Manual KP/KD domain randomization ────────────────────────────────
        # FIX 3: Range widened to [0.60, 1.20] × nominal (was [0.80, 1.20]).
        # Real hardware can show effective KP as low as 60% of nominal due to
        # motor compliance, cable routing, and gear backlash. KD also widened.
        rand_factor_kp = torch.empty(len(env_ids), 12, device=self.device).uniform_(
            self._kp_rand_lo, self._kp_rand_hi)
        rand_factor_kd = torch.empty(len(env_ids), 12, device=self.device).uniform_(
            self._kd_rand_lo, self._kd_rand_hi)
        self._kp_live[env_ids] = self._kp_nominal * rand_factor_kp
        self._kd_live[env_ids] = self._kd_nominal * rand_factor_kd

        # Write randomized gains to physics — correct Isaac Lab Articulation API
        self._robot.write_joint_stiffness_to_sim(self._kp_live[env_ids], env_ids=env_ids)
        self._robot.write_joint_damping_to_sim(  self._kd_live[env_ids], env_ids=env_ids)

        # Verify once early to confirm rand is actually applied
        if self._global_step < 50 and len(env_ids) >= 2:
            kp0 = self._kp_live[env_ids[0]].cpu().numpy().round(1)
            kp1 = self._kp_live[env_ids[1]].cpu().numpy().round(1)
            diff = abs(self._kp_live[env_ids[0]] - self._kp_live[env_ids[1]]).max().item()
            status = "✓ rand active" if diff > 0.5 else "⚠ identical — check API"
            print(f"  [KP rand] env{env_ids[0].item()}: {kp0}  {status}")
            print(f"  [KP rand] env{env_ids[1].item()}: {kp1}")

        super()._reset_idx(env_ids)
        self.command_manager.reset(env_ids)

        # Clear buffers
        # FIX 4: Randomize prev_actions instead of zeros.
        # Real deploy script seeds prev_actions from warmup loop (never zero).
        # ±30% of [lo,hi] range → small but non-zero, covers realistic start states.
        _mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5   # (12,)
        _half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        _rand_prev = (_mid + _half * self._prev_act_init_scale
                      * (torch.rand(len(env_ids), 12, device=self.device) * 2 - 1))
        self._actions[env_ids]           = _rand_prev
        self._prev_actions[env_ids]      = _rand_prev
        self._prev_prev_actions[env_ids] = _rand_prev
        self._target_pos[env_ids]        = self._robot.data.default_joint_pos[env_ids]

        # Reset actuator lag state for these envs to current joint pos
        # (avoids lag_pos being stale from previous episode)
        if self._lag_pos is not None:
            self._lag_pos[env_ids] = self._robot.data.joint_pos[env_ids].clone()

        for k in self._ep_sums:
            self._ep_sums[k][env_ids]   = 0.0

        # Gentle init with small noise
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += (torch.rand_like(joint_pos) - 0.5) * 0.05
        joint_vel  = torch.zeros_like(joint_pos)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        # write_root_pose_to_sim expects WORLD-FRAME coordinates.
        # default_root_state pos is (0,0,0.35) — local to env origin.
        # Without env_origins, ALL robots write to world (0,0,0.35) = same spot.
        root_state[:, :3]   = self.scene.env_origins[env_ids]  # world x,y + terrain z
        root_state[:, 2]   += 0.35                              # height above ground
        root_state[:, 3:7]  = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        root_state[:, 7:10] = 0.0   # zero linear vel
        root_state[:, 10:13]= 0.0   # zero angular vel

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ── Step (with periodic debug) ────────────────────────────────────────────
    def step(self, action):
        self._global_step += 1

        if self._global_step % 500 == 0:
            idx = 0
            print(f"\n{'='*80}")
            print(f"[DEBUG] step {self._global_step} | env 0 | cmd_vx={self.command_manager.command[idx,0]:.2f}")
            print(f"  [S2R] lag_alpha={self._lag_alpha}  noise_jvel_std={self._obs_noise_std[15].item():.3f}"
                  f"  KP_range=[{self._kp_rand_lo:.2f},{self._kp_rand_hi:.2f}]×nom"
                  f"  prev_act_init_scale={self._prev_act_init_scale}")

            # ── ACTUATOR GAIN CHECK (domain rand verification) ────────────────
            # _kp_live holds the actual per-env randomized KP written to physics.
            # env0 and env1 must differ — proves rand is active each episode.
            kp0 = self._kp_live[0].cpu().numpy().round(1)
            kp1 = self._kp_live[1].cpu().numpy().round(1) if self.num_envs > 1 else kp0
            diff = abs(self._kp_live[0] - self._kp_live[1]).max().item() if self.num_envs > 1 else 0
            status = "✓ rand active" if diff > 0.5 else "⚠ same — rand not applied"
            print(f"  Live KP env0: {kp0}  {status}")
            print(f"  Live KP env1: {kp1}")
            print(f"  Live KD env0: {self._kd_live[0].cpu().numpy().round(2)}")

            # ── TORQUE SANITY CHECK ───────────────────────────────────────────
            # Cross-check: does Isaac's internal torque match our reward estimate?
            # robot.data.applied_torque = what PhysX actually applied last step
            if hasattr(self._robot.data, "applied_torque"):
                tau_real = self._robot.data.applied_torque[idx].cpu().numpy()
                print(f"  Applied torque env0: {tau_real.round(2)}")
                q_err_now = (self._target_pos[idx] - self._robot.data.joint_pos[idx]).cpu().numpy()
                dq_now    = self._robot.data.joint_vel[idx].cpu().numpy()
                # Expected using per-type KP/KD
                kp_np = [35]*4 + [65]*4 + [80]*4
                kd_np = [4.0]*4 + [4.5]*4 + [5.0]*4
                tau_est = [kp_np[i]*q_err_now[i] - kd_np[i]*dq_now[i] for i in range(12)]
                import numpy as np
                tau_est = np.array(tau_est).round(2)
                err = np.abs(tau_real - tau_est)
                print(f"  Reward-estimated torque: {tau_est}")
                print(f"  Max mismatch: {err.max():.3f} Nm "
                      f"({'✓ KP/KD match' if err.max()<3.0 else '⚠ mismatch — check actuator cfg'})")

            if self.last_obs is not None:
                o = self.last_obs
                i = 0
                print(f"  cmd:        {o[i:i+3].cpu().numpy()}");  i+=3
                print(f"  jpos_delta: {o[i:i+12].cpu().numpy()}"); i+=12
                print(f"  jvel:       {o[i:i+12].cpu().numpy()}"); i+=12
                print(f"  ang_vel:    {o[i:i+3].cpu().numpy()}");  i+=3
                print(f"  proj_grav:  {o[i:i+3].cpu().numpy()}");  i+=3
                print(f"  prev_act:   {o[i:i+12].cpu().numpy()}")
                # Sanity checks on raw obs (no normaliser applied in env)
                grav_mag = o[30:33].norm().item()
                print(f"  grav_mag:   {grav_mag:.3f} (should be ~1.0 if upright)")

            # ── ACTUATOR LAG VISUALISATION ────────────────────────────────────
            # Shows the gap between what the policy commanded and what the lag
            # filter actually sent to PhysX this step.
            #
            # Real CAN-bus: 2-6ms lag at 500Hz resolution
            # Our filter:   τ=step_dt*(1-α)/α = 0.02*0.4/0.6 = 13ms at 50Hz
            #
            # Columns:
            #   policy_target — what tanh squash produced (desired pos)
            #   lag_sent      — what _lag_pos actually sent to PhysX
            #   lag_error     — lag_sent - policy_target  (lag-induced offset)
            #   actual_q      — where joint actually is right now
            #   track_err     — actual_q - lag_sent  (PD controller residual)
            if self._lag_pos is not None:
                import numpy as _np
                tgt  = self._target_pos[idx].cpu().numpy()
                lag  = self._lag_pos[idx].cpu().numpy()
                act  = self._robot.data.joint_pos[idx].cpu().numpy()
                lerr = lag - tgt                # how far lag_sent is from policy intent
                terr = act - lag                # how far actual is from lag command
                JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
                          'FL_th', 'FR_th', 'RL_th', 'RR_th',
                          'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']
                print(f"\n  [LAG DEBUG env0]  α={self._lag_alpha}  "
                      f"τ≈{0.02*(1-self._lag_alpha)/self._lag_alpha*1000:.0f}ms")
                print(f"  {'joint':8s}  {'policy_tgt':>10}  {'lag_sent':>10}  "
                      f"{'lag_err':>10}  {'actual_q':>10}  {'track_err':>10}")
                print(f"  {'-'*62}")
                for j in range(12):
                    lag_flag   = "LAG"   if abs(lerr[j]) > 0.02 else ""
                    track_flag = "TRACK" if abs(terr[j]) > 0.05 else ""
                    flag = f"  *** {lag_flag}{'+' if lag_flag and track_flag else ''}{track_flag}" if (lag_flag or track_flag) else ""
                    print(f"  {JNAMES[j]:8s}  {tgt[j]:+10.4f}  {lag[j]:+10.4f}  "
                          f"{lerr[j]:+10.4f}  {act[j]:+10.4f}  {terr[j]:+10.4f}{flag}")
                # Summary stats
                mean_lag_err   = _np.abs(lerr).mean()
                mean_track_err = _np.abs(terr).mean()
                total_err      = _np.abs(act - tgt).mean()
                print(f"  mean |lag_err|={mean_lag_err:.4f} rad  "
                      f"mean |track_err|={mean_track_err:.4f} rad  "
                      f"mean |total_err|={total_err:.4f} rad")
                print(f"  (real hardware target: total_err ~ 0.05-0.12 rad  "
                      f"{'✓ in range' if 0.03 < total_err < 0.15 else '⚠ check lag_alpha'})")

            print(f"  action out: {action[idx].cpu().numpy()}")
            print(f"  root_z:     {self._robot.data.root_pos_w[idx,2]:.3f}")
            print(f"  lin_vel_xy: {self._robot.data.root_lin_vel_b[idx,:2].norm():.3f}")
            print("  rewards:")
            for k in self._ep_sums:
                print(f"    {k:12s}: {self._ep_sums[k][idx]:+.3f}")

        result = super().step(action)
        self.last_obs = result[0]["policy"][0].clone()

        # ── Sim logger — write env 0 every policy step ───────────────────────
        if self._slog_active and self._slog_step < self._sim_log_maxsteps:
            s = self._slog_step
            obs0 = result[0]["policy"][0]          # pre-normalizer obs, env 0, (45,)
            g    = self._robot.data.projected_gravity_b[0]
            tilt = float(torch.sqrt(g[0]**2 + g[1]**2).item())
            try:
                fz = self.scene.sensors["contact_sensor"].data.net_forces_w[0, :, 2]
                contact_np = fz.cpu().numpy()
            except Exception:
                contact_np = np.zeros(4, np.float32)

            self._slog["obs_raw"][s]    = obs0.cpu().numpy()
            self._slog["tanh_delta"][s] = self._actions[0].cpu().numpy()
            # raw_net[s] filled by play_log.py hook before this executes
            self._slog["target_q"][s]   = self._target_pos[0].cpu().numpy()
            self._slog["actual_q"][s]   = self._robot.data.joint_pos[0].cpu().numpy()
            self._slog["actual_qd"][s]  = self._robot.data.joint_vel[0].cpu().numpy()
            self._slog["proj_grav"][s]  = g.cpu().numpy()
            self._slog["ang_vel"][s]    = self._robot.data.root_ang_vel_b[0].cpu().numpy()
            self._slog["lin_vel"][s]    = self._robot.data.root_lin_vel_b[0].cpu().numpy()
            self._slog["cmd"][s]        = self.command_manager.command[0, :3].cpu().numpy()
            self._slog["contact"][s]    = contact_np
            self._slog["tilt_deg"][s]   = tilt * (180.0 / 3.14159265)
            self._slog["reward"][s]     = float(result[1][0].item())
            self._slog_step += 1

        return result