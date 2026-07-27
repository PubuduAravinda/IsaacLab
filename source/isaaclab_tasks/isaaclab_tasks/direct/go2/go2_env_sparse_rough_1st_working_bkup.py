'''
python train.py --task Isaac-Go2-Sparse-Rough-Direct-v0 --num_envs 1000 --headless \
  --checkpoint /path/to/go2_flat/.../model_XXXXX.pt
'''

# go2_env_sparse_rough.py — Natural Gait on Rough Terrain (Go2, blind, no LiDAR)
#
# Direct port of go1_env_sparse_rough.py's approach to Go2. Inherits ALL
# infrastructure from go2_env.py (verified to already provide everything
# this needs: command_manager, _actions/_prev_actions/_prev_prev_actions,
# _rate_weights, _episode_wz, _WZ_MAX, _terrain_z, _prev_feet_contact,
# _foot_body_ids) — same "only override _get_rewards" pattern as Go1.
#
# THEORETICAL BASIS (same as Go1):
#   Hoyt & Taylor 1981 (Nature doi:10.1038/292239a0):
#     Animals select gaits to minimise metabolic cost of transport.
#   Heess et al. 2017 (arXiv:1707.02286):
#     Displacement + survive on rich terrain -> trot/bound emerge naturally.
#   Alexander 1976: stride length balances metabolic cost vs collision
#     losses per touchdown -- longer strides reduce collision term.
#
# REMOVED vs go1_env_sparse_rough.py -- these were Go1 FAULT-SPECIFIC,
# not general sparse-reward technique, and Go2 has no known fault:
#   r_fr_binding — Go1's FR_th mechanical-binding hardware safety term.
#     No equivalent joint issue on Go2; nothing to protect against.
#   r_wz_excess  — explicitly framed in Go1's code as "RL_th stiction
#     escape suppression". Go2's r_ang_vel straight/turn split already
#     handles yaw damping generically; this term would just be solving
#     a problem Go2 doesn't have.
# FR_BINDING_WEIGHT constant removed along with r_fr_binding.
#
# KEPT AND GENERAL (not fault-specific, applies to any healthy quadruped):
#   r_displacement, r_lateral, r_traj_track, r_ang_vel (straight/turn split)
#   r_metabolic (Hoyt & Taylor), r_alive, r_upright, r_fall
#   r_action_rate/jerk, r_stride_quality (foot clearance), r_stride_len
#   (Alexander 1976), r_hip_posture, r_hip_reg
#
# WEIGHT VALUES — STARTING POINT, NOT YET GO2-HARDWARE-TUNED:
#   Go1's exact weights (METABOLIC_WEIGHT=-0.005, STRIDE_LEN_WEIGHT=5.0,
#   etc.) were tuned against real Go1 hardware logs at Go1's specific
#   actuator bandwidth (KP=35/65/80 per joint type) and mass (12kg).
#   Go2 is heavier (15kg) with uniform KP=60 — different dynamics. The
#   values below start at the SAME weights as Go1's (same order of
#   magnitude, reasonable prior) but should be re-tuned once you have a
#   first Go2 sparse-rough real hardware log, the same way Go1's values
#   evolved from hardware evidence over multiple iterations.
#
# ONE DELIBERATE CARRYOVER FROM GO1, KEPT ON PURPOSE:
#   r_stride_quality weights front legs [FL,FR]=2.0 vs rear [RL,RR]=1.0.
#   Go1's comment cites "front legs historically dragged on real hardware"
#   as the reason. Your real Go2 deploy log (real_log_go2_..._115920.npz)
#   independently found front-leg-biased thigh tracking lag (front thigh
#   err 0.15-0.17 rad vs rear 0.08-0.10 rad) -- same asymmetry, different
#   robot. This is legitimate supporting evidence to keep this weighting,
#   not just copying Go1 blindly.
#
# WHAT IS INHERITED UNCHANGED from go2_env.py:
#   PACE-equivalent nulls (Ia=0, d=0, tau_f=0, q~b not modeled)
#   command_manager, obs noise (Test1/2), delay FIFO (Test3),
#   lag filter (Test5), KP/KD, _reset_idx DR, _get_dones, debug prints
#
# REGISTER in __init__.py:
#   gym.register(
#     id="Isaac-Go2-Sparse-Rough-Direct-v0",
#     entry_point="isaaclab_tasks.direct.go2.go2_env_sparse_rough:Go2EnvSparseRough",
#     disable_env_checker=True,
#     kwargs={
#       "env_cfg_entry_point": "isaaclab_tasks.direct.go2.go2_rough_env_cfg:Go2RoughEnvCfg",
#       "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2SparsePPORunnerCfg",
#     },
#   )
#
# *** Go2SparsePPORunnerCfg MAY NOT EXIST YET in your agents/rsl_rl_ppo_cfg.py ***
# Go1's has one (Go1SparsePPORunnerCfg, referenced in train.py). If Go2's
# doesn't exist yet, add one mirroring it -- train.py/play.py below both
# import it with a fallback to Go2RslRlPpoCfg + a printed warning so
# nothing hard-crashes if it's missing, but you should still add the real
# one for correct sparse-reward-scale PPO hyperparameters.
#
# TRAIN:
#   python train.py --task Isaac-Go2-Sparse-Rough-Direct-v0 \
#     --num_envs 1000 --max_iterations 15000 \
#     --checkpoint logs/.../go2_flat_model_XXXXX.pt --headless

import torch
import numpy as np

from isaaclab_tasks.direct.go2.go2_env import Go2Env

# ── Natural reward knobs — Go1-derived starting point, see header note ────
METABOLIC_WEIGHT  = -0.005
STRIDE_LEN_WEIGHT = 5.0
UPRIGHT_WEIGHT    = -0.5

# RATE_WEIGHT/JERK_WEIGHT — UPDATED after real Go2 hardware deployment of
# model_9950 (sparse_v1 run) showed severe action jitter: real action_rate
# was ~3.5x higher than the same checkpoint's sim value (0.15 vs 0.043),
# with target_q swinging close to the full joint range between consecutive
# 20ms inference cycles. Confirmed NOT a rate-limiter bug (go2_rl_flat_1.py
# is unchanged from the flat deployment that worked cleanly) -- the
# policy's own raw output was genuinely that jerky, traced to these two
# weights being 20-25x weaker than the flat reward's proven-safe values
# (flat: -1.0 / -0.5, this file previously: -0.05 / -0.02).
# Now matched to flat's values directly, since flat deployed cleanly on
# real hardware and there's no evidence yet that a weaker penalty is safe
# for Go2 specifically (unlike whatever informed Go1's original -0.05/-0.02
# choice). If you want to reintroduce some "natural gait freedom" margin
# once this is confirmed safe on hardware, loosen gradually from here
# rather than starting from the value that caused the joint-stress issue.
RATE_WEIGHT       = -1.0    # was -0.05
JERK_WEIGHT       = -0.5    # was -0.02


class Go2EnvSparseRough(Go2Env):
    """
    Natural gait emergence on rough terrain, blind (no LiDAR).
    Inherits ALL infrastructure from Go2Env. Only _get_rewards differs --
    natural/metabolic rewards replace prescriptive gait rewards
    (air_time threshold, trot_bias/trot_penalty from the flat reward set).

    Research hypothesis (Hoyt & Taylor 1981 + Heess 2017), same as Go1:
      With only displacement + metabolic cost + survive + stride-quality:
        - Trot emerges because it minimises cost-of-transport at medium speed
        - No explicit air_time threshold, trot_bias, f_cmd prescription needed
    Unlike Go1, there's no known joint fault to compensate for, so the
    hypothesis here is simpler: does natural gait emerge cleanly on a
    healthy platform, as a baseline before ever needing fault compensation.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "="*72)
        print("Go2EnvSparseRough | NATURAL GAIT — inherits Go2Env infrastructure")
        print("  Rewards: displacement + |tau*qdot| metabolic + survive")
        print("  Removed: air_time threshold, trot_bias/penalty, ang_vel-only tracking")
        print("  Also removed vs Go1: fr_binding, wz_excess (both Go1-fault-specific)")
        print(f"  METABOLIC={METABOLIC_WEIGHT}  UPRIGHT={UPRIGHT_WEIGHT}  "
              f"RATE={RATE_WEIGHT}  (matched to flat reward's proven-safe "
              f"values after real-hw jitter finding, was -0.05)")
        print("="*72 + "\n")

        _n = self.num_envs
        self._prev_pos_xy = self._robot.data.root_pos_w[:, :2].clone()

        self._stride_vx_sum = torch.zeros(_n, 4, device=self.device)
        self._foot_was_up   = torch.zeros(_n, 4, device=self.device)

        # Commanded-trajectory tracking (integrate cmd vx/wz -> expected path)
        self._expected_pos_local = torch.zeros(_n, 2, device=self.device)
        self._expected_heading   = torch.zeros(_n, device=self.device)
        self._episode_start_xy   = torch.zeros(_n, 2, device=self.device)

        # Gait emergence logging (research output — NOT reward)
        self._gait_log = {
            "n_contact_hist" : torch.zeros(_n, 5,  device=self.device),
            "air_time_sum"   : torch.zeros(_n, 4,  device=self.device),
            "air_time_count" : torch.zeros(_n, 4,  device=self.device),
            "power_sum"      : torch.zeros(_n,     device=self.device),
            "disp_sum"       : torch.zeros(_n,     device=self.device),
            "step_count"     : torch.zeros(_n,     device=self.device),
        }

        _rk = ["displacement", "lateral", "traj_track", "ang_vel",
               "metabolic", "upright", "alive", "fall",
               "action_rate", "action_jerk",
               "stride_quality", "stride_len", "hip_posture", "hip_reg"]
        self._ep_sums = {k: torch.zeros(_n, device=self.device) for k in _rk}

    # =========================================================================
    def _get_rewards(self):
        lin_vel  = self._robot.data.root_lin_vel_b
        ang_vel  = self._robot.data.root_ang_vel_b
        gravity  = self._robot.data.projected_gravity_b
        height   = self._robot.data.root_pos_w[:, 2]
        height_rel = height - self._terrain_z

        # ── 1. DISPLACEMENT — Gaussian velocity tracking, not raw speed ────
        r_displacement = 1.5 * torch.exp(
            -(lin_vel[:, 0] - self.command_manager.command[:, 0])**2 / 0.25)
        self._gait_log["disp_sum"]   += torch.clamp(lin_vel[:, 0], 0.0)
        self._gait_log["step_count"] += 1.0

        # ── 1b. LATERAL DRIFT ────────────────────────────────────────────
        r_lateral = -1.0 * lin_vel[:, 1]**2

        # ── 1c. COMMANDED TRAJECTORY TRACKING ───────────────────────────
        vx_cmd_traj = self.command_manager.command[:, 0]
        cos_h = torch.cos(self._expected_heading)
        sin_h = torch.sin(self._expected_heading)
        self._expected_pos_local[:, 0] += vx_cmd_traj * cos_h * self.step_dt
        self._expected_pos_local[:, 1] += vx_cmd_traj * sin_h * self.step_dt
        self._expected_heading         += self._episode_wz * self.step_dt

        pos_world   = self._robot.data.root_pos_w[:, :2]
        pos_local   = pos_world - self.scene.env_origins[:, :2]
        cross_track = torch.norm(pos_local - self._expected_pos_local, dim=1)
        r_traj_track = -0.5 * torch.clamp(cross_track, max=0.5)**2

        # ── 1d. HEADING CONTROL — straight damping / turn tracking split ──
        is_straight = (self._episode_wz.abs() < 0.10)
        r_ang_vel = torch.where(
            is_straight,
            -0.5 * ang_vel[:, 2]**2,
            0.8  * torch.exp(-(ang_vel[:, 2] - self._episode_wz)**2 / 0.25)
        )

        # ── 2. METABOLIC COST — Hoyt & Taylor 1981 ─────────────────────
        torques  = self._robot.data.applied_torque
        jvel     = self._robot.data.joint_vel
        mech_pow = torch.sum(torch.abs(torques * jvel), dim=1)
        r_metabolic = METABOLIC_WEIGHT * mech_pow * self.step_dt
        self._gait_log["power_sum"] += mech_pow.detach()

        # ── 3. ALIVE — terrain-relative ────────────────────────────────
        tilt     = torch.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)
        vel_gate = torch.clamp(lin_vel[:, 0] / 0.2, 0.0, 1.0)
        r_alive  = 0.3 * vel_gate * ((height_rel > 0.30) & (tilt < 0.3)).float()
        # NOTE: 0.30 threshold (not Go1's 0.26/0.28) — Go2 default calf
        # pose is -1.3 rad vs Go1's -1.5, different nominal stand height;
        # verify against your Go2 flat cfg's actual default standing
        # height before trusting this exact number on hardware.

        contact_fz   = self.scene.sensors["contact_sensor"].data.net_forces_w[:, :, 2]
        feet_contact = (contact_fz > 1.0).float()

        # ── 4. UPRIGHT — small weight, allow terrain-adaptive lean ──────
        r_upright = UPRIGHT_WEIGHT * (gravity[:, 0]**2 + gravity[:, 1]**2)

        # ── 5. FALL — terrain-relative ───────────────────────────────────
        r_fall = -10.0 * (height_rel < 0.20).float()

        first_touch = (feet_contact - self._prev_feet_contact).clamp(min=0.0)

        # ── 6. SMOOTHNESS — small weight, allow natural gait freedom ─────
        d1 = self._actions - self._prev_actions
        d2 = self._actions - 2*self._prev_actions + self._prev_prev_actions
        r_action_rate = RATE_WEIGHT * torch.sum(self._rate_weights * d1**2, dim=1)
        r_action_jerk = JERK_WEIGHT * torch.sum(self._rate_weights * d2**2, dim=1)

        # ── 7. STRIDE QUALITY — foot clearance during swing ──────────────
        # Front legs weighted 2.0 vs rear 1.0 — see header note: this is a
        # deliberate carryover, independently supported by Go2's own real
        # deploy log showing front-leg tracking lag, not blind copying.
        r_stride_quality = torch.zeros(self.num_envs, device=self.device)
        if self._foot_body_ids is not None:
            foot_pos    = self._robot.data.body_pos_w[:, self._foot_body_ids, :]
            foot_z_rel  = foot_pos[:, :, 2] - self._terrain_z.unsqueeze(1)
            swing_mask  = 1.0 - feet_contact
            clearance_w = torch.tensor([2.0, 2.0, 1.0, 1.0], device=self.device)
            clearance   = torch.clamp(foot_z_rel, 0.0, 0.10)
            r_stride_quality = 0.5 * torch.sum(
                swing_mask * clearance * clearance_w, dim=1) * vel_gate

        # ── 8. STRIDE LENGTH — Alexander 1976 collision mechanics ────────
        sensor_sl   = self.scene.sensors["contact_sensor"]
        last_air_sl = sensor_sl.data.last_air_time[:, :4]
        vx_clamp    = lin_vel[:, 0].clamp(min=0.0)
        stride_len  = last_air_sl * vx_clamp.unsqueeze(1)
        r_stride_len = STRIDE_LEN_WEIGHT * torch.sum(
            stride_len * first_touch, dim=1) * vel_gate

        # ── 9. HIP POSTURE — natural outward stance width ────────────────
        hip_deviation = (self._robot.data.joint_pos[:, :4]
                         - self._robot.data.default_joint_pos[:, :4])
        r_hip_posture = -3.0 * torch.sum(hip_deviation**2, dim=1)
        r_hip_reg     = -2.0 * torch.sum(self._actions[:, :4]**2, dim=1)

        # ── Gait contact logging (NOT reward — research output) ──────────
        n_contact = feet_contact.sum(dim=1).long().clamp(0, 4)
        for nc in range(5):
            self._gait_log["n_contact_hist"][:, nc] += (n_contact == nc).float()
        sensor   = self.scene.sensors["contact_sensor"]
        last_air = sensor.data.last_air_time[:, :4]
        self._gait_log["air_time_sum"]   += last_air * first_touch
        self._gait_log["air_time_count"] += first_touch
        self._prev_feet_contact[:]        = feet_contact.detach()

        for k, v in zip(
            ["displacement", "lateral", "traj_track", "ang_vel",
             "metabolic", "upright", "alive", "fall",
             "action_rate", "action_jerk",
             "stride_quality", "stride_len", "hip_posture", "hip_reg"],
            [r_displacement, r_lateral, r_traj_track, r_ang_vel,
             r_metabolic, r_upright, r_alive, r_fall,
             r_action_rate, r_action_jerk,
             r_stride_quality, r_stride_len, r_hip_posture, r_hip_reg]
        ):
            self._ep_sums[k] += v

        return self.step_dt * (
            r_displacement + r_lateral   + r_traj_track + r_ang_vel
            + r_metabolic  + r_upright   + r_alive
            + r_action_rate + r_action_jerk
            + r_stride_quality + r_stride_len
            + r_hip_posture + r_hip_reg
        ) + r_fall

    # =========================================================================
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        if self._global_step % 300 == 0 and self._global_step > 0:
            idx = env_ids
            print(f"\n=== Go2 Sparse Rough Rewards @ step {self._global_step} ===")
            for k, v in self._ep_sums.items():
                print(f"  {k:16s}: {v[idx].mean().item():+.3f}")

            hist  = self._gait_log["n_contact_hist"][idx]
            total = hist.sum(dim=1, keepdim=True).clamp(min=1)
            frac  = (hist / total).mean(dim=0)
            air_s = self._gait_log["air_time_sum"][idx]
            air_c = self._gait_log["air_time_count"][idx].clamp(min=1)
            mean_air  = (air_s / air_c).mean(dim=0)
            # BUG FIX: avg_power was never divided by step count (raw sum
            # printed as if already an average), and avg_speed divided by
            # (step_count * dt) when disp_sum is already a sum of
            # velocities (m/s), not distances -- disp_sum/step_count alone
            # gives the correct mean velocity; the extra /dt introduced a
            # spurious ~50x inflation (1/dt at dt=0.02s). Confirmed by
            # comparing against physically impossible printed values
            # (Speed: 22-31 m/s, Mech power: ~750,000W for a Go2).
            # This is a DISPLAY-ONLY bug in the research/gait-logging
            # block -- it does not affect _get_rewards() or the actual
            # PPO training signal, only these printed COT diagnostics.
            step_ct   = self._gait_log["step_count"][idx].mean().item()
            avg_power = (self._gait_log["power_sum"][idx].mean().item()
                         / max(step_ct, 1e-6))
            avg_speed = (self._gait_log["disp_sum"][idx].mean().item()
                         / max(step_ct, 1e-6))
            cot = avg_power / max(avg_speed, 1e-3)

            print(f"\n  ── EMERGENT GAIT (Hoyt & Taylor COT metric) ──")
            labels = ["0-foot", "1-foot", "2-foot(TROT)", "3-foot", "4-foot(stance)"]
            for nc, (f, lbl) in enumerate(zip(frac, labels)):
                bar = "#" * int(f.item() * 30)
                mk = ""
                if nc == 2 and f.item() > 0.35: mk = " <- TROT ok"
                elif nc == 4 and f.item() > 0.50: mk = " <- SHUFFLE warn"
                print(f"    {lbl:<22}: {bar:<30} {f.item()*100:5.1f}%{mk}")

            print(f"  Air [FL FR RL RR]: "
                  f"{mean_air[0]:.3f}  {mean_air[1]:.3f}  "
                  f"{mean_air[2]:.3f}  {mean_air[3]:.3f} s")
            print(f"  Mech power: {avg_power:.0f}W  Speed: {avg_speed:.3f}m/s  "
                  f"COT: {cot:.1f} W*s/m  "
                  # NOTE: the "500" threshold below was presumably chosen
                  # while avg_power/avg_speed were still inflated by the
                  # bug fixed above -- it's now comparing against
                  # correctly-scaled numbers it was never calibrated
                  # against. Treat this ok/high-cost label as unverified
                  # until you've seen a few corrected readings and checked
                  # they land in a sane range for Go2's real motor power.
                  f"({'ok efficient' if cot < 500 else '<- high cost (threshold unverified post-fix)'})")

        super()._reset_idx(env_ids)

        for k in self._gait_log:
            self._gait_log[k][env_ids] = 0.0
        self._prev_pos_xy[env_ids] = (
            self._robot.data.root_pos_w[env_ids, :2].clone())
        self._stride_vx_sum[env_ids] = 0.0
        self._foot_was_up[env_ids]   = 0.0

        spawn_pos = self._robot.data.root_pos_w[env_ids, :2]
        self._expected_pos_local[env_ids] = (
            spawn_pos - self.scene.env_origins[env_ids, :2])
        self._expected_heading[env_ids] = 0.0