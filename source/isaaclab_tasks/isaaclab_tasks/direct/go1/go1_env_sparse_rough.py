'''
python train.py --task Isaac-Go1-Sparse-Rough-Direct-v0 --num_envs 4096 --headless --checkpoint /home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-06-01_20-06-31/model_13700.pt
'''

# go1_env_sparse_rough.py — Natural Gait on Rough Terrain
# v2: Inherits ALL infrastructure from go1_env.py — only _get_rewards changed
#
# APPROACH: go1_env.py handles EVERYTHING (PACE, command_manager, terrain,
#   lighting, _reset_idx DR, _pre_physics_step delay FIFO, debug prints).
#   This class ONLY overrides _get_rewards with natural/metabolic rewards.
#
# THEORETICAL BASIS:
#   Hoyt & Taylor 1981 (Nature doi:10.1038/292239a0):
#     Animals select gaits to minimise metabolic cost of transport.
#     No explicit gait reward needed — physics selects optimal gait.
#   Heess et al. 2017 (arXiv:1707.02286):
#     Displacement + survive on rich terrain → trot/bound emerge naturally.
#
# WHAT IS REMOVED vs go1_env.py:
#   r_trot (air_time + trot_bias + trot_penalty) — prescriptive, gameable
#   r_ang_vel — wz tracking not primary objective for emergent gait
#   r_lin_vel — replaced by displacement (simpler, more biological)
#   r_hip_sat, r_hip_reg — allow natural hip compensation for impaired leg
#   r_foot_clear, r_foot_drag — allow natural foot pattern to emerge
#   r_lat_vel — allow lateral drift as impaired gait compensation
#
# WHAT IS KEPT (survival/safety, no gait prescription):
#   r_displacement — forward progress reward (Heess 2017)
#   r_metabolic    — |τ × q̇| mechanical power cost (Hoyt & Taylor 1981)
#   r_alive        — survive and keep moving (terrain-relative)
#   r_upright      — very small weight (rough terrain needs lean tolerance)
#   r_fall         — terrain-relative hard penalty
#   r_fr_binding   — hardware safety (not gait prescription)
#   r_action_rate/jerk — very small weight (prevent thermal runaway)
#   r_wz_excess    — RL_th stiction escape suppression
#
# WHAT IS INHERITED UNCHANGED from go1_env.py:
#   All PACE: Ia, τf DR, d DR, encoder bias, KP/KD DR, delay FIFO
#   command_manager: vx resampling, wz episode assignment
#   _reset_idx: all DR, terrain curriculum, friction DR
#   _setup_scene: lighting, terrain colours, marker poles
#   _pre_physics_step: delay buffer, FR_th cap, tanh squash
#   _get_dones: terrain-relative height termination
#   debug prints, sim logging
#
# REGISTER in __init__.py:
#   gym.register(
#     id="Isaac-Go1-Sparse-Rough-Direct-v0",
#     entry_point=".go1_env_sparse_rough:Go1EnvSparseRough",
#     disable_env_checker=True,
#     kwargs={
#       "env_cfg_entry_point": ".go1_rough_env_cfg:Go1RoughEnvCfg",
#       "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:Go1SparsePPORunnerCfg",
#     },
#   )
#
# TRAIN:
#   python train.py --task Isaac-Go1-Sparse-Rough-Direct-v0 \
#     --num_envs 4096 --max_iterations 15000 \
#     --checkpoint logs/.../rough_model_13700.pt --headless

import torch
import numpy as np

# ── Import parent — inherits ALL infrastructure ────────────────────────────
from isaaclab_tasks.direct.go1.go1_env import Go1Env

# ── Natural reward knobs ───────────────────────────────────────────────────
# Start conservative — adjust after confirming forward walking.
METABOLIC_WEIGHT  = -0.005   # 5× larger than original -0.001
                               # Hoyt & Taylor: 8Hz costs 3× more than 3Hz per metre
                               # At -0.001 the diff was 0.006/step (0.4% of displacement → ignored)
                               # At -0.005 the diff is 0.031/step (2%) → policy adapts frequency
                               # This IS the biological mechanism — metabolic cost selects gait Hz
STRIDE_LEN_WEIGHT = 5.0      # was 3.0 → stronger signal for longer strides on hardware
                               # 14Hz hw (35ms swing): 0.035×0.5=0.018m → tiny
                               # 4Hz  hw (125ms swing): 0.125×0.5=0.063m → 3.5× more
                               # Policy discovers FR forward reach naturally (no explicit target)
                               # Alexander 1976: longer strides = fewer collision losses/metre
                               # 200ms swing at 0.5m/s = 0.100m → 10 impacts/m (natural)
                               # NOT gameable: needs BOTH long air time AND forward velocity
UPRIGHT_WEIGHT    = -0.5
FR_BINDING_WEIGHT = -3.0     # was -2.0 → stronger: onset now at 0.750 where
                               # hardware FR_th actually operates (0.766 mean)
                               # Combined with onset 0.750: clear gradient to
                               # keep FR_th below binding zone in training
WZ_EXCESS_WEIGHT  = -0.3
RATE_WEIGHT       = -0.05
JERK_WEIGHT       = -0.02


class Go1EnvSparseRough(Go1Env):
    """
    Natural gait emergence on rough terrain.
    Inherits ALL infrastructure from Go1Env.
    Only _get_rewards is different — natural/metabolic rewards replace
    the prescriptive gait rewards (air_time, trot_bias, trot_penalty).

    Research hypothesis (Hoyt & Taylor 1981 + Heess 2017):
      With only displacement + metabolic cost + survive:
        - Trot emerges because it minimises cost-of-transport at medium speed
        - Impaired RL_th → tripod gait emerges as natural compensation
        - No explicit air_time, trot_bias, f_cmd prescription needed
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        # ── Call parent __init__ — ALL setup happens here ────────────────
        # PACE, command_manager, terrain_z, DR bounds, delay FIFO,
        # _setup_scene (lighting, colours, poles) — all inherited.
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "="*72)
        print("Go1EnvSparseRough | NATURAL GAIT — inherits Go1Env infrastructure")
        print("  Rewards: displacement + |τ×q̇| metabolic + survive")
        print("  Removed: air_time, trot_bias, trot_penalty, ang_vel tracking")
        print(f"  METABOLIC={METABOLIC_WEIGHT}  UPRIGHT={UPRIGHT_WEIGHT}  "
              f"FR_BINDING={FR_BINDING_WEIGHT}  RATE={RATE_WEIGHT}")
        print("="*72 + "\n")

        # ── Extra state for displacement tracking and gait logging ────────
        _n = self.num_envs
        self._prev_pos_xy = self._robot.data.root_pos_w[:, :2].clone()

        # ── Stride length tracking (Alexander 1976) ───────────────────
        self._stride_vx_sum  = torch.zeros(_n, 4, device=self.device)
        self._foot_was_up    = torch.zeros(_n, 4, device=self.device)

        # ── Commanded trajectory tracking (user's idea) ───────────────
        # Integrate the commanded (vx, wz) each step to get the EXPECTED
        # world position. Penalise deviation from this expected path.
        #
        # For wz=0.00: expected path = straight line → only x changes
        # For wz=0.10: expected path = arc radius=vx/wz → robot must
        #              follow THAT arc, not drift inside or outside it
        #
        # Cross-track error = distance(actual_local, expected_local)
        # Saturated at 0.5m to prevent divergence if FR_th binding
        # causes unavoidable drift (caps max penalty, keeps training stable)
        self._expected_pos_local = torch.zeros(_n, 2, device=self.device)
        self._expected_heading   = torch.zeros(_n, device=self.device)
        # For wz≈0 episodes: y-drift from start = cumulative path error.
        # Unlike r_lateral (instantaneous vy²), this catches sustained drift
        # that back-forth rocking produces (vy cancels but y drifts).
        # Training: perfect Isaac world [x,y] coordinates.
        # Deploy: use IMU yaw integral → corrective obs[2] → same response.
        self._episode_start_xy  = torch.zeros(_n, 2, device=self.device)
        # heading_integral NOT used — obs[2] stays fixed wz_cmd during training
        # Path straightness enforced by r_lateral + r_ang_vel (step-by-step)
        # FR_th PACE DR in go1_env.py handles physical binding compensation

        # Gait emergence logging (research output — NOT reward)
        self._gait_log = {
            "n_contact_hist" : torch.zeros(_n, 5,  device=self.device),
            "air_time_sum"   : torch.zeros(_n, 4,  device=self.device),
            "air_time_count" : torch.zeros(_n, 4,  device=self.device),
            "power_sum"      : torch.zeros(_n,     device=self.device),
            "disp_sum"       : torch.zeros(_n,     device=self.device),
            "step_count"     : torch.zeros(_n,     device=self.device),
        }

        # Override episode sums to match new reward keys
        _rk = ["displacement", "lateral", "traj_track", "ang_vel",
               "metabolic", "upright", "alive",
               "fall", "fr_binding", "wz_excess", "action_rate", "action_jerk",
               "stride_quality", "stride_len", "hip_posture", "hip_reg"]
        self._ep_sums = {k: torch.zeros(_n, device=self.device) for k in _rk}

    # =========================================================================
    # _get_rewards — ONLY OVERRIDE: natural/metabolic sparse rewards
    # All other methods (_pre_physics_step, _get_dones, _reset_idx, _setup_scene,
    # step, _get_observations) are inherited unchanged from Go1Env.
    # =========================================================================
    def _get_rewards(self):
        lin_vel  = self._robot.data.root_lin_vel_b
        ang_vel  = self._robot.data.root_ang_vel_b
        gravity  = self._robot.data.projected_gravity_b
        height   = self._robot.data.root_pos_w[:, 2]

        # Terrain-relative height (inherited from parent _terrain_z)
        height_rel = height - self._terrain_z

        # ── 1. DISPLACEMENT — velocity tracking (not raw speed) ───────
        # Original: clamp(vx, -2, 2) → rewards going as FAST as possible
        # Problem (model_3200): robot overshoots vx_cmd by 57% (1.19 vs 0.76)
        #   Policy learned: extreme hips + crouched knees → faster = more reward
        #   On rough terrain: high speed + crouched stance = falls on bumps
        #
        # Fix: Gaussian tracking like go1_env.py r_lin_vel
        #   Reward peaks when vx == vx_cmd, falls off for both too fast AND too slow
        #   At deploy vx_cmd=0.35: robot optimises for 0.35 m/s — not 1.19 m/s
        #   Biologically natural: animals match intended pace on rough terrain
        #   (horses don't gallop across rocky ground when they intended to trot)
        r_displacement = 1.5 * torch.exp(
            -(lin_vel[:, 0] - self.command_manager.command[:, 0])**2 / 0.25)
        self._gait_log["disp_sum"]  += torch.clamp(lin_vel[:, 0], 0.0)
        self._gait_log["step_count"] += 1.0

        # ── 1b. LATERAL DRIFT — natural biological efficiency ──────────
        # Biological basis (Hoyt & Taylor 1981):
        #   Lateral body velocity = zero net displacement toward destination
        #   = pure metabolic waste. Animals minimise lateral sway naturally.
        #
        # Exploit it prevents (model_5000 spiral):
        #   Spiraling robot: high vx (body frame) from centripetal motion,
        #   but vy is large (body curves sideways continuously).
        #   r_lateral = -1.0 × vy² directly penalises the sideways component
        #   of the spiral. Can't circle without vy → exploit breaks.
        #
        # Weight -1.0: same order as in go1_env.py r_lat_vel.
        r_lateral = -1.0 * lin_vel[:, 1]**2

        # ── 1c. COMMANDED TRAJECTORY TRACKING ─────────────────────────
        # User idea: integrate commanded (vx_cmd, wz_cmd) each step to get
        # expected world position. Penalise deviation from expected path.
        #
        #   wz=0.00, vx=0.5: expected = straight +x line
        #   wz=0.10, vx=0.5: expected = arc radius=5m curving left
        #   wz=-0.20, vx=0.5: expected = arc radius=2.5m curving right
        #
        # Cross-track = ||actual_local - expected_local||
        # Saturated at 0.5m → max penalty -0.125/step (never diverges)
        vx_cmd_traj = self.command_manager.command[:, 0]
        cos_h = torch.cos(self._expected_heading)
        sin_h = torch.sin(self._expected_heading)
        self._expected_pos_local[:, 0] += vx_cmd_traj * cos_h * self.step_dt
        self._expected_pos_local[:, 1] += vx_cmd_traj * sin_h * self.step_dt
        self._expected_heading          += self._episode_wz * self.step_dt

        pos_world    = self._robot.data.root_pos_w[:, :2]
        pos_local    = pos_world - self.scene.env_origins[:, :2]
        cross_track  = torch.norm(pos_local - self._expected_pos_local, dim=1)
        r_traj_track = -0.5 * torch.clamp(cross_track, max=0.5)**2

        # ── 1d. HEADING CONTROL — natural directional intent ────────────
        # Biological basis:
        #   Straight walking: an animal moving to a destination doesn't
        #   spin randomly — maintaining heading costs nothing extra and
        #   prevents RL_th stiction drift from growing into a spiral.
        #
        #   Turning: an animal turning efficiently matches its intended arc
        #   (horse rounding a corner doesn't overshoot or undershoot).
        #
        # Two behaviours from one term:
        #   Straight episode (|wz_cmd| < 0.05):
        #     penalise ANY yaw rate → -0.5 × yaw²
        #     Prevents stiction impulse from accumulating into drift
        #     (model_150 real HW: +27° drift in 14s at wz=0 → fixed here)
        #
        #   Turning episode (|wz_cmd| ≥ 0.05):
        #     reward tracking commanded wz → +0.3 × exp(-err²/0.25)
        #     Turning is natural when intended — just don't overshoot
        #
        # Straight threshold raised 0.05→0.10:
        # wz=-0.031 was classified as TURNING → got +0.3×exp() instead of -0.5×yaw²
        # → yaw drift -37.7° in 8s despite r_ang_vel being active.
        # 0.10 threshold: wz<0.10 = straight = damping applied → drift suppressed
        is_straight = (self._episode_wz.abs() < 0.10)
        r_ang_vel = torch.where(
            is_straight,
            -0.5  * ang_vel[:, 2]**2,                                   # straight: gentle damping (was -2.0 → killed turning)
            0.8   * torch.exp(-(ang_vel[:, 2] - self._episode_wz)**2    # turning: stronger tracking (was 0.3 → too weak)
                               / 0.25)
        )

        # ── 2. METABOLIC COST — Hoyt & Taylor 1981 ────────────────────
        # |τ × q̇| = mechanical power per joint.
        # Biologically: cost proportional to work done, not just force.
        # Key advantage: RL_th stiction (high τ, low q̇) → near-zero penalty.
        # τ² (previous version) over-penalised stiction compensation.
        torques   = self._robot.data.applied_torque    # [N, 12] Nm
        jvel      = self._robot.data.joint_vel          # [N, 12] rad/s
        mech_pow  = torch.sum(torch.abs(torques * jvel), dim=1)  # [N] W
        r_metabolic = METABOLIC_WEIGHT * mech_pow * self.step_dt
        self._gait_log["power_sum"] += mech_pow.detach()

        # ── 3. ALIVE — terrain-relative (from parent _terrain_z) ──────
        # Robot must be upright AND moving to earn alive reward.
        # Identical to go1_env.py r_alive but without gait_gate dependency.
        tilt     = torch.sqrt(gravity[:, 0]**2 + gravity[:, 1]**2)
        vel_gate = torch.clamp(lin_vel[:, 0] / 0.2, 0.0, 1.0)
        r_alive  = 0.3 * vel_gate * ((height_rel > 0.26) & (tilt < 0.3)).float()
        # ── Contact — computed early: needed by stride_quality and gait logging
        contact_fz   = self.scene.sensors["contact_sensor"].data.net_forces_w[:, :, 2]
        feet_contact = (contact_fz > 1.0).float()   # [N, 4]

        # ── 4. UPRIGHT — very small weight for lean tolerance ──────────
        # Tripod gait (impaired RL_th) requires body lean — don't over-penalise.
        r_upright = UPRIGHT_WEIGHT * (gravity[:, 0]**2 + gravity[:, 1]**2)

        # ── 5. FALL — terrain-relative ─────────────────────────────────
        r_fall = -10.0 * (height_rel < 0.18).float()

        # ── first_touch — computed here, used by stride_quality and stride_len ──
        # Must be before both those rewards (line ~260 and ~300).
        # Also used by gait logging at end — gait logging section skips recomputing it.
        first_touch = (feet_contact - self._prev_feet_contact).clamp(min=0.0)

        # ── 6. FR_th BINDING — hardware safety ─────────────────────────
        # Identical to go1_env.py — protects FR_th from mechanical binding.
        fr_th_q  = self._robot.data.joint_pos[:, 5]
        # Onset tightened 0.790→0.750: hardware FR_th mean=0.766 was BELOW
        # old onset → zero penalty where FR_th actually operates!
        # New onset 0.750: penalty fires at 0.766 (fr_prox=0.267)
        # Policy learns to keep FR_th below 0.750 → 3-leg compensation gait
        # On hardware: FR_th still goes to 0.842 but policy adapted gait
        # to not RELY on FR_th contribution beyond 0.750 range
        fr_prox  = torch.clamp((fr_th_q - 0.750) / 0.060, 0.0, 1.0)
        wz_scale = torch.clamp(self._episode_wz.abs() / self._WZ_MAX, 0.0, 1.0)
        r_fr_binding = -(abs(FR_BINDING_WEIGHT) + 1.5 * wz_scale) * fr_prox**2

        # ── 7. WZ EXCESS — RL_th stiction escape suppression ───────────
        # Hardware-specific: RL_th stiction creates impulsive yaw.
        wz_excess   = torch.clamp(
            ang_vel[:, 2].abs() - self._episode_wz.abs() * 1.5, min=0.0)
        r_wz_excess = WZ_EXCESS_WEIGHT * wz_excess**2

        # ── 8. SMOOTHNESS — very small (allow natural gait freedom) ────
        d1 = self._actions - self._prev_actions
        d2 = self._actions - 2*self._prev_actions + self._prev_prev_actions
        r_action_rate = RATE_WEIGHT * torch.sum(self._rate_weights * d1**2, dim=1)
        r_action_jerk = JERK_WEIGHT * torch.sum(self._rate_weights * d2**2, dim=1)

        # ── 9. STRIDE QUALITY — foot clearance during swing ────────────
        # What it does: rewards foot HEIGHT above terrain during swing phase.
        # NOT prescriptive about WHEN or HOW OFTEN to lift — only HOW HIGH.
        # Biological basis: animals naturally clear feet higher on rough terrain
        # to avoid stumbling. Higher clearance per swing = fewer terrain catches.
        #
        # Front legs [FL, FR]: weight 2.0 — historically dragged on real hardware
        #   (FL_kn mean was -0.060 before fix). Extra signal guides them upward.
        # Rear legs [RL, RR]: weight 1.0 — already lifting adequately.
        # vel_gate: only fires when moving — prevents static leg-raising exploit.
        # clamp(0, 0.10): 10cm window — realistic Go1 clearance range.
        r_stride_quality = torch.zeros(self.num_envs, device=self.device)
        if self._foot_body_ids is not None:
            foot_pos   = self._robot.data.body_pos_w[:, self._foot_body_ids, :]
            foot_z_rel = foot_pos[:, :, 2] - self._terrain_z.unsqueeze(1)
            swing_mask = 1.0 - feet_contact
            clearance_w = torch.tensor([2.0, 2.0, 1.0, 1.0], device=self.device)
            clearance   = torch.clamp(foot_z_rel, 0.0, 0.10)
            r_stride_quality = 0.5 * torch.sum(
                swing_mask * clearance * clearance_w, dim=1) * vel_gate

        # ── 9b. STRIDE LENGTH — Alexander 1976 collision mechanics ─────
        # Biological basis:
        #   Every foot touchdown = collision energy loss ∝ 1/stride_length.
        #   Cost of transport = metabolic + collision losses.
        #   Animals minimise TOTAL COT → long strides reduce collision term.
        #   Alexander (1976): optimal stride length balances metabolic cost
        #   (longer strides = higher peak force) vs collision cost (shorter
        #   strides = more impacts/metre). Trot ~3Hz is the sweet spot.
        #
        # Implementation:
        #   stride_length = air_time × avg_vx during swing
        #   This fires at each TOUCHDOWN using contact sensor last_air_time.
        #   The reward is proportional to actual ground covered per stride —
        #   not a threshold, not a minimum, just: "longer stride = bigger reward"
        #
        # Why NOT gameable (unlike air_time threshold):
        #   To cheat: need long air_time AND high vx simultaneously.
        #   Keeping a foot up without moving → vx=0 → stride_len=0 → no reward.
        #   Only actual locomotion earns the reward.
        #
        # At 15Hz real HW (35ms swing at 0.5m/s):
        #   stride_len = 0.035 × 0.5 = 0.0175m → r = 3.0 × 0.0175 = 0.053/touch
        # At 3Hz sim target (200ms swing at 0.5m/s):
        #   stride_len = 0.200 × 0.5 = 0.100m  → r = 3.0 × 0.100  = 0.300/touch
        #   → 5.7× more reward for proper stride → policy selects longer strides
        sensor_sl   = self.scene.sensors["contact_sensor"]
        last_air_sl = sensor_sl.data.last_air_time[:, :4]   # [N,4] seconds
        vx_clamp    = lin_vel[:, 0].clamp(min=0.0)          # only forward motion
        stride_len  = last_air_sl * vx_clamp.unsqueeze(1)   # [N,4] metres
        r_stride_len = STRIDE_LEN_WEIGHT * torch.sum(
            stride_len * first_touch, dim=1) * vel_gate

        # ── 10. HIP POSTURE — natural outward stance width ──────────────
        # Problem observed: FR_hip rotates inward on rough terrain (narrower
        # base of support → less stable on uneven ground).
        # Biological fact: quadrupeds WIDEN stance on unstable terrain —
        # wider BOS reduces tipping moment when foot hits unexpected height.
        #
        # WHY r_hip_reg (go1_env.py) didn't prevent this:
        #   r_hip_reg = -w × sum(actions[:, :4]²)  ← penalises COMMAND delta
        #   If the policy sends a persistent small command, joint drifts inward
        #   but action is small → r_hip_reg barely fires.
        #
        # THIS reward (r_hip_posture) penalises ACTUAL joint position deviation:
        #   hip_deviation = actual_q[:, :4] - default_q[:, :4]
        #   At default: deviation=0 → r=0 (free)
        #   Hip drifts inward: deviation grows → penalty fires
        #   Policy learns: stay near default = free; drift inward = costly
        #
        # Weight -1.5: gentle enough to allow terrain-adaptive lean
        # but strong enough to hold natural stance width.
        hip_deviation = (self._robot.data.joint_pos[:, :4]
                         - self._robot.data.default_joint_pos[:, :4])
        r_hip_posture = -3.0 * torch.sum(hip_deviation**2, dim=1)

        # ── Fix 2: Hip action penalty (complements position penalty) ───
        # r_hip_posture penalises accumulated drift in joint POSITION.
        # But policy can drift hips via persistent small commands that
        # individually look small but sum to large position error.
        # r_hip_reg penalises the ACTION sent each step — blocks the commands
        # that drive the drift before they accumulate.
        # go1_env.py proved this pattern works (r_hip_reg -1.5 effective).
        # Combined: position keeps hips at default, action prevents drift commands.
        r_hip_reg = -2.0 * torch.sum(self._actions[:, :4]**2, dim=1)

        # ── Gait contact logging (NOT reward — research output) ────────
        n_contact    = feet_contact.sum(dim=1).long().clamp(0, 4)
        for nc in range(5):
            self._gait_log["n_contact_hist"][:, nc] += (n_contact == nc).float()
        sensor      = self.scene.sensors["contact_sensor"]
        last_air    = sensor.data.last_air_time[:, :4]
        # first_touch already computed above — reuse here
        self._gait_log["air_time_sum"]   += last_air * first_touch
        self._gait_log["air_time_count"] += first_touch
        self._prev_feet_contact[:]        = feet_contact.detach()

        # ── Episode sums ───────────────────────────────────────────────
        for k, v in zip(
            ["displacement", "lateral", "traj_track", "ang_vel",
             "metabolic", "upright", "alive",
             "fall", "fr_binding", "wz_excess", "action_rate", "action_jerk",
             "stride_quality", "stride_len", "hip_posture", "hip_reg"],
            [r_displacement, r_lateral, r_traj_track, r_ang_vel,
             r_metabolic, r_upright, r_alive,
             r_fall, r_fr_binding, r_wz_excess, r_action_rate, r_action_jerk,
             r_stride_quality, r_stride_len, r_hip_posture, r_hip_reg]
        ):
            self._ep_sums[k] += v

        # ── Total reward ───────────────────────────────────────────────
        return self.step_dt * (
            r_displacement + r_lateral   + r_traj_track + r_ang_vel
            + r_metabolic  + r_upright   + r_alive
            + r_fr_binding + r_wz_excess
            + r_action_rate + r_action_jerk
            + r_stride_quality + r_stride_len
            + r_hip_posture + r_hip_reg
        ) + r_fall

    # =========================================================================
    # _reset_idx — extend parent to add gait log reset + displacement tracking
    # =========================================================================
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        # ── Gait emergence report ─────────────────────────────────────
        if self._global_step % 300 == 0 and self._global_step > 0:
            idx = env_ids
            print(f"\n=== Sparse Rough Rewards @ step {self._global_step} ===")
            for k, v in self._ep_sums.items():
                print(f"  {k:16s}: {v[idx].mean().item():+.3f}")

            hist  = self._gait_log["n_contact_hist"][idx]
            total = hist.sum(dim=1, keepdim=True).clamp(min=1)
            frac  = (hist / total).mean(dim=0)
            air_s = self._gait_log["air_time_sum"][idx]
            air_c = self._gait_log["air_time_count"][idx].clamp(min=1)
            mean_air = (air_s / air_c).mean(dim=0)
            avg_power = self._gait_log["power_sum"][idx].mean().item()
            step_ct   = self._gait_log["step_count"][idx].mean().item()
            avg_speed = (self._gait_log["disp_sum"][idx].mean().item()
                         / max(step_ct * self.step_dt, 1e-6))
            cot = avg_power / max(avg_speed, 1e-3)

            print(f"\n  ── EMERGENT GAIT (Hoyt & Taylor COT metric) ──")
            labels = ["0-foot", "1-foot", "2-foot(TROT)", "3-foot", "4-foot(stance)"]
            for nc, (f, lbl) in enumerate(zip(frac, labels)):
                bar = "█" * int(f.item() * 30)
                mk  = ""
                if nc == 2 and f.item() > 0.35: mk = " ← TROT ✓"
                elif nc == 3 and f.item() > 0.60: mk = " ← TRIPOD (impaired leg)"
                elif nc == 4 and f.item() > 0.50: mk = " ← SHUFFLE ⚠"
                print(f"    {lbl:<22}: {bar:<30} {f.item()*100:5.1f}%{mk}")

            print(f"  Air [FL FR RL RR]: "
                  f"{mean_air[0]:.3f}  {mean_air[1]:.3f}  "
                  f"{mean_air[2]:.3f}  {mean_air[3]:.3f} s")
            if mean_air[2].item() < mean_air.mean().item() * 0.7:
                print(f"  ⚠ RL impaired: RL={mean_air[2]:.3f} vs avg={mean_air.mean():.3f}")
            print(f"  Mech power: {avg_power:.0f}W  Speed: {avg_speed:.3f}m/s  "
                  f"COT: {cot:.1f} W·s/m  "
                  f"({'✓ efficient' if cot < 500 else '← high cost'})")

        # ── Call parent _reset_idx — handles ALL DR, PACE, terrain ────
        super()._reset_idx(env_ids)

        # ── Reset gait logs and displacement tracking ──────────────────
        for k in self._gait_log:
            self._gait_log[k][env_ids] = 0.0
        self._prev_pos_xy[env_ids] = (
            self._robot.data.root_pos_w[env_ids, :2].clone())
        self._stride_vx_sum[env_ids]  = 0.0
        self._foot_was_up[env_ids]    = 0.0

        # Reset commanded trajectory — robot spawns at env origin, heading=+x
        # Expected position starts at actual spawn local position
        spawn_pos = self._robot.data.root_pos_w[env_ids, :2]
        self._expected_pos_local[env_ids] = (
            spawn_pos - self.scene.env_origins[env_ids, :2])
        self._expected_heading[env_ids]   = 0.0   # spawn faces +x direction