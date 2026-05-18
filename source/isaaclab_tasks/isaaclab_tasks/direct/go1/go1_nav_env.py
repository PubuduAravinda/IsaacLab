# go1_nav_env.py — High-Level Navigation Policy Environment v8
#
# ═══════════════════════════════════════════════════════════════════════════
# KEY FIXES vs v7
# ═══════════════════════════════════════════════════════════════════════════
# 1. NORMALIZER STD DEBUG — printed at startup
#    Shows exact std for each obs dimension.
#    obs[2]=wz std tells you the safe HL wz range (stay within ±2×std).
#    If wz_std > 0.05 → widen cfg wz range. If < 0.03 → wz is dangerous.
#
# 2. CYCLE DEBUG — prints one full HL→LL cycle every 50 HL steps (env 0)
#    Shows: HL cmd → normalized LL obs[0:3] → LL raw output → joint delta
#    KEY: if norm_wz > 3.0, the LL is overreacting → reduce wz range
#
# 3. wz RANGE — reduced to ±0.10 in cfg (was ±0.35)
#    ±0.35 caused yaw=1.2 rad/s (3.4× commanded) due to small normalizer std
#    Adjust after seeing "[LL NORMALIZER STD] safe wz range" at startup
#
# 4. wz DEADBAND — reduced to 0.02 (was 0.03/0.05)
#    Prevents tiny constant wz that resonates in gait
#
# 5. DIVERGENCE GUARD — clips log_std if noise_std > 2.0
#    Prevents the noise_std=1465 explosion seen in 18h training run
#
# ═══════════════════════════════════════════════════════════════════════════
# HL POLICY FLAG
# ═══════════════════════════════════════════════════════════════════════════
# False → fixed cmd [0.5, 0.0, 0.0] — confirm LL walks
# True  → HL PPO policy controls [vx, wz]
# ═══════════════════════════════════════════════════════════════════════════

import torch
import numpy as np
from torch import nn

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG

# ─────────────────────────────────────────────────────────────────────────────
# ★ TOGGLE THIS FLAG ★
# False → fixed cmd [0.5, 0, 0] — verify LL walks before HL training
# True  → HL PPO policy controls [vx, wz]
# ─────────────────────────────────────────────────────────────────────────────
HL_POLICY_ACTIVE = True

_DEBUG_FIXED_CMD = [0.5, 0.0, 0.0]   # [vx, vy=0, wz=0]

# ─────────────────────────────────────────────────────────────────────────────
# Timing constants
# ─────────────────────────────────────────────────────────────────────────────
_LL_DECIMATION = 10
_HL_DECIMATION = 50
_N_SUBSTEPS    = _HL_DECIMATION // _LL_DECIMATION    # 5
_MAX_DELAY     = 8     # 8 × 2ms = 16ms FIFO delay

# wz deadband — clamp |wz| < this to zero (prevents tiny-turn gait resonance)
_WZ_DEADBAND = 0.02


# =============================================================================
# FROZEN LOW-LEVEL POLICY LOADER
# =============================================================================
class _LLPolicyLoader(nn.Module):

    def __init__(self, checkpoint_path, obs_dim=46, action_dim=12,
                 hidden=(512, 256, 128), activation="elu"):
        super().__init__()
        act_fn = nn.ELU if activation == "elu" else nn.ReLU
        layers, in_dim = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim)]
        self.actor = nn.Sequential(*layers)
        self.register_buffer("_norm_mean", torch.zeros(obs_dim))
        self.register_buffer("_norm_var",  torch.ones(obs_dim))
        self._has_normalizer = False
        if checkpoint_path:
            self._load(checkpoint_path)

    def _load(self, path):
        print(f"\n[LL] Loading frozen policy from: {path}")
        ckpt     = torch.load(path, map_location="cpu", weights_only=False)
        state    = ckpt.get("policy_state_dict", ckpt)
        remapped = {k[len("actor."):]: v
                    for k, v in state.items() if k.startswith("actor.")}
        self.actor.load_state_dict(remapped, strict=True)
        print(f"[LL] Actor loaded  — {len(remapped)} tensors")

        ns = ckpt.get("normalizer_state", None)
        if ns is not None:
            for mk in ("_mean", "mean", "running_mean"):
                if mk in ns:
                    self._norm_mean.copy_(ns[mk].squeeze())
                    break
            for vk in ("_var", "var", "running_var"):
                if vk in ns:
                    self._norm_var.copy_(ns[vk].squeeze())
                    break
            self._has_normalizer = True

            # ── NORMALIZER STD DEBUG ──────────────────────────────────────
            # This tells you the safe HL wz range.
            # Keep HL wz within ±2×wz_std to stay in LL training distribution.
            stds = (self._norm_var + 1e-8).sqrt()
            vx_std  = stds[0].item()
            vy_std  = stds[1].item()
            wz_std  = stds[2].item()
            safe_wz = 2.0 * wz_std

            print(f"\n[LL NORMALIZER STD] — key values for HL design:")
            print(f"  obs[0] vx  : mean={self._norm_mean[0]:.4f}  "
                  f"std={vx_std:.4f}")
            print(f"  obs[1] vy  : mean={self._norm_mean[1]:.4f}  "
                  f"std={vy_std:.4f}  (always 0 in LL training)")
            print(f"  obs[2] wz  : mean={self._norm_mean[2]:.4f}  "
                  f"std={wz_std:.4f}")
            print(f"  ★ Safe HL wz range : ±{safe_wz:.3f} rad/s  "
                  f"(±2×wz_std — stay within LL training distribution)")
            print(f"  ★ Current cfg range: ±0.10 rad/s")
            if safe_wz < 0.08:
                print(f"  ⚠ wz_std is very small ({wz_std:.4f}) — "
                      f"even ±0.10 may cause overreaction!")
                print(f"  → Consider reducing cfg wz range to ±{safe_wz:.3f}")
            elif safe_wz > 0.20:
                print(f"  ✓ wz_std is healthy ({wz_std:.4f}) — "
                      f"±0.10 is safe, could widen to ±{safe_wz:.3f}")
            else:
                print(f"  ✓ wz_std is moderate ({wz_std:.4f}) — "
                      f"±0.10 is appropriate")

            # Full normalizer table (first 15 obs dimensions)
            print(f"\n[LL NORMALIZER] First 15 obs dimensions:")
            labels = (["vx","vy","wz"]
                      + [f"jd{i}" for i in range(12)]
                      + ["jv0"])
            for i in range(min(15, len(self._norm_mean))):
                lbl = labels[i] if i < len(labels) else f"obs{i}"
                print(f"  [{i:2d}] {lbl:4s}  "
                      f"mean={self._norm_mean[i]:.4f}  "
                      f"std={stds[i]:.4f}")
            print()
        else:
            print("[LL] WARNING: No normalizer_state — obs not normalised")

    @torch.no_grad()
    def forward(self, obs):
        if self._has_normalizer:
            std = (self._norm_var + 1e-8).sqrt().clamp(min=1e-5)
            obs = (obs - self._norm_mean) / std
        return self.actor(obs)

    def get_normalized(self, obs):
        """Return normalized obs for debug inspection."""
        if self._has_normalizer:
            std = (self._norm_var + 1e-8).sqrt().clamp(min=1e-5)
            return (obs - self._norm_mean) / std
        return obs


# =============================================================================
# HIGH-LEVEL NAVIGATION ENVIRONMENT
# =============================================================================
class Go1NavEnv(DirectRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):

        # Step 1: Isaac Lab base init (calls _setup_scene inside)
        super().__init__(cfg, render_mode, **kwargs)

        # Step 2: Buffers that need robot.data (available after super returns)
        _n = self.num_envs
        _j = 12
        self._current_ll_target = self._robot.data.default_joint_pos.clone()

        print("\n" + "="*72)
        print(f"Go1NavEnv v8  |  HRL Navigation")
        print(f"  HL_POLICY_ACTIVE = {HL_POLICY_ACTIVE}")
        if not HL_POLICY_ACTIVE:
            print(f"  ★ HL DISABLED — fixed cmd {_DEBUG_FIXED_CMD}")
        else:
            print(f"  ★ HL ACTIVE — PPO controls [vx, wz]  vy=0 always")
        print(f"  HL:{500//_HL_DECIMATION}Hz  LL:{500//_LL_DECIMATION}Hz  "
              f"{_N_SUBSTEPS} LL calls/HL step  FIFO:{_MAX_DELAY}×2ms")
        print(f"  wz range: [{cfg.action_space.low[1]:.3f}, "
              f"{cfg.action_space.high[1]:.3f}] rad/s  "
              f"deadband: ±{_WZ_DEADBAND}")
        print("="*72 + "\n")

        # Step 3: Visual markers
        self._markers_ok = False
        try:
            _gcfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/NavGoals")
            _gcfg.markers["sphere"].radius = 0.35
            _gcfg.markers["sphere"].visual_material = \
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.1, 0.1), opacity=0.9)
            self._goal_markers = VisualizationMarkers(_gcfg)

            _scfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/NavStarts")
            _scfg.markers["sphere"].radius = 0.25
            _scfg.markers["sphere"].visual_material = \
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 1.0, 0.1), opacity=0.9)
            self._start_markers = VisualizationMarkers(_scfg)
            self._markers_ok = True
            print("[Markers] 🟢 Start=green  🔴 Goal=red  ✓")
        except Exception as e:
            print(f"[Markers] Disabled ({e})")

        # Step 4: PACE physics — identical to Go1Env
        _z = torch.zeros(_n, _j, device=self.device)
        self._robot.write_joint_armature_to_sim(_z)
        Ia = torch.tensor([
            0.0026,0.0037,0.0029,0.0031,
            0.0052,0.0045,0.1343,0.0064,
            0.0070,0.0080,0.0065,0.0070,
        ], device=self.device).unsqueeze(0).expand(_n, -1)
        self._robot.write_joint_armature_to_sim(Ia)
        self._robot.write_joint_damping_to_sim(_z)

        self._tau_f_nominal = torch.tensor([
            0.026,0.028,0.029,0.031,
            0.052,0.044,4.944,0.045,
            0.070,0.071,0.078,0.079,
        ], device=self.device)
        tau_f_init = self._tau_f_nominal.unsqueeze(0).expand(_n, -1)
        self._tau_fn = None
        for _fn in ("write_joint_friction_coefficient_to_sim",
                    "write_joint_friction_to_sim"):
            if hasattr(self._robot, _fn):
                getattr(self._robot, _fn)(tau_f_init)
                self._tau_fn = _fn
                break
        self._friction_env_ids_ok = False
        if self._tau_fn:
            try:
                _t = torch.zeros(1, 12, device=self.device)
                _t[0] = self._tau_f_nominal
                getattr(self._robot, self._tau_fn)(
                    _t, env_ids=torch.tensor([0], device=self.device))
                getattr(self._robot, self._tau_fn)(tau_f_init)
                self._friction_env_ids_ok = True
            except TypeError:
                pass

        self._d_nominal = torch.tensor([
            0.039,0.040,0.040,0.042,
            0.045,0.050,3.459,0.050,
            0.124,0.092,0.108,0.108,
        ], device=self.device)
        actuator = self._robot.actuators["legs"]
        if hasattr(actuator, "viscous_friction"):
            actuator.viscous_friction[:] = (
                self._d_nominal.unsqueeze(0).expand(_n, -1))

        self._bias_values = torch.tensor([
            -0.006,-0.002,-0.005, 0.004,
            -0.018,-0.022, 0.063,-0.020,
            -0.001,-0.011, 0.012,-0.021,
        ], device=self.device)
        for _attr in ("encoder_bias", "position_offset", "bias"):
            if hasattr(actuator, _attr):
                getattr(actuator, _attr)[:] = (
                    self._bias_values.unsqueeze(0).expand(_n, -1))
                break

        # Step 5: Tanh squash (must match Go1Env exactly)
        self._delta_soft_lo = torch.tensor(
            [-0.08,-0.08,-0.08,-0.08,
             -0.35,-0.35,-0.35,-0.35,
             -0.35,-0.35,-0.35,-0.35], device=self.device)
        self._delta_soft_hi = torch.tensor(
            [ 0.08, 0.08, 0.08, 0.08,
              0.35, 0.35, 0.35, 0.35,
              0.35, 0.35, 0.35, 0.35], device=self.device)
        self._tanh_mid  = (self._delta_soft_hi + self._delta_soft_lo) * 0.5
        self._tanh_half = (self._delta_soft_hi - self._delta_soft_lo) * 0.5
        self._fr_th_max_delta = 0.02

        # Step 6: KP/KD nominal
        self._kp_nominal = torch.tensor(
            [35,35,35,35,65,65,65,65,80,80,80,80],
            device=self.device, dtype=torch.float)
        self._kd_nominal = torch.tensor(
            [4.0,4.0,4.0,4.0,4.5,4.5,4.5,4.5,5.0,5.0,5.0,5.0],
            device=self.device, dtype=torch.float)
        self._rl_fault_level = torch.ones(_n, device=self.device)

        # Step 7: FIFO delay buffer
        self._delay_buf  = torch.zeros(_n, _MAX_DELAY+1, _j, device=self.device)
        self._delay_head = 0

        # Step 8: HL/LL runtime state
        self._ll_prev_actions = torch.zeros(_n, _j, device=self.device)
        self._substep_counter = 0
        self._current_hl_cmd  = torch.zeros(_n, 3, device=self.device)

        # Action bounds for rescaling PPO [-1,1] → [lo,hi]
        self._cmd_lo = torch.tensor(cfg.action_space.low,  device=self.device)
        self._cmd_hi = torch.tensor(cfg.action_space.high, device=self.device)

        # Fixed cmd for HL=OFF
        self._fixed_cmd_vals = torch.tensor(
            _DEBUG_FIXED_CMD, device=self.device)

        # Gait frequency 2Hz (obs[45]=0.333)
        f_lo, f_hi = 1.5, 3.0
        self._ll_f_cmd_norm = torch.full(
            (_n, 1), (cfg.ll_f_cmd_fixed - f_lo) / (f_hi - f_lo),
            device=self.device)

        # Step 9: Navigation state
        self._goal_pos_w  = torch.zeros(_n, 2, device=self.device)
        self._start_pos_w = torch.zeros(_n, 2, device=self.device)
        self._prev_dist   = torch.full((_n,), cfg.goal_dist_max,
                                       device=self.device)

        # Step 10: Debug counters
        self._dbg_fall_count    = 0
        self._dbg_success_count = 0
        self._dbg_timeout_count = 0
        self._dbg_total_eps     = 0
        self._dbg_hl_step       = 0
        self._dbg_ll_calls      = 0
        self._dbg_cmd_buf       = []
        self._dbg_vx_buf        = []
        self._dbg_cycle_enabled = True  # set False to disable per-step debug

        # Step 11: Load frozen LL policy
        assert cfg.ll_checkpoint, "[Go1NavEnv] ll_checkpoint must be set!"
        self._ll_policy = _LLPolicyLoader(
            checkpoint_path = cfg.ll_checkpoint,
            obs_dim         = cfg.ll_obs_dim,
            action_dim      = cfg.ll_action_dim,
            hidden          = cfg.ll_actor_hidden,
            activation      = cfg.ll_activation,
        ).to(self.device)
        self._ll_policy.eval()
        for p in self._ll_policy.parameters():
            p.requires_grad_(False)
        n_p = sum(p.numel() for p in self._ll_policy.parameters())
        print(f"[LL] Policy frozen ✓  ({n_p} params)")
        print(f"[Go1NavEnv] Ready.\n")

    # =========================================================================
    def _setup_scene(self):
        # Called inside super().__init__() BEFORE robot.data is populated.
        # ONLY register robot here — no robot.data access.
        self._robot = self.scene["robot"]

    # =========================================================================
    def _update_markers(self):
        if not self._markers_ok:
            return
        n = self.num_envs
        goal_xyz = torch.cat([
            self._goal_pos_w,
            torch.full((n,1), 0.5, device=self.device)], dim=-1)
        start_xyz = torch.cat([
            self._start_pos_w,
            torch.full((n,1), 0.3, device=self.device)], dim=-1)
        self._goal_markers.visualize(goal_xyz)
        self._start_markers.visualize(start_xyz)

    # =========================================================================
    # _pre_physics_step — store HL cmd, reset counter
    # =========================================================================
    def _pre_physics_step(self, action: torch.Tensor):
        self._dbg_hl_step += 1
        self._substep_counter = 0

        if HL_POLICY_ACTIVE:
            # Rescale PPO [-1,1] → physical [lo, hi]
            alpha  = (action.clamp(-1.0, 1.0) + 1.0) / 2.0
            cmd_2d = self._cmd_lo + alpha*(self._cmd_hi - self._cmd_lo)
            self._current_hl_cmd[:, 0] = cmd_2d[:, 0]   # vx
            self._current_hl_cmd[:, 1] = 0.0             # vy always zero
            # wz deadband: clamp tiny turns to zero
            wz = cmd_2d[:, 1]
            wz = torch.where(wz.abs() < _WZ_DEADBAND, torch.zeros_like(wz), wz)
            self._current_hl_cmd[:, 2] = wz
        else:
            self._current_hl_cmd[:] = self._fixed_cmd_vals

        self._dbg_cmd_buf.append(self._current_hl_cmd.mean(dim=0).cpu())
        self._dbg_vx_buf.append(
            self._robot.data.root_lin_vel_b[:, 0].mean().item())

    # =========================================================================
    # _apply_action — LL at 50Hz via substep counter + CYCLE DEBUG
    # =========================================================================
    def _apply_action(self):
        if self._substep_counter % _LL_DECIMATION == 0:

            # Build 46D LL obs
            ll_obs = torch.cat([
                self._current_hl_cmd,                                        # [0:3]
                self._robot.data.joint_pos
                    - self._robot.data.default_joint_pos,                    # [3:15]
                torch.clamp(self._robot.data.joint_vel, -5.0, 5.0),         # [15:27]
                torch.clamp(self._robot.data.root_ang_vel_b, -5.0, 5.0),   # [27:30]
                self._robot.data.projected_gravity_b,                        # [30:33]
                self._ll_prev_actions,                                       # [33:45]
                self._ll_f_cmd_norm,                                         # [45]
            ], dim=-1)

            # Forward frozen LL
            with torch.inference_mode():
                raw = self._ll_policy(ll_obs)
            self._dbg_ll_calls += 1

            # Tanh squash → joint targets
            joint_delta = self._tanh_mid + self._tanh_half * torch.tanh(raw)
            target_pos  = self._robot.data.default_joint_pos + joint_delta

            # FIFO delay buffer
            self._delay_buf[:, self._delay_head] = target_pos
            delayed_idx    = (self._delay_head - _MAX_DELAY) % (_MAX_DELAY + 1)
            delayed_target = self._delay_buf[:, delayed_idx].clone()
            self._delay_head = (self._delay_head + 1) % (_MAX_DELAY + 1)

            # FR_th safety cap
            fr_th_max = (self._robot.data.default_joint_pos[:, 5]
                         + self._fr_th_max_delta)
            delayed_target[:, 5] = torch.minimum(delayed_target[:, 5], fr_th_max)

            self._current_ll_target = delayed_target
            self._ll_prev_actions   = joint_delta.detach().clone()

            # ── CYCLE DEBUG — one full HL→LL cycle (env 0, every 50 HL steps)
            if (self._dbg_cycle_enabled
                    and self._dbg_hl_step > 0
                    and self._dbg_hl_step % 50 == 0
                    and self._substep_counter == 0):   # first substep only
                self._print_cycle_debug(ll_obs, raw, joint_delta)

        self._robot.set_joint_position_target(self._current_ll_target)
        self._substep_counter += 1

    # =========================================================================
    # _print_cycle_debug — key diagnostic for HL→LL interaction
    # =========================================================================
    def _print_cycle_debug(self, ll_obs, raw, joint_delta):
        e = 0  # env 0
        cmd0   = self._current_hl_cmd[e].cpu().numpy()
        raw0   = raw[e].cpu().numpy()
        delta0 = joint_delta[e].cpu().numpy()
        h0     = self._robot.data.root_pos_w[e, 2].item()
        vx0    = self._robot.data.root_lin_vel_b[e, 0].item()
        vy0    = self._robot.data.root_lin_vel_b[e, 1].item()
        yr0    = self._robot.data.root_ang_vel_b[e, 2].item()

        # Normalized obs[0:3] — this is what LL actually sees after normalizer
        norm_obs = self._ll_policy.get_normalized(ll_obs[e:e+1])[0]
        norm_cmd = norm_obs[:3].cpu().numpy()

        # Normalizer std for obs[2] (wz)
        if self._ll_policy._has_normalizer:
            wz_std = (self._ll_policy._norm_var[2] + 1e-8).sqrt().item()
        else:
            wz_std = 1.0

        print(f"\n{'─'*65}")
        print(f"[CYCLE DEBUG]  HL_step={self._dbg_hl_step}")
        print(f"  ── HL command (raw physical) ────────────────────────────")
        print(f"     vx = {cmd0[0]:.4f} m/s  "
              f"vy = {cmd0[1]:.4f}(always0)  "
              f"wz = {cmd0[2]:.4f} rad/s")
        print(f"  ── LL normalized obs[0:3] (what LL network sees) ────────")
        print(f"     norm_vx = {norm_cmd[0]:+.3f}  "
              f"norm_vy = {norm_cmd[1]:+.3f}  "
              f"norm_wz = {norm_cmd[2]:+.3f}")
        if abs(norm_cmd[2]) > 3.0:
            print(f"  ⚠ norm_wz={norm_cmd[2]:.2f} > 3.0 — LL OVERREACTING! "
                  f"Reduce wz range. wz_std={wz_std:.4f}")
        elif abs(norm_cmd[2]) > 2.0:
            print(f"  ⚠ norm_wz={norm_cmd[2]:.2f} — approaching LL limits. "
                  f"wz_std={wz_std:.4f}")
        else:
            print(f"  ✓ norm_wz={norm_cmd[2]:.2f} — within LL training range. "
                  f"wz_std={wz_std:.4f}")
        print(f"  ── LL raw network output (before tanh) ──────────────────")
        print(f"     min={raw0.min():.3f}  max={raw0.max():.3f}  "
              f"mean={raw0.mean():.3f}  std={raw0.std():.3f}")
        print(f"  ── LL joint delta (after tanh squash) ───────────────────")
        print(f"     hips : {delta0[:4]}")
        print(f"     thigh: {delta0[4:8]}")
        print(f"     calf : {delta0[8:12]}")
        print(f"  ── Robot state ───────────────────────────────────────────")
        print(f"     height={h0:.3f}m  vx={vx0:+.3f}  vy={vy0:+.3f}  "
              f"yaw_rate={yr0:+.3f} rad/s")
        if abs(yr0) > abs(cmd0[2]) * 2.5 and abs(cmd0[2]) > 0.01:
            print(f"  ⚠ yaw_rate ({yr0:.3f}) >> wz_cmd ({cmd0[2]:.3f}) — "
                  f"overreaction ratio={abs(yr0/cmd0[2]):.1f}×")
        print(f"{'─'*65}\n")

    # =========================================================================
    # _get_observations — 5D HL obs + marker update
    # =========================================================================
    def _get_observations(self) -> dict:
        robot_pos    = self._robot.data.root_pos_w[:, :2]
        goal_vec_w   = self._goal_pos_w - robot_pos
        dist         = torch.norm(goal_vec_w, dim=-1)
        goal_angle_w = torch.atan2(goal_vec_w[:, 1], goal_vec_w[:, 0])
        q   = self._robot.data.root_quat_w
        yaw = torch.atan2(
            2.0*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
            1.0 - 2.0*(q[:,2]**2 + q[:,3]**2))
        heading_err = goal_angle_w - yaw
        cos_yaw  = torch.cos(yaw)
        sin_yaw  = torch.sin(yaw)
        dx_robot =  cos_yaw*goal_vec_w[:,0] + sin_yaw*goal_vec_w[:,1]
        dy_robot = -sin_yaw*goal_vec_w[:,0] + cos_yaw*goal_vec_w[:,1]
        obs = torch.stack([
            dx_robot.clamp(-10., 10.),
            dy_robot.clamp(-10., 10.),
            dist.clamp(0., 10.),
            torch.cos(heading_err),
            torch.sin(heading_err),
        ], dim=-1)
        self._update_markers()
        return {"policy": obs}

    # =========================================================================
    # _get_rewards — navigation + survival
    # =========================================================================
    def _get_rewards(self) -> torch.Tensor:
        robot_pos = self._robot.data.root_pos_w[:, :2]
        curr_dist = torch.norm(self._goal_pos_w - robot_pos, dim=-1)
        height    = self._robot.data.root_pos_w[:, 2]
        gravity   = self._robot.data.projected_gravity_b
        tilt      = torch.sqrt(gravity[:,0]**2 + gravity[:,1]**2)

        dt_hl      = _HL_DECIMATION * self.cfg.sim.dt
        r_progress = ((self._prev_dist - curr_dist) / dt_hl).clamp(-2., 2.)

        goal_vec_w = self._goal_pos_w - robot_pos
        goal_angle = torch.atan2(goal_vec_w[:,1], goal_vec_w[:,0])
        q   = self._robot.data.root_quat_w
        yaw = torch.atan2(
            2.0*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
            1.0 - 2.0*(q[:,2]**2 + q[:,3]**2))
        r_heading  = 0.3 * torch.cos(goal_angle - yaw)

        lin_vel_x = self._robot.data.root_lin_vel_b[:, 0]
        vel_gate  = torch.clamp(lin_vel_x / 0.2, 0.0, 1.0)
        r_alive   = 0.5 * vel_gate * ((height > 0.28) & (tilt < 0.3)).float()

        r_success = 20.0 * (curr_dist < self.cfg.success_radius).float()
        r_fall    = -10.0 * (height < 0.25).float()
        r_time    = torch.full((self.num_envs,), -0.01, device=self.device)

        self._prev_dist = curr_dist.detach().clone()
        return r_progress + r_heading + r_alive + r_success + r_fall + r_time

    # =========================================================================
    # _get_dones
    # =========================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_pos    = self._robot.data.root_pos_w[:, :2]
        curr_dist    = torch.norm(self._goal_pos_w - robot_pos, dim=-1)
        height       = self._robot.data.root_pos_w[:, 2]
        goal_reached = curr_dist < self.cfg.success_radius
        fell         = height < 0.22
        done         = goal_reached | fell
        time_out     = self.episode_length_buf >= self.max_episode_length
        newly_done   = done | time_out
        self._dbg_fall_count    += (fell         & newly_done).sum().item()
        self._dbg_success_count += (goal_reached & newly_done).sum().item()
        self._dbg_timeout_count += (time_out & ~done).sum().item()
        self._dbg_total_eps     += newly_done.sum().item()
        return done, time_out

    # =========================================================================
    # _reset_idx
    # =========================================================================
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        if 0 in env_ids and self._dbg_hl_step > 0:
            self._print_episode_summary(env_id=0)

        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Robot state reset
        joint_pos  = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += (torch.rand_like(joint_pos) - 0.5) * 0.05
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3]   = self.scene.env_origins[env_ids]
        root_state[:, 2]   += 0.35
        root_state[:, 3:7]  = torch.tensor([1.,0.,0.,0.], device=self.device)
        root_state[:, 7:13] = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), None, env_ids)

        # PACE: RL_th fault DR
        fault = torch.rand(n, device=self.device)
        self._rl_fault_level[env_ids] = fault
        tau_f = self._tau_f_nominal.unsqueeze(0).expand(n,-1).clone()
        tau_f[:, 6] = 0.007 + fault*(4.944-0.007)
        if self._tau_fn and self._friction_env_ids_ok:
            try:
                getattr(self._robot, self._tau_fn)(tau_f, env_ids=env_ids)
            except TypeError:
                pass
        d_env = self._d_nominal.unsqueeze(0).expand(n,-1).clone()
        d_env[:, 6] = 0.048 + fault*(3.459-0.048)
        actuator = self._robot.actuators["legs"]
        if hasattr(actuator, "viscous_friction"):
            actuator.viscous_friction[env_ids] = d_env

        # PACE: KP/KD DR
        kp_sc = torch.ones(n,12,device=self.device)
        kp_sc[:,:8] = torch.empty(n,8,device=self.device).uniform_(0.75,1.25)
        kp_sc[:,8:] = torch.empty(n,4,device=self.device).uniform_(0.80,1.20)
        kd_sc = torch.ones(n,12,device=self.device)
        kd_sc[:,:8] = torch.empty(n,8,device=self.device).uniform_(0.75,1.25)
        kd_sc[:,8:] = torch.empty(n,4,device=self.device).uniform_(0.80,1.20)
        if hasattr(actuator, "stiffness"):
            actuator.stiffness[env_ids] = self._kp_nominal * kp_sc
        if hasattr(actuator, "damping"):
            actuator.damping[env_ids]   = self._kd_nominal * kd_sc

        # Reset LL state
        self._ll_prev_actions[env_ids]   = 0.0
        self._current_ll_target[env_ids] = self._robot.data.default_joint_pos[env_ids]
        default_tgt = self._robot.data.default_joint_pos[env_ids].unsqueeze(1)
        self._delay_buf[env_ids] = default_tgt.expand(-1, _MAX_DELAY+1, -1)

        # Reset HL cmd
        if HL_POLICY_ACTIVE:
            self._current_hl_cmd[env_ids] = 0.0
        else:
            self._current_hl_cmd[env_ids] = self._fixed_cmd_vals

        # Sample new goal
        dist_range = self.cfg.goal_dist_max - self.cfg.goal_dist_min
        goal_dist  = (torch.rand(n, device=self.device)*dist_range
                      + self.cfg.goal_dist_min)
        goal_angle = torch.empty(n, device=self.device).uniform_(-np.pi, np.pi)
        spawn_xy   = self.scene.env_origins[env_ids, :2]
        self._goal_pos_w[env_ids,0] = spawn_xy[:,0]+goal_dist*torch.cos(goal_angle)
        self._goal_pos_w[env_ids,1] = spawn_xy[:,1]+goal_dist*torch.sin(goal_angle)
        robot_pos = self._robot.data.root_pos_w[env_ids, :2]
        self._start_pos_w[env_ids]  = robot_pos
        self._prev_dist[env_ids]    = torch.norm(
            self._goal_pos_w[env_ids] - robot_pos, dim=-1)

        self._update_markers()

        if self._dbg_hl_step > 0 and self._dbg_hl_step % 500 == 0:
            self._print_aggregate_debug()

    # =========================================================================
    # _print_episode_summary
    # =========================================================================
    def _print_episode_summary(self, env_id=0):
        e       = env_id
        start   = self._start_pos_w[e].cpu().numpy()
        goal    = self._goal_pos_w[e].cpu().numpy()
        current = self._robot.data.root_pos_w[e,:2].cpu().numpy()
        height  = self._robot.data.root_pos_w[e,2].item()
        ep_len  = self.episode_length_buf[e].item()
        dist_r  = float(np.linalg.norm(goal-current))
        dist_i  = float(np.linalg.norm(goal-start))
        covered = dist_i - dist_r
        angle   = float(np.degrees(np.arctan2(goal[1]-start[1],
                                               goal[0]-start[0])))
        if height < 0.25:
            outcome = "FELL ✗"
        elif dist_r < self.cfg.success_radius:
            outcome = "SUCCESS ✓"
        else:
            outcome = f"TIMEOUT (ep_len={ep_len:.0f})"
        lv = self._robot.data.root_lin_vel_b[e].cpu().numpy()
        av = self._robot.data.root_ang_vel_b[e,2].item()
        print(f"\n[ENV0 EPISODE END]")
        print(f"  Start   : ({start[0]:+.2f}, {start[1]:+.2f}) m  [🟢]")
        print(f"  Goal    : ({goal[0]:+.2f}, {goal[1]:+.2f}) m  "
              f"→ {angle:+.1f}°  |  dist={dist_i:.2f}m  [🔴]")
        print(f"  Final   : ({current[0]:+.2f}, {current[1]:+.2f}) m  "
              f"|  remaining={dist_r:.2f}m  "
              f"|  covered={covered:+.2f}m  "
              f"|  height={height:.3f}m")
        print(f"  Outcome : {outcome}")
        if self._dbg_cmd_buf:
            c    = self._dbg_cmd_buf[-1].numpy()
            mode = "HL=ON" if HL_POLICY_ACTIVE else "HL=OFF-fixed"
            print(f"  HL cmd  : vx={c[0]:.3f}  vy=0.000  "
                  f"wz={c[2]:.3f}  [{mode}]")
        print(f"  Real vel: vx={lv[0]:.3f}  vy={lv[1]:.3f}  "
              f"vz={lv[2]:.3f} m/s  |  yaw={av:.3f} rad/s")

    # =========================================================================
    # _print_aggregate_debug
    # =========================================================================
    def _print_aggregate_debug(self):
        s = self._dbg_hl_step
        print(f"\n{'='*72}")
        print(f"[NAV AGGREGATE]  HL_step={s}  "
              f"HL={'ON' if HL_POLICY_ACTIVE else 'OFF(fixed)'}")
        expected = s * _N_SUBSTEPS
        ok = "✓" if self._dbg_ll_calls == expected else "✗ MISMATCH"
        print(f"  LL calls={self._dbg_ll_calls}  expected={expected}  {ok}")
        tot = self._dbg_total_eps
        if tot > 0:
            print(f"  Episodes: total={tot}  "
                  f"falls={self._dbg_fall_count}({100*self._dbg_fall_count/tot:.1f}%)  "
                  f"success={self._dbg_success_count}({100*self._dbg_success_count/tot:.1f}%)  "
                  f"timeout={self._dbg_timeout_count}({100*self._dbg_timeout_count/tot:.1f}%)")
        h    = self._robot.data.root_pos_w[:, 2]
        lv   = self._robot.data.root_lin_vel_b[:, 0]
        av   = self._robot.data.root_ang_vel_b[:, 2]
        grav = self._robot.data.projected_gravity_b
        tilt = torch.sqrt(grav[:,0]**2 + grav[:,1]**2)
        print(f"  Height  : mean={h.mean():.3f}m  min={h.min():.3f}m  "
              f"fallen: {(h<0.25).sum().item()}/{self.num_envs}")
        print(f"  Real vx : mean={lv.mean():.3f}  min={lv.min():.3f}  "
              f"max={lv.max():.3f} m/s")
        print(f"  Yaw rate: mean={av.mean():.3f}  max={av.abs().max():.3f} rad/s")
        print(f"  Tilt    : mean={tilt.mean():.3f}  "
              f"upright: {(tilt<0.3).sum().item()}/{self.num_envs}")
        if self._dbg_cmd_buf:
            cmds = torch.stack(self._dbg_cmd_buf[-500:])
            m, sd = cmds.mean(0), cmds.std(0)
            print(f"  HL cmd  : vx={m[0]:.3f}±{sd[0]:.3f}  vy=0.000  "
                  f"wz={m[2]:.3f}±{sd[2]:.3f}  "
                  f"→ {'forward ✓' if m[0]>0.2 else 'STALLED ✗'}")
        if self._dbg_vx_buf:
            real_vx = float(np.mean(self._dbg_vx_buf[-500:]))
            cmd_vx  = self._dbg_cmd_buf[-1][0].item() if self._dbg_cmd_buf else 0.0
            diff    = abs(real_vx - cmd_vx)
            print(f"  vx track: cmd={cmd_vx:.3f}  real={real_vx:.3f}  "
                  f"err={diff:.3f}  "
                  f"{'TRACKING ✓' if diff<0.25 else 'NOT TRACKING ✗'}")
        rp = self._robot.data.root_pos_w[:, :2]
        d  = torch.norm(self._goal_pos_w - rp, dim=-1)
        print(f"  Goal    : dist mean={d.mean():.2f}m  "
              f"success: {(d<self.cfg.success_radius).sum().item()}/{self.num_envs}")
        s0 = self._start_pos_w[0].cpu().numpy()
        g0 = self._goal_pos_w[0].cpu().numpy()
        c0 = self._robot.data.root_pos_w[0,:2].cpu().numpy()
        print(f"  [Env0]  🟢({s0[0]:+.2f},{s0[1]:+.2f}) → "
              f"🔴({g0[0]:+.2f},{g0[1]:+.2f})  "
              f"now=({c0[0]:+.2f},{c0[1]:+.2f})  "
              f"dist={np.linalg.norm(g0-c0):.2f}m")
        print(f"{'='*72}\n")