#!/usr/bin/env python3
"""
Export Go1 PPO policy + obs normalizer for real robot deployment.
Phase 3b — Per-Episode Delay DR + Full PACE calibration.

CRITICAL FIX from previous version:
  OLD: DELTA_LO hip = -0.15, DELTA_HI hip = +0.25  (asymmetric, old go1_env.py)
       → _mid_hip = +0.05 → every hip joint offset +0.05 rad on real robot (legs splayed)
  NEW: DELTA_LO hip = -0.20, DELTA_HI hip = +0.20  (symmetric, current go1_env.py)
       → _mid_hip = 0.00 → no standing offset, correct sim match
  RL_hip: lo=-0.25, hi=+0.25 (wider, matches RL_thigh mechanical fault compensation)

Training setup (go1_env.py Phase 3b):
  - 45D obs (no foot contacts)
  - MLP: 45 → 512 → 256 → 128 → 12   activation: ELU
  - EmpiricalNormalization inside policy
  - checkpoint: {"policy_state_dict": ..., "normalizer_state": ...}

Action formulation — must match go1_env.py _pre_physics_step exactly:
  raw_net  = MLP(normalise(obs))
  _mid     = (delta_hi + delta_lo) / 2
  _half    = (delta_hi - delta_lo) / 2
  delta    = _mid + _half * tanh(raw_net)
  target_q = DEFAULT_JOINT_POS + delta        (absolute joint pos, Isaac order)

  policy.pt bakes: normalise + MLP + tanh_squash → output is delta directly.
  SDK: target_q_isaac = policy(obs) + DEFAULT_JOINT_POS  (after hw sign flip)

Joint order (Isaac Lab type-grouped):
  [0]=FL_hip  [1]=FR_hip  [2]=RL_hip  [3]=RR_hip
  [4]=FL_th   [5]=FR_th   [6]=RL_th   [7]=RR_th
  [8]=FL_kn   [9]=FR_kn  [10]=RL_kn  [11]=RR_kn

Go1 SDK motor order (actual hardware — FR FL RR RL per-leg):
  SDK[0]=FR_hip  [1]=FR_th  [2]=FR_kn
  SDK[3]=FL_hip  [4]=FL_th  [5]=FL_kn
  SDK[6]=RR_hip  [7]=RR_th  [8]=RR_kn
  SDK[9]=RL_hip  [10]=RL_th [11]=RL_kn
  sdk_to_isaac = [3,0,9,6, 4,1,10,7, 5,2,11,8]  (verified on hardware)
"""

import os
import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE: point to the checkpoint you want to deploy
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT = (
    "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-05-15_20-33-22/model_35000.pt"
)
OUTPUT_DIR = os.path.join(os.path.dirname(CHECKPOINT), "go1_deploy")

# ─────────────────────────────────────────────────────────────────────────────
# Architecture — must match rsl_rl_ppo_cfg.py exactly
# ─────────────────────────────────────────────────────────────────────────────
OBS_DIM     = 46
ACTION_DIM  = 12
HIDDEN_DIMS = [512, 256, 128]   # actor_hidden_dims

# ─────────────────────────────────────────────────────────────────────────────
# tanh_squash parameters — MUST match go1_env.py _delta_soft_lo / _delta_soft_hi
#
# Phase 3b (current go1_env.py):
#   _delta_soft_lo = [-0.20,-0.20,-0.25,-0.20, -0.35×8]
#   _delta_soft_hi = [+0.20,+0.20,+0.25,+0.20, +0.35×8]
#
# Result:
#   _mid  = (hi+lo)/2 = [0.00, 0.00, 0.00, 0.00, 0.00×8]  ← all zero, symmetric
#   _half = (hi-lo)/2 = [0.20, 0.20, 0.25, 0.20, 0.35×8]  ← RL_hip wider
#
# Isaac order: [FL_hip, FR_hip, RL_hip, RR_hip, FL_th, FR_th, RL_th, RR_th,
#               FL_kn,  FR_kn,  RL_kn,  RR_kn]
# ─────────────────────────────────────────────────────────────────────────────
# DELTA_LO = np.array([
#     -0.20, -0.20, -0.25, -0.20,   # FL FR RL RR hip  (RL wider for fault compensation)
#     -0.35, -0.35, -0.35, -0.35,   # thighs
#     -0.35, -0.35, -0.35, -0.35,   # calves
# ], dtype=np.float32)
#
# DELTA_HI = np.array([
#      0.20,  0.20,  0.25,  0.20,   # FL FR RL RR hip
#      0.35,  0.35,  0.35,  0.35,   # thighs
#      0.35,  0.35,  0.35,  0.35,   # calves
# ], dtype=np.float32)
DELTA_LO = np.array(
            [-0.08, -0.08, -0.08, -0.08,   # hips: ±0.20 → ±0.08
             -0.35, -0.35, -0.35, -0.35,   # thighs unchanged
             -0.35, -0.35, -0.35, -0.35,   # calves unchanged
            ], dtype=np.float32)

DELTA_HI = np.array(
            [ 0.08,  0.08,  0.08,  0.08,   # hips: ±0.20 → ±0.08
              0.35,  0.35,  0.35,  0.35,   # thighs unchanged
              0.35,  0.35,  0.35,  0.35,   # calves unchanged
            ], dtype=np.float32)

_MID  = torch.tensor((DELTA_HI + DELTA_LO) / 2.0)   # all zeros — symmetric
_HALF = torch.tensor((DELTA_HI - DELTA_LO) / 2.0)   # [0.20,0.20,0.25,0.20, 0.35×8]

# Verify symmetry
assert np.allclose(_MID.numpy(), 0.0, atol=1e-6), \
    f"_mid should be all-zero (symmetric limits). Got: {_MID.numpy()}"
print(f"[CHECK] _mid  = {_MID.numpy()}  (all zero ✓)")
print(f"[CHECK] _half = {_HALF.numpy()}")

# ─────────────────────────────────────────────────────────────────────────────
# Default joint positions — Isaac order (from go1_env_cfg.py init_state)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,   # FL FR RL RR hip
    0.8,  0.8,  0.8,  0.8,   # FL FR RL RR thigh
   -1.5, -1.5, -1.5, -1.5,   # FL FR RL RR knee
], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# SDK joint index mapping (verified on real Go1 hardware)
# Go1 SDK actual motor order: FR FL RR RL (NOT FL FR RL RR as comment may say)
#   SDK[0]=FR_hip, [1]=FR_th, [2]=FR_kn
#   SDK[3]=FL_hip, [4]=FL_th, [5]=FL_kn
#   SDK[6]=RR_hip, [7]=RR_th, [8]=RR_kn
#   SDK[9]=RL_hip, [10]=RL_th,[11]=RL_kn
# sdk_to_isaac[isaac_idx] = sdk_idx that holds that joint
# ─────────────────────────────────────────────────────────────────────────────
SDK_TO_ISAAC = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]


# ─────────────────────────────────────────────────────────────────────────────
# Neural network modules
# ─────────────────────────────────────────────────────────────────────────────
class Go1Normalizer(nn.Module):
    """(obs - mean) / std  from EmpiricalNormalization state."""
    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std",  torch.sqrt(var.float() + 1e-8))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.std


class Go1Actor(nn.Module):
    """MLP: normalised_obs(45) → raw_net(12).  No tanh here."""
    def __init__(self):
        super().__init__()
        dims   = [OBS_DIM] + HIDDEN_DIMS + [ACTION_DIM]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Go1Policy(nn.Module):
    """
    Full policy: raw_obs(45) → normalise → MLP → tanh_squash → delta(12)

    Output is joint delta in Isaac order, tanh-squashed to [DELTA_LO, DELTA_HI].
    Guaranteed within bounds — no additional clamping needed.

    SDK usage:
        delta_isaac = policy(obs_45d)              # this forward
        delta_isaac[1] = -delta_isaac[1]           # FR_hip hw sign flip
        delta_isaac[3] = -delta_isaac[3]           # RR_hip hw sign flip
        target_q_isaac = DEFAULT_JOINT_POS + delta_isaac
        target_q_sdk   = target_q_isaac[isaac_to_sdk]
        robot.setJointPos(target_q_sdk, kp, kd)
    """
    def __init__(self, normalizer: Go1Normalizer, actor: Go1Actor,
                 mid: torch.Tensor, half: torch.Tensor):
        super().__init__()
        self.normalizer   = normalizer
        self.actor        = actor
        self.register_buffer("delta_mid",  mid.float())
        self.register_buffer("delta_half", half.float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.normalizer(obs)
        raw_net  = self.actor(norm_obs)
        return self.delta_mid + self.delta_half * torch.tanh(raw_net)


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_actor_weights(policy_sd: dict) -> dict:
    """Map RSL-RL 'actor.*' or 'actor_body.*' → Go1Actor 'net.*' keys."""
    # Try 'actor.' prefix first (standard RSL-RL)
    out = {"net." + k[len("actor."):]: v
           for k, v in policy_sd.items() if k.startswith("actor.")}
    if not out:
        out = {"net." + k[len("actor_body."):]: v
               for k, v in policy_sd.items() if k.startswith("actor_body.")}
    print(f"  Actor tensors: {len(out)}  keys[:4]: {list(out.keys())[:4]}")
    return out


def extract_normalizer_params(norm_sd: dict):
    """Return (mean, var) from EmpiricalNormalization state_dict."""
    mean_key = next(k for k in norm_sd if "mean" in k.lower())
    var_key  = next(k for k in norm_sd if "var"  in k.lower())
    mean = norm_sd[mean_key].squeeze()
    var  = norm_sd[var_key].squeeze()
    print(f"  mean key '{mean_key}'  shape {mean.shape}")
    print(f"  var  key '{var_key}'   shape {var.shape}")
    return mean, var


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*65}")
    print("Go1 Policy Export — Phase 3b")
    print(f"{'='*65}")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  delta_lo   : {DELTA_LO.tolist()}")
    print(f"  delta_hi   : {DELTA_HI.tolist()}")
    print(f"  _mid       : {_MID.numpy().tolist()}")
    print(f"  _half      : {_HALF.numpy().tolist()}")

    # 1. Load checkpoint
    print(f"\n[1/5] Loading checkpoint ...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    print(f"  Top-level keys: {list(ckpt.keys())}")
    assert "policy_state_dict" in ckpt, \
        "Missing 'policy_state_dict' — check custom_save in your training script"
    assert "normalizer_state"  in ckpt, \
        "Missing 'normalizer_state' — check custom_save in your training script"

    # 2. Build actor
    print(f"\n[2/5] Building actor ...")
    actor_sd = extract_actor_weights(ckpt["policy_state_dict"])
    actor    = Go1Actor()
    missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
    assert not missing,    f"Missing actor keys: {missing}"
    assert not unexpected, f"Unexpected actor keys: {unexpected}"
    actor.eval()
    print("  ✓ Actor loaded")

    # 3. Build normalizer
    print(f"\n[3/5] Building normalizer ...")
    mean, var = extract_normalizer_params(ckpt["normalizer_state"])
    std = torch.sqrt(var + 1e-8)
    normalizer = Go1Normalizer(mean, var)
    normalizer.eval()
    print(f"  mean[:5] : {mean[:5].numpy().round(5)}")
    print(f"  std[:5]  : {std[:5].numpy().round(5)}")
    assert mean.shape == (OBS_DIM,), f"mean shape wrong: {mean.shape}"

    # 4. Verify forward pass
    print(f"\n[4/5] Verifying full forward pass ...")
    policy = Go1Policy(normalizer, actor, _MID, _HALF)
    policy.eval()

    # Canonical test: upright robot at rest, cmd_vx=0.5
    dummy       = torch.zeros(1, OBS_DIM)
    dummy[0, 0] = 0.5    # cmd_vx
    dummy[0,32] = -1.0   # proj_gravity_z (obs index 30+2=32)

    with torch.no_grad():
        delta = policy(dummy)

    print(f"  Input shape  : {dummy.shape}")
    print(f"  Delta shape  : {delta.shape}")
    print(f"  Delta (Isaac): {delta[0].numpy().round(4)}")

    target_q = delta[0].numpy() + DEFAULT_JOINT_POS
    print(f"  Target_q     : {target_q.round(4)}")
    print(f"  Target_q range: [{target_q.min():.3f}, {target_q.max():.3f}]")

    # Verify tanh guarantee: delta must be within [DELTA_LO, DELTA_HI]
    delta_np = delta[0].numpy()
    assert (delta_np >= DELTA_LO - 1e-4).all(), \
        f"delta below lo! {delta_np[delta_np < DELTA_LO - 1e-4]}"
    assert (delta_np <= DELTA_HI + 1e-4).all(), \
        f"delta above hi! {delta_np[delta_np > DELTA_HI + 1e-4]}"
    print("  ✓ delta within [DELTA_LO, DELTA_HI] — tanh_squash correct")

    # Verify hip _mid is zero (symmetric limits check)
    # At rest canonical obs, hips should output near _mid = 0.0
    hip_delta = delta_np[:4]
    print(f"  Hip delta at rest: {hip_delta.round(4)}")
    print(f"  (expected near 0.0 — if +0.05 offset, DELTA_LO/HI mismatch with training)")

    # 5. Export
    print(f"\n[5/5] Exporting ...")

    # policy.pt — TorchScript: raw_obs(45) → delta(12)
    scripted = torch.jit.script(policy)
    pt_path  = os.path.join(OUTPUT_DIR, "policy.pt")
    scripted.save(pt_path)
    print(f"  ✓ policy.pt      → raw_obs(45) → delta(12)  [normalise+MLP+tanh baked in]")

    # normalizer.npz — for compare_sim_real.py
    np.savez(os.path.join(OUTPUT_DIR, "normalizer.npz"),
             mean=mean.numpy().astype(np.float32),
             var=var.numpy().astype(np.float32),
             std=std.numpy().astype(np.float32))
    print(f"  ✓ normalizer.npz → mean/var/std  shape=({OBS_DIM},)")

    # deploy_info.txt
    isaac_to_sdk = [0] * 12
    for sdk_i, isa_i in enumerate(SDK_TO_ISAAC):
        isaac_to_sdk[isa_i] = sdk_i

    jnames = ["FL_hip","FR_hip","RL_hip","RR_hip",
              "FL_thigh","FR_thigh","RL_thigh","RR_thigh",
              "FL_knee","FR_knee","RL_knee","RR_knee"]

    info_path = os.path.join(OUTPUT_DIR, "deploy_info.txt")
    with open(info_path, "w") as f:
        f.write("Go1 Deployment Constants — Phase 3b\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"checkpoint  : {CHECKPOINT}\n")
        f.write(f"obs_dim     : {OBS_DIM}\n")
        f.write(f"action_dim  : {ACTION_DIM}\n\n")

        f.write("── Observation layout (45D) ─────────────────────────────────\n")
        f.write("  [0:3]   cmd [vx, vy, wz]             joystick / fixed cmd\n")
        f.write("  [3:15]  jpos - default_q              Isaac order, hardware sign-flipped\n")
        f.write("  [15:27] jvel  clipped ±5              Isaac order\n")
        f.write("  [27:30] gyro  clipped ±5 [x,y,z]     IMU\n")
        f.write("  [30:33] proj_gravity [x,y,z]          -acc/|acc| from IMU\n")
        f.write("  [33:45] prev_delta                    policy output at t-1 (pre hw-flip)\n\n")

        f.write("── Isaac joint order ─────────────────────────────────────────\n")
        for i, n in enumerate(jnames):
            f.write(f"  [{i:2d}] {n}\n")
        f.write("\n")

        f.write("── SDK motor order (actual Go1 hardware) ─────────────────────\n")
        f.write("  SDK[0]=FR_hip  [1]=FR_th  [2]=FR_kn\n")
        f.write("  SDK[3]=FL_hip  [4]=FL_th  [5]=FL_kn\n")
        f.write("  SDK[6]=RR_hip  [7]=RR_th  [8]=RR_kn\n")
        f.write("  SDK[9]=RL_hip  [10]=RL_th [11]=RL_kn\n")
        f.write(f"  sdk_to_isaac = {SDK_TO_ISAAC}\n")
        f.write(f"  isaac_to_sdk = {isaac_to_sdk}\n\n")

        f.write("── Hip sign convention ───────────────────────────────────────\n")
        f.write("  Hardware FR_hip+: inward (adduction)\n")
        f.write("  Isaac    FR_hip+: outward (abduction)\n")
        f.write("  obs sign flip (hw → Isaac): obs[4] = -obs[4]  (FR_hip)\n")
        f.write("                              obs[6] = -obs[6]  (RR_hip)\n")
        f.write("  delta sign flip (Isaac → hw): delta[1] = -delta[1]  (FR_hip)\n")
        f.write("                                delta[3] = -delta[3]  (RR_hip)\n")
        f.write("  prev_delta stored PRE-hw-flip (same frame as training obs)\n\n")

        f.write("── tanh_squash bounds (Phase 3b — symmetric hips) ────────────\n")
        f.write(f"  delta_lo = {DELTA_LO.tolist()}\n")
        f.write(f"  delta_hi = {DELTA_HI.tolist()}\n")
        f.write(f"  _mid     = {_MID.numpy().tolist()}  (all zero — symmetric)\n")
        f.write(f"  _half    = {_HALF.numpy().tolist()}\n\n")

        f.write("── Default joint positions (Isaac order) ─────────────────────\n")
        f.write(f"  {DEFAULT_JOINT_POS.tolist()}\n\n")

        f.write("── Training KP/KD (must match on hardware) ───────────────────\n")
        f.write("  KP: hip=35  thigh=65  knee=80  (Nm/rad)\n")
        f.write("  KD: hip=4.0 thigh=4.5 knee=5.0 (Nm·s/rad)\n\n")

        f.write("── SDK deployment pseudocode ──────────────────────────────────\n")
        f.write("  # 1. Read hardware, convert to Isaac order\n")
        f.write("  jpos_isaac = encoder_q[sdk_to_isaac]   # SDK → Isaac reorder\n")
        f.write("  jvel_isaac = encoder_dq[sdk_to_isaac]\n\n")
        f.write("  # 2. Build 45D obs\n")
        f.write("  obs[0:3]   = [cmd_vx, 0, 0]\n")
        f.write("  obs[3:15]  = (jpos_isaac - default_q) - jdelta_offset  # zero at real equilibrium\n")
        f.write("  obs[4]     = -obs[4]   # FR_hip hw→Isaac\n")
        f.write("  obs[6]     = -obs[6]   # RR_hip hw→Isaac\n")
        f.write("  obs[15:27] = clip(jvel_isaac, ±5)\n")
        f.write("  obs[27:30] = clip(gyro_xyz, ±5)\n")
        f.write("  obs[30:33] = -acc / |acc|   # proj_gravity\n")
        f.write("  obs[33:45] = prev_delta      # policy output t-1, PRE hw-flip\n\n")
        f.write("  # 3. Policy inference\n")
        f.write("  delta_isaac = policy(obs)    # normalise + MLP + tanh baked in\n\n")
        f.write("  # 4. Store prev_delta BEFORE hw sign flip\n")
        f.write("  prev_delta[:] = delta_isaac.copy()\n\n")
        f.write("  # 5. Hardware sign flip\n")
        f.write("  delta_hw      = delta_isaac.copy()\n")
        f.write("  delta_hw[1]   = -delta_hw[1]   # FR_hip Isaac→hw\n")
        f.write("  delta_hw[3]   = -delta_hw[3]   # RR_hip Isaac→hw\n\n")
        f.write("  # 6. Compute target, reorder, send\n")
        f.write("  target_q_isaac = default_q + delta_hw\n")
        f.write("  target_q_sdk   = target_q_isaac[isaac_to_sdk]\n")
        f.write("  robot.setJointPos(target_q_sdk, kp, kd)\n")

    print(f"  ✓ deploy_info.txt")

    # Summary
    print(f"\n{'='*65}")
    print("✓ EXPORT COMPLETE")
    print(f"{'='*65}")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fp   = os.path.join(OUTPUT_DIR, fn)
        size = os.path.getsize(fp) / 1024
        print(f"  {fn:30s} {size:8.1f} KB")

    print(f"""
policy.pt interface:
  Input:  45D obs (raw, unnormalised — build exactly as deploy_info.txt)
  Output: 12D delta Isaac order — tanh-squashed, bounded in [DELTA_LO, DELTA_HI]
  Usage:  target_q_isaac = policy(obs) + DEFAULT_JOINT_POS
          (after hw sign flip on delta[1] and delta[3])

Phase 3b specific:
  _mid = 0.0 for all joints (symmetric limits) — hip targets centred on default_q
  Previous version had _mid=0.05 for hips → legs splayed +0.05 rad at rest
  This version: hip delta at rest ≈ 0.0 → natural standing pose
""")


if __name__ == "__main__":
    main()