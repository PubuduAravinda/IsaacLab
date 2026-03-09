#!/usr/bin/env python3
"""
Export Go1 PPO policy + obs normalizer for real robot deployment.

Training setup:
  - 45D observation (no foot contacts)
  - MLP: 45 → 512 → 256 → 128 → 12
  - actor_obs_normalizer (EmpiricalNormalization) inside policy
  - checkpoint saved by custom_save:
      {"policy_state_dict": ..., "normalizer_state": ...}

CRITICAL — action formulation (must match go1_env.py exactly):
  raw_net  = MLP(normalise(obs))           # network output, no tanh
  delta    = _mid + _half * tanh(raw_net)  # tanh_squash in go1_env.py
  target_q = DEFAULT_JOINT_POS + delta     # absolute joint position

  hip:   _mid= 0.05  _half= 0.20   (lo=-0.15  hi=+0.25)
  thigh: _mid= 0.00  _half= 0.35   (lo=-0.35  hi=+0.35)
  knee:  _mid= 0.00  _half= 0.35   (lo=-0.35  hi=+0.35)

  This policy.pt bakes in normalise + MLP + tanh_squash.
  Output is already delta — just add DEFAULT_JOINT_POS in SDK.

JOINT ORDER — Isaac Lab order (NOT SDK per-leg order):
  [0] FL_hip   [1] FR_hip   [2] RL_hip   [3] RR_hip
  [4] FL_thigh [5] FR_thigh [6] RL_thigh [7] RR_thigh
  [8] FL_knee  [9] FR_knee  [10]RL_knee  [11]RR_knee

  SDK Go1 order is per-leg: FL(hip,th,kn) FR(hip,th,kn) RL(hip,th,kn) RR(hip,th,kn)
  sdk_to_isaac = [3,0,9,6, 4,1,10,7, 5,2,11,8]  (same as go1_deploy_final.py)

Output files:
  policy.pt       — TorchScript: raw_obs(45) → delta(12), ready for SDK
  normalizer.npz  — mean/var/std as numpy (for debugging)
  deploy_info.txt — all constants needed in SDK
"""

import os
import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS to your checkpoint path
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT = (
    "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/"
    "logs/rsl_rl/go1_himloco/2026-03-07_18-39-24/model_4900.pt"
)
OUTPUT_DIR = os.path.join(os.path.dirname(CHECKPOINT), "go1_deploy")

# ─────────────────────────────────────────────────────────────────────────────
# Architecture — must match training config exactly
# ─────────────────────────────────────────────────────────────────────────────
OBS_DIM     = 45
ACTION_DIM  = 12
HIDDEN_DIMS = [512, 256, 128]

# ─────────────────────────────────────────────────────────────────────────────
# tanh_squash parameters — from go1_env.py delta_soft limits
# Verified from sim_log delta_lo/delta_hi keys.
# Isaac order: [FL_hip FR_hip RL_hip RR_hip  FL_th FR_th RL_th RR_th  FL_kn FR_kn RL_kn RR_kn]
# ─────────────────────────────────────────────────────────────────────────────
DELTA_LO = np.array([-0.15, -0.15, -0.15, -0.15,
                     -0.35, -0.35, -0.35, -0.35,
                     -0.35, -0.35, -0.35, -0.35], dtype=np.float32)
DELTA_HI = np.array([ 0.25,  0.25,  0.25,  0.25,
                       0.35,  0.35,  0.35,  0.35,
                       0.35,  0.35,  0.35,  0.35], dtype=np.float32)
_HALF = torch.tensor((DELTA_HI - DELTA_LO) / 2.0)  # [0.20×4, 0.35×8]
_MID  = torch.tensor((DELTA_HI + DELTA_LO) / 2.0)  # [0.05×4, 0.00×8]

# ─────────────────────────────────────────────────────────────────────────────
# Default joint positions — Isaac order (verified from sim_log default_q)
# [h,h,h,h,  th,th,th,th,  kn,kn,kn,kn]
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,   # FL FR RL RR hip
    0.8,  0.8,  0.8,  0.8,   # FL FR RL RR thigh
   -1.5, -1.5, -1.5, -1.5,   # FL FR RL RR knee
], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# SDK index mapping — for deploy_info.txt reference
# ─────────────────────────────────────────────────────────────────────────────
# Go1 SDK order: FL(hip,th,kn), FR(hip,th,kn), RL(hip,th,kn), RR(hip,th,kn)
# sdk_to_isaac[sdk_idx] = isaac_idx
SDK_TO_ISAAC = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class Go1Normalizer(nn.Module):
    """(obs - mean) / std  using trained EmpiricalNormalization params."""
    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std",  torch.sqrt(var.float() + 1e-8))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.std


class Go1Actor(nn.Module):
    """Raw MLP: normalised_obs(45) → raw_net(12). No tanh here."""
    def __init__(self):
        super().__init__()
        dims = [OBS_DIM] + HIDDEN_DIMS + [ACTION_DIM]
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

    Output is joint DELTA in Isaac order.
    SDK usage:
        delta     = policy(obs_45d)             # this forward pass
        target_q  = delta + DEFAULT_JOINT_POS   # in Isaac order
        target_sdk = target_q[isaac_to_sdk]     # reorder for Go1 SDK
        robot.setJointPos(target_sdk)
    """
    def __init__(self, normalizer: Go1Normalizer, actor: Go1Actor,
                 mid: torch.Tensor, half: torch.Tensor):
        super().__init__()
        self.normalizer = normalizer
        self.actor      = actor
        # Note: 'half' and 'float' are reserved nn.Module method names — use prefix
        self.register_buffer("delta_mid",  mid.float())
        self.register_buffer("delta_half", half.float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.normalizer(obs)
        raw_net  = self.actor(norm_obs)
        delta    = self.delta_mid + self.delta_half * torch.tanh(raw_net)
        return delta


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_actor_weights(policy_sd: dict) -> dict:
    """Map RSL-RL 'actor.N.*' keys → Go1Actor 'net.N.*' keys."""
    out = {("net." + k[len("actor."):]): v
           for k, v in policy_sd.items() if k.startswith("actor.")}
    if not out:  # fallback for actor_body naming
        out = {("net." + k[len("actor_body."):]): v
               for k, v in policy_sd.items() if k.startswith("actor_body.")}
    print(f"  Actor tensors extracted: {len(out)}")
    print(f"  Keys: {list(out.keys())[:6]} ...")
    return out


def extract_normalizer_params(norm_sd: dict):
    """Return (mean, var) tensors from EmpiricalNormalization state_dict."""
    mean_key = next(k for k in norm_sd if "mean" in k.lower())
    var_key  = next(k for k in norm_sd if "var"  in k.lower())
    mean = norm_sd[mean_key].squeeze()   # (1,45) → (45,)
    var  = norm_sd[var_key].squeeze()
    print(f"  mean key: '{mean_key}'  shape: {mean.shape}")
    print(f"  var  key: '{var_key}'   shape: {var.shape}")
    return mean, var


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*65}")
    print("Go1 Policy Export")
    print(f"{'='*65}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    print(f"\n[1/5] Loading checkpoint ...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    print(f"  Top-level keys: {list(ckpt.keys())}")
    assert "policy_state_dict" in ckpt,  "Missing 'policy_state_dict' — check custom_save"
    assert "normalizer_state"  in ckpt,  "Missing 'normalizer_state' — check custom_save"

    # ── 2. Build actor ────────────────────────────────────────────────────────
    print(f"\n[2/5] Building actor ...")
    actor_sd = extract_actor_weights(ckpt["policy_state_dict"])
    actor = Go1Actor()
    missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
    assert not missing,    f"Missing actor keys: {missing}"
    assert not unexpected, f"Unexpected actor keys: {unexpected}"
    actor.eval()
    print("  ✓ Actor loaded and verified")

    # ── 3. Build normalizer ───────────────────────────────────────────────────
    print(f"\n[3/5] Building normalizer ...")
    mean, var = extract_normalizer_params(ckpt["normalizer_state"])
    std = torch.sqrt(var + 1e-8)
    normalizer = Go1Normalizer(mean, var)
    normalizer.eval()
    print(f"  mean[:5]: {mean[:5].numpy().round(5)}")
    print(f"  std[:5]:  {std[:5].numpy().round(5)}")
    assert mean.shape == (OBS_DIM,), f"mean shape wrong: {mean.shape}"

    # ── 4. Verify forward pass ────────────────────────────────────────────────
    print(f"\n[4/5] Verifying full forward pass ...")
    policy = Go1Policy(normalizer, actor, _MID, _HALF)
    policy.eval()

    # Canonical test obs: upright robot, zero everything except proj_grav_z=-1
    dummy = torch.zeros(1, OBS_DIM)
    dummy[0, 0]  = 0.5    # cmd_vx centre of training range
    dummy[0, 32] = -1.0   # proj_gravity z (channel 30+2=32)

    with torch.no_grad():
        delta = policy(dummy)

    print(f"  Input obs shape:  {dummy.shape}")
    print(f"  Delta out shape:  {delta.shape}")
    print(f"  Delta (Isaac):    {delta[0].numpy().round(4)}")

    target_q = delta[0].numpy() + DEFAULT_JOINT_POS
    print(f"  Target_q (Isaac): {target_q.round(4)}")
    print(f"  Target_q range:   [{target_q.min():.3f}, {target_q.max():.3f}]")

    # Sanity: delta should be within [DELTA_LO, DELTA_HI] always (tanh guarantee)
    assert (delta[0].numpy() >= DELTA_LO - 1e-4).all(), "delta below lo limit!"
    assert (delta[0].numpy() <= DELTA_HI + 1e-4).all(), "delta above hi limit!"
    print("  ✓ delta within [DELTA_LO, DELTA_HI] — tanh_squash correct")

    # ── 5. Export ─────────────────────────────────────────────────────────────
    print(f"\n[5/5] Exporting ...")

    # policy.pt — TorchScript: raw_obs → delta
    scripted = torch.jit.script(policy)
    pt_path  = os.path.join(OUTPUT_DIR, "policy.pt")
    scripted.save(pt_path)
    print(f"  ✓ policy.pt      — raw_obs(45) → delta(12)  [normalise+MLP+tanh_squash]")

    # normalizer.npz — for debugging / numpy-only environments
    np.savez(os.path.join(OUTPUT_DIR, "normalizer.npz"),
             mean=mean.numpy().astype(np.float32),
             var=var.numpy().astype(np.float32),
             std=std.numpy().astype(np.float32))
    print(f"  ✓ normalizer.npz — mean/var/std  shape=({OBS_DIM},)")

    # deploy_info.txt — complete constants for SDK
    isaac_to_sdk = [0] * 12
    for sdk_i, isa_i in enumerate(SDK_TO_ISAAC):
        isaac_to_sdk[isa_i] = sdk_i

    info = os.path.join(OUTPUT_DIR, "deploy_info.txt")
    with open(info, "w") as f:
        f.write("Go1 Deployment Constants\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"checkpoint : {CHECKPOINT}\n")
        f.write(f"obs_dim    : {OBS_DIM}\n")
        f.write(f"action_dim : {ACTION_DIM}\n\n")

        f.write("── Observation layout (45D, policy input) ──────────────────\n")
        f.write("  [0:3]   cmd [vx, vy, wz]       — joystick / fixed\n")
        f.write("  [3:15]  jpos_delta              — (joint_pos - default_q), Isaac order\n")
        f.write("  [15:27] jvel                    — joint velocities,         Isaac order\n")
        f.write("  [27:30] ang_vel                 — IMU gyro [x,y,z]\n")
        f.write("  [30:33] proj_grav               — gravity unit vec [x,y,z], IMU\n")
        f.write("  [33:45] prev_actions            — tanh delta from t-1,      Isaac order\n\n")

        f.write("── Joint ORDER: Isaac Lab (NOT Go1 SDK per-leg order) ───────\n")
        f.write("  Isaac index → joint name\n")
        jnames = ["FL_hip","FR_hip","RL_hip","RR_hip",
                  "FL_thigh","FR_thigh","RL_thigh","RR_thigh",
                  "FL_knee","FR_knee","RL_knee","RR_knee"]
        for i, n in enumerate(jnames):
            f.write(f"    [{i:2d}] {n}\n")
        f.write("\n")

        f.write("── SDK index mapping ─────────────────────────────────────────\n")
        f.write("  Go1 SDK order: FL(hip,th,kn) FR(hip,th,kn) RL(hip,th,kn) RR(hip,th,kn)\n")
        f.write(f"  sdk_to_isaac = {SDK_TO_ISAAC}\n")
        f.write(f"  isaac_to_sdk = {isaac_to_sdk}\n\n")

        f.write("── Default joint positions (Isaac order) ─────────────────────\n")
        f.write(f"  {DEFAULT_JOINT_POS.tolist()}\n\n")

        f.write("── tanh_squash parameters (Isaac order) ──────────────────────\n")
        f.write(f"  delta_lo = {DELTA_LO.tolist()}\n")
        f.write(f"  delta_hi = {DELTA_HI.tolist()}\n")
        f.write(f"  _mid     = {_MID.numpy().tolist()}\n")
        f.write(f"  _half    = {_HALF.numpy().tolist()}\n\n")

        f.write("── SDK deployment pseudocode ──────────────────────────────────\n")
        f.write("  # 1. Build obs in Isaac order (apply sdk_to_isaac to encoder readings)\n")
        f.write("  jpos_isaac = encoder_q[sdk_to_isaac]\n")
        f.write("  jvel_isaac = encoder_dq[sdk_to_isaac]\n")
        f.write("  obs[0:3]   = [cmd_vx, cmd_vy, cmd_wz]\n")
        f.write("  obs[3:15]  = jpos_isaac - default_q          # jpos_delta\n")
        f.write("  obs[3]     = -obs[3]  # FR_hip sign flip: hardware→Isaac\n")
        f.write("  obs[5]     = -obs[5]  # RR_hip sign flip: hardware→Isaac\n")
        f.write("  obs[15:27] = jvel_isaac\n")
        f.write("  obs[27:30] = gyro_xyz\n")
        f.write("  obs[30:33] = proj_gravity_xyz\n")
        f.write("  obs[33:45] = prev_delta_isaac                 # tanh output t-1\n\n")
        f.write("  # 2. Run policy\n")
        f.write("  delta_isaac = policy(obs)                     # TorchScript forward\n\n")
        f.write("  # 3. Store prev_actions BEFORE hardware sign flip\n")
        f.write("  prev_delta_isaac[:] = delta_isaac.copy()\n\n")
        f.write("  # 4. Hardware sign flip for FR/RR hips\n")
        f.write("  delta_hw = delta_isaac.copy()\n")
        f.write("  delta_hw[1] = -delta_hw[1]  # FR_hip: Isaac → hardware\n")
        f.write("  delta_hw[3] = -delta_hw[3]  # RR_hip: Isaac → hardware\n\n")
        f.write("  # 5. Compute target, reorder to SDK, send\n")
        f.write("  target_q_isaac = default_q + delta_hw\n")
        f.write("  target_q_sdk   = target_q_isaac[isaac_to_sdk]\n")
        f.write("  robot.setJointPos(target_q_sdk, kp, kd)\n")

    print(f"  ✓ deploy_info.txt — Isaac joint order, tanh_squash params, SDK pseudocode")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("✓ EXPORT COMPLETE")
    print(f"{'='*65}")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fp   = os.path.join(OUTPUT_DIR, fn)
        size = os.path.getsize(fp) / 1024
        print(f"  {fn:30s} {size:8.1f} KB")

    print(f"""
Key facts about policy.pt:
  Input:  45D obs (raw, unnormalised — same as go1_deploy_final.py obs array)
  Output: 12D delta in Isaac order — ALREADY tanh-squashed
          guaranteed in [DELTA_LO, DELTA_HI]
  Usage:  target_q_isaac = policy(obs) + DEFAULT_JOINT_POS
          (no separate ACTION_SCALE needed — tanh_squash is baked in)
""")


if __name__ == "__main__":
    main()