#!/usr/bin/env python3
"""
Export Go1 PPO policy + obs normalizer for real robot deployment.

Your training setup:
  - 45D observation (no foot contacts)
  - Simple MLP: 45 → 512 → 256 → 128 → 12
  - actor_obs_normalizer (EmpiricalNormalization) inside policy
  - checkpoint saved by custom_save in train.py:
      {"policy_state_dict": ..., "normalizer_state": ...}

Output files:
  - policy.pt       (TorchScript — run on Go1 RPi or NUC)
  - normalizer.npz  (mean/var as numpy — easy to load in any Python SDK)
  - deploy_info.txt (constants needed in SDK code)
"""

import os
import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS to your best checkpoint path
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT = (
    "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/"
    "logs/rsl_rl/go1_himloco/2026-03-03_16-08-46/model_7000.pt"
)
OUTPUT_DIR = os.path.join(os.path.dirname(CHECKPOINT), "go1_deploy")

# ─────────────────────────────────────────────────────────────────────────────
# These must match your training config exactly
# ─────────────────────────────────────────────────────────────────────────────
OBS_DIM       = 45
ACTION_DIM    = 12
HIDDEN_DIMS   = [512, 256, 128]
ACTIVATION    = "elu"

# From go1_env.py _pre_physics_step
ACTION_SCALE  = np.array([
    0.25, 0.25, 0.25, 0.25,   # hip   FL FR RL RR
    0.5,  0.5,  0.5,  0.5,    # thigh FL FR RL RR
    0.5,  0.5,  0.5,  0.5,    # knee  FL FR RL RR
], dtype=np.float32)

# From go1_env_cfg.py init_state — standing pose
DEFAULT_JOINT_POS = np.array([
    0.1, 0.8, -1.5,   # FL hip thigh knee
    0.1, 0.8, -1.5,   # FR
    0.1, 0.8, -1.5,   # RL
    0.1, 0.8, -1.5,   # RR
], dtype=np.float32)

# Safety rate limiter (SDK code) — max joint change per 20ms step
MAX_DELTA_RAD = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild the actor MLP (same architecture as rsl_rl_ppo_cfg.py)
# ─────────────────────────────────────────────────────────────────────────────
class Go1Actor(nn.Module):
    """
    Standalone actor MLP. Input is ALREADY normalised by the normalizer.
    Output is raw network values — apply action_scale in SDK.
    """
    def __init__(self):
        super().__init__()
        act = nn.ELU if ACTIVATION == "elu" else nn.ReLU
        dims = [OBS_DIM] + HIDDEN_DIMS + [ACTION_DIM]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone normalizer (apply BEFORE passing to actor)
# ─────────────────────────────────────────────────────────────────────────────
class Go1Normalizer(nn.Module):
    """
    Applies the trained EmpiricalNormalization: (obs - mean) / std
    Saved separately so you can also use it as plain numpy in SDK.
    """
    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std",  torch.sqrt(var.float() + 1e-8))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.std


# ─────────────────────────────────────────────────────────────────────────────
# Combined model: normalise + actor in one forward pass
# ─────────────────────────────────────────────────────────────────────────────
class Go1Policy(nn.Module):
    """
    Full policy: raw_obs (45D) → normalise → MLP → action (12D)
    This is what you load on the real Go1.
    action output ∈ roughly [-2, 2] — multiply by ACTION_SCALE in SDK.
    """
    def __init__(self, normalizer: Go1Normalizer, actor: Go1Actor):
        super().__init__()
        self.normalizer = normalizer
        self.actor      = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))


# ─────────────────────────────────────────────────────────────────────────────
# Extract weights from RSL-RL checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def extract_actor_weights(policy_state_dict: dict) -> dict:
    """
    RSL-RL ActorCritic stores actor as:
      actor.0.weight, actor.0.bias, actor.2.weight, ...
    Map to our Go1Actor:
      net.0.weight,   net.0.bias,   net.2.weight, ...
    """
    actor_sd = {}
    for k, v in policy_state_dict.items():
        if k.startswith("actor."):
            new_k = "net." + k[len("actor."):]
            actor_sd[new_k] = v

    if not actor_sd:
        # Try alternative: some RSL-RL versions use 'actor_body'
        for k, v in policy_state_dict.items():
            if k.startswith("actor_body."):
                new_k = "net." + k[len("actor_body."):]
                actor_sd[new_k] = v

    print(f"  Actor weights extracted: {len(actor_sd)} tensors")
    print(f"  Keys: {list(actor_sd.keys())[:6]} ...")
    return actor_sd


def extract_normalizer_params(normalizer_state: dict):
    """
    EmpiricalNormalization state_dict has keys like _mean, _var (or mean, var).
    Returns (mean_tensor, var_tensor).
    """
    mean_key = next((k for k in normalizer_state if "mean" in k.lower()), None)
    var_key  = next((k for k in normalizer_state if "var"  in k.lower()), None)

    if mean_key is None or var_key is None:
        raise ValueError(
            f"Cannot find mean/var in normalizer keys: {list(normalizer_state.keys())}"
        )

    mean = normalizer_state[mean_key].squeeze()
    var  = normalizer_state[var_key].squeeze()
    print(f"  Normalizer mean key: '{mean_key}', shape: {mean.shape}")
    print(f"  Normalizer var  key: '{var_key}',  shape: {var.shape}")
    return mean, var


# ─────────────────────────────────────────────────────────────────────────────
# Main export
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*70}")
    print("Go1 Policy Export")
    print(f"{'='*70}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Output dir : {OUTPUT_DIR}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\n[1/5] Loading checkpoint ...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    print(f"  Keys: {list(ckpt.keys())}")

    if "policy_state_dict" not in ckpt:
        raise ValueError(
            "Expected 'policy_state_dict' key — was this saved by custom_save in train.py?"
        )

    # ── Extract actor ─────────────────────────────────────────────────────────
    print(f"\n[2/5] Extracting actor weights ...")
    actor_weights = extract_actor_weights(ckpt["policy_state_dict"])
    actor = Go1Actor()
    missing, unexpected = actor.load_state_dict(actor_weights, strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected}")
    actor.eval()
    print("  ✓ Actor loaded")

    # ── Extract normalizer ────────────────────────────────────────────────────
    print(f"\n[3/5] Extracting normalizer ...")
    if "normalizer_state" not in ckpt or ckpt["normalizer_state"] is None:
        raise ValueError(
            "No 'normalizer_state' in checkpoint — was it saved by custom_save?"
        )
    mean, var = extract_normalizer_params(ckpt["normalizer_state"])
    std = torch.sqrt(var + 1e-8)
    normalizer = Go1Normalizer(mean, var)
    normalizer.eval()
    print("  ✓ Normalizer loaded")
    print(f"  mean[:5]: {mean[:5].numpy()}")
    print(f"  std[:5]:  {std[:5].numpy()}")

    # ── Verify combined forward pass ──────────────────────────────────────────
    print(f"\n[4/5] Verifying combined forward pass ...")
    policy = Go1Policy(normalizer, actor)
    policy.eval()
    dummy_obs = torch.zeros(1, OBS_DIM)
    dummy_obs[0, 32] = -1.0   # proj_gravity z ≈ -1 (upright robot)
    with torch.no_grad():
        action_out = policy(dummy_obs)
    print(f"  Input obs shape:  {dummy_obs.shape}")
    print(f"  Action out shape: {action_out.shape}")
    print(f"  Action out: {action_out[0].numpy().round(4)}")
    joint_targets = action_out[0].numpy() * ACTION_SCALE + DEFAULT_JOINT_POS
    print(f"  Joint targets: {joint_targets.round(4)}")
    print(f"  Joint target range: [{joint_targets.min():.3f}, {joint_targets.max():.3f}]")

    # ── Export TorchScript ────────────────────────────────────────────────────
    print(f"\n[5/5] Exporting files ...")

    # 1. Full policy as TorchScript (normalizer + actor combined)
    scripted = torch.jit.script(policy)
    pt_path = os.path.join(OUTPUT_DIR, "policy.pt")
    scripted.save(pt_path)
    print(f"  ✓ TorchScript policy: {pt_path}")

    # 2. Normalizer stats as numpy (easiest to use in SDK without PyTorch)
    npz_path = os.path.join(OUTPUT_DIR, "normalizer.npz")
    np.savez(npz_path,
             mean=mean.numpy().astype(np.float32),
             var=var.numpy().astype(np.float32),
             std=std.numpy().astype(np.float32))
    print(f"  ✓ Normalizer numpy:   {npz_path}")

    # 3. Actor weights only (if you want to rebuild in C++ or ONNX)
    actor_only = torch.jit.script(actor)
    actor_path = os.path.join(OUTPUT_DIR, "actor_only.pt")
    actor_only.save(actor_path)
    print(f"  ✓ Actor only:         {actor_path}")

    # 4. Deploy info text file — constants for SDK code
    info_path = os.path.join(OUTPUT_DIR, "deploy_info.txt")
    with open(info_path, "w") as f:
        f.write("Go1 Deployment Constants\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"checkpoint: {CHECKPOINT}\n")
        f.write(f"obs_dim: {OBS_DIM}\n")
        f.write(f"action_dim: {ACTION_DIM}\n\n")
        f.write("# Observation layout (45D):\n")
        f.write("#  [0:3]   velocity commands (vx, vy, wz)  — from joystick\n")
        f.write("#  [3:15]  joint pos delta from default     — encoders\n")
        f.write("#  [15:27] joint velocity                   — encoders\n")
        f.write("#  [27:30] base angular velocity            — IMU gyro\n")
        f.write("#  [30:33] projected gravity                — IMU orientation\n")
        f.write("#  [33:45] previous actions                 — buffer\n\n")
        f.write(f"action_scale: {ACTION_SCALE.tolist()}\n\n")
        f.write(f"default_joint_pos: {DEFAULT_JOINT_POS.tolist()}\n\n")
        f.write(f"max_delta_per_step_rad: {MAX_DELTA_RAD}\n\n")
        f.write("# Joint order (must match SDK):\n")
        f.write("# [0]FL_hip [1]FL_thigh [2]FL_calf\n")
        f.write("# [3]FR_hip [4]FR_thigh [5]FR_calf\n")
        f.write("# [6]RL_hip [7]RL_thigh [8]RL_calf\n")
        f.write("# [9]RR_hip [10]RR_thigh [11]RR_calf\n\n")
        f.write("# SDK deployment pseudocode:\n")
        f.write("# obs = build_obs_45d(cmd, joint_pos, joint_vel, ang_vel, proj_grav, prev_action)\n")
        f.write("# action = policy(obs)              # TorchScript forward\n")
        f.write("# target = action * action_scale + default_joint_pos\n")
        f.write("# target = clip(target, prev_target - max_delta, prev_target + max_delta)\n")
        f.write("# robot.setJointPosition(target)\n")
    print(f"  ✓ Deploy info:        {info_path}")

    print(f"\n{'='*70}")
    print("✓ EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"\nFiles in {OUTPUT_DIR}:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f:30s} {size:8.1f} KB")
    print()
    print("Next step: share your SDK code and we'll write the deployment loop.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()