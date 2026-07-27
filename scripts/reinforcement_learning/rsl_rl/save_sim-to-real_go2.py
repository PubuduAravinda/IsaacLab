#!/usr/bin/env python3
"""
Export Go2 PPO policy + obs normalizer for real robot deployment.

Adapted from save_sim-to-real_models.py (Go1). Go2 differences baked in:
  - 45D obs (no f_cmd — Go2 uses fixed 2Hz gait, unlike Go1's commanded f_cmd)
  - Symmetric, UNIFORM delta bounds (hip ±0.30, thigh/knee ±0.35) —
    Go1 had asymmetric per-joint bounds to compensate for the RL_th PACE
    fault; Go2 is unfaulted hardware so bounds are the same shape on
    every leg.
  - Default joint pose from go2_env_cfg.py: hip=0.0, thigh=0.67, calf=-1.3
    (NOT Go1's 0.1 / 0.8 / -1.5 — do not reuse Go1 deploy constants)
  - No PACE terms anywhere. PACE (Ia/tau_f/d/q~b) only ever affected sim
    training dynamics in go1_env.py; it was never baked into policy.pt,
    so there is nothing PACE-related to port into this script.

*** VERIFIED ON REAL GO2 HARDWARE ***

  1. SDK_TO_ISAAC = [1,5,9, 0,4,8, 3,7,11, 2,6,10] — confirmed via
     go2_test3/test5 calibration runs and go2_test_sign_check.py.

  2. Hip sign convention — confirmed via go2_test_sign_check.py
     (hanging-rack manual check): FL_hip/RL_hip (left) need no flip,
     FR_hip/RR_hip (right) are mirrored vs Isaac and need a sign flip.
     Thigh/knee joints checked too — same direction on both sides,
     no flip needed anywhere outside the two right-side hips.
     This script's exported policy.pt is Isaac-frame only and doesn't
     apply this flip itself — it belongs in the deployment script
     (go2_deploy_model35000.py), which already has it as
     HIP_SIGN_FLIP_IDX = [1, 3].

  3. HIDDEN_DIMS below assumes the same [512,256,128] MLP as Go1's
     rsl_rl_ppo_cfg. Check your actual Go2RslRlPpoCfg (agents/rsl_rl_ppo_cfg.py)
     actor_hidden_dims and correct this constant if it differs — a mismatch
     will fail to load state_dict (which will at least fail loudly), but
     double check anyway.

Training setup (go2_env.py):
  - 45D obs, no PACE, no fault modelling, uniform KP=60 KD=5
  - MLP: 45 -> HIDDEN_DIMS -> 12   activation: ELU (matches Go1's actor)
  - EmpiricalNormalization inside policy
  - checkpoint: {"policy_state_dict": ..., "normalizer_state": ...}

Action formulation — must match go2_env.py _pre_physics_step exactly:
  raw_net  = MLP(normalise(obs))
  _mid     = (delta_hi + delta_lo) / 2      == 0 for all joints (symmetric)
  _half    = (delta_hi - delta_lo) / 2
  delta    = _mid + _half * tanh(raw_net)
  target_q = DEFAULT_JOINT_POS + delta        (absolute joint pos, Isaac order)

  policy.pt bakes: normalise + MLP + tanh_squash -> output is delta directly.
  SDK: target_q_isaac = policy(obs) + DEFAULT_JOINT_POS  (after hw sign flip)

Joint order (Isaac Lab type-grouped — same convention as Go1):
  [0]=FL_hip  [1]=FR_hip  [2]=RL_hip  [3]=RR_hip
  [4]=FL_th   [5]=FR_th   [6]=RL_th   [7]=RR_th
  [8]=FL_kn   [9]=FR_kn  [10]=RL_kn  [11]=RR_kn
"""

import os
import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE: point to the checkpoint you want to deploy
#
# Recommended based on sim_log comparison (35000 vs 44999):
#   model_35000 has lower tilt (4.89 vs 5.38 deg), better velocity tracking
#   (RMSE 0.098 vs 0.103), more symmetric trot (diag balance 0.98/0.92 vs
#   1.00/0.88), smaller hip excursions (+/-0.16 vs +/-0.23 rad of +/-0.30
#   limit), and much lower peak action jerk (0.36 vs 0.50). model_44999
#   has marginally higher training reward (0.0511 vs 0.0503) but that's
#   consistent with late-PPO reward creep decoupled from actual gait
#   quality — worth validating with a couple more --log rollouts before
#   trusting this fully, but 35000 is the safer first hardware candidate.
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT = (
    "/home/sripu715/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go2_rough_sparse/2026-07-24_20-11-40_sparse_v1/model_9950.pt"
)
OUTPUT_DIR = os.path.join(os.path.dirname(CHECKPOINT), "go2_deploy")

# ─────────────────────────────────────────────────────────────────────────────
# Architecture — VERIFY against Go2RslRlPpoCfg (agents/rsl_rl_ppo_cfg.py)
# ─────────────────────────────────────────────────────────────────────────────
OBS_DIM     = 45          # Go2: no f_cmd term (Go1 uses 46)
ACTION_DIM  = 12
HIDDEN_DIMS = [512, 256, 128]   # TODO: confirm matches Go2RslRlPpoCfg.actor_hidden_dims

# ─────────────────────────────────────────────────────────────────────────────
# tanh_squash parameters — MUST match go2_env.py _delta_soft_lo / _delta_soft_hi
#
#   _delta_soft_lo = [-0.30]*4 hip, [-0.35]*8 thigh+knee
#   _delta_soft_hi = [ 0.30]*4 hip, [ 0.35]*8 thigh+knee
#
# Result:
#   _mid  = 0.0 for ALL joints (fully symmetric, unlike Go1's RL_hip skew)
#   _half = [0.30]*4, [0.35]*8
# ─────────────────────────────────────────────────────────────────────────────
DELTA_LO = np.array([
    -0.30, -0.30, -0.30, -0.30,   # FL FR RL RR hip
    -0.35, -0.35, -0.35, -0.35,   # thighs
    -0.35, -0.35, -0.35, -0.35,   # calves
], dtype=np.float32)

DELTA_HI = np.array([
     0.30,  0.30,  0.30,  0.30,   # FL FR RL RR hip
     0.35,  0.35,  0.35,  0.35,   # thighs
     0.35,  0.35,  0.35,  0.35,   # calves
], dtype=np.float32)

_MID  = torch.tensor((DELTA_HI + DELTA_LO) / 2.0)   # all zeros — symmetric
_HALF = torch.tensor((DELTA_HI - DELTA_LO) / 2.0)   # [0.30]*4, [0.35]*8

assert np.allclose(_MID.numpy(), 0.0, atol=1e-6), \
    f"_mid should be all-zero (symmetric limits). Got: {_MID.numpy()}"
print(f"[CHECK] _mid  = {_MID.numpy()}  (all zero \u2713)")
print(f"[CHECK] _half = {_HALF.numpy()}")

# ─────────────────────────────────────────────────────────────────────────────
# Default joint positions — Isaac order (from go2_env_cfg.py init_state)
# NOTE: different from Go1! hip=0.0 (not 0.1), thigh=0.67 (not 0.8),
# calf=-1.3 (not -1.5). Do not reuse Go1's DEFAULT_JOINT_POS here.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_JOINT_POS = np.array([
    0.0,   0.0,   0.0,   0.0,    # FL FR RL RR hip
    0.67,  0.67,  0.67,  0.67,   # FL FR RL RR thigh
   -1.3,  -1.3,  -1.3,  -1.3,    # FL FR RL RR knee
], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# SDK joint index mapping — VERIFIED on real Go2 hardware
# (go2_test3_latency.py, go2_test5_freq_sweep.py, go2_test_sign_check.py all
# use this same mapping and it has been exercised across many hardware runs).
# ─────────────────────────────────────────────────────────────────────────────
SDK_TO_ISAAC = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]   # VERIFIED on Go2 hardware


# ─────────────────────────────────────────────────────────────────────────────
# Neural network modules
# ─────────────────────────────────────────────────────────────────────────────
class Go2Normalizer(nn.Module):
    """(obs - mean) / std  from EmpiricalNormalization state."""
    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std",  torch.sqrt(var.float() + 1e-8))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.std


class Go2Actor(nn.Module):
    """MLP: normalised_obs(45) -> raw_net(12).  No tanh here."""
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


class Go2Policy(nn.Module):
    """
    Full policy: raw_obs(45) -> normalise -> MLP -> tanh_squash -> delta(12)

    Output is joint delta in Isaac order, tanh-squashed to [DELTA_LO, DELTA_HI].
    Guaranteed within bounds — no additional clamping needed.

    SDK usage (once mapping/sign are verified on hardware):
        delta_isaac = policy(obs_45d)              # this forward
        # apply any hw sign flip(s) found during verification
        target_q_isaac = DEFAULT_JOINT_POS + delta_isaac
        target_q_sdk   = target_q_isaac[isaac_to_sdk]
        robot.setJointPos(target_q_sdk, kp, kd)
    """
    def __init__(self, normalizer: Go2Normalizer, actor: Go2Actor,
                 mid: torch.Tensor, half: torch.Tensor):
        super().__init__()
        self.normalizer = normalizer
        self.actor      = actor
        self.register_buffer("delta_mid",  mid.float())
        self.register_buffer("delta_half", half.float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.normalizer(obs)
        raw_net  = self.actor(norm_obs)
        return self.delta_mid + self.delta_half * torch.tanh(raw_net)


def extract_normalizer_params(normalizer_state):
    """Pull (mean, var) tensors out of an EmpiricalNormalization state_dict,
    regardless of exact key naming across rsl_rl versions.

    Shape handling: older checkpoints (e.g. the go2_flat run, before
    actor_obs_normalization was enabled) stored mean/var as (OBS_DIM,).
    Newer ones (actor_obs_normalization=True, confirmed on the
    go2_rough_sparse run) store them as (1, OBS_DIM) -- a leading batch
    dim from rsl_rl's own normalizer module shape convention. Squeeze any
    leading singleton dims so this works for BOTH checkpoint generations
    without needing to know in advance which one you're loading.
    """
    mean_key = next(k for k in normalizer_state if "mean" in k.lower())
    var_key  = next(k for k in normalizer_state if "var" in k.lower())
    mean = normalizer_state[mean_key].float()
    var  = normalizer_state[var_key].float()

    # Squeeze leading singleton dims: (1, 45) -> (45,), (1, 1, 45) -> (45,)
    # but leave a genuine (45,) untouched. Works regardless of how many
    # leading 1-dims a given rsl_rl version happens to add.
    while mean.dim() > 1 and mean.shape[0] == 1:
        mean = mean.squeeze(0)
    while var.dim() > 1 and var.shape[0] == 1:
        var = var.squeeze(0)

    return mean, var


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[1/5] Loading checkpoint ...\n  {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    assert "policy_state_dict" in ckpt, \
        f"checkpoint missing 'policy_state_dict'. Keys: {list(ckpt.keys())}"
    assert "normalizer_state" in ckpt, \
        f"checkpoint missing 'normalizer_state'. Keys: {list(ckpt.keys())}"

    print(f"\n[2/5] Building actor ...")
    actor = Go2Actor()

    raw_keys = list(ckpt["policy_state_dict"].keys())
    print(f"  checkpoint policy_state_dict keys ({len(raw_keys)}):")
    for k in raw_keys:
        print(f"    {k}  {tuple(ckpt['policy_state_dict'][k].shape)}")

    # Try known rsl_rl ActorCritic naming conventions, in order, remapping
    # the found prefix -> "net." to match Go2Actor's nn.Sequential("net").
    candidate_prefixes = ["actor.", "a2c_network.actor_mlp.", "actor_mlp."]
    actor_state = {}
    used_prefix = None
    for prefix in candidate_prefixes:
        matched = {k: v for k, v in ckpt["policy_state_dict"].items() if k.startswith(prefix)}
        if matched:
            actor_state = {k.replace(prefix, "net.", 1): v for k, v in matched.items()}
            used_prefix = prefix
            break

    if not actor_state:
        raise RuntimeError(
            "Could not find actor weights under any known prefix "
            f"{candidate_prefixes}. Inspect the printed keys above and adjust "
            "candidate_prefixes to match this checkpoint's actual naming."
        )
    print(f"  Matched prefix: '{used_prefix}' -> remapped to 'net.'")

    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    assert not missing,    f"Missing actor keys: {missing}"
    assert not unexpected, f"Unexpected actor keys: {unexpected}"
    actor.eval()
    print("  \u2713 Actor loaded")

    print(f"\n[3/5] Building normalizer ...")
    print(f"  top-level checkpoint keys: {list(ckpt.keys())}")
    norm_state = ckpt["normalizer_state"]
    print(f"  normalizer_state type: {type(norm_state)}")
    if isinstance(norm_state, dict):
        print(f"  normalizer_state keys ({len(norm_state)}):")
        for k, v in norm_state.items():
            shape = tuple(v.shape) if hasattr(v, "shape") else v
            print(f"    {k}  {shape}")
    else:
        print(f"  normalizer_state repr: {norm_state!r}")

    if isinstance(norm_state, dict) and len(norm_state) > 0:
        mean, var = extract_normalizer_params(norm_state)
    else:
        # Empty/absent normalizer_state -> this checkpoint's training run
        # did not use EmpiricalNormalization (or it wasn't saved). Fall
        # back to an identity normalizer rather than guessing at keys.
        # NOTE: if go2_env.py / your rsl_rl_ppo_cfg DOES enable obs
        # normalization during training, deploying with an identity
        # normalizer here will be WRONG (obs scale mismatch -> bad
        # policy behavior on hardware even though it loads fine). Check
        # your Go2RslRlPpoCfg for `empirical_normalization` before
        # trusting this fallback.
        print("  WARNING: normalizer_state empty/unrecognized -> using "
              "IDENTITY normalizer (mean=0, var=1). Verify your "
              "Go2RslRlPpoCfg.empirical_normalization setting before "
              "trusting this on hardware.")
        mean = torch.zeros(OBS_DIM)
        var  = torch.ones(OBS_DIM)

    std = torch.sqrt(var + 1e-8)
    normalizer = Go2Normalizer(mean, var)
    normalizer.eval()
    print(f"  mean[:5] : {mean[:5].numpy().round(5)}")
    print(f"  std[:5]  : {std[:5].numpy().round(5)}")
    assert mean.shape == (OBS_DIM,), f"mean shape wrong: {mean.shape}"

    print(f"\n[4/5] Verifying full forward pass ...")
    policy = Go2Policy(normalizer, actor, _MID, _HALF)
    policy.eval()

    # Canonical test: upright robot at rest, cmd_vx=0.5
    dummy       = torch.zeros(1, OBS_DIM)
    dummy[0, 0] = 0.5    # cmd_vx
    dummy[0, 32] = -1.0  # proj_gravity_z (obs index 30+2=32)

    with torch.no_grad():
        delta = policy(dummy)

    print(f"  Input shape  : {dummy.shape}")
    print(f"  Delta shape  : {delta.shape}")
    print(f"  Delta (Isaac): {delta[0].numpy().round(4)}")

    target_q = delta[0].numpy() + DEFAULT_JOINT_POS
    print(f"  Target_q     : {target_q.round(4)}")
    print(f"  Target_q range: [{target_q.min():.3f}, {target_q.max():.3f}]")

    delta_np = delta[0].numpy()
    assert (delta_np >= DELTA_LO - 1e-4).all(), \
        f"delta below lo! {delta_np[delta_np < DELTA_LO - 1e-4]}"
    assert (delta_np <= DELTA_HI + 1e-4).all(), \
        f"delta above hi! {delta_np[delta_np > DELTA_HI + 1e-4]}"
    print("  \u2713 delta within [DELTA_LO, DELTA_HI] — tanh_squash correct")

    hip_delta = delta_np[:4]
    print(f"  Hip delta at rest: {hip_delta.round(4)}")
    print(f"  (expected near 0.0 — symmetric limits, no PACE skew for Go2)")

    print(f"\n[5/5] Exporting ...")

    scripted = torch.jit.script(policy)
    pt_path  = os.path.join(OUTPUT_DIR, "policy.pt")
    scripted.save(pt_path)
    print(f"  \u2713 policy.pt      -> raw_obs(45) -> delta(12)  [normalise+MLP+tanh baked in]")

    np.savez(os.path.join(OUTPUT_DIR, "normalizer.npz"),
             mean=mean.numpy().astype(np.float32),
             var=var.numpy().astype(np.float32),
             std=std.numpy().astype(np.float32))
    print(f"  \u2713 normalizer.npz -> mean/var/std  shape=({OBS_DIM},)")

    isaac_to_sdk = [0] * 12
    for sdk_i, isa_i in enumerate(SDK_TO_ISAAC):
        isaac_to_sdk[isa_i] = sdk_i

    jnames = ["FL_hip","FR_hip","RL_hip","RR_hip",
              "FL_thigh","FR_thigh","RL_thigh","RR_thigh",
              "FL_knee","FR_knee","RL_knee","RR_knee"]

    info_path = os.path.join(OUTPUT_DIR, "deploy_info.txt")
    with open(info_path, "w") as f:
        f.write("Go2 Deployment Constants\n")
        f.write("=" * 65 + "\n\n")
        f.write("*** SDK mapping & hip sign convention VERIFIED on real Go2 ***\n")
        f.write("*** hardware (go2_test3/test5, go2_test_sign_check.py).    ***\n\n")
        f.write(f"checkpoint  : {CHECKPOINT}\n")
        f.write(f"obs_dim     : {OBS_DIM}\n")
        f.write(f"action_dim  : {ACTION_DIM}\n\n")

        f.write("── Observation layout (45D) ─────────────────────────────────\n")
        f.write("  [0:3]   cmd [vx, vy, wz]             joystick / fixed cmd\n")
        f.write("  [3:15]  jpos - default_q              Isaac order\n")
        f.write("  [15:27] jvel  clipped ±5              Isaac order\n")
        f.write("  [27:30] gyro  clipped ±5 [x,y,z]     IMU\n")
        f.write("  [30:33] proj_gravity [x,y,z]          -acc/|acc| from IMU\n")
        f.write("  [33:45] prev_actions                  policy output at t-1\n\n")
        f.write("  NOTE: no f_cmd term (Go1 has 46D with f_cmd; Go2 uses fixed 2Hz gait)\n\n")

        f.write("── Isaac joint order ─────────────────────────────────────────\n")
        for i, n in enumerate(jnames):
            f.write(f"  [{i:2d}] {n}\n")
        f.write("\n")

        f.write("── SDK motor order — VERIFIED on real Go2 hardware ──\n")
        f.write(f"  sdk_to_isaac (VERIFIED) = {SDK_TO_ISAAC}\n")
        f.write(f"  isaac_to_sdk (VERIFIED) = {isaac_to_sdk}\n\n")

        f.write("── Hip sign convention — VERIFIED via go2_test_sign_check.py ──\n")
        f.write("  Left hips  (FL, RL) : no flip needed.\n")
        f.write("  Right hips (FR, RR) : mirrored vs Isaac, NEEDS flip.\n")
        f.write("  Thigh/knee (all 8)  : checked, no flip needed.\n")
        f.write("  -> HIP_SIGN_FLIP_IDX = [1, 3] in go2_deploy_model35000.py\n\n")

        f.write("── tanh_squash bounds (symmetric, no PACE skew) ───────────────\n")
        f.write(f"  delta_lo = {DELTA_LO.tolist()}\n")
        f.write(f"  delta_hi = {DELTA_HI.tolist()}\n")
        f.write(f"  _mid     = {_MID.numpy().tolist()}  (all zero)\n")
        f.write(f"  _half    = {_HALF.numpy().tolist()}\n\n")

        f.write("── Default joint positions (Isaac order) ─────────────────────\n")
        f.write(f"  {DEFAULT_JOINT_POS.tolist()}\n\n")

        f.write("── Training KP/KD (must match on hardware) ───────────────────\n")
        f.write("  KP: 60 (uniform all joints)  KD: 5 (uniform all joints)\n\n")

        f.write("── SDK deployment pseudocode ──────────────────────────────────\n")
        f.write("  # 1. Read hardware, convert to Isaac order\n")
        f.write("  jpos_isaac = encoder_q[sdk_to_isaac]   # VERIFY mapping first\n")
        f.write("  jvel_isaac = encoder_dq[sdk_to_isaac]\n\n")
        f.write("  # 2. Build 45D obs\n")
        f.write("  obs[0:3]   = [cmd_vx, 0, 0]\n")
        f.write("  obs[3:15]  = jpos_isaac - default_q   # apply hip sign flip if verified\n")
        f.write("  obs[15:27] = clip(jvel_isaac, ±5)\n")
        f.write("  obs[27:30] = clip(gyro_xyz, ±5)\n")
        f.write("  obs[30:33] = -acc / |acc|   # proj_gravity\n")
        f.write("  obs[33:45] = prev_delta      # policy output t-1\n\n")
        f.write("  # 3. Policy inference\n")
        f.write("  delta_isaac = policy(obs)    # normalise + MLP + tanh baked in\n\n")
        f.write("  # 4. Store prev_delta BEFORE any hw sign flip\n")
        f.write("  prev_delta[:] = delta_isaac.copy()\n\n")
        f.write("  # 5. Apply hw sign flip if verification finds one, reorder, send\n")
        f.write("  target_q_isaac = default_q + delta_hw\n")
        f.write("  target_q_sdk   = target_q_isaac[isaac_to_sdk]\n")
        f.write("  robot.setJointPos(target_q_sdk, kp, kd)\n")

    print(f"  \u2713 deploy_info.txt")

    print(f"\n{'='*65}")
    print("\u2713 EXPORT COMPLETE")
    print(f"{'='*65}")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fp   = os.path.join(OUTPUT_DIR, fn)
        size = os.path.getsize(fp) / 1024
        print(f"  {fn:30s} {size:8.1f} KB")

    print("""
policy.pt interface:
  Input:  45D obs (raw, unnormalised — build exactly as deploy_info.txt)
  Output: 12D delta Isaac order — tanh-squashed, bounded in [DELTA_LO, DELTA_HI]
  Usage:  target_q_isaac = policy(obs) + DEFAULT_JOINT_POS

REMINDER: SDK_TO_ISAAC and hip sign convention are VERIFIED for Go2 —
see deploy_info.txt. Apply the flip in go2_deploy_model35000.py
(HIP_SIGN_FLIP_IDX = [1, 3]), not in this exported policy.pt itself.
""")


if __name__ == "__main__":
    main()