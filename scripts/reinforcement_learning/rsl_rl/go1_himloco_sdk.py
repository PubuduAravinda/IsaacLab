#!/usr/bin/env python3
"""
HIMLoco Real Go1 Deployment
Loads trained policy + encoder and runs on real robot
Based on working crawl_sequence.py with proper joint reordering
"""

import time
import numpy as np
import robot_interface as sdk
import torch
import sys
from collections import deque

# ─── CONFIG ─────────────────────────────────────────────────────────────────
# PD gains (same as calibration script)
KP_START = 5.0
KD_FIXED = 3.0
KP_STEP = 5.0
RAMP_INTERVAL_SECONDS = 3.0
RAMP_MAX_LEVEL = 10

# Model paths (update these!)
ACTOR_PATH = "/home/unitree/go1_deployment/actor.pt"
ENCODER_PATH = "/home/unitree/go1_deployment/encoder.pt"

# HIMLoco parameters
HISTORY_LENGTH = 5  # Keep last 5 observations
OBS_DIM = 45  # Base observation dimension
CONTROL_FREQ = 50  # Hz (decimation=8, 400Hz physics → 50Hz policy)

# Real stand pose (default_q) - matches your calibration
stand_q = np.array([0.0, 0.77, -1.5, 0.0, 0.77, -1.5, 0.0, 0.77, -1.5, 0.0, 0.77, -1.5])

# Joint names for debugging
joint_names = ["FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_th", "FR_th",
               "RL_th", "RR_th", "FL_cal", "FR_cal", "RL_cal", "RR_cal"]


# ─── JOINT ORDER MAPPING ───────────────────────────────────────────────────
# Sim order: [FL_hip, FR_hip, RL_hip, RR_hip, FL_th, FR_th, RL_th, RR_th, FL_cal, FR_cal, RL_cal, RR_cal]
# Real SDK order: [FR_hip, FL_hip, RR_hip, RL_hip, FR_th, FL_th, RR_th, RL_th, FR_cal, FL_cal, RR_cal, RL_cal]

def sim_to_real_order(sim_array):
    """Convert from sim joint order to real SDK motor order"""
    real_array = np.zeros(12)
    real_array[0] = sim_array[1]  # FR_hip
    real_array[1] = sim_array[5]  # FR_th
    real_array[2] = sim_array[9]  # FR_cal
    real_array[3] = sim_array[0]  # FL_hip
    real_array[4] = sim_array[4]  # FL_th
    real_array[5] = sim_array[8]  # FL_cal
    real_array[6] = sim_array[3]  # RR_hip
    real_array[7] = sim_array[7]  # RR_th
    real_array[8] = sim_array[11]  # RR_cal
    real_array[9] = sim_array[2]  # RL_hip
    real_array[10] = sim_array[6]  # RL_th
    real_array[11] = sim_array[10]  # RL_cal
    return real_array


def real_to_sim_order(real_array):
    """Convert from real SDK motor order to sim joint order"""
    sim_array = np.zeros(12)
    sim_array[0] = real_array[3]  # FL_hip
    sim_array[1] = real_array[0]  # FR_hip
    sim_array[2] = real_array[9]  # RL_hip
    sim_array[3] = real_array[6]  # RR_hip
    sim_array[4] = real_array[4]  # FL_th
    sim_array[5] = real_array[1]  # FR_th
    sim_array[6] = real_array[10]  # RL_th
    sim_array[7] = real_array[7]  # RR_th
    sim_array[8] = real_array[5]  # FL_cal
    sim_array[9] = real_array[2]  # FR_cal
    sim_array[10] = real_array[11]  # RL_cal
    sim_array[11] = real_array[8]  # RR_cal
    return sim_array


# ─── OBSERVATION CONSTRUCTION ──────────────────────────────────────────────
def build_base_obs(state, stand_q_sim, prev_actions_sim, commands):
    """
    Build 45D base observation matching Isaac Lab training
    Args:
        state: LowState from SDK
        stand_q_sim: default joint positions in SIM order (12,)
        prev_actions_sim: previous actions in SIM order (12,)
        commands: [lin_vel_x, lin_vel_y, ang_vel_z] (3,)
    Returns:
        obs: (45,) numpy array
    """
    # Get real measurements and convert to sim order
    real_joint_pos = np.array([state.motorState[i].q for i in range(12)])
    real_joint_vel = np.array([state.motorState[i].dq for i in range(12)])

    joint_pos_sim = real_to_sim_order(real_joint_pos)
    joint_vel_sim = real_to_sim_order(real_joint_vel)

    # Projected gravity (IMU orientation)
    # Note: SDK gives accelerometer, we need to normalize for gravity direction
    accel = np.array(state.imu.accelerometer)
    gravity_norm = max(np.linalg.norm(accel), 0.1)
    projected_gravity = -accel / gravity_norm  # (3,)

    # Angular velocity from gyroscope
    ang_vel = np.array(state.imu.gyroscope)  # (3,)

    # Build observation matching training
    obs = np.concatenate([
        commands,  # (3,) velocity commands
        projected_gravity,  # (3,) gravity direction in body frame
        ang_vel,  # (3,) body angular velocity
        joint_pos_sim - stand_q_sim,  # (12,) joint position error
        joint_vel_sim,  # (12,) joint velocities
        prev_actions_sim,  # (12,) previous actions
    ])  # Total: 3 + 3 + 3 + 12 + 12 + 12 = 45

    return obs


# ─── MAIN DEPLOYMENT ───────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 90)
    print("HIMLoco Go1 Real Robot Deployment")
    print("=" * 90)

    # Load models
    print(f"\nLoading models...")
    print(f"  Actor:   {ACTOR_PATH}")
    print(f"  Encoder: {ENCODER_PATH}")

    try:
        actor = torch.jit.load(ACTOR_PATH, map_location='cpu')
        encoder = torch.jit.load(ENCODER_PATH, map_location='cpu')
        actor.eval()
        encoder.eval()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        sys.exit(1)

    # Initialize robot communication
    print(f"\nInitializing robot communication...")
    udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)

    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    print("✓ Robot connected")
    print("\nStarting in 10 seconds...")
    print("  Press Ctrl+C to stop")
    print("=" * 90 + "\n")

    time.sleep(10)

    # Initialize state
    obs_history = deque(maxlen=HISTORY_LENGTH)  # Keep last 5 obs
    prev_actions_sim = np.zeros(12)  # Previous actions in sim order
    stand_q_sim = real_to_sim_order(stand_q)  # Convert default pose to sim order

    # Command: forward walking at 0.8 m/s (adjust as needed)
    commands = np.array([0.8, 0.0, 0.0])  # [lin_vel_x, lin_vel_y, ang_vel_z]

    # Fill initial history with zeros
    for _ in range(HISTORY_LENGTH):
        obs_history.append(np.zeros(OBS_DIM))

    step = 0
    t0 = time.time()
    dt = 1.0 / CONTROL_FREQ

    print("Policy running...")

    while True:
        step_start = time.time()
        step += 1
        t = time.time() - t0

        # Receive state
        try:
            udp.Recv()
            udp.GetRecv(state)
        except Exception as e:
            print(f"UDP RECV ERROR: {e}", flush=True)
            break

        # KP ramp (same as calibration)
        current_level = int(t // RAMP_INTERVAL_SECONDS)
        ramp_level = min(RAMP_MAX_LEVEL, current_level)
        current_kp = KP_START + ramp_level * KP_STEP
        current_kd = KD_FIXED

        # Build base observation
        base_obs = build_base_obs(state, stand_q_sim, prev_actions_sim, commands)
        obs_history.append(base_obs)

        # Prepare encoder input: last 5 obs flattened (225D)
        history_array = np.array(list(obs_history))  # (5, 45)
        history_flat = history_array.flatten()  # (225,)

        # Run encoder
        with torch.no_grad():
            history_tensor = torch.from_numpy(history_flat).float().unsqueeze(0)  # (1, 225)
            latent = encoder(history_tensor)  # (1, 19)

        # Concatenate base_obs + latent for policy input
        policy_input = np.concatenate([base_obs, latent.numpy().flatten()])  # (64,)

        # Run policy
        with torch.no_grad():
            policy_tensor = torch.from_numpy(policy_input).float().unsqueeze(0)  # (1, 64)
            actions_sim = actor(policy_tensor).numpy().flatten()  # (12,) in sim order

        # Convert actions to real motor order
        actions_real = sim_to_real_order(actions_sim)

        # Compute target joint positions (actions are offsets from stand pose)
        target_q_real = stand_q + actions_real

        # Send commands to motors
        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q = target_q_real[i]
            cmd.motorCmd[i].dq = 0.0
            cmd.motorCmd[i].Kp = current_kp
            cmd.motorCmd[i].Kd = current_kd
            cmd.motorCmd[i].tau = 0.0

        try:
            udp.SetSend(cmd)
            udp.Send()
        except Exception as e:
            print(f"UDP SEND ERROR: {e}", flush=True)
            break

        # Debug output every second
        if step % CONTROL_FREQ == 0:
            real_joint_pos = np.array([state.motorState[i].q for i in range(12)])
            gravity = -np.array(state.imu.accelerometer) / max(np.linalg.norm(state.imu.accelerometer), 0.1)
            tilt = np.degrees(np.sqrt(gravity[0] ** 2 + gravity[1] ** 2))
            forces = state.footForce

            print(
                f"t={t:5.1f}s | kp={current_kp:4.1f} | cmd=[{commands[0]:+.2f}, {commands[1]:+.2f}, {commands[2]:+.2f}]")
            print(
                f"  Tilt: {tilt:5.1f}° | Forces: FL={forces[0]:4.0f} FR={forces[1]:4.0f} RL={forces[2]:4.0f} RR={forces[3]:4.0f}")
            print(
                f"  Actions (sim): min={actions_sim.min():+.3f} max={actions_sim.max():+.3f} mean={actions_sim.mean():+.3f}")

        # Maintain control frequency
        elapsed = time.time() - step_start
        sleep_time = max(0, dt - elapsed)
        time.sleep(sleep_time)

        # Update previous actions
        prev_actions_sim = actions_sim.copy()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()