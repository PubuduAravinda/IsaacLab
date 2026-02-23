import torch
import numpy as np
from isaaclab.envs import DirectRLEnv


class Go1Env(DirectRLEnv):
    """
    Simplified Go1 environment for replaying real-robot SDK calibration sequence.
    No RL/policy/rewards/encoders/SwAV — only position control + logging.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "=" * 90)
        print("GO1 SDK CALIBRATION SEQUENCE REPLAY MODE")
        print("One leg at a time — large deflections — no RL/policy/rewards")
        print("Robot held in air — gravity effects visible")
        print("=" * 90 + "\n")

        # Your original crouch / reference pose (Isaac Sim joint order)
        self.crouch_q = torch.tensor(
            [0.0, 0.0, 0.0, 0.0,
             1.8, 1.8, 1.8, 1.8,
             -2.8, -2.8, -2.8, -2.8],
            device=self.device, dtype=torch.float32
        )

        # Your original phase deltas — already in Isaac Sim order
        deltas_isaac = [
            np.zeros(12),  # baseline

            # FL leg
            [0,0,0,0, -np.pi/2, 0,0,0, 0, -0.15, 0.10, -0.10],
            [0,0,0,0, -np.pi/2, 0,0,0, np.pi/2, -0.15, 0.10, -0.10],
            [np.deg2rad(20), 0,0,0, -np.pi/2, 0,0,0, np.pi/2, -0.15, 0.10, -0.10],
            [0,0,0,0, -np.pi/2, 0,0,0, np.pi/2, -0.15, 0.10, -0.10],
            [0,0,0,0, -np.pi/2, 0,0,0, 0, -0.15, 0.10, -0.10],
            [0,0,0,0, 0,0,0,0, 0, -0.15, 0.10, -0.10],

            # FR leg
            [0,0,0,0, 0, -np.pi/2, 0,0, -0.15, 0, -0.10, 0.10],
            [0,0,0,0, 0, -np.pi/2, 0,0, -0.15, np.pi/2, -0.10, 0.10],
            [0, -np.deg2rad(20), 0,0, 0, -np.pi/2, 0,0, -0.15, np.pi/2, -0.10, 0.10],
            [0,0,0,0, 0, -np.pi/2, 0,0, -0.15, np.pi/2, -0.10, 0.10],
            [0,0,0,0, 0, -np.pi/2, 0,0, -0.15, 0, -0.10, 0.10],
            [0,0,0,0, 0,0,0,0, -0.15, 0, -0.10, 0.10],

            # RL leg
            [0,0,0,0, 0,0, -np.pi/2, 0, 0.10, -0.10, 0, -0.15],
            [0,0,0,0, 0,0, -np.pi/2, 0, 0.10, -0.10, np.pi/2, -0.15],
            [0,0, np.deg2rad(20), 0, 0,0, -np.pi/2, 0, 0.10, -0.10, np.pi/2, -0.15],
            [0,0,0,0, 0,0, -np.pi/2, 0, 0.10, -0.10, np.pi/2, -0.15],
            [0,0,0,0, 0,0, -np.pi/2, 0, 0.10, -0.10, 0, -0.15],
            [0,0,0,0, 0,0, 0,0, 0.10, -0.10, 0, -0.15],

            # RR leg
            [0,0,0,0, 0,0,0, -np.pi/2, -0.10, 0.10, -0.15, 0],
            [0,0,0,0, 0,0,0, -np.pi/2, -0.10, 0.10, -0.15, np.pi/2],
            [0,0,0, -np.deg2rad(20), 0,0,0, -np.pi/2, -0.10, 0.10, -0.15, np.pi/2],
            [0,0,0,0, 0,0,0, -np.pi/2, -0.10, 0.10, -0.15, np.pi/2],
            [0,0,0,0, 0,0,0, -np.pi/2, -0.10, 0.10, -0.15, 0],
            [0,0,0,0, 0,0,0,0, -0.10, 0.10, -0.15, 0],
        ]

        self.deltas = torch.tensor(deltas_isaac, device=self.device, dtype=torch.float32)

        self.phase_names = [
            "crouch (baseline)",
            "FL thigh forward", "FL knee forward", "FL abd forward",
            "FL abd back", "FL knee back", "FL thigh back",
            "FR thigh forward", "FR knee forward", "FR abd forward",
            "FR abd back", "FR knee back", "FR thigh back",
            "RL thigh forward", "RL knee forward", "RL abd forward",
            "RL abd back", "RL knee back", "RL thigh back",
            "RR thigh forward", "RR knee forward", "RR abd forward",
            "RR abductor back", "RR knee back", "RR thigh back",
        ]

        # Sequence control
        self.phase_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.steps_in_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Timing to match real: 5s hold, 3s settle, average over last 2s
        dt_per_step = self.cfg.sim.dt * self.cfg.decimation
        self.steps_per_phase = int(5.0 / dt_per_step)
        self.settle_steps = int(3.0 / dt_per_step)
        print(f"Timing: steps_per_phase={self.steps_per_phase} (~5s), settle_steps={self.settle_steps} (~3s)")

        print(f"Physics dt = {self.cfg.sim.dt:.4f} s")
        print(f"Decimation = {self.cfg.decimation}")
        print(f"Control interval = {self.cfg.sim.dt * self.cfg.decimation:.4f} s")
        print(f"Actual hold time = {self.steps_per_phase * self.cfg.sim.dt * self.cfg.decimation:.3f} s")
        print(f"Actual settle time = {self.settle_steps * self.cfg.sim.dt * self.cfg.decimation:.3f} s")

        # Data collection for averages (only for first 4 envs to avoid memory overhead)
        self.num_monitored = min(4, self.num_envs)
        self.phase_real_buffers = [[] for _ in range(self.num_monitored)]  # list of lists for real_pos

        # Cycle control (to match real summaries)
        self.cycle_count = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.current_kp = 5.0  # start from real initial KP
        self.kp_step = 5.0
        self.ramp_max_level = 10  # up to KP=55.0
        self.cycle_abs_errors = [[[] for _ in range(12)] for _ in range(self.num_envs)]  # per-env per-joint abs errors

        # Joint names for logging (Isaac Sim order)
        self.joint_names = [
            "FL_hip", "FR_hip", "RL_hip", "RR_hip",
            "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
            "FL_calf", "FR_calf", "RL_calf", "RR_calf"
        ]

        print(f"Loaded {len(self.deltas)} phases — sequence will loop with 5s pauses per phase")

    def _setup_scene(self):
        self._robot = self.scene["robot"]

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply hardcoded SDK calibration sequence — ignore policy actions."""

        # Current phase delta for each env
        current_delta = self.deltas[self.phase_idx]  # [num_envs, 12]

        # Target = crouch + delta (no clamping)
        target_positions = self.crouch_q + current_delta

        self._robot.set_joint_position_target(target_positions)

        # ─── Data collection for averages (only monitored envs, after settle) ────────
        settle_mask = self.steps_in_phase >= self.settle_steps
        for eid in range(self.num_monitored):
            if settle_mask[eid]:
                real_pos = self._robot.data.joint_pos[eid].cpu().numpy()
                self.phase_real_buffers[eid].append(real_pos.copy())

        # ─── Advance steps ───────────────────────────────────────────────────────────
        self.steps_in_phase += 1

        # ─── Phase end detection (vectorized for all envs) ───────────────────────────
        phase_end_mask = self.steps_in_phase >= self.steps_per_phase

        if phase_end_mask.any():
            # Vectorized advance for ALL envs
            self.phase_idx[phase_end_mask] += 1

            # Handle cycle end + KP ramp (vectorized)
            cycle_end_mask = self.phase_idx >= len(self.deltas)
            if cycle_end_mask.any():
                for eid in range(self.num_monitored):
                    if cycle_end_mask[eid]:
                        print("\n" + "=" * 90)
                        print(f" CYCLE {self.cycle_count[eid].item() + 1} SUMMARY | KP = {self.current_kp:.1f}")
                        print(" Motor | Joint name     | Min |err| | Avg |err| | Max |err| | Notes")
                        print("-" * 90)

                        motor_avg_errors = []
                        has_errors = False

                        for j in range(12):
                            errs = self.cycle_abs_errors[eid][j]
                            if errs:
                                has_errors = True
                                min_e = np.min(errs)
                                avg_e = np.mean(errs)
                                max_e = np.max(errs)
                            else:
                                min_e = avg_e = max_e = 0.0

                            note = "HIGH resistance / poor tracking" if avg_e > 0.30 else "moderate tracking" if avg_e > 0.15 else ""
                            print(
                                f" {j:2d}   | {self.joint_names[j]:14} | {min_e:8.3f} | {avg_e:8.3f} | {max_e:8.3f} | {note}")

                            motor_avg_errors.append(avg_e)

                        print("-" * 90)

                        if has_errors:
                            overall_min = min(motor_avg_errors)
                            overall_avg = np.mean(motor_avg_errors)
                            overall_max = max(motor_avg_errors)
                            print(
                                f"Overall robot error stats | Min: {overall_min:5.3f} | Avg: {overall_avg:5.3f} | Max: {overall_max:5.3f}")
                        else:
                            print("No error data collected for this cycle (check collection logic)")

                        print("=" * 90 + "\n")

                        # Reset for next cycle
                        self.cycle_abs_errors[eid] = [[] for _ in range(12)]

                # Increment cycle count and KP (scalar)
                self.cycle_count[cycle_end_mask] += 1
                self.current_kp += self.kp_step

                # Clamp KP
                max_kp = 5.0 + self.ramp_max_level * self.kp_step
                self.current_kp = min(self.current_kp, max_kp)

                if self.current_kp >= max_kp:
                    print("\nMax KP reached — no more increment\n")

                print(f"=== New cycle | KP = {self.current_kp:.1f} ===\n")

                # Reset phase
                self.phase_idx[cycle_end_mask] = 0

            # ─── Print phase summary for monitored envs that just ended a phase ───────
            for eid in range(self.num_monitored):
                if phase_end_mask[eid]:
                    if self.phase_real_buffers[eid]:
                        avg_real = np.mean(self.phase_real_buffers[eid], axis=0)
                        target_q = target_positions[eid].cpu().numpy()
                        avg_error = target_q - avg_real
                        abs_errors = np.abs(avg_error)

                        # Store for cycle
                        for j in range(12):
                            self.cycle_abs_errors[eid][j].append(abs_errors[j])

                        # ─── Add wall-clock timestamp ───────────────────────────────────────────────
                        import time
                        current_time = time.time()
                        if not hasattr(self, '_last_phase_time'):
                            self._last_phase_time = [0.0] * self.num_monitored
                        elapsed = current_time - self._last_phase_time[eid] if self._last_phase_time[eid] > 0 else 0.0
                        self._last_phase_time[eid] = current_time

                        print("─────────────────────────────────────────────────────────────────────────────────────")
                        print(
                            f"KP = {self.current_kp:.1f} | Phase {self.phase_idx[eid].item()}/24 | {self.phase_names[self.phase_idx[eid].item()]}")
                        print(f"Wall-clock time since last phase end: {elapsed:.2f} seconds")
                        print("Given target pos: " + ", ".join(f"{x:+6.3f}" for x in target_q))
                        print("Avg real pos (settled): " + ", ".join(f"{x:+6.3f}" for x in avg_real))
                        print("Avg error (given - real): " + ", ".join(f"{x:+6.3f}" for x in avg_error))
                        print("(based on last 2.0 s)")
                        print("─────────────────────────────────────────────────────────────────────────────────────")

                    # Clear buffer
                    self.phase_real_buffers[eid] = []

            # Print next phase announcement (only once per phase end, for monitored)
            if phase_end_mask[:self.num_monitored].any():
                print(f"→ Next: {self.phase_names[self.phase_idx[0].item()]}")  # use env 0 as reference

            # Reset steps for ended phases
            self.steps_in_phase[phase_end_mask] = 0

    def _apply_action(self):
        """No-op — targets are set in _pre_physics_step."""
        pass

    def _get_observations(self) -> dict:
        # Dummy observation — rsl_rl expects "policy" key with shape [num_envs, 64]
        return {"policy": torch.zeros(self.num_envs, 64, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        # No rewards
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Never terminate — loop sequence forever
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # Get per-env origins for grid spacing
        env_origins = self.scene.env_origins[env_ids]

        # Default root state with offsets
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += env_origins

        # Low height for trunk on ground
        root_state[:, 2] = 0.12

        # Upside-down orientation (180° roll)
        upside_down_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(len(env_ids), -1)
        root_state[:, 3:7] = upside_down_quat

        # Zero velocities
        root_state[:, 7:] = 0.0

        # Apply to sim
        self._robot.write_root_state_to_sim(root_state, env_ids)

        # Reset sequence and buffers
        self.phase_idx[env_ids] = 0
        self.steps_in_phase[env_ids] = 0
        self.cycle_count[env_ids] = 0
        self.current_kp = 5.0  # reset KP
        for eid in env_ids.tolist():
            if eid < self.num_monitored:
                self.phase_real_buffers[eid] = []
            self.cycle_abs_errors[eid] = [[] for _ in range(12)]

        print(f"[RESET] {len(env_ids)} envs → placed upside down on ground with env offsets, sequence restarted")