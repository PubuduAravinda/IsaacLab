import torch
import numpy as np
from isaaclab.envs import DirectRLEnv
import sys

class Go1Env(DirectRLEnv):
    """
    Simplified Go1 environment for replaying real-robot SDK calibration sequence.
    Prints full cycle summary table after every 24 phases (like real SDK).
    """
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\n" + "=" * 90)
        print("GO1 SDK CALIBRATION SEQUENCE REPLAY MODE")
        print("One leg at a time — large deflections — no RL/policy/rewards")
        print("Robot placed upside down on ground")
        print("=" * 90 + "\n")

        # Your original crouch pose (Isaac Sim joint order)
        self.crouch_q = torch.tensor(
            [0.0, 0.0, 0.0, 0.0,
             1.8, 1.8, 1.8, 1.8,
             -2.8, -2.8, -2.8, -2.8],
            device=self.device, dtype=torch.float32
        )

        # Numerical constants (matching your real SDK)
        th_delta = -1.5708  # -90° in radians
        k_delta = 1.5708  # +90° in radians
        abd_delta = 0.3491  # +20° in radians

        # Full 25 phase deltas — all numerical, Isaac Sim joint order
        deltas_isaac = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0: baseline crouch

            # FL leg
            [0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, 0.0, -0.15, 0.10, -0.10],
            [0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, k_delta, -0.15, 0.10, -0.10],
            [abd_delta, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, k_delta, -0.15, 0.10, -0.10],
            [0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, k_delta, -0.15, 0.10, -0.10],
            [0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, 0.0, -0.15, 0.10, -0.10],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.10, -0.10],

            # FR leg
            [0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, 0.0, -0.10, 0.10],
            [0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, k_delta, -0.10, 0.10],
            [0.0, -abd_delta, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, k_delta, -0.10, 0.10],
            [0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, k_delta, -0.10, 0.10],
            [0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, 0.0, -0.10, 0.10],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0, -0.10, 0.10],

            # RL leg
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, 0.0, -0.15],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, k_delta, -0.15],
            [0.0, 0.0, abd_delta, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, k_delta, -0.15],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, k_delta, -0.15],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, 0.0, -0.15],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, -0.10, 0.0, -0.15],

            # RR leg
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, k_delta],
            [0.0, 0.0, 0.0, -abd_delta, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, k_delta],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, k_delta],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.10, 0.10, -0.15, 0.0],
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
            "RR abd back", "RR knee back", "RR thigh back",
        ]

        # Timing
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.steps_per_phase = int(5.0 / dt)
        self.settle_steps = int(3.0 / dt)
        print(f"Phase timing: {self.steps_per_phase} steps ≈ 5s hold")

        # State
        self.phase_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.steps_in_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Cycle & error tracking (for env 0)
        self.current_kp = 50.0
        self.kp_step = 10.0
        self.ramp_max_level = 5
        self.cycle_count = 0
        self.cycle_abs_errors = []  # list of [12] abs error arrays, one per phase

        self.joint_names = [
            "FL_hip", "FR_hip", "RL_hip", "RR_hip",
            "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
            "FL_calf", "FR_calf", "RL_calf", "RR_calf"
        ]

        print(f"Loaded {len(self.deltas)} phases — sequence will loop with 5s pauses")

    def _setup_scene(self):
        self._robot = self.scene["robot"]

    def _pre_physics_step(self, actions: torch.Tensor):
        import sys
        current_delta = self.deltas[self.phase_idx]
        target_positions = self.crouch_q + current_delta
        self._robot.set_joint_position_target(target_positions)

        # Collect abs errors for cycle summary (env 0 only, after settle)
        if self.steps_in_phase[0] >= self.settle_steps:
            real_pos = self._robot.data.joint_pos[0].cpu().numpy()
            target_q = target_positions[0].cpu().numpy()
            error = target_q - real_pos
            self.cycle_abs_errors.append(np.abs(error))

        self.steps_in_phase += 1

        # Phase end detection (env 0 triggers print & advance)
        if self.steps_in_phase[0] >= self.steps_per_phase:
            avg_real = self._robot.data.joint_pos[0].cpu().numpy()
            target_q = target_positions[0].cpu().numpy()
            avg_error = target_q - avg_real

            sys.stdout.write("─" * 89 + "\n")
            sys.stdout.write(
                f"KP = {self.current_kp:.1f} | Phase {self.phase_idx[0].item()}/24 | {self.phase_names[self.phase_idx[0].item()]}\n")
            sys.stdout.write("Given target pos:       " + ", ".join(f"{x:+6.3f}" for x in target_q) + "\n")
            sys.stdout.write("Avg real pos (settled): " + ", ".join(f"{x:+6.3f}" for x in avg_real) + "\n")
            sys.stdout.write("Avg error (given-real): " + ", ".join(f"{x:+6.3f}" for x in avg_error) + "\n")
            sys.stdout.write("(based on last 2.0 s)\n")
            sys.stdout.write("─" * 89 + "\n")
            sys.stdout.flush()

            # Advance phase for ALL envs
            self.phase_idx += 1
            self.steps_in_phase.fill_(0)

            # Cycle end detection & summary
            if self.phase_idx[0] >= len(self.deltas):
                sys.stdout.write("\n" + "=" * 90 + "\n")
                sys.stdout.write(f" CYCLE {self.cycle_count + 1} SUMMARY | KP = {self.current_kp:.1f}\n")
                sys.stdout.write(" Motor | Joint name       | Min |err| | Avg |err| | Max |err| | Notes\n")
                sys.stdout.write("-" * 90 + "\n")

                motor_avg_errors = []
                for j in range(12):
                    errs = [e[j] for e in self.cycle_abs_errors if len(e) > j]
                    min_e = np.min(errs) if errs else 0.0
                    avg_e = np.mean(errs) if errs else 0.0
                    max_e = np.max(errs) if errs else 0.0
                    note = "HIGH resistance / poor tracking" if avg_e > 0.30 else "moderate tracking" if avg_e > 0.15 else ""
                    sys.stdout.write(
                        f" {j:2d}  | {self.joint_names[j]:16} | {min_e:9.3f} | {avg_e:9.3f} | {max_e:9.3f} | {note}\n")
                    motor_avg_errors.append(avg_e)

                overall_min = min(motor_avg_errors)
                overall_avg = np.mean(motor_avg_errors)
                overall_max = max(motor_avg_errors)
                sys.stdout.write("-" * 90 + "\n")
                sys.stdout.write(
                    f" Overall | Min: {overall_min:5.3f} | Avg: {overall_avg:5.3f} | Max: {overall_max:5.3f}\n")
                sys.stdout.write("=" * 90 + "\n\n")
                sys.stdout.flush()

                # Reset cycle data
                self.cycle_abs_errors = []
                self.cycle_count += 1
                self.current_kp += self.kp_step

                if self.current_kp > 5.0 + self.ramp_max_level * self.kp_step:
                    self.current_kp = 5.0 + self.ramp_max_level * self.kp_step
                    sys.stdout.write("\nMax KP reached — no more increment\n")
                    sys.stdout.flush()


                sys.stdout.write(f"\n=== New cycle | KP = {self.current_kp:.1f} ===\n\n")
                sys.stdout.flush()
                self.phase_idx.fill_(0)

            sys.stdout.write(f"→ Next: {self.phase_names[self.phase_idx[0].item()]}\n")
            sys.stdout.flush()


    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros(self.num_envs, 64, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
               torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        env_origins = self.scene.env_origins[env_ids]
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += env_origins
        root_state[:, 2] = 0.12
        upside_down_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(len(env_ids), -1)
        root_state[:, 3:7] = upside_down_quat
        root_state[:, 7:] = 0.0
        self._robot.write_root_state_to_sim(root_state, env_ids)

        self.phase_idx[env_ids] = 0
        self.steps_in_phase[env_ids] = 0
        print(f"[RESET] {len(env_ids)} envs → upside down on ground, sequence restarted")