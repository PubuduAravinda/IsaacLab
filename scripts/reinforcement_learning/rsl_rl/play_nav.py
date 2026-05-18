# play_nav.py — Visualize trained HL+LL navigation policy
#
# ═══════════════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════════════
#
#   # Step 1: Set HL_POLICY_ACTIVE = True in go1_nav_env.py
#
#   # Step 2: Run play
#   python play_nav.py \
#       --ll_checkpoint  logs/rsl_rl/go1_himloco/.../model_34999.pt \
#       --hl_checkpoint  logs/rsl_rl/go1_nav_hl/.../model_5000.pt \
#       --num_envs 4
#
# ═══════════════════════════════════════════════════════════════════════════
# WHAT YOU SEE IN THE VIEWPORT
# ═══════════════════════════════════════════════════════════════════════════
#   🟢 Green sphere   = episode start position for each env
#   🔴 Red sphere     = goal target position for each env
#   ── Yellow line    = straight-line path from start → goal
#   Robot path        = the arc the Go1 actually walks
#
#   With only 4 envs you can clearly match each robot to its markers.
#
# ═══════════════════════════════════════════════════════════════════════════
# HL POLICY LOADING
# ═══════════════════════════════════════════════════════════════════════════
# RSL-RL saves: {"policy_state_dict": ..., "normalizer_state": ...}
# We reconstruct the actor MLP [obs_dim→128→64→32→act_dim] and run
# inference only — no critic needed for play.

"""Launch Isaac Sim first."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Go1 HRL navigation policy.")
parser.add_argument("--ll_checkpoint",  type=str, required=True,
                    help="Frozen LL policy .pt checkpoint")
parser.add_argument("--hl_checkpoint",  type=str, required=True,
                    help="Trained HL PPO policy .pt checkpoint")
parser.add_argument("--num_envs",       type=int, default=4,
                    help="Number of envs to visualize (keep small: 2-8)")
parser.add_argument("--episode_length", type=float, default=30.0,
                    help="Episode length in seconds")
parser.add_argument("--goal_dist_min",  type=float, default=3.0)
parser.add_argument("--goal_dist_max",  type=float, default=8.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Play never runs headless — always show viewport
args_cli.headless = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after Isaac Sim launch."""

import torch
import numpy as np
import gymnasium as gym
from torch import nn

# Isaac Lab / Sim imports
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG

# Nav env
from isaaclab_tasks.direct.go1.go1_nav_env     import Go1NavEnv       # noqa
from isaaclab_tasks.direct.go1.go1_nav_env_cfg import Go1NavEnvCfg


# =============================================================================
# CONNECTING LINE DRAWER
# Uses omni.isaac.debug_draw — draws lines directly in the Isaac Sim viewport
# =============================================================================
class _LineDrawer:
    """Thin wrapper around omni.isaac.debug_draw for start→goal lines."""

    def __init__(self):
        self._ok = False
        try:
            from omni.isaac.debug_draw import _debug_draw
            self._draw = _debug_draw.acquire_debug_draw_interface()
            self._ok = True
            print("[LineDrawer] debug_draw acquired ✓  "
                  "— yellow lines between 🟢 and 🔴")
        except Exception as e:
            print(f"[LineDrawer] Disabled ({e}) "
                  "— markers still visible, lines won't draw")

    def update(self, starts_xy: torch.Tensor, goals_xy: torch.Tensor,
               z_start: float = 0.3, z_goal: float = 0.5):
        """
        Draw one line per env from start sphere to goal sphere.
        starts_xy, goals_xy: [N, 2] world-frame XY tensors (CPU or GPU)
        """
        if not self._ok:
            return
        n = starts_xy.shape[0]
        s = starts_xy.cpu().numpy()
        g = goals_xy.cpu().numpy()

        p0, p1, cols, widths = [], [], [], []
        for i in range(n):
            p0.append((float(s[i, 0]), float(s[i, 1]), z_start))
            p1.append((float(g[i, 0]), float(g[i, 1]), z_goal))
            cols.append((1.0, 0.85, 0.1, 0.9))   # yellow line
            widths.append(3.0)

        self._draw.clear_lines()
        self._draw.draw_lines(p0, p1, cols, widths)

    def clear(self):
        if self._ok:
            self._draw.clear_lines()


# =============================================================================
# HL POLICY LOADER (actor only — no critic needed for inference)
# =============================================================================
class _HLPolicyLoader(nn.Module):
    """
    Loads the trained HL ActorCritic actor from an RSL-RL checkpoint.
    Network: [obs_dim→128→64→32→act_dim]  (matches train_nav.py config)
    Applies obs normalisation if saved.
    """

    def __init__(self, checkpoint_path: str, obs_dim: int = 5,
                 act_dim: int = 2, hidden: tuple = (128, 64, 32)):
        super().__init__()
        layers, in_dim = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim)]
        self.actor = nn.Sequential(*layers)
        self.register_buffer("_norm_mean", torch.zeros(obs_dim))
        self.register_buffer("_norm_var",  torch.ones(obs_dim))
        self._has_normalizer = False
        self._load(checkpoint_path)

    def _load(self, path: str):
        print(f"\n[HL] Loading trained policy from: {path}")
        ckpt     = torch.load(path, map_location="cpu", weights_only=False)
        state    = ckpt.get("policy_state_dict", ckpt)

        # RSL-RL keys: "actor.0.weight", "actor.0.bias", ...
        remapped = {k[len("actor."):]: v
                    for k, v in state.items() if k.startswith("actor.")}
        if remapped:
            self.actor.load_state_dict(remapped, strict=True)
            print(f"[HL] Actor loaded  — {len(remapped)} tensors")
        else:
            print("[HL] WARNING: No actor.* keys found — check checkpoint format")

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
            print(f"[HL] Normalizer   — mean={self._norm_mean.numpy()}")
        else:
            print("[HL] No normalizer_state — raw obs used")

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [N,5] → raw action [N,2] in PPO space (before rescaling)"""
        if self._has_normalizer:
            std = (self._norm_var + 1e-8).sqrt().clamp(min=1e-5)
            obs = (obs - self._norm_mean) / std
        return self.actor(obs)


# =============================================================================
# MAIN PLAY LOOP
# =============================================================================
def main():
    device = getattr(args_cli, "device", "cuda:0") or "cuda:0"
    n      = args_cli.num_envs

    print(f"\n{'='*70}")
    print(f"[NAV PLAY]  num_envs={n}")
    print(f"  LL: {args_cli.ll_checkpoint}")
    print(f"  HL: {args_cli.hl_checkpoint}")
    print(f"  Viewport: 🟢 start  🔴 goal  ── yellow line A→B")
    print(f"{'='*70}\n")

    # ── Build nav env cfg ──────────────────────────────────────────────────
    env_cfg                  = Go1NavEnvCfg()
    env_cfg.num_envs         = n
    env_cfg.scene.num_envs   = n
    env_cfg.env_spacing      = 14.0      # more space between envs for visibility
    env_cfg.ll_checkpoint    = args_cli.ll_checkpoint
    env_cfg.episode_length_s = args_cli.episode_length
    env_cfg.goal_dist_min    = args_cli.goal_dist_min
    env_cfg.goal_dist_max    = args_cli.goal_dist_max
    env_cfg.sim.device       = device

    env = gym.make("Isaac-Go1-Nav-v0", cfg=env_cfg)
    nav_env: Go1NavEnv = env.unwrapped   # direct access to our env

    # ── Load trained HL policy ─────────────────────────────────────────────
    hl_policy = _HLPolicyLoader(
        checkpoint_path = args_cli.hl_checkpoint,
        obs_dim         = 5,
        act_dim         = 2,
        hidden          = (128, 64, 32),
    ).to(device)
    hl_policy.eval()

    # ── Line drawer ────────────────────────────────────────────────────────
    line_drawer = _LineDrawer()

    # ── Reset env ──────────────────────────────────────────────────────────
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]   # [N, 5]

    print(f"\n[PLAY] Running... Close the Isaac Sim window to stop.\n")

    step          = 0
    ep_rewards    = torch.zeros(n, device=device)
    ep_lengths    = torch.zeros(n, device=device)
    ep_count      = 0

    while simulation_app.is_running():
        step += 1

        # ── HL policy inference ────────────────────────────────────────────
        with torch.no_grad():
            raw_action = hl_policy(obs.to(device))   # [N, 2] PPO space

        # ── Update markers + connecting lines ──────────────────────────────
        # Spheres are updated inside nav_env._get_observations() each step.
        # Lines are updated here with the latest start/goal positions.
        line_drawer.update(
            nav_env._start_pos_w,
            nav_env._goal_pos_w,
            z_start = 0.3,
            z_goal  = 0.5,
        )

        # ── Step environment ───────────────────────────────────────────────
        # raw_action is [N,2]; the env's _pre_physics_step rescales it
        # from [-1,1] to [vx_lo..hi, wz_lo..hi] with vy=0 forced.
        obs_dict, rewards, terminated, truncated, info = env.step(raw_action)
        obs = obs_dict["policy"]

        ep_rewards  += rewards
        ep_lengths  += 1
        done_mask    = terminated | truncated

        # ── Episode summary on terminal ────────────────────────────────────
        if done_mask.any():
            for i in done_mask.nonzero(as_tuple=True)[0]:
                idx      = i.item()
                start    = nav_env._start_pos_w[idx].cpu().numpy()
                goal     = nav_env._goal_pos_w[idx].cpu().numpy()
                pos      = nav_env._robot.data.root_pos_w[idx, :2].cpu().numpy()
                height   = nav_env._robot.data.root_pos_w[idx, 2].item()
                dist_rem = float(np.linalg.norm(goal - pos))
                dist_ini = float(np.linalg.norm(goal - start))
                covered  = dist_ini - dist_rem

                if height < 0.25:
                    outcome = "FELL ✗"
                elif dist_rem < env_cfg.success_radius:
                    outcome = "SUCCESS ✓"
                else:
                    outcome = "TIMEOUT"

                ep_count += 1
                print(f"[Env {idx}]  ep#{ep_count:03d}  "
                      f"steps={ep_lengths[idx]:.0f}  "
                      f"reward={ep_rewards[idx]:.1f}  "
                      f"{outcome}  "
                      f"covered={covered:+.2f}m / {dist_ini:.2f}m total  "
                      f"🟢({start[0]:+.1f},{start[1]:+.1f})"
                      f"→🔴({goal[0]:+.1f},{goal[1]:+.1f})")

            ep_rewards[done_mask] = 0.0
            ep_lengths[done_mask] = 0.0

        # ── Periodic status print ──────────────────────────────────────────
        if step % 100 == 0:
            h   = nav_env._robot.data.root_pos_w[:, 2]
            lv  = nav_env._robot.data.root_lin_vel_b[:, 0]
            rp  = nav_env._robot.data.root_pos_w[:, :2]
            d   = torch.norm(nav_env._goal_pos_w - rp, dim=-1)

            # Current HL cmd for env 0
            cmd = nav_env._current_hl_cmd[0].cpu().numpy()

            print(f"[step {step:5d}]  "
                  f"height={h.mean():.3f}m  "
                  f"vx={lv.mean():.3f}m/s  "
                  f"dist_to_goal={d.mean():.2f}m  "
                  f"HL_cmd=[vx={cmd[0]:.2f}, wz={cmd[2]:.2f}]")

    line_drawer.clear()
    env.close()
    print("[PLAY] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
