# train_nav.py — Train High-Level Navigation Policy v8
#
# CHANGES vs v7:
#   entropy_coef = 0.008  (0.005 → collapse, 0.02 → diverge, 0.008 balanced)
#   max_grad_norm = 0.5   (unchanged — already clips large gradients)
#   init_noise_std = 0.3  (unchanged — gentle exploration)
#   Adds divergence guard callback: warns if noise_std > 2.0
#
# FRESH START PROCEDURE:
#   1. Set HL_POLICY_ACTIVE = False in go1_nav_env.py
#   2. Run this script → see "[LL NORMALIZER STD]" → note safe wz range
#   3. Update wz range in go1_nav_env_cfg.py if needed
#   4. Set HL_POLICY_ACTIVE = True in go1_nav_env.py
#   5. Run this script fresh (NO --resume, no HL checkpoint loading)

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Go1 HL Navigation Policy v8.")
parser.add_argument("--task",           type=str,   default="Isaac-Go1-Nav-v0")
parser.add_argument("--ll_checkpoint",  type=str,   required=True)
parser.add_argument("--num_envs",       type=int,   default=4096)
parser.add_argument("--max_iterations", type=int,   default=20000)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int,   default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after Isaac Sim launch."""

import os
import torch
import gymnasium as gym
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml

from isaaclab_tasks.direct.go1.go1_nav_env     import Go1NavEnv       # noqa
from isaaclab_tasks.direct.go1.go1_nav_env_cfg import Go1NavEnvCfg
from isaaclab_tasks.direct.go1.agents.rsl_rl_ppo_cfg import Go1RslRlPpoCfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark        = False


def main():
    device = getattr(args_cli, "device", "cuda:0") or "cuda:0"

    env_cfg                = Go1NavEnvCfg()
    env_cfg.num_envs       = args_cli.num_envs
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.ll_checkpoint  = args_cli.ll_checkpoint
    env_cfg.sim.device     = device

    print(f"\n{'='*70}")
    print(f"[NAV TRAIN v8]  LL: {args_cli.ll_checkpoint}")
    print(f"  num_envs={env_cfg.num_envs}  max_iter={args_cli.max_iterations}")
    print(f"  action: [vx∈[0.0,0.8], wz∈[-0.10,0.10]]  vy=0 always")
    print(f"  entropy_coef=0.008  init_noise_std=0.3")
    print(f"  ★ See [LL NORMALIZER STD] at startup to verify wz range")
    print(f"{'='*70}\n")

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", "go1_nav_hl"))
    log_dir  = os.path.join(log_root,
                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, **{
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % 2000 == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        })
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # Build HL PPO cfg from existing working cfg (gets obs_groups etc)
    agent_cfg = Go1RslRlPpoCfg()
    agent_cfg.experiment_name   = "go1_nav_hl"
    agent_cfg.max_iterations    = args_cli.max_iterations
    agent_cfg.save_interval     = 200
    agent_cfg.num_steps_per_env = 16
    agent_cfg.seed              = args_cli.seed
    agent_cfg.device            = device

    # Small network for 5D→2D nav problem
    agent_cfg.policy.actor_hidden_dims  = [128, 64, 32]
    agent_cfg.policy.critic_hidden_dims = [128, 64, 32]
    agent_cfg.policy.init_noise_std     = 0.3

    # Balanced entropy: 0.005 → collapse, 0.02 → diverge, 0.008 → stable
    agent_cfg.algorithm.entropy_coef  = 0.008
    agent_cfg.algorithm.learning_rate = 3e-4
    agent_cfg.algorithm.max_grad_norm = 0.5   # clips large gradients

    agent_cfg_dict               = agent_cfg.to_dict()
    agent_cfg_dict["clip_actions"] = 1.0

    runner = OnPolicyRunner(
        env, agent_cfg_dict, log_dir=log_dir, device=device)

    # ── Save hook ─────────────────────────────────────────────────────────
    def custom_save(self, path, infos=None):
        normalizer_state = None
        if hasattr(self.alg.policy, "actor_obs_normalizer"):
            normalizer_state = (
                self.alg.policy.actor_obs_normalizer.state_dict())
        state = {
            "policy_state_dict":    self.alg.policy.state_dict(),
            "optimizer_state_dict": (self.alg.optimizer.state_dict()
                                     if hasattr(self.alg, "optimizer")
                                     else None),
            "normalizer_state":     normalizer_state,
        }
        torch.save(state, path)
        print(f"[SAVE] {path}")
        if infos is not None:
            torch.save(infos, path.replace(".pt", "_infos.pt"))

    runner.save = custom_save.__get__(runner)

    # ── Divergence guard ──────────────────────────────────────────────────
    # Wraps runner.learn to print warnings if noise_std explodes.
    # If noise_std > 2.0 before iter 50, training is diverging — stop and
    # increase entropy_coef or reduce learning_rate.
    _orig_learn = runner.learn

    def _guarded_learn(num_learning_iterations, init_at_random_ep_len=False):
        import functools
        orig_log = runner.log
        @functools.wraps(orig_log)
        def _log_with_guard(locs, width=80, pad=35):
            orig_log(locs, width, pad)
            # Check for divergence
            if hasattr(runner.alg.policy, "std"):
                noise_std = runner.alg.policy.std.mean().item()
                if noise_std > 2.0:
                    it = locs.get("it", 0)
                    print(f"\n⚠ [DIVERGENCE WARNING] iter={it}  "
                          f"noise_std={noise_std:.2f} > 2.0")
                    print(f"  Policy is diverging! Options:")
                    print(f"  1. Reduce entropy_coef: 0.008 → 0.005")
                    print(f"  2. Reduce learning_rate: 3e-4 → 1e-4")
                    print(f"  3. Check wz range (see CYCLE DEBUG norm_wz)")
                    print(f"  Action: continuing but monitor closely.\n")
        runner.log = _log_with_guard
        return _orig_learn(num_learning_iterations, init_at_random_ep_len)

    runner.learn = _guarded_learn

    dump_yaml(os.path.join(log_dir, "params", "nav_env.yaml"),   env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "nav_agent.yaml"), agent_cfg)

    print(f"[INFO] Logging to   : {log_dir}")
    print(f"[INFO] Training for : {args_cli.max_iterations} iterations\n")

    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )
    env.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()