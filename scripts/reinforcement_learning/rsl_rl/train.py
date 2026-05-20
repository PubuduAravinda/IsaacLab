# train.py
# Standard OnPolicyRunner (no custom classes).
# Monkey-patched save/load for obs normalisation stats.
# Go1 task detection rebuilds cfg from scratch — no runtime scene patching.
#
# CHANGES vs original:
#   + --checkpoint arg: warm-start fine-tuning from any .pt file
#   + is_sparse detection: Isaac-Go1-Sparse-Direct-v0 uses Go1SparsePPORunnerCfg
#   + [WARM-START] block: loads weights + normalizer, prints noise std to verify
#   + removed duplicate --distributed (was conflicting with cli_args)

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int,  default=200)
parser.add_argument("--video_interval", type=int,  default=2000)
parser.add_argument("--num_envs",       type=int,  default=None)
parser.add_argument("--task",           type=str,  default=None)
parser.add_argument("--agent",          type=str,  default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed",           type=int,  default=None)
parser.add_argument("--max_iterations", type=int,  default=None)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
# --checkpoint, --resume, --load_run, --distributed etc. come from cli_args below
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.direct.go1.go1_env_cfg import Go1FlatEnvCfg
from isaaclab_tasks.direct.go1.agents.rsl_rl_ppo_cfg import (
    Go1RslRlPpoCfg,
    Go1SparsePPORunnerCfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    env_cfg.seed       = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(args_cli, "distributed") and args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device   = f"cuda:{app_launcher.local_rank}"
        env_cfg.seed = agent_cfg.seed = agent_cfg.seed + app_launcher.local_rank

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging to: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # ── Go1 task: rebuild cfg from scratch ───────────────────────────────────
    # @configclass bakes scene(num_envs) at class definition time.
    # Detect sparse vs flat to select the correct PPO runner cfg.
    is_go1    = "go1"    in args_cli.task.lower()
    is_sparse = "sparse" in args_cli.task.lower()

    if is_go1:
        if is_sparse:
            env_cfg   = Go1FlatEnvCfg()          # same physics/scene as flat
            agent_cfg = Go1SparsePPORunnerCfg()  # 10k iters, sparse experiment name
            print("[INFO] Go1 SPARSE task — Go1FlatEnvCfg + Go1SparsePPORunnerCfg")
        else:
            env_cfg   = Go1FlatEnvCfg()
            agent_cfg = Go1RslRlPpoCfg()
            print("[INFO] Go1 FLAT task — Go1FlatEnvCfg + Go1RslRlPpoCfg")

        # Re-apply CLI overrides after cfg rebuild
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
            agent_cfg.device   = args_cli.device
        if args_cli.max_iterations is not None:
            agent_cfg.max_iterations = args_cli.max_iterations
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs

        print(f"[INFO] num_envs={env_cfg.scene.num_envs}  "
              f"max_iters={agent_cfg.max_iterations}  "
              f"device={agent_cfg.device}")

    env_cfg.log_dir = log_dir

    # ── Create environment ────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, **{
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        })

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Standard runner ───────────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(),
                            log_dir=log_dir, device=agent_cfg.device)

    # ── Monkey-patch save: include obs normalisation stats ────────────────────
    def custom_save(self, path, infos=None):
        normalizer_state = None
        if hasattr(self.alg, "actor_critic") and hasattr(self.alg.actor_critic, "actor_obs_normalizer"):
            norm = self.alg.actor_critic.actor_obs_normalizer
            normalizer_state = norm.state_dict()
        elif hasattr(self.alg, "policy") and hasattr(self.alg.policy, "actor_obs_normalizer"):
            norm = self.alg.policy.actor_obs_normalizer
            normalizer_state = norm.state_dict()

        if normalizer_state is not None:
            for k, v in normalizer_state.items():
                if v.numel() <= 6:
                    print(f"[SAVE] normalizer {k}: {v.cpu().numpy()}")
                else:
                    print(f"[SAVE] normalizer {k}[:5]: {v[:5].cpu().numpy()}")

        state = {
            "policy_state_dict":    self.alg.actor_critic.state_dict()
                                    if hasattr(self.alg, "actor_critic")
                                    else self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict()
                                    if hasattr(self.alg, "optimizer") else None,
            "normalizer_state":     normalizer_state,
        }
        torch.save(state, path)
        print(f"[SAVE] {path}")
        if infos is not None:
            torch.save(infos, path.replace(".pt", "_infos.pt"))

    runner.save = custom_save.__get__(runner)

    # ── Monkey-patch load: restore obs normalisation stats ────────────────────
    def custom_load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device)

        # Support both actor_critic and policy attribute names (RSL-RL version diff)
        ac = getattr(self.alg, "actor_critic", None) or getattr(self.alg, "policy", None)
        if ac is None:
            print(f"[LOAD] ERROR: cannot find actor_critic or policy on runner.alg")
            return

        ac.load_state_dict(ckpt["policy_state_dict"])

        if load_optimizer and ckpt.get("optimizer_state_dict") is not None:
            self.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "normalizer_state" in ckpt and ckpt["normalizer_state"] is not None:
            if hasattr(ac, "actor_obs_normalizer"):
                ac.actor_obs_normalizer.load_state_dict(ckpt["normalizer_state"])
                print(f"[LOAD] ✓ Normalizer restored from {path}")
            else:
                print(f"[LOAD] WARNING: checkpoint has normalizer_state "
                      f"but actor_critic has no actor_obs_normalizer attr")
        else:
            print(f"[LOAD] WARNING: No normalizer_state in checkpoint — "
                  f"obs normalizer stats reset to default")

        print(f"[LOAD] ✓ Policy weights restored from {path}")

    runner.load = custom_load.__get__(runner)

    # ── Resume from a previous run (--resume flag, unchanged) ────────────────
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(resume_path)

    # ── Warm-start from checkpoint (--checkpoint flag) ────────────────────────
    # Separate from --resume: loads weights+normalizer into a NEW run's log dir.
    # Use this for fine-tuning (e.g. sparse reward experiment from flat policy).
    if args_cli.checkpoint is not None:
        print(f"\n{'='*60}")
        print(f"[WARM-START] Loading checkpoint:")
        print(f"  {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)

        # Verify noise std — if v14 checkpoint loaded correctly, std << 0.10
        ac = getattr(runner.alg, "actor_critic", None) or getattr(runner.alg, "policy", None)
        if ac is not None:
            try:
                std_val = ac.std.mean().item()
                loaded  = std_val < 0.08
                print(f"[WARM-START] Actor noise std = {std_val:.4f}  "
                      f"({'✓ checkpoint loaded' if loaded else '⚠ WARNING: std=0.10 = fresh weights — checkpoint may not have loaded'})")
                if not loaded:
                    print(f"[WARM-START] Expected std ~0.03–0.05 for model_35000.pt")
                    print(f"[WARM-START] Check that checkpoint was saved with custom_save "
                          f"(needs 'policy_state_dict' key, not raw state_dict)")
                    # Try loading as raw state dict (older checkpoint format)
                    ckpt_raw = torch.load(args_cli.checkpoint, map_location=runner.device)
                    if isinstance(ckpt_raw, dict) and "policy_state_dict" not in ckpt_raw:
                        print(f"[WARM-START] Keys in checkpoint: {list(ckpt_raw.keys())}")
                        print(f"[WARM-START] Checkpoint was NOT saved with custom_save — "
                              f"it may be a raw state_dict. Attempting direct load...")
                        ac.load_state_dict(ckpt_raw)
                        std_val2 = ac.std.mean().item()
                        print(f"[WARM-START] After direct load: std = {std_val2:.4f}  "
                              f"({'✓ loaded' if std_val2 < 0.08 else '⚠ still 0.10 — architecture mismatch?'})")
            except AttributeError:
                print(f"[WARM-START] Could not read .std — check runner.alg attribute names")
        print(f"{'='*60}\n")
    # ─────────────────────────────────────────────────────────────────────────

    runner.add_git_repo_to_log(__file__)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"),   env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[INFO] Training for {agent_cfg.max_iterations} iterations ...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)

    env.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()