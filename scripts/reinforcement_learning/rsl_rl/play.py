# play.py
# Standard OnPolicyRunner. Loads obs normalisation stats from checkpoint.
# Exports policy as JIT + ONNX for real Go1 deployment.

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a trained Go1 policy.")
parser.add_argument("--video",        action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs",     type=int, default=None)
parser.add_argument("--task",         type=str, default=None)
parser.add_argument("--agent",        type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed",         type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time",    action="store_true", default=False)
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
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg, RslRlVecEnvWrapper,
    export_policy_as_jit, export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.direct.go1.go1_env_cfg import Go1FlatEnvCfg
from isaaclab_tasks.direct.go1.agents.rsl_rl_ppo_cfg import Go1RslRlPpoCfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── Always rebuild Go1 cfg for play ──────────────────────────────────────
    is_go1 = "go1" in args_cli.task.lower()
    if is_go1:
        env_cfg   = Go1FlatEnvCfg()
        agent_cfg = Go1RslRlPpoCfg()
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
            agent_cfg.device   = args_cli.device

    # For play: 1 env is enough. Change num_envs class field in go1_env_cfg.py
    # before running play if you want more. CLI --num_envs also works here
    # because we're just overriding scene.num_envs after fresh construction.
    if args_cli.num_envs is not None:
        env_cfg.num_envs        = args_cli.num_envs
        env_cfg.scene.num_envs  = args_cli.num_envs
        env_cfg.scene.env_spacing = 4.0
        print(f"[PLAY] num_envs overridden to {args_cli.num_envs}")

    # ── Checkpoint path ───────────────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    print(f"[PLAY] checkpoint : {resume_path}")
    print(f"[PLAY] num_envs   : {env_cfg.scene.num_envs}")

    # ── Create environment ────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, **{
            "video_folder":   os.path.join(log_dir, "videos", "play"),
            "step_trigger":   lambda step: step == 0,
            "video_length":   args_cli.video_length,
            "disable_logger": True,
        })

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Runner + custom load ──────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    def custom_load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device)
        self.alg.policy.load_state_dict(ckpt["policy_state_dict"])
        # Restore normalizer — critical for correct inference on real Go1
        if "normalizer_state" in ckpt and ckpt["normalizer_state"] is not None:
            if hasattr(self.alg.policy, "actor_obs_normalizer"):
                self.alg.policy.actor_obs_normalizer.load_state_dict(ckpt["normalizer_state"])
                print(f"[LOAD] Normalizer restored")
                for k, v in ckpt["normalizer_state"].items():
                    if v.numel() <= 6:
                        print(f"  {k}: {v.cpu().numpy()}")
                    else:
                        print(f"  {k}[:5]: {v.flatten()[:5].cpu().numpy()}")
        else:
            print("[LOAD] WARNING: No normalizer_state in checkpoint!")
            print("  Policy will use untrained normalizer — may behave incorrectly on real robot")
        print(f"[LOAD] Policy restored from {path}")

    runner.load = custom_load.__get__(runner)
    runner.load(resume_path)
    print("[INFO] Checkpoint loaded.")

    # ── Export policy for real robot ──────────────────────────────────────────
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_dir = os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn,  normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print(f"[INFO] Exported to: {export_dir}")

    # ── Inference loop ────────────────────────────────────────────────────────
    policy   = runner.get_inference_policy(device=env.unwrapped.device)
    dt       = env.unwrapped.step_dt
    obs      = env.get_observations()
    timestep = 0

    print("[INFO] Running ...")
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions          = policy(obs)
            obs, _, _, _     = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep = dt - (time.time() - t0)
        if args_cli.real_time and sleep > 0:
            time.sleep(sleep)

    env.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()