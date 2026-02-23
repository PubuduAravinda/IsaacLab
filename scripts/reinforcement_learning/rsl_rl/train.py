# train.py - Fixed version for HIMLoco replication with proper PPO initialization
# (Cleaned: commented out noisy PPO / debug prints)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

import torch.nn as nn

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# Import PPO directly from rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic, MLP   # ← MLP added here

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Custom Go1 imports
from isaaclab_tasks.direct.go1.go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg
from isaaclab_tasks.direct.go1.agents.rsl_rl_ppo_cfg import Go1RslRlPpoCfg

import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class CustomPPO(PPO):
    """Custom PPO that adds HIMLoco auxiliary losses."""

    def __init__(self, original_env, **kwargs):
        super().__init__(**kwargs)
        self.original_env = original_env
        self.momentum = 0.99
        # print("[INFO] CustomPPO initialized")  # commented

        # Initialize HIMLoco optimizer for encoder and prototypes
        him_params = []
        if hasattr(original_env, 'encoder_source'):
            him_params.extend(list(original_env.encoder_source.parameters()))
        if hasattr(original_env, 'prototypes'):
            him_params.append(original_env.prototypes)

        if him_params:
            self.him_optimizer = torch.optim.Adam(him_params, lr=1e-3, eps=1e-8)
            # print("[INFO] HIMLoco optimizer initialized with encoder_source and prototypes")  # commented
        else:
            self.him_optimizer = None
            # print("[WARNING] No encoder_source or prototypes found - HIMLoco aux losses disabled")  # commented

    def update(self):
        """Override update to include HIMLoco auxiliary losses."""

        # ─── Commented out all PPO debug prints ─────────────────────────────────────
        # if hasattr(self, 'storage') and self.storage.step > 0:
        #     print(f"\n[PPO DEBUG] Update called")
        #     adv = self.storage.advantages
        #     print(f"  Advantages: min={adv.min():.6f}, max={adv.max():.6f}, mean={adv.mean():.6f}, std={adv.std():.6f}")
        #     adv_unnorm = self.storage.returns - self.storage.values
        #     print(f"  Advantages (unnormalized): min={adv_unnorm.min():.6f}, max={adv_unnorm.max():.6f}, std={adv_unnorm.std():.6f}")
        #     if adv.std() < 1e-6:
        #         print(f"  [WARNING] Advantage std too small! Policy won't update!")

        # Update HIMLoco components if available
        if self.him_optimizer is not None and hasattr(self.original_env, '_get_aux_losses'):
            try:
                aux_losses = self.original_env._get_aux_losses()

                if aux_losses:
                    self.him_optimizer.zero_grad()
                    total_aux = sum(aux_losses.values())

                    if torch.isnan(total_aux) or torch.isinf(total_aux):
                        # print(f"[WARNING] NaN/Inf detected in aux losses, skipping this update")  # commented
                        pass
                    else:
                        total_aux.backward()
                        him_params = []
                        if hasattr(self.original_env, 'encoder_source'):
                            him_params.extend(list(self.original_env.encoder_source.parameters()))
                        if hasattr(self.original_env, 'prototypes'):
                            him_params.append(self.original_env.prototypes)
                        torch.nn.utils.clip_grad_norm_(him_params, 10.0)
                        self.him_optimizer.step()

                        if hasattr(self.original_env, 'encoder_target') and hasattr(self.original_env, 'encoder_source'):
                            with torch.no_grad():
                                for t_p, s_p in zip(
                                    self.original_env.encoder_target.parameters(),
                                    self.original_env.encoder_source.parameters()
                                ):
                                    t_p.data = self.momentum * t_p.data + (1.0 - self.momentum) * s_p.data

                        # if hasattr(self.original_env, '_global_step'):
                        #     global_step = self.original_env._global_step
                        #     if global_step % 150 == 0 and global_step > 0:
                        #         loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in aux_losses.items()])
                        #         print(f"[HIMLoco AUX] step {global_step} | {loss_str}")  # commented
            except Exception as e:
                # print(f"[WARNING] Error in auxiliary loss update: {e}")  # commented
                # import traceback
                # traceback.print_exc()  # commented
                pass

        # Call parent PPO update silently
        loss_dict = super().update()

        # ─── Commented out post-update debug ────────────────────────────────────────
        # print(f"[PPO DEBUG] Update complete")
        # print(f"  Loss dict: {[(k, f'{v:.6f}') for k, v in loss_dict.items()]}")

        # if hasattr(self, 'policy'):
        #     policy = self.policy
        # elif hasattr(self, 'actor_critic'):
        #     policy = self.actor_critic
        # else:
        #     policy = None

        # if policy is not None:
        #     total_grad_norm = 0.0
        #     for p in policy.parameters():
        #         if p.grad is not None:
        #             total_grad_norm += p.grad.norm().item() ** 2
        #     total_grad_norm = total_grad_norm ** 0.5
        #     print(f"  Policy gradient norm: {total_grad_norm:.6f}")
        #     if total_grad_norm < 1e-6:
        #         print(f"  [WARNING] Policy gradients near zero! No learning happening!")

        return loss_dict


class CustomGo1Runner(OnPolicyRunner):
    """Custom runner with HIMLoco support — only Go1Env prints visible."""

    def __init__(self, env, train_cfg, log_dir=None, device="cuda:0"):
        self.original_env = env.unwrapped

        obs_dim = env.observation_space.shape[-1]
        num_actions = env.action_space.shape[-1]

        # Simple ActorCritic (same as before)
        class SimpleActorCritic(nn.Module):
            def __init__(self, num_obs, num_actions, actor_hidden_dims, critic_hidden_dims, activation, init_noise_std):
                super().__init__()
                assert num_actions == 12, f"Expected 12 actions for Go1, got {num_actions}"

                self.actor = MLP(num_obs, num_actions, actor_hidden_dims, activation)
                self.critic = MLP(num_obs, 1, critic_hidden_dims, activation)

                # Force std to [12]
                self.std = nn.Parameter(
                    torch.full((num_actions,), float(init_noise_std), dtype=torch.float32, device="cuda")
                )

                self.distribution = None
                self.is_recurrent = False

            def reset(self, dones=None):
                pass

            def update_normalization(self, obs):
                pass

            def act(self, obs, masks=None, hidden_states=None):
                self.update_distribution(obs)
                actions = self.distribution.sample()
                # Force actions to [batch, 12]
                return actions

            def get_actions_log_prob(self, actions):
                return self.distribution.log_prob(actions).sum(dim=-1)

            def update_distribution(self, obs, masks=None, hidden_states=None):
                if hasattr(obs, 'get'):
                    obs = obs.get("policy", obs)
                elif isinstance(obs, dict):
                    obs = obs["policy"]

                mean = self.actor(obs)
                # Force mean and std to same shape
                std_expanded = self.std.expand_as(mean)  # [batch, 12]

                self.distribution = torch.distributions.Normal(mean, std_expanded)

                self.action_mean = mean
                self.action_std = std_expanded
                self.entropy = self.distribution.entropy().sum(dim=-1)

            def evaluate(self, obs, actions=None, masks=None, hidden_states=None):
                if hasattr(obs, 'get'):
                    obs = obs.get("policy", obs)
                elif isinstance(obs, dict):
                    obs = obs["policy"]
                # No evaluate prints
                if actions is None:
                    return self.critic(obs)
                value = self.critic(obs)
                self.update_distribution(obs)
                log_prob = self.get_actions_log_prob(actions)
                entropy = self.distribution.entropy().sum(dim=-1)
                return value, log_prob, entropy

            def act_inference(self, obs):
                if hasattr(obs, 'get'):
                    obs = obs.get("policy", obs)
                elif isinstance(obs, dict):
                    obs = obs["policy"]
                return self.actor(obs)

        actor_critic = SimpleActorCritic(
            num_obs=obs_dim,
            num_actions=num_actions,
            actor_hidden_dims=train_cfg["policy"]["actor_hidden_dims"],
            critic_hidden_dims=train_cfg["policy"]["critic_hidden_dims"],
            activation=train_cfg["policy"]["activation"],
            init_noise_std=train_cfg["policy"]["init_noise_std"],
        ).to(device)


        # Initialize parent runner (creates storage, git_status_repos, etc.)
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)

        # Replace alg with CustomPPO
        original_alg = self.alg
        self.alg = CustomPPO(original_env=self.original_env, policy=actor_critic, device=device)

        # Copy parent alg attributes
        for attr in ['storage', 'transition', 'optimizer', 'learning_rate', 'num_learning_epochs',
                     'num_mini_batches', 'clip_param', 'gamma', 'lam', 'value_loss_coef',
                     'entropy_coef', 'max_grad_norm', 'use_clipped_value_loss', 'schedule', 'desired_kl']:
            if hasattr(original_alg, attr):
                setattr(self.alg, attr, getattr(original_alg, attr))

        self.alg.policy = actor_critic
        if hasattr(self.alg, 'actor_critic'):
            self.alg.actor_critic = actor_critic

        # Optimizer for our policy
        learning_rate = train_cfg["algorithm"].get("learning_rate", 1e-3)
        self.alg.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Run parent learn() but suppress its built-in printing."""

        import builtins
        original_print = builtins.print

        # Silent print during parent learn()
        def silent_print(*args, **kwargs):
            # Only allow prints that contain your calibration keywords
            msg = " ".join(str(a) for a in args)
            if any(kw in msg.lower() for kw in ["phase", "kp =", "given target pos", "avg real pos", "avg error", "cycle", "next:"]):
                original_print(*args, **kwargs)  # allow your Go1Env prints
            # else: silence everything else

        builtins.print = silent_print

        try:
            super().learn(num_learning_iterations=num_learning_iterations,
                          init_at_random_ep_len=init_at_random_ep_len)
        finally:
            builtins.print = original_print  # restore normal print

        original_print("[INFO] Training loop finished (calibration mode)")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    # print(f"[INFO] Logging experiment in directory: {log_root_path}")  # commented

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    is_go1_task = "Go1-Direct-v0" in args_cli.task or "go1" in args_cli.task.lower()

    if is_go1_task:
        # print("[INFO] Detected Go1 HIMLoco task - using custom configuration")  # commented
        if "Flat" in args_cli.task or "flat" in args_cli.task.lower():
            env_cfg = Go1FlatEnvCfg()
            # print("[INFO] Using Go1FlatEnvCfg")  # commented
        else:
            env_cfg = Go1RoughEnvCfg()
            # print("[INFO] Using Go1RoughEnvCfg")  # commented
        agent_cfg = Go1RslRlPpoCfg()
        # print("[INFO] Using Go1RslRlPpoCfg")  # commented

    env_cfg.log_dir = log_dir

    # print(f"[INFO] Creating environment: {args_cli.task}")  # commented
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if is_go1_task:
        # print("[INFO] Using CustomGo1Runner with HIMLoco support")  # commented
        runner = CustomGo1Runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        # print("[INFO] Using standard OnPolicyRunner")  # commented
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        # print(f"[INFO] Resuming from checkpoint: {resume_path}")  # commented
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # print(f"[INFO] Starting training for {agent_cfg.max_iterations} iterations")  # commented
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    # print("[INFO] Training completed successfully")  # commented


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()