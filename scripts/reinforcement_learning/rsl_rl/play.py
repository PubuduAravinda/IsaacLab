# play.py - Fixed for HIMLoco with embedded CustomGo1Runner

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pre-trained checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
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
import torch.nn as nn
import numpy as np

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import MLP

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class CustomPPO(PPO):
    """Minimal CustomPPO for loading checkpoints."""

    def __init__(self, original_env, **kwargs):
        super().__init__(**kwargs)
        self.original_env = original_env
        self.momentum = 0.99


class CustomGo1Runner(OnPolicyRunner):
    """Custom runner for HIMLoco checkpoints with SimpleActorCritic."""

    def __init__(self, env, train_cfg, log_dir=None, device="cuda:0"):
        # Unwrap to get original env
        self.original_env = env.unwrapped

        # Enable training mode if available
        if hasattr(self.original_env, 'training'):
            self.original_env.training = False  # Inference mode for play

        # Get dims
        obs_dim = env.observation_space.shape[-1]
        num_actions = env.action_space.shape[-1]

        print(f"[INFO] Play mode - Observation dim: {obs_dim}, Action dim: {num_actions}")

        # Create SimpleActorCritic (same as in train.py)
        class SimpleActorCritic(nn.Module):
            def __init__(self, num_obs, num_actions, actor_hidden_dims, critic_hidden_dims, activation, init_noise_std):
                super().__init__()
                self.actor = MLP(num_obs, num_actions, actor_hidden_dims, activation)
                self.critic = MLP(num_obs, 1, critic_hidden_dims, activation)
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
                self.distribution = None
                self.is_recurrent = False

            def reset(self, dones=None):
                pass

            def update_normalization(self, obs):
                pass

            def act(self, obs, masks=None, hidden_states=None):
                if hasattr(obs, 'get'):
                    obs = obs.get("policy", obs)
                elif isinstance(obs, dict):
                    obs = obs["policy"]
                self.update_distribution(obs)
                return self.distribution.sample()

            def get_actions_log_prob(self, actions):
                return self.distribution.log_prob(actions).sum(dim=-1)

            def update_distribution(self, obs, masks=None, hidden_states=None):
                if hasattr(obs, 'get'):
                    obs = obs.get("policy", obs)
                elif isinstance(obs, dict):
                    obs = obs["policy"]
                mean = self.actor(obs)
                self.distribution = torch.distributions.Normal(mean, self.std)
                self.action_mean = mean
                self.action_std = self.std
                self.entropy = self.distribution.entropy().sum(dim=-1)

            def evaluate(self, obs, actions=None, masks=None, hidden_states=None):
                if hasattr(obs, 'get'):
                    obs = obs.get("policy", obs)
                elif isinstance(obs, dict):
                    obs = obs["policy"]

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

        print(f"[INFO] SimpleActorCritic created for play mode")

        # Initialize parent runner
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)

        # Replace alg with CustomPPO
        original_alg = self.alg
        self.alg = CustomPPO(original_env=self.original_env, policy=actor_critic, device=device)

        # Copy attributes from parent's PPO
        for attr in ['storage', 'transition', 'optimizer', 'learning_rate', 'num_learning_epochs',
                     'num_mini_batches', 'clip_param', 'gamma', 'lam', 'value_loss_coef',
                     'entropy_coef', 'max_grad_norm', 'use_clipped_value_loss', 'schedule', 'desired_kl']:
            if hasattr(original_alg, attr):
                setattr(self.alg, attr, getattr(original_alg, attr))

        print("[INFO] CustomGo1Runner initialized for play mode")

        # Store the checkpoint path for later HIMLoco loading
        self._resume_path = None

    def load(self, path, load_optimizer=True):
        """Override load to restore HIMLoco components."""
        print(f"[DEBUG] CustomGo1Runner.load() called with path: {path}")
        self._resume_path = path

        # Call parent load first
        super().load(path, load_optimizer)

        # Now load HIMLoco components
        self._load_himloco_components(path)

    def _load_himloco_components(self, path):
        """Load HIMLoco encoder and prototypes."""
        himloco_path = path.replace('.pt', '_himloco.pt')
        print(f"[DEBUG] Looking for HIMLoco checkpoint at: {himloco_path}")
        print(f"[DEBUG] File exists: {os.path.exists(himloco_path)}")

        if os.path.exists(himloco_path):
            print(f"[INFO] Loading HIMLoco components...")
            himloco_dict = torch.load(himloco_path, map_location=self.device)
            if hasattr(self.original_env, 'encoder_source'):
                self.original_env.encoder_source.load_state_dict(himloco_dict['encoder_source'])
                self.original_env.encoder_target.load_state_dict(himloco_dict['encoder_target'])
                self.original_env.prototypes.data = himloco_dict['prototypes']
                # DON'T load obs_history - it needs to match current num_envs
                # self.original_env.obs_history will be initialized correctly by the env
                print(f"[INFO] ✓ Successfully loaded HIMLoco encoder and prototypes!")
            else:
                print(f"[WARNING] Environment doesn't have encoder_source")
        else:
            print(f"[WARNING] HIMLoco checkpoint not found at: {himloco_path}")
            print(f"[WARNING] Encoder will use RANDOM weights - robot behavior will be poor!")
            checkpoint_dir = os.path.dirname(himloco_path)
            if os.path.exists(checkpoint_dir):
                files = [f for f in os.listdir(checkpoint_dir) if 'himloco' in f.lower()]
                print(f"[DEBUG] HIMLoco files in directory: {files}")

    def load(self, path, load_optimizer=True):
        """Override load to handle action_std -> std parameter name mismatch."""
        # Load checkpoint
        loaded_dict = torch.load(path, map_location=self.device)

        # Fix parameter name mismatch: action_std -> std
        if "model_state_dict" in loaded_dict:
            state_dict = loaded_dict["model_state_dict"]
            if "action_std" in state_dict:
                print("[INFO] Renaming 'action_std' to 'std' for compatibility")
                state_dict["std"] = state_dict.pop("action_std")
                # Save the modified dict back
                torch.save(loaded_dict, path)

        # Now call parent load which will load the modified state_dict
        return super().load(path, load_optimizer)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Get checkpoint path
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    print(f"[INFO] Loading checkpoint from: {resume_path}")

    # Check if Go1 HIMLoco task (match any task with 'go1' in the name)
    is_go1_task = "go1" in args_cli.task.lower()
    print(f"[DEBUG] Task: {args_cli.task}, is_go1_task: {is_go1_task}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner
    if is_go1_task:
        print("[INFO] Using CustomGo1Runner for HIMLoco checkpoint")
        runner = CustomGo1Runner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        print("[INFO] Using standard OnPolicyRunner")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)
    print("[INFO] Checkpoint loaded successfully")

    # Manually load HIMLoco components for CustomGo1Runner
    if is_go1_task and hasattr(runner, '_load_himloco_components'):
        print("[INFO] Loading HIMLoco encoder components...")
        runner._load_himloco_components(resume_path)

    # Get policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Extract policy network
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Extract normalizer if exists
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    else:
        normalizer = None

    # Export policy
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    print(f"[INFO] Policy exported to: {export_model_dir}")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting playback...")
    # Simulate
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()
    print("[INFO] Playback completed")


if __name__ == "__main__":
    main()
    simulation_app.close()