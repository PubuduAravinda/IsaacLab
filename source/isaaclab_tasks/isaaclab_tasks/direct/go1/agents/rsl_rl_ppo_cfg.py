# rsl_rl_ppo_cfg.py - HIMLoco-accurate RSL_RL config for Go1 (flat & rough)
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go1RslRlPpoCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for RSL-RL PPO runner — tuned for HIMLoco replication."""

    seed = 42
    device = "cuda:0"

    num_steps_per_env = 24                # Critical: ~2s rollout, matches paper
    max_iterations = 10000                 # Long training (adjust as needed)
    save_interval = 100 #500                    # Save every 500 updates
    experiment_name = "go1_himloco"
    run_name = ""                          # Optional suffix
    empirical_normalization = False        # No running stats on obs (paper uses raw)
    policy_obs_normalization = False       # Important for proprioception

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,
        actor_hidden_dims=[512, 256, 128],   # Exact HIMLoco policy MLP
        critic_hidden_dims=[512, 256, 128],  # Same backbone
        activation="elu",
        # CRITICAL: Enable observation normalization to stabilize learning
        actor_obs_normalization=True,
        critic_obs_normalization = True,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,                  # Paper value — better exploration
        num_learning_epochs=5,
        num_mini_batches=4,                  # Good balance with large env count
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,                  # Paper clipping
    )


# Use the same config for both flat and rough (just change task name)
Go1FlatPPORunnerCfg = Go1RslRlPpoCfg
Go1RoughPPORunnerCfg = Go1RslRlPpoCfg