# go2/agents/rsl_rl_ppo_cfg.py
#
# CORRECTED — matches go1_rsl_rl_ppo_cfg.py's proven-stable hyperparameters.
# The previous version of this file was hand-typed separately from Go1's
# ("does NOT inherit from Go1... field names copied to guarantee
# compatibility") and several values differed from Go1's without
# justification, all in the direction that destabilizes PPO entropy:
#
#   init_noise_std:          1.0  -> 0.1   (was 10x Go1's proven start)
#   entropy_coef:            0.005 -> 0.001 (was 5x Go1's proven value —
#                             almost certainly the primary cause of the
#                             action-noise-std runaway seen in training:
#                             10 -> 15 -> 40 -> 42, never recovering,
#                             while task reward barely moved)
#   actor_obs_normalization: unset(False) -> True  (Go1 has this ON;
#                             Go2 was missing it entirely)
#   critic_obs_normalization:unset(False) -> True  (same)
#   max_grad_norm:           1.0 -> 0.5   (Go1's tighter clipping)
#   use_clipped_value_loss:  unset -> True (Go1 has this ON)
#
# Architecture (MLP [512,256,128], ELU) and Go2-specific fields
# (experiment_name, empirical_normalization=False — matches the
# confirmed-empty normalizer_state finding from the save-policy debugging
# session) are UNCHANGED from the working checkpoint you've already
# successfully exported and deployed on real hardware.
#
# *** VALIDATE BEFORE COMMITTING THE FULL 2-3 DAY RUN ***
# Given the cost of being wrong again, run a SHORT sanity check first —
# a few thousand iterations, not all 45000 — and confirm in the debug
# logs that "Mean action noise std" is actually shrinking (not growing)
# before walking away for multiple days:
#   python train.py --task Isaac-Velocity-Rough-Go2-v0 --num_envs 1000 \
#       --max_iterations 3000 --headless
# Expect: init_noise_std=0.1 at iteration 0, and by iteration ~2000-3000
# it should be trending DOWN toward something like 0.05-0.2, not climbing.
# If it's still climbing at 3000 iterations, stop and investigate further
# before committing the multi-day run — don't assume this fix worked
# just because the numbers changed, confirm the actual trend.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2RslRlPpoCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for Go2 flat/rough locomotion — matches Go1's
    proven-stable HIMLoco-derived hyperparameters."""

    seed                  : int   = 42
    device                : str   = "cuda:0"
    num_steps_per_env     : int   = 24
    max_iterations        : int   = 45000
    save_interval         : int   = 500
    experiment_name       : str   = "go2_flat"
    empirical_normalization: bool = False
    run_name              : str   = ""
    resume                : bool  = False
    load_run              : str   = ".*"
    load_checkpoint       : str   = "model_.*.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std           = 0.1,     # was 1.0
        actor_hidden_dims        = [512, 256, 128],
        critic_hidden_dims       = [512, 256, 128],
        activation                = "elu",
        actor_obs_normalization   = True,   # was missing entirely
        critic_obs_normalization  = True,   # was missing entirely
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef       = 1.0,
        use_clipped_value_loss= True,       # was missing entirely
        clip_param             = 0.2,
        entropy_coef            = 0.001,    # was 0.005
        num_learning_epochs    = 5,
        num_mini_batches       = 4,
        learning_rate           = 1.0e-3,
        schedule                = "adaptive",
        gamma                   = 0.99,
        lam                     = 0.95,
        desired_kl              = 0.01,
        max_grad_norm           = 0.5,      # was 1.0
    )


# ── Rough terrain uses the same hyperparameters — matches Go1's pattern
#    (Go1FlatPPORunnerCfg = Go1RoughPPORunnerCfg = Go1RslRlPpoCfg) ─────────
Go2RoughPPORunnerCfg = Go2RslRlPpoCfg


@configclass
class Go2SparsePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Sparse/natural reward experiment — same network, shorter run,
    matches Go1SparsePPORunnerCfg's structure. Referenced by train.py/
    play.py with a fallback to Go2RslRlPpoCfg if this didn't exist yet —
    it now does, so that fallback path should no longer trigger."""

    seed                  : int   = 42
    device                : str   = "cuda:0"
    num_steps_per_env     : int   = 24
    max_iterations        : int   = 10000
    save_interval         : int   = 50
    experiment_name       : str   = "go2_sparse_emergent"
    run_name              : str   = "sparse_v1"
    empirical_normalization: bool = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std           = 0.1,
        actor_hidden_dims        = [512, 256, 128],
        critic_hidden_dims       = [512, 256, 128],
        activation                = "elu",
        actor_obs_normalization   = True,
        critic_obs_normalization  = True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef       = 1.0,
        use_clipped_value_loss= True,
        clip_param             = 0.2,
        entropy_coef            = 0.001,
        num_learning_epochs    = 5,
        num_mini_batches       = 4,
        learning_rate           = 1.0e-3,
        schedule                = "adaptive",
        gamma                   = 0.99,
        lam                     = 0.95,
        desired_kl              = 0.01,
        max_grad_norm           = 0.5,
    )