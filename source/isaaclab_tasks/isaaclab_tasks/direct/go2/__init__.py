# /home/sripu715/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2/__init__.py
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Go2-v0",
    entry_point="isaaclab_tasks.direct.go2.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.go2.go2_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RslRlPpoCfg",
    },
)

# ── Rough/outdoor terrain — SAME Go2Env class as flat, only cfg differs.
#    Blind (no LiDAR) locomotion. Requires a train.py branch that calls
#    make_go2_rough_scene(n) before gym.make() -- see go2_rough_env_cfg.py
#    header for why (configclass num_envs freezing), same as Go1's rough task.
gym.register(
    id="Isaac-Velocity-Rough-Go2-v0",
    entry_point="isaaclab_tasks.direct.go2.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.go2.go2_rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RslRlPpoCfg",
    },
)

# ── Sparse/natural rewards on rough terrain — mirrors Isaac-Go1-Sparse-
#    Rough-Direct-v0. Same rough cfg as above, different env class (reward
#    override) and PPO cfg (sparse-reward-scale hyperparameters).
#    Go2SparsePPORunnerCfg must exist in agents/rsl_rl_ppo_cfg.py -- add
#    one mirroring Go1SparsePPORunnerCfg if it doesn't yet.
gym.register(
    id="Isaac-Go2-Sparse-Rough-Direct-v0",
    entry_point="isaaclab_tasks.direct.go2.go2_env_sparse_rough:Go2EnvSparseRough",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.go2.go2_rough_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2SparsePPORunnerCfg",
    },
)

# No separate Go2 baseline task. Go1's baseline (Isaac-Velocity-Flat-Go1-
# Baseline-v0) exists to force the PACE-identified fault parameters to
# their brand-new/healthy null values for an ablation against the faulted
# calibrated env. Go2 has no known fault — Go2Env's PACE-equivalent terms
# (Ia, d, tau_f, q~b) are ALREADY null by construction (see go2_env.py
# header comment for the full mapping), so Go2Env itself already IS that
# condition. A separate baseline task would be an ablation with nothing
# to ablate.