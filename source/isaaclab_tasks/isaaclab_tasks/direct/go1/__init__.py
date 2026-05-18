# /home/sripu715/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go1/__init__.py
import gymnasium as gym
from . import agents  # Your agents folder (contains rsl_rl_ppo_cfg.py)

from isaaclab_tasks.direct.go1.go1_nav_env_cfg import Go1NavEnvCfg

gym.register(
    id="Isaac-Go1-Nav-v0",
    entry_point="isaaclab_tasks.direct.go1.go1_nav_env:Go1NavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.go1.go1_nav_env_cfg:Go1NavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go1RslRlPpoCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Go1-Direct-v0",
    entry_point="isaaclab_tasks.direct.go1.go1_env:Go1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.go1.go1_env_cfg:Go1FlatEnvCfg",
        # Keep rl_games if you still want to support it (optional)
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        # RSL_RL entry point — points to the config class
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go1RslRlPpoCfg",
        # Keep skrl if needed
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go1-Direct-v0",
    entry_point="isaaclab_tasks.direct.go1.go1_env:Go1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.go1.go1_env_cfg:Go1RoughEnvCfg",
        # Optional: rl_games for rough
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        # Same RSL_RL config works for both flat and rough (as recommended)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go1RslRlPpoCfg",
        # Keep skrl if needed
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)