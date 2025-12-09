# /home/sripu715/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go1/__init__.py
import gymnasium as gym
from . import agents  # Your agents folder

gym.register(
    id="Isaac-Velocity-Flat-Go1-Direct-v0",
    entry_point="isaaclab_tasks.direct.go1.go1_env:Go1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.go1.go1_env_cfg:Go1FlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",  # FIXED: Your file name
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go1-Direct-v0",
    entry_point="isaaclab_tasks.direct.go1.go1_env:Go1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.go1.go1_env_cfg:Go1RoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",  # Use flat config for rough too
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)