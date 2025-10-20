import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Go1-Direct-v0",
    entry_point=f"{__name__}.go1_env:Go1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_env_cfg:Go1FlatEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go1-Direct-v0",
    entry_point=f"{__name__}.go1_env:Go1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_env_cfg:Go1RoughEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
