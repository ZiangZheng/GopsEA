import gymnasium as gym
from . import agents_ppo, agents_sac, agents_mbpo, tasks

names = [
    "UnitreeA1RoughEnvCfg",
    "UnitreeA1FlatEnvCfg",
    "UnitreeGo1RoughEnvCfg",
    "UnitreeGo1FlatEnvCfg",
    "UnitreeGo2RoughEnvCfg",
    "UnitreeGo2FlatEnvCfg",
    "AnymalBRoughEnvCfg",
    "AnymalBFlatEnvCfg",
    "AnymalCRoughEnvCfg",
    "AnymalCFlatEnvCfg",
    "AnymalDRoughEnvCfg",
    "AnymalDFlatEnvCfg",
    "H1RoughEnvCfg",
    "H1FlatEnvCfg",
]

for name in names:
    gym.register(
        id=f"GopsEA-{name[:-6]}-PPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "GopsEA_entry_point": getattr(agents_ppo, name),
        },
    )
    
for name in names:
    gym.register(
        id=f"GopsEA-{name[:-6]}-SAC",
        entry_point="GopsEA.utils.isaaclab.envs:ManagerBasedOffRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "GopsEA_entry_point": getattr(agents_sac, name),
        },
    )
    
for name in names:
    gym.register(
        id=f"GopsEA-{name[:-6]}-MBPO",
        entry_point="GopsEA.utils.isaaclab.envs:ManagerBasedOffRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "GopsEA_entry_point": getattr(agents_mbpo, name),
        },
    )
