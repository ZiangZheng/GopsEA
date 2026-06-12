import gymnasium as gym
from . import (
    agents_dsact,
    agents_ppo, 
    agents_sac, 
    agents_mbpo, 
    tasks
)

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
    "G1RoughEnvCfg",
    "G1FlatEnvCfg",
    "G1TrackingFlatEnvCfg",
]

for name in names:
    gym.register(
        id=f"RenforceRL-{name[:-6]}-PPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RenforceRL_entry_point": getattr(agents_ppo, name),
        },
    )
    
for name in names:
    gym.register(
        id=f"RenforceRL-{name[:-6]}-SAC",
        entry_point="RenforceRL.utils.isaaclab.envs:ManagerBasedOffRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RenforceRL_entry_point": getattr(agents_sac, name),
        },
    )
    
for name in names:
    gym.register(
        id=f"RenforceRL-{name[:-6]}-MBPO",
        entry_point="RenforceRL.utils.isaaclab.envs:ManagerBasedOffRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RenforceRL_entry_point": getattr(agents_mbpo, name),
        },
    )

for name in names:
    gym.register(
        id=f"RenforceRL-{name[:-6]}-DSACT",
        entry_point="RenforceRL.utils.isaaclab.envs:ManagerBasedOffRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RenforceRL_entry_point": getattr(agents_dsact, name),
        },
    )