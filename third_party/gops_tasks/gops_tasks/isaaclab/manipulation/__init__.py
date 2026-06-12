import gymnasium as gym

from . import agents_ppo, pickplace_tasks


TASK_NAMES = [
    "FrankaCubeLiftEnvCfg",
    "PickPlaceSimpleEnvCfg",
]

SKRL_CFG_ENTRY_POINTS = {
    "FrankaCubeLiftEnvCfg": (
        "PickPlace.tasks.manager_based.pickplace.config.franka.agents:"
        "skrl_ppo_cfg.yaml"
    ),
    "PickPlaceSimpleEnvCfg": (
        "PickPlace_simple.tasks.manager_based.pickplace_simple.config.franka.agents:"
        "skrl_ppo_cfg.yaml"
    ),
}


for name in TASK_NAMES:
    gym.register(
        id=f"GopsEA-{name[:-6]}-PPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(pickplace_tasks, name),
            "GopsEA_entry_point": agents_ppo.PPOCfg().replace(
                experiment_name=f"{name[:-6]}"
            ),
            "skrl_cfg_entry_point": SKRL_CFG_ENTRY_POINTS[name],
        },
    )
