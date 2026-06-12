import gymnasium as gym

from . import rsl_rl_ppo_cfg, flat_env_cfg, agents_dsact, agents_sac

##
# Register Gym environments.
##

gym.register(
    id="beyondMimic-Tracking-G1-Flat-PPO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "RenforceRL_entry_point": rsl_rl_ppo_cfg.G1TrackingFlatPPORunnerCfg,
    },
)

gym.register(
    id="beyondMimic-Tracking-G1-Flat-DSACT",
    entry_point="RenforceRL.utils.isaaclab.envs:ManagerBasedOffRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "RenforceRL_entry_point": agents_dsact.G1TrackingFlatDSACTRunnerCfg,
    },
)

gym.register(
    id="beyondMimic-Tracking-G1-Flat-SAC",
    entry_point="RenforceRL.utils.isaaclab.envs:ManagerBasedOffRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "RenforceRL_entry_point": agents_sac.G1TrackingFlatSACRunnerCfg,
    },
)

gym.register(
    id="beyondMimic-Tracking-G1-Flat-Wo-State-Estimation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "RenforceRL_entry_point": rsl_rl_ppo_cfg.G1TrackingFlatWoStateEstimationPPORunnerCfg,
    },
)
