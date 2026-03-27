from gymnasium.envs import register


def _make_sac_cfg():
    from gops_tasks.gym_agent_cfg import GymSACRunnerCfg

    return GymSACRunnerCfg()


def _make_ppo_cfg():
    from gops_tasks.gym_agent_cfg import GymPPORunnerCfg

    return GymPPORunnerCfg()

_DMC_AVAILABLE = True
try:
    from dm_control import suite
except Exception:
    _DMC_AVAILABLE = False

_DMC_FALLBACK_TASKS = [
    ("acrobot", "swingup"),
    ("ball_in_cup", "catch"),
    ("cartpole", "balance"),
    ("cartpole", "balance_sparse"),
    ("cartpole", "swingup"),
    ("cartpole", "swingup_sparse"),
    ("cheetah", "run"),
    ("finger", "spin"),
    ("finger", "turn_easy"),
    ("finger", "turn_hard"),
    ("fish", "swim"),
    ("fish", "upright"),
    ("hopper", "hop"),
    ("hopper", "stand"),
    ("humanoid", "run"),
    ("humanoid", "stand"),
    ("humanoid", "walk"),
    ("manipulator", "bring_ball"),
    ("manipulator", "bring_peg"),
    ("pendulum", "swingup"),
    ("point_mass", "easy"),
    ("reacher", "easy"),
    ("reacher", "hard"),
    ("swimmer", "swimmer6"),
    ("swimmer", "swimmer15"),
    ("walker", "run"),
    ("walker", "stand"),
    ("walker", "walk"),
]


def _to_env_id(domain_name: str, task_name: str) -> str:
    return f"dmc-{domain_name.replace('_', '-')}-{task_name.replace('_', '-')}-v0"


all_tasks = suite.ALL_TASKS if _DMC_AVAILABLE else _DMC_FALLBACK_TASKS
for domain_name, task_name in all_tasks:
    register(
        id=_to_env_id(domain_name, task_name),
        entry_point="gops_tasks.dm_control.env:DMCGymEnv",
        kwargs={
            "domain_name": domain_name,
            "task_name": task_name,
            "GopsEA_entry_point": {"sac": _make_sac_cfg, "ppo": _make_ppo_cfg},
        },
        max_episode_steps=500,
        disable_env_checker=True,
    )
