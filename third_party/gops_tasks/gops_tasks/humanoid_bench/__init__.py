from __future__ import annotations

import gymnasium as gym

from .env import _prepare_humanoid_bench_import


def _make_sac_cfg():
    from gops_tasks.gym_agent_cfg import GymSACRunnerCfg

    return GymSACRunnerCfg()


def _make_ppo_cfg():
    from gops_tasks.gym_agent_cfg import GymPPORunnerCfg

    return GymPPORunnerCfg()


def _register_once(env_id: str, entry_point: str, kwargs: dict, max_episode_steps: int) -> None:
    if env_id in gym.envs.registry:
        return
    gym.register(
        id=env_id,
        entry_point=entry_point,
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        disable_env_checker=True,
    )


try:
    _prepare_humanoid_bench_import()
    from humanoid_bench.env import ROBOTS, TASKS
except Exception:
    ROBOTS = {
        "h1": None,
        "h1hand": None,
        "h1simplehand": None,
        "h1strong": None,
        "h1touch": None,
        "g1": None,
    }
    TASKS = {
        "stand": None,
        "walk": None,
        "run": None,
        "kitchen": None,
        "maze": None,
        "hurdle": None,
        "cube": None,
        "bookshelf_simple": None,
        "bookshelf_hard": None,
        "highbar_simple": None,
        "highbar_hard": None,
        "crawl": None,
        "window": None,
        "spoon": None,
        "door": None,
        "push": None,
        "reach": None,
        "basketball": None,
        "truck": None,
        "package": None,
        "cabinet": None,
        "sit_simple": None,
        "sit_hard": None,
        "balance_simple": None,
        "balance_hard": None,
        "stair": None,
        "slide": None,
        "pole": None,
        "room": None,
        "insert_normal": None,
        "insert_small": None,
        "powerlift": None,
    }


for robot in ROBOTS:
    control = "torque" if robot in {"g1", "digit"} else "pos"
    for task_name, task_cls in TASKS.items():
        max_episode_steps = 1000
        task_kwargs = {}
        if task_cls is not None:
            task_info = task_cls()
            task_kwargs = task_info.kwargs.copy()
            max_episode_steps = task_info.max_episode_steps
        task_kwargs.update(
            {
                "robot": robot,
                "task": task_name,
                "control": control,
            }
        )

        # Stable, namespaced IDs under gops_tasks.
        _register_once(
            env_id=f"gops-hb-{robot}-{task_name}-v0",
            entry_point="gops_tasks.humanoid_bench.env:make_humanoid_env",
            kwargs={
                "robot": robot,
                "task": task_name,
                "control": control,
                "GopsEA_entry_point": {"sac": _make_sac_cfg, "ppo": _make_ppo_cfg},
            },
            max_episode_steps=max_episode_steps,
        )

        # Optional alias mirroring upstream naming (only if not already occupied).
        _register_once(
            env_id=f"{robot}-{task_name}-v0",
            entry_point="humanoid_bench.env:HumanoidEnv",
            kwargs=task_kwargs,
            max_episode_steps=max_episode_steps,
        )
