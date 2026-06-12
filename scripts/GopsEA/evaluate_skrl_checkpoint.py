import argparse
import os
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a skrl checkpoint on an IsaacLab task.")
parser.add_argument("--task", required=True, help="Gym task id.")
parser.add_argument("--checkpoint", required=True, help="Path to a skrl checkpoint.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments.")
parser.add_argument("--steps", type=int, default=500, help="Number of environment steps.")
parser.add_argument("--output", type=str, default=None, help="Optional file path for the EVAL_RESULT line.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="ML framework used by the skrl checkpoint.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="skrl algorithm used by the checkpoint.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import skrl
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from packaging import version

import gops_tasks  # noqa: F401
import PickPlace.tasks  # noqa: F401
import PickPlace_simple.tasks  # noqa: F401

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner


SKRL_VERSION = "1.4.2"


def main():
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        raise RuntimeError(
            f"Unsupported skrl version: {skrl.__version__}. "
            f"Install skrl>={SKRL_VERSION}."
        )
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    algorithm = args_cli.algorithm.lower()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    try:
        experiment_cfg = load_cfg_from_registry(
            args_cli.task,
            f"skrl_{algorithm}_cfg_entry_point",
        )
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    checkpoint = os.path.abspath(args_cli.checkpoint)
    runner.agent.load(checkpoint)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    total_returns = torch.zeros(env.num_envs, device=env.device)
    episode_returns = torch.zeros(env.num_envs, device=env.device)
    completed_returns = []
    reward_min = float("inf")
    reward_max = float("-inf")
    terminations = 0
    truncations = 0
    start_time = time.time()

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {
                    agent: outputs[-1][agent].get("mean_actions", outputs[0][agent])
                    for agent in env.possible_agents
                }
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rewards, terminated, truncated, _ = env.step(actions)

        rewards = rewards.reshape(-1)
        done = (terminated.reshape(-1) | truncated.reshape(-1)).bool()
        total_returns += rewards
        episode_returns += rewards
        reward_min = min(reward_min, float(rewards.min().item()))
        reward_max = max(reward_max, float(rewards.max().item()))
        terminations += int(terminated.sum().item())
        truncations += int(truncated.sum().item())
        if done.any():
            completed_returns.extend(episode_returns[done].detach().cpu().tolist())
            episode_returns[done] = 0

    elapsed = time.time() - start_time
    returns = total_returns.detach().cpu()
    mean_return = returns.mean().item()
    std_return = returns.std(unbiased=False).item()
    min_return = returns.min().item()
    max_return = returns.max().item()
    mean_step_reward = mean_return / args_cli.steps
    fps = (args_cli.steps * args_cli.num_envs) / elapsed
    result_line = (
        "EVAL_RESULT "
        f"task={args_cli.task} "
        f"envs={args_cli.num_envs} "
        f"steps={args_cli.steps} "
        f"checkpoint={checkpoint} "
        f"seed={args_cli.seed} "
        f"mean_return={mean_return:.6f} "
        f"std_return={std_return:.6f} "
        f"min_return={min_return:.6f} "
        f"max_return={max_return:.6f} "
        f"mean_step_reward={mean_step_reward:.6f} "
        f"reward_min={reward_min:.6f} "
        f"reward_max={reward_max:.6f} "
        f"terminations={terminations} "
        f"truncations={truncations} "
        f"completed_episodes={len(completed_returns)} "
        f"fps={fps:.2f}"
    )
    if args_cli.output:
        with open(args_cli.output, "w", encoding="utf-8") as f:
            f.write(result_line + "\n")
    print(result_line, flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
