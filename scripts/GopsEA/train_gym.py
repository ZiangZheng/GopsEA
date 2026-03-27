import argparse
import os
import gymnasium as gym
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from GopsEA import argtool
from GopsEA.runners import BaseRunner
from GopsEA.utils.env_wrapper.simple_gym import SimpleGymWrapper

from isaaclab.utils.io import dump_yaml
from default_gym_cfg import make_default_runner_cfg

# trigger task registration (isaaclab / dmc / humanoid-bench)
import gops_tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Gymnasium task id.")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"], help="Default agent config.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override runner max_iterations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rldevice", type=str, default="cuda:0")
    argtool.add_args_group(parser)
    args = parser.parse_args()

    task = args.task
    agent_cfg = make_default_runner_cfg(task=task, algo=args.algo)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    
    # ---- agent cfg & log dir ----
    log_dir = argtool.make_log_dir(agent_cfg)
    agent_cfg.seed = args.seed

    # ---- gym env ----
    env = gym.make(task)
    env.reset(seed=args.seed)
    env = SimpleGymWrapper(env)
    # ---- runner ----
    runner: BaseRunner = agent_cfg.construct_from_cfg(
        env=env,
        log_dir=log_dir,
        device=args.rldevice,
    )
    if agent_cfg.resume:
        runner.load(agent_cfg.load_checkpoint)

    dump_yaml(os.path.join(log_dir, "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    env.close()

if __name__ == "__main__":
    main()
