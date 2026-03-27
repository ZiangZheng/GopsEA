import argparse
import os
from isaaclab import __version__ as omni_isaac_lab_version
assert omni_isaac_lab_version > "0.21.0"
from isaaclab.app import AppLauncher
from GopsEA.utils import argtool # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--cfg", type=str, default=None, help="Directly using the target cfg object.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--replicate", type=str, default=None, help="Replicate old experiment with same configuration.")

parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--local", action="store_true", default=False, help="Using asset in local buffer")

argtool.add_args_group(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
import gops_tasks  # noqa: F401

from GopsEA.runners import BaseRunner
from GopsEA.utils.env_wrapper.lab_wrapper import GopsEAEnvWrapper

def main():
    task_name, env_cfg, agent_cfg, log_dir = argtool.make_cfgs(args_cli, parse_env_cfg, None)
    env_cfg.sim.device, env_cfg.seed = args_cli.device, args_cli.seed
    env_cfg.recorders = None    
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    agent_cfg.seed = args_cli.seed
    env = GopsEAEnvWrapper(env)
    runner: "BaseRunner" = agent_cfg.construct_from_cfg(env, log_dir, device=args_cli.rldevice)
    print(runner.alg)
    if agent_cfg.resume:
        resume_path = agent_cfg.load_checkpoint
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "args.yaml"), vars(args_cli))
    argtool.dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    argtool.dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    argtool.dump_pickle(os.path.join(log_dir, "params", "args.pkl"), args_cli)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
