from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Tuple
import yaml, os
from datetime import datetime
import pickle
from .gym import load_cfg_from_registry

if TYPE_CHECKING:
    from GopsEA.runners import BaseRunnerCfg
    from isaaclab.envs import ManagerBasedEnvCfg

def add_args_group(parser: argparse.ArgumentParser):
    """Add arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("renforce", description="Arguments for renforce rl agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--algor", type=str, default=None, help="Algorithm Type."
    )
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )

def parse_rl_cfg(task_name: str, args_cli: argparse.Namespace, rl_cfg=None) -> "BaseRunnerCfg":
    """Parse configuration for agent based on inputs.
    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.
    Returns:
        The parsed configuration for agent based on inputs.
    """
    if rl_cfg is None:
        rl_cfg:dict = load_cfg_from_registry(task_name, "GopsEA_entry_point")
        if isinstance(rl_cfg, dict):
            if getattr(args_cli, "algor", None) is not None:
                algor = getattr(args_cli, "algor")
                rl_cfg = rl_cfg.get(algor)
            else:
                rl_cfg = next(iter(rl_cfg.values()))
            rl_cfg = rl_cfg() # not init before

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        rl_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        rl_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        rl_cfg.load_run = args_cli.load_run
    if args_cli.run_name is not None:
        rl_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        rl_cfg.logger_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if rl_cfg.logger_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        rl_cfg.logger_cfg.wandb_project = args_cli.log_project_name
        rl_cfg.logger_cfg.neptune_project = args_cli.log_project_name

    return rl_cfg

def update_object_from_dict(obj, data):
    for key, value in data.items():
        if hasattr(obj, key):
            if isinstance(value, dict):
                nested_obj = getattr(obj, key)
                update_object_from_dict(nested_obj, value)
            else:
                setattr(obj, key, value)
        else:
            print(f"Warning: Object has no attribute '{key}'")
    return obj

def update_object_from_yaml(obj, file_path:str):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        obj = update_object_from_dict(obj, data)
    return obj

def make_log_dir(agent_cfg):
    log_root_path = os.path.join("logs", "RFRL", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    return log_dir

def make_cfgs(args_cli, parse_env_cfg, runner_cfg=None):
    task_name = args_cli.task
    env_cfg: "ManagerBasedEnvCfg" = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: "BaseRunnerCfg" = parse_rl_cfg(task_name, args_cli, runner_cfg)
    log_dir = make_log_dir(agent_cfg)
    agent_cfg.seed = args_cli.seed
    return task_name, env_cfg, agent_cfg, log_dir

def load_cfgs(args_cli, modified=False):
    task_dir = args_cli.replicate
    task_dir = os.path.abspath(task_dir)
    log_root_path = os.path.dirname(task_dir)
    with open(os.path.join(task_dir, "params", "args.pkl"), "rb") as f:
        args_old = pickle.load(f)
        task_name: str = args_old.task
    with open(os.path.join(task_dir, "params", "env.pkl"), "rb") as f:
        env_cfg: "ManagerBasedEnvCfg" = pickle.load(f)
    with open(os.path.join(task_dir, "params", "agent.pkl"), "rb") as f:
        agent_cfg: "BaseRunnerCfg" = pickle.load(f)
        if modified: agent_cfg: "BaseRunnerCfg" = update_object_from_yaml(agent_cfg, os.path.join(task_dir, "params", "agent.yaml"))

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    return task_name, env_cfg, agent_cfg, log_dir

def dump_pickle(fpath, obj):
    with open(fpath, "wb") as f:
        pickle.dump(obj, f)        
