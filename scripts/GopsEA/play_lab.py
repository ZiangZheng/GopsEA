import argparse, pickle, tqdm
from isaaclab import __version__ as omni_isaac_lab_version
from isaaclab.app import AppLauncher
from GopsEA.utils import argtool  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
if omni_isaac_lab_version < "0.21.0":
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--target", type=str, default=None, help="If use, direct point to the target ckpt")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")
# argtool.add_args_group(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video: args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import os
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from GopsEA.runners import BaseRunner, BaseRunnerCfg
from GopsEA.utils.env_wrapper.lab_wrapper import GopsEAEnvWrapper
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.common import ViewerCfg

def modify_env(env_cfg: ManagerBasedEnvCfg):
    env_cfg.viewer = ViewerCfg(
        eye = (4.0, 4.0, 4.0),
        lookat = (0.0, 0.0, 0.0),
        env_index = 20,
        origin_type = "asset_root",
        # origin_type = "env",
        asset_name = "robot",
    )


def main():
    task_name = args_cli.task
    if args_cli.target is None:
        raise ValueError("Please using the target specified way.")
    else:
        resume_path = os.path.abspath(args_cli.target)
        log_root_path = os.path.dirname(os.path.dirname(resume_path))
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        run_path = os.path.dirname(resume_path)
    
    if task_name is None:
        assert os.path.exists(os.path.join(run_path, "params", "args.pkl")), "No task specified."
        with open(os.path.join(run_path, "params", "args.pkl"), "rb") as f:
            args_old = pickle.load(f)
        task_name = args_old.task
        with open(os.path.join(run_path, "params", "env.pkl"), "rb") as f:
            env_cfg: ManagerBasedEnvCfg = pickle.load(f)
        with open(os.path.join(run_path, "params", "agent.pkl"), "rb") as f:
            agent_cfg: BaseRunnerCfg = pickle.load(f)
    else:
        import demo_tasks
        env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task_name, "GopsEA_entry_point")

    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    if args_cli.num_envs is not None: env_cfg.scene.num_envs = args_cli.num_envs
    modify_env(env_cfg)
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.length if args_cli.length > 0 else 0,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.rldevice
    env = GopsEAEnvWrapper(env)
    runner: BaseRunner = agent_cfg.construct_from_cfg(env, log_dir, device=args_cli.rldevice)
    runner.load(resume_path, False)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    # torch.save(runner.alg.actor_critic, os.path.join(export_model_dir, "policy.pth"))
    # export_policy_as_onnx(runner.alg.actor_critic, export_model_dir, filename="policy.onnx")
    print(f"[INFO]: Saving policy to: {export_model_dir}")
    
    # reset environment
    obs, _ = env.get_observations()
    pbar = tqdm.tqdm(range(args_cli.length)) if args_cli.length>0  else tqdm.tqdm()
    step = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)
            step += 1
            pbar.update() 
            if args_cli.length > 0 and args_cli.length < step:
                break
    except KeyboardInterrupt:
        pass
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
