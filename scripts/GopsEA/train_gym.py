# train_pendulum_gym.py
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
from isaaclab.utils import configclass

from GopsEA.runners import OnPolicyRunnerCfg, OffPolicyRunnerCfg, LoggerBaseCfg
from GopsEA.components.normalizer import NormalizerEmpiricalCfg, NormalizerBaseCfg
from GopsEA.components.actor_critic_pack import ActorCriticPackCfg
from GopsEA.buffer import ReplayBufferChunkCfg, PipeBufferTransitionCfg
from GopsEA.components.actor import SACActorCfg, StateIndStdActorCfg
from GopsEA.components.critic import MultiQNetworkCfg, VNetworkCfg
from GopsEA.algorithms.off_policy.sac import SACCfg, SACTransCfg
from GopsEA.algorithms.on_policy.ppo import PPOCfg
from GopsEA.networks.mlp import MLPCfg

@configclass
class SACRunnerCfg(OffPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 1
    max_iterations = 200_000
    save_interval = 10_000
    experiment_name = ""
    run_name = "sac"
    # ---- policy ----
    policy = ActorCriticPackCfg(
        actor_cfg = SACActorCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [128, 128],
                activations = [[('ELU', {})]] * 3
            ),
            use_tanh = True,
            log_std_min = -5,
            log_std_max = 2,
            hidden_dim = 32,
            action_bias = 0.0,
            action_scale = 1.0,            # Pendulum action range
        ),
        critic_cfg = MultiQNetworkCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [128, 128],
                activations = [[('ELU', {})]] * 2 + [[]]
            )
        )
    )
    algorithm = SACTransCfg(
        gamma              = 0.99,
        tau                = 0.005,
        actor_lr           = 3e-4,
        critic_lr          = 3e-4,
        alpha_lr           = 3e-4,
        auto_entropy       = True,
        alpha              = 0.2,           # 稳定默认值（target_entropy = -1）
        max_grad_norm      = 5.0,
        actor_update_freq  = 1,
        target_update_freq = 1,
    )
    replay_cfg = OffPolicyRunnerCfg.ReplayCfg(
        replay_buffer_cfg = PipeBufferTransitionCfg(
            capacity = 100_000,
        ),
        replay_num_epoches = 1,
        replay_mini_batch_size = 256,
        replay_num_batch_per_epoch = 4,
    )
    logger_cfg = LoggerBaseCfg(
        logger = "tensorboard",
        is_log_sample = False,
    )
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()

@configclass
class PPORunnerCfg(OnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 128
    max_iterations = 3_000 
    save_interval = 200
    experiment_name = "pendulum"
    run_name = "ppo"
    policy = ActorCriticPackCfg(
        actor_cfg = StateIndStdActorCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [128, 128],
                activations = [[('ELU', {})]] * 2
            ),
            use_log_std = True,
        ),
        critic_cfg = VNetworkCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [128, 128],
                activations = [[('ELU', {})]] * 2
            )
        )
    )
    algorithm = PPOCfg(
        gamma = 0.99,
        lam = 0.95,
        clip_param = 0.2,
        value_loss_coef = 0.5,
        entropy_coef = 0.0,
        num_learning_epochs = 10,
        num_mini_batches = 4,
        learning_rate = 3e-4,
        schedule = "fixed",
        max_grad_norm = 1.0,
        desired_kl = 0.02,
    )
    logger_cfg = LoggerBaseCfg(logger = "tensorboard",)
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rldevice", type=str, default="cuda:0")
    argtool.add_args_group(parser)
    args = parser.parse_args()

    task = "HalfCheetah-v4" # "Walker2d-v4" # "BipedalWalker-v3" # "Pendulum-v1"
    agent_cfg = SACRunnerCfg()
    agent_cfg.experiment_name = task
    
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
    # env.close()

if __name__ == "__main__":
    main()
