# train_pendulum_gym.py
import argparse
import os
import gymnasium as gym
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from RenforceRL.utils import argtool
from RenforceRL.runners import BaseRunner
from RenforceRL.utils import env_wrapper

from isaaclab.utils.io import dump_yaml
from isaaclab.utils import configclass

from RenforceRL.runners import OnPolicyRunnerCfg, OffPolicyRunnerCfg, LoggerBaseCfg
from RenforceRL.components.normalizer import NormalizerEmpiricalCfg, NormalizerBaseCfg
from RenforceRL.components.actor_critic_pack import ActorCriticPackCfg
from RenforceRL.buffer import PipeBufferTransitionCfg, replay_bundle, DirectTransitionBuffer, DirectTransitionBufferCfg
from RenforceRL.components.actor import SACActorCfg, StateIndStdActorCfg
from RenforceRL.components.critic import MultiQNetworkCfg, VNetworkCfg, GaussianQNetworkCfg
from RenforceRL.algorithms.off_policy.sac import SACCfg, SACTransCfg
from RenforceRL.algorithms.on_policy.ppo import PPOCfg
from RenforceRL.algorithms.off_policy.dsac import DSACCfg, DSACTCfg
from RenforceRL.networks.mlp import MLPCfg
import math

from RenforceRL import runners, algorithms, components, networks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rldevice", type=str, default="cuda:0")
    argtool.add_args_group(parser)
    args = parser.parse_args()

    task = args.task # "HalfCheetah-v4" # "Walker2d-v4" # "BipedalWalker-v3" # "Pendulum-v1"
    agent_cfg = DSACTRunnerCfg() # SACRunnerCfg()
    agent_cfg.experiment_name = task
    
    # ---- agent cfg & log dir ----
    log_dir = argtool.make_log_dir(agent_cfg)
    agent_cfg.seed = args.seed

    # ---- gym env ----
    env = gym.make(task)
    env.reset(seed=args.seed)
    env = env_wrapper.SimpleGymWrapper(env)
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

# Default runner params
@configclass
class SACRunnerCfg(OffPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 20
    max_iterations = 200_000
    save_interval = 10_000
    experiment_name = ""
    run_name = "sac"
    # ---- policy ----
    policy = ActorCriticPackCfg(
        actor_cfg = SACActorCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [256, 256, 256],
                activations = [[('ReLU', {})]] * 4
            ),
            use_tanh = True,
            log_std_min = -5,
            log_std_max = 2,
            # hidden_dim = 32,
            action_bias = 0.0,
            action_scale = 1.0,            # Pendulum action range
        ),
        critic_cfg = MultiQNetworkCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [256, 256, 256],
                activations = [[('ReLU', {})]] * 3 + [[]]
            )
        )
    )
    algorithm = SACTransCfg(
        gamma              = 0.99,
        tau                = 0.005,
        actor_lr           = 3e-4,
        critic_lr          = 3e-4,
        alpha_lr           = 3e-5,
        auto_entropy       = True,
        alpha              = math.e,           # 稳定默认值（target_entropy = -1）
        max_grad_norm      = 5.0,
        actor_update_freq  = 2,
        target_update_freq = 1,
    )
    replay_cfg=replay_bundle.ReplayBundle(
        replay_buffer_cfg=DirectTransitionBufferCfg(
            max_steps=1000_000,  # Large buffer for off-policy learning
            warmup_steps=10_000,
        ),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=100 #100_000,
    )
    logger_cfg = LoggerBaseCfg(
        logger = "tensorboard",
        is_log_sample = False,
    )
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()

@configclass
class DSACRunnerCfg(OffPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 20
    max_iterations = 200_000
    save_interval = 10_000
    experiment_name = ""
    run_name = "dsac"
    # ---- policy ----
    policy = ActorCriticPackCfg(
        actor_cfg = SACActorCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [256, 256, 256],
                activations = [[('GELU', {})]] * 4
            ),
            use_tanh = True,
            log_std_min = -20,
            log_std_max = 1,
            # hidden_dim = 32,
            action_bias = 0.0,
            action_scale = 1.0,            # Pendulum action range
        ),
        critic_cfg = GaussianQNetworkCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [256, 256, 256],
                activations = [[('GELU', {})]] * 3 + [[]]
            )
        )
    )
    algorithm = DSACCfg(
        gamma              = 0.99,
        tau                = 0.005,
        actor_lr           = 3e-4,
        critic_lr          = 3e-4,
        alpha_lr           = 3e-5,
        auto_entropy       = True,
        bound              = True,
        alpha              = math.e,           # 稳定默认值（target_entropy = -1）
        max_grad_norm      = 5.0,
        actor_update_freq  = 2,
        target_update_freq = 2,
    )
    replay_cfg=replay_bundle.ReplayBundle(
        replay_buffer_cfg=DirectTransitionBufferCfg(
            max_steps=1000_000,  # Large buffer for off-policy learning
            warmup_steps=1_000,
        ),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=100 #100_000,
    )
    logger_cfg = LoggerBaseCfg(
        logger = "tensorboard",
        is_log_sample = False,
    )
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()
    
@configclass
class DSACTRunnerCfg(OffPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 20
    max_iterations = 200_000
    save_interval = 10_000
    experiment_name = ""
    run_name = "dsact"
    # ---- policy ----
    policy = ActorCriticPackCfg(
        actor_cfg = SACActorCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [256, 256, 256],
                activations = [[('GELU', {})]] * 4
            ),
            use_tanh = True,
            log_std_min = -20,
            log_std_max = 1,
            # hidden_dim = 32,
            action_bias = 0.0,
            action_scale = 1.0,            # Pendulum action range
        ),
        critic_cfg = components.ModuleList(
            module_list=[
                components.GaussianQNetworkCfg(
                    backbone_cfg = MLPCfg(
                        hidden_features = [256, 256, 256],
                        activations = [[('GELU', {})]] * 3 + [[]]
                    )
                ),
                components.GaussianQNetworkCfg(
                    backbone_cfg = MLPCfg(
                        hidden_features = [256, 256, 256],
                        activations = [[('GELU', {})]] * 3 + [[]]
                    )
                )
            ]
        )
    )
    algorithm = DSACTCfg(
        gamma              = 0.99,
        tau                = 0.005,
        tau_b              = 0.005,
        actor_lr           = 3e-4,
        critic_lr          = 3e-4,
        alpha_lr           = 3e-5,
        auto_entropy       = True,
        alpha              = math.e,           # 稳定默认值（target_entropy = -1）
        max_grad_norm      = 5.0,
        actor_update_freq  = 2,
        target_update_freq = 2,
    )
    replay_cfg=replay_bundle.ReplayBundle(
        replay_buffer_cfg=DirectTransitionBufferCfg(
            max_steps=1000_000,  # Large buffer for off-policy learning
            warmup_steps=10_000,
        ),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=100 #100_000,
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
    max_iterations = 200_000 
    save_interval = 200
    experiment_name = ""
    run_name = "ppo"
    policy = ActorCriticPackCfg(
        actor_cfg = StateIndStdActorCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [128, 128],
                activations = [[('ELU', {})]] * 3
            ),
            use_log_std = True,
        ),
        critic_cfg = VNetworkCfg(
            backbone_cfg = MLPCfg(
                hidden_features = [128, 128],
                activations = [[('ELU', {})]] * 3
            )
        )
    )
    algorithm = PPOCfg(
        gamma = 0.99,
        lam = 0.95,
        clip_param = 0.2,
        value_loss_coef = 0.25,
        entropy_coef = 0.01,
        num_learning_epochs = 10,
        num_mini_batches = 8,
        learning_rate = 3e-4,
        schedule = "fixed",
        max_grad_norm = 1.0,
        desired_kl = 0.02,
    )
    logger_cfg = LoggerBaseCfg(logger = "tensorboard",)
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()

if __name__ == "__main__":
    main()
