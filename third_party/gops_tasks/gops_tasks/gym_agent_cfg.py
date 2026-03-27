try:
    from isaaclab.utils import configclass
except Exception:
    def configclass(cls):
        return cls

from GopsEA.algorithms.off_policy.sac import SACTransCfg
from GopsEA.algorithms.on_policy.ppo import PPOCfg
from GopsEA.buffer import PipeBufferTransitionCfg
from GopsEA.components.actor import SACActorCfg, StateIndStdActorCfg
from GopsEA.components.actor_critic_pack import ActorCriticPackCfg
from GopsEA.components.critic import MultiQNetworkCfg, VNetworkCfg
from GopsEA.components.normalizer import NormalizerBaseCfg
from GopsEA.networks.mlp import MLPCfg
from GopsEA.runners import LoggerBaseCfg, OffPolicyRunnerCfg, OnPolicyRunnerCfg


@configclass
class GymSACRunnerCfg(OffPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 1
    max_iterations = 200_000
    save_interval = 10_000
    experiment_name = ""
    run_name = "sac"
    policy = ActorCriticPackCfg(
        actor_cfg=SACActorCfg(
            backbone_cfg=MLPCfg(hidden_features=[256, 256], activations=[[("ReLU", {})]] * 3),
            use_tanh=True,
            log_std_min=-5.0,
            log_std_max=2.0,
            action_bias=0.0,
            action_scale=1.0,
        ),
        critic_cfg=MultiQNetworkCfg(
            backbone_cfg=MLPCfg(hidden_features=[256, 256], activations=[[("ReLU", {})]] * 2 + [[]])
        ),
    )
    algorithm = SACTransCfg(
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        auto_entropy=True,
        alpha=0.2,
        max_grad_norm=5.0,
        actor_update_freq=1,
        target_update_freq=1,
    )
    replay_cfg = OffPolicyRunnerCfg.ReplayCfg(
        replay_buffer_cfg=PipeBufferTransitionCfg(capacity=1_000_000),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=4,
    )
    logger_cfg = LoggerBaseCfg(logger="tensorboard", is_log_sample=False)
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()


@configclass
class GymPPORunnerCfg(OnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 256
    max_iterations = 20_000
    save_interval = 500
    experiment_name = ""
    run_name = "ppo"
    policy = ActorCriticPackCfg(
        actor_cfg=StateIndStdActorCfg(
            backbone_cfg=MLPCfg(hidden_features=[256, 256], activations=[[("ELU", {})]] * 2),
            use_log_std=True,
        ),
        critic_cfg=VNetworkCfg(
            backbone_cfg=MLPCfg(hidden_features=[256, 256], activations=[[("ELU", {})]] * 2)
        ),
    )
    algorithm = PPOCfg(
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        num_learning_epochs=10,
        num_mini_batches=8,
        learning_rate=3e-4,
        schedule="fixed",
        max_grad_norm=1.0,
        desired_kl=0.02,
    )
    logger_cfg = LoggerBaseCfg(logger="tensorboard")
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()

