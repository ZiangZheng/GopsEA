from RenforceRL import configclass
from RenforceRL import runners, algorithms, components, networks
from RenforceRL.buffer import PipeBufferTransitionCfg, replay_bundle, DirectTransitionBuffer, DirectTransitionBufferCfg
import math

@configclass
class G1TrackingSACRunnerCfgBase(runners.OffPolicyRunnerCfg):
    seed=42
    num_steps_per_env=1
    max_iterations=720_000
    save_interval=4000
    experiment_name="None"
    run_name="sac"
    # ---- Policy: Actor-Critic for SAC ----
    policy=components.ActorCriticPackCfg(
        actor_cfg=components.SACActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('ELU', {})],
                    [('ELU', {})],
                    [('ELU', {})],
                    []
                ]
            ),
            hidden_dim=128,
            use_tanh=True,
            log_std_min=-5.0,
            log_std_max=0.0,
            # action_scale=3.14,
            action_bias=0.0
        ),
        critic_cfg=components.MultiQNetworkCfg(
            num_q=2,  # Clipped double Q-learning
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('LayerNorm', {}), ('ReLU', {})],
                    [('LayerNorm', {}), ('ReLU', {})],
                    [('LayerNorm', {}), ('ReLU', {})],
                    []
                ]
            )
        )
    )
    # ---- SAC Algorithm Parameters ----
    algorithm=algorithms.SACTransCfg(
        gamma=0.97,
        tau=0.125,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,  # 3e-5
        auto_entropy=True,
        alpha=math.e, # 0.2, 0.001
        # target_entropy=0,  # None -> -action_dim # 0.0
        max_grad_norm=5.0,
        actor_update_freq=4,
        target_update_freq=2
    )
    # ---- Replay Buffer Pipeline Bundle Configuration ----
    replay_cfg=replay_bundle.ReplayBundle(
        replay_buffer_cfg=DirectTransitionBufferCfg(
            max_steps=128*1024,  # Large buffer for off-policy learning
            warmup_steps=128*10,
        ),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=8
    )
    # ---- Logger Configuration ----
    logger_cfg = runners.LoggerBaseCfg(
        logger = "tensorboard",
        is_log_sample = False,
        is_log_ep_info= True
    )
    # ---- Normalizers ----
    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()
    
@configclass
class G1TrackingFlatSACRunnerCfg(G1TrackingSACRunnerCfgBase):
    experiment_name = "G1TrackingFlat"