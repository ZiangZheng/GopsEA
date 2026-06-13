from GopsEA import configclass
from GopsEA import runners, algorithms, components, networks
from GopsEA.buffer import PipeBufferTransitionCfg, replay_bundle, DirectTransitionBuffer, DirectTransitionBufferCfg
import math

@configclass
class G1TrackingDSACTRunnerCfgBase(runners.OffPolicyRunnerCfg):
    seed=42
    num_steps_per_env=1
    max_iterations=720_000
    save_interval=4000
    experiment_name="None"
    run_name="dsact"
    # ---- Policy: Actor-Critic for DSAC ----
    policy=components.ActorCriticPackCfg(
        actor_cfg=components.SACActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations = [[("LayerNorm", {}), ('SiLU', {})]] * 4
            ),
            hidden_dim=128,
            use_tanh=True,
            log_std_min=-5.0,
            log_std_max=0.0,
            # action_scale=1.57,
            action_bias=0.0
        ),
        critic_cfg = components.ModuleList(
            module_list=[
                components.GaussianQNetworkCfg(
                    backbone_cfg = networks.MLPCfg(
                        hidden_features = [768, 768//2, 768//4],
                        activations = [[("LayerNorm", {}), ('SiLU', {})]] * 3 + [[]]
                    )
                ),
                components.GaussianQNetworkCfg(
                    backbone_cfg = networks.MLPCfg(
                        hidden_features = [768, 768//2, 768//4],
                        activations = [[("LayerNorm", {}), ('SiLU', {})]] * 3 + [[]]
                    )
                )
            ]
        )
    )
    # ---- DSAC Algorithm Parameters ----
    algorithm=algorithms.DSACTCfg(
        gamma=0.97,
        tau=0.125,
        tau_b=0.125,#TODO: check this
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        auto_entropy=True,#TODO: False here
        # target_entropy=0.0,
        alpha=math.e, # 0.2, 0.001
        max_grad_norm=5.0,
        actor_update_freq=4,
        target_update_freq=2
    )
    # ---- Replay Buffer Configuration ----
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
class G1TrackingFlatDSACTRunnerCfg(G1TrackingDSACTRunnerCfgBase):
    experiment_name = "G1TrackingFlat"