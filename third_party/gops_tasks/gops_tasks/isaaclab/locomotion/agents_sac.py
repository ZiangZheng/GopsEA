from GopsEA import configclass
from GopsEA import runners, algorithms, components, networks
from GopsEA.buffer import PipeBufferTransitionCfg, replay_bundle

@configclass
class LocoRLCfgBase(runners.OffPolicyRunnerCfg):
    seed=42
    num_steps_per_env=1
    max_iterations=3000
    save_interval=200
    experiment_name="None"
    run_name="sac"
    # ---- Policy: Actor-Critic for SAC ----
    policy=components.ActorCriticPackCfg(
        actor_cfg=components.SACActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('SiLU', {})],
                    [('SiLU', {})],
                    [('SiLU', {})],
                    []
                ]
            ),
            hidden_dim=256,
            use_tanh=True,
            log_std_min=-5.0,
            log_std_max=2.0,
            action_scale=1.0,
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
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        auto_entropy=True,
        alpha=0.2,
        target_entropy=None,  # None -> -action_dim
        max_grad_norm=1.0,
        actor_update_freq=1,
        target_update_freq=1
    )
    # ---- Replay Buffer Configuration ----
    replay_cfg=replay_bundle.ReplayPipelineBundle(
        replay_buffer_cfg=PipeBufferTransitionCfg(
            capacity=1_000_000,  # Large buffer for off-policy learning
        ),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=4
    )
    # ---- Logger Configuration ----
    logger_cfg = runners.LoggerBaseCfg(
        logger="tensorboard",
        is_log_ep_info=False,
        is_log_update=False,
        is_log_sample=False,
        width=75,
        pad=20,
    )
    # ---- Normalizers ----
    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()
    
@configclass
class UnitreeA1RoughEnvCfg(LocoRLCfgBase):
    experiment_name = "UnitreeA1Rough"
    
@configclass
class UnitreeA1FlatEnvCfg(LocoRLCfgBase):
    experiment_name = "UnitreeA1Rough"

@configclass
class UnitreeGo1RoughEnvCfg(LocoRLCfgBase):
    experiment_name = "UnitreeGo1Rough"
    
@configclass
class UnitreeGo1FlatEnvCfg(LocoRLCfgBase):
    experiment_name = "UnitreeGo1Flat"
    
@configclass
class UnitreeGo2RoughEnvCfg(LocoRLCfgBase):
    experiment_name = "UnitreeGo2Rough"
    
@configclass
class UnitreeGo2FlatEnvCfg(LocoRLCfgBase):
    experiment_name = "UnitreeGo2Flat"
    
@configclass
class AnymalBRoughEnvCfg(LocoRLCfgBase):
    experiment_name = "AnymalBRough"
    
@configclass
class AnymalBFlatEnvCfg(LocoRLCfgBase):
    experiment_name = "AnymalBFlat"
    
@configclass
class AnymalCRoughEnvCfg(LocoRLCfgBase):
    experiment_name = "AnymalCRough"
    
@configclass
class AnymalCFlatEnvCfg(LocoRLCfgBase):
    experiment_name = "AnymalCFlat"

@configclass
class AnymalDRoughEnvCfg(LocoRLCfgBase):
    experiment_name = "AnymalDRough"
    
@configclass
class AnymalDFlatEnvCfg(LocoRLCfgBase):
    experiment_name = "AnymalDFlat"
    
@configclass
class H1RoughEnvCfg(LocoRLCfgBase):
    experiment_name = "H1Rough"
    
@configclass
class H1FlatEnvCfg(LocoRLCfgBase):
    experiment_name = "H1Flat"