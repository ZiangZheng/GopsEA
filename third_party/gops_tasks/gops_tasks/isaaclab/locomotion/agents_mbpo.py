from GopsEA import configclass
from GopsEA import runners, algorithms, components, networks
from GopsEA.buffer.direct_based.dynamic_replay_buffer import DynamicReplayBufferCfg
from GopsEA.components.world_models.system_dynamics import SystemDynamicsMLPCfg


@configclass
class LocoRLCfgBase(runners.OnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 100
    experiment_name = "None"
    run_name = "mbpo"
    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            ),
            use_log_std=False
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            )
        )
    )
    # ---- MBPO Algorithm Parameters ----
    algorithm = algorithms.MBPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # System dynamics trainer config
        system_dynamics_trainer_cfg=algorithms.SystemDynamicsTrainerCfg(
            dynamic_update_learning_rate=1e-3,
            dynamic_update_weight_decay=0.0,
            dynamic_update_forecast_horizon=10,
            dynamic_update_num_mini_batches=4,
            dynamic_update_mini_batch_size=256,
            max_grad_norm=1.0,
        ),
    )
    # ---- System Dynamics Model Configuration ----
    system_dynamics_cfg = SystemDynamicsMLPCfg(
        history_horizon=1,
        backbone_cfg=networks.MLPCfg(
            hidden_features=[256, 256],
            activations=[
                [('ReLU', {})],
                [('ReLU', {})],
                [],  # No activation on last layer
            ]
        ),
    )
    # ---- Dynamic Replay Buffer Configuration ----
    replay_cfg = DynamicReplayBufferCfg(
        buffer_size=100_000,  # Per environment
        extension_dim=0,
        contact_dim=0,
        termination_dim=0,
        reward_dim=1,
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
