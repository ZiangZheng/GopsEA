from RenforceRL import configclass
from RenforceRL import algorithms, components, networks, runners


@configclass
class G1TrackingPPORunnerCfgBase(runners.OnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 60000
    save_interval = 100
    experiment_name = "G1Tracking"
    run_name = "ppo"

    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]],
            ),
            use_log_std=False,
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]],
            )
        ),
    )

    algorithm = algorithms.PPOCfg(
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
    )

    logger_cfg = runners.LoggerBaseCfg(
        logger="tensorboard",
        is_log_sample=False,
        is_log_ep_info=True,
    )

    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()


@configclass
class G1TrackingFlatPPORunnerCfg(G1TrackingPPORunnerCfgBase):
    experiment_name = "G1TrackingFlat"


@configclass
class G1TrackingFlatWoStateEstimationPPORunnerCfg(G1TrackingPPORunnerCfgBase):
    experiment_name = "G1TrackingFlatWoStateEstimation"


@configclass
class G1TrackingFlatAMPHardTrackCfg(G1TrackingPPORunnerCfgBase):
    experiment_name = "G1TrackingFlatAMPHardTrack"
