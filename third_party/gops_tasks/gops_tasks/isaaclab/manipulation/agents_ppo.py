from GopsEA import configclass
from GopsEA import algorithms, components, networks, runners


@configclass
class PPOCfg(runners.OnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "None"
    run_name = "ppo"
    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[256, 128, 64],
                activations=[[("ELU", {})]] * 3 + [[]],
            ),
            use_log_std=False,
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[256, 128, 64],
                activations=[[("ELU", {})]] * 3 + [[]],
            )
        ),
    )
    algorithm = algorithms.PPOCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
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
