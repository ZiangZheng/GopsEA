import os

import tqdm
from RenforceRL import configclass
from dataclasses import MISSING
from RenforceRL.utils.logging import timeit
from RenforceRL.runners.off_policy.off_policy_runner import OffPolicyRunner, OffPolicyRunnerCfg
from RenforceRL.components.world_model_bundle import WorldModelBundleCfg, RenforceRLEnvWrapper
from RenforceRL.buffer import ComponentCfg, DataPipeline, DataPipelineCfg
from RenforceRL.buffer import ReplayBufferBaseCfg, ReplayBufferBase

class WorldModelRunner(OffPolicyRunner):
    def init_components(self):
        self.actor_critic = self.policy_cfg.construct_from_cfg(dim_params=self.env.dim_params)
        self.alg = self.alg_cfg.construct_from_cfg(actor_critic=self.actor_critic, device=self.device)
        self.actor_critic.to(self.device)

        self.obs_normalizer = self.cfg.obs_normalize_cfg.construct_from_cfg(shape=self.env.dim_params["policy_dim"])
        self.critic_normalizer = self.cfg.critic_normalize_cfg.construct_from_cfg(shape=self.env.dim_params["critic_dim"])
        self.obs_normalizer.to(self.device)
        self.critic_normalizer.to(self.device)
        
        self.data_pipeline: DataPipeline = self.replay_cfg.data_pipeline_cfg.construct_from_cfg()
        self.replay_buffer: ReplayBufferBase = self.replay_cfg.replay_buffer_cfg.construct_from_cfg(
            self.data_pipeline.REPLAY_BUFFER_COMP,
            dim_params=self.data_pipeline.buffer_dim_params(self.env.dim_params),
            env_dim_params=self.env.dim_params,
            device=self.device,
            num_envs=self.env.num_envs
        )
        
    def _learn(self, num_learning_iterations: int):
        ep_infos = []
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)
        for it in range(start_iter, tot_iter):
            sample_infos, obs, critic_obs  = self.sample_rollout(ep_infos, obs, critic_obs)
            collection_time = sample_infos["collection_time"]

            alg_update_infos = self.update(show_pbar=True)
            learn_time = alg_update_infos["learn_time"]
            
            self.current_learning_iteration = it
            if self.logger.log_dir is not None:
                self.logger.log(self, locals())
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        
    @timeit("learn_time")
    def update(self, show_pbar=False):
        total_iters = self.replay_cfg.replay_num_epoches * self.replay_cfg.replay_num_batch_per_epoch if self.replay_cfg.replay_num_batch_per_epoch else None
        generator = self.replay_buffer.mini_batch_generator(
            num_epochs=self.replay_cfg.replay_num_epoches, 
            batch_size=self.replay_cfg.replay_mini_batch_size, 
            max_batches_per_epoch=self.replay_cfg.replay_num_batch_per_epoch
        )
        if show_pbar:
            generator = tqdm.tqdm(generator, desc="update epoch", total=total_iters)

        update_infos = self.alg.update(generator)
        return update_infos

@configclass
class WorldModelRunnerCfg(OffPolicyRunnerCfg):
    runner_type:                type[WorldModelRunner] = WorldModelRunner
    world_model_bundle_cfg:     WorldModelBundleCfg = MISSING
    