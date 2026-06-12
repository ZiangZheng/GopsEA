from __future__ import annotations

import os
from typing import Dict, List, Tuple, Union
import torch
import tqdm
from collections import deque
from dataclasses import MISSING

from RenforceRL import configclass
from RenforceRL.utils.env_wrapper import lab_wrapper
from RenforceRL.utils.logging import timeit
from RenforceRL.algorithms import AlgorithmBaseCfg
from RenforceRL.runners.logger import LoggerBaseCfg
from RenforceRL.runners.base_runner import BaseRunner, BaseRunnerCfg
from RenforceRL.buffer import replay_bundle
from RenforceRL.algorithms.off_policy.sac import SAC, SACCfg
from RenforceRL.components.actor_critic_pack import ActorCriticPackCfg
from RenforceRL.components.normalizer import NormalizerBaseCfg, NormalizerEmpiricalCfg, ActionDenormalizerCfg

class OffPolicyRunner(BaseRunner):
    env: "lab_wrapper.RFDynamicEnvWrapper"
    cfg: "OffPolicyRunnerCfg"
    replay_buffer: "replay_bundle.ReplayBufferBase"
    replay_cfg: "replay_bundle.ReplayBundle"
    def __init__(self, train_cfg: "OffPolicyRunnerCfg", env: "lab_wrapper.RFDynamicEnvWrapper", log_dir=None, device="cpu"):
        self.alg_cfg: SACCfg = train_cfg.algorithm
        self.policy_cfg = train_cfg.policy
        self.replay_cfg = train_cfg.replay_cfg
        super().__init__(train_cfg=train_cfg, env=env, log_dir=log_dir, device=device)

    def init_components(self):
        self.actor_critic = self.policy_cfg.construct_from_cfg(dim_params=self.env.dim_params)
        self.alg: SAC = self.alg_cfg.construct_from_cfg(actor_critic=self.actor_critic, device=self.device)
        self.actor_critic.to(self.device)

        self.obs_normalizer = self.cfg.obs_normalize_cfg.construct_from_cfg(shape=self.env.dim_params["policy_dim"])
        self.critic_normalizer = self.cfg.critic_normalize_cfg.construct_from_cfg(shape=self.env.dim_params["critic_dim"])
        self.obs_normalizer.to(self.device)
        self.critic_normalizer.to(self.device)
        
        self.replay_buffer: "replay_bundle.ReplayBufferBase" = self.replay_cfg.construct_replay_buffer(
            env = self.env,
            device = self.device,
        )
        
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self.logger.init_logger()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
            
        # Prepare buffer terms
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        self._learn(num_learning_iterations = num_learning_iterations)
        
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

            if not self.replay_buffer.is_warmingup():
                alg_update_infos = self.update(show_pbar=True)
                learn_time = alg_update_infos["learn_time"]
            else:
                alg_update_infos = {}
                learn_time = 0
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
        # if show_pbar:
        #     generator = tqdm.tqdm(generator, desc="update epoch", total=total_iters)

        update_infos = self.alg.update(generator)
        return update_infos

    @timeit("collection_time")
    def sample_rollout(self, ep_infos, obs, critic_obs, **kwargs):
        rollout_datas = []
        
        def arrange_obs(obs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            obs = obs_dict.get("policy").to(self.device)
            critic_obs = obs_dict.get("critic", obs).to(self.device)
            obs = self.obs_normalizer(obs)
            critic_obs = self.critic_normalizer(critic_obs)
            return obs, critic_obs
        
        with torch.inference_mode():
            for i in range(self.cfg.num_steps_per_env):
                _obs, _critic_obs = obs, critic_obs
                actions = self.alg.act(obs, critic_obs).to(self.env.device)
                # actions = actions[0]
                obs, reward, done, infos = self.env.step(actions)
                reward, done = reward.to(self.device), done.to(self.device)
                obs, critic_obs = arrange_obs(infos["observations"])
                td_obs, td_critic_obs = arrange_obs(infos["td_observations"])
                
                self.replay_buffer.add(_obs, _critic_obs, actions, reward, td_obs, td_critic_obs, infos["termination"], infos["timeout"])
                self.alg.process_env_step(reward, done, infos)
                rollout_datas.append((obs, critic_obs, actions, reward, done, infos))
                 
                if self.logger.log_dir is not None:
                    if "episode" in infos   : ep_infos.append(infos["episode"])
                    elif "log" in infos     : ep_infos.append(infos["log"])
                    self.cur_reward_sum += reward
                    self.cur_episode_length += 1
                    new_ids = (done > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0
            process_infos = self.process_rollout(rollout_datas)
            sample_infos = {
                "cur_reward_sum": self.cur_reward_sum,
                "cur_episode_length": self.cur_episode_length, 
            }
            sample_infos.update(process_infos)
            return sample_infos, obs, critic_obs

    @timeit("data_process_time")
    def process_rollout(
        self, rollout_datas: 
            List[Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            ]]
        ):
        return super().process_rollout(rollout_datas)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        self.logger.save_model(saved_dict, path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        if device is not None: self.actor_critic.to(device)
        return lambda x: self.actor_critic.act_inference(self.obs_normalizer(x))

    def train_mode(self):
        self.actor_critic.train()
        self.obs_normalizer.train()
        self.critic_normalizer.train()

    def eval_mode(self):
        self.actor_critic.eval()
        self.obs_normalizer.eval()
        self.critic_normalizer.eval()

# not stack dict
def stack_dict(targets: List[dict]):
    out = {}
    for k in targets[0].keys():
         if k not in ["log", "episode"]:
            v = [tar[k] for tar in targets]
            if isinstance(v[0], torch.Tensor): 
                out[k] = torch.stack(v, dim=1)
                if out[k].ndim == 2: out[k] = out[k].unsqueeze(-1)
            else: out[k] = stack_dict(v)
    return out

@configclass
class OffPolicyRunnerCfg(BaseRunnerCfg):
    class_type: type[OffPolicyRunner] = OffPolicyRunner
    
    seed: int = 42
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    
    policy: ActorCriticPackCfg = MISSING
    algorithm: AlgorithmBaseCfg = MISSING
    obs_normalize_cfg: NormalizerBaseCfg = NormalizerEmpiricalCfg()
    critic_normalize_cfg: NormalizerBaseCfg = NormalizerEmpiricalCfg()
    
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    resume: bool = False
    load_checkpoint: str = "model_.*.pt"
    
    logger_cfg: LoggerBaseCfg = LoggerBaseCfg()
        
    replay_cfg: replay_bundle.ReplayBundle = replay_bundle.ReplayPipelineBundle()
