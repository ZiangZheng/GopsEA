from __future__ import annotations

import os
import time
import torch
from collections import deque
from GopsEA import configclass
from dataclasses import MISSING

from GopsEA.utils.logging import timeit
from GopsEA.algorithms import AlgorithmBaseCfg
from GopsEA.components.actor_critic_pack import ActorCriticPackCfg
from GopsEA.runners.logger import LoggerBaseCfg
from GopsEA.runners.base_runner import BaseRunner, BaseRunnerCfg
from GopsEA.utils.env_wrapper.lab_wrapper.robo_env_wrapper import GopsEAEnvWrapper

from GopsEA.algorithms.on_policy.ppo import PPO, PPOCfg
from GopsEA.components.normalizer import NormalizerBaseCfg

class OnPolicyRunner(BaseRunner):
    env: "GopsEAEnvWrapper"
    cfg: "OnPolicyRunnerCfg"
    def __init__(self, train_cfg: "OnPolicyRunnerCfg", env: "GopsEAEnvWrapper", log_dir=None, device="cpu"):
        self.alg_cfg: PPOCfg = train_cfg.algorithm
        self.policy_cfg = train_cfg.policy
        super().__init__(train_cfg=train_cfg, env=env, log_dir=log_dir, device=device)

    def init_components(self):
        num_obs, num_critic_obs = self.env.dim_params["policy_dim"], self.env.dim_params["critic_dim"]
        self.actor_critic = self.policy_cfg.construct_from_cfg(dim_params=self.env.dim_params)
        self.alg: PPO = self.alg_cfg.construct_from_cfg(actor_critic=self.actor_critic, device=self.device)
        self.alg.init_storage(
            self.env.num_envs,
            self.cfg.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        self.obs_normalizer = self.cfg.obs_normalize_cfg.construct_from_cfg(shape=self.env.dim_params["policy_dim"])
        self.critic_normalizer = self.cfg.critic_normalize_cfg.construct_from_cfg(shape=self.env.dim_params["critic_dim"])
        self.obs_normalizer.to(self.device)
        self.critic_normalizer.to(self.device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self.logger.init_logger()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
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
            # log_infos.update(sample_infos)

            start = time.time()
            alg_update_infos = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            self.current_learning_iteration = it
            if self.logger.log_dir is not None:
                self.logger.log(self, locals())
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    @timeit("collection_time")
    def sample_rollout(self, ep_infos, obs, critic_obs, **kwargs):
        rollout_datas = []
        with torch.inference_mode():
            for i in range(self.cfg.num_steps_per_env):
                actions = self.alg.act(obs, critic_obs)
                obs, reward, done, infos = self.env.step(actions.to(self.env.device))
                critic_obs = infos["observations"].get("critic", obs)
                obs, critic_obs, reward, done = (
                    obs.to(self.device),
                    critic_obs.to(self.device),
                    reward.to(self.device),
                    done.to(self.device),
                )
                obs = self.obs_normalizer(obs)
                critic_obs = self.critic_normalizer(critic_obs)
                self.process_env_step(reward, done, infos)
                
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
            self.alg.compute_returns(critic_obs)
            process_infos = self.process_rollout(rollout_datas)
            sample_infos = {
                "cur_reward_sum": self.cur_reward_sum,
                "cur_episode_length": self.cur_episode_length, 
            }
            sample_infos.update(process_infos)
            return sample_infos, obs, critic_obs
        
    def process_env_step(self, reward, done, infos, **kwargs):
        self.alg.process_env_step(reward, done, infos)

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

@configclass
class OnPolicyRunnerCfg(BaseRunnerCfg):
    class_type: type[OnPolicyRunner] = OnPolicyRunner
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    
    policy: ActorCriticPackCfg = MISSING
    algorithm: AlgorithmBaseCfg = MISSING
    obs_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    critic_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    resume: bool = False
    load_checkpoint: str = "model_.*.pt"
    
    logger_cfg: LoggerBaseCfg = LoggerBaseCfg()
