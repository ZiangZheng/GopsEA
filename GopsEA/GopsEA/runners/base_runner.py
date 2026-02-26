from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple, Union
import torch
from collections import deque
from GopsEA import configclass
from dataclasses import MISSING

from GopsEA.utils.logging import timeit
from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg
from GopsEA.utils.env_wrapper.lab_wrapper.robo_env_wrapper import GopsEAEnvWrapper
from GopsEA.runners.logger import LoggerBase, LoggerBaseCfg

from GopsEA.components.normalizer import NormalizerBaseCfg

class BaseRunner(ModuleBase):
    env: "GopsEAEnvWrapper"
    logger: "LoggerBase"
    def __init__(self, train_cfg: "BaseRunnerCfg", env: "GopsEAEnvWrapper", log_dir=None, device="cpu"):
        super().__init__()
        self.cfg = train_cfg
        self.device = device
        self.env = env
        
        self.log_dir =  log_dir
        self.logger = self.cfg.logger_cfg.construct_from_cfg(log_dir=log_dir)
        self.current_learning_iteration = 0
        self.init_components()
        
    def init_components(self):
        ...

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        ...

    @timeit("collection_time")
    def sample_rollout(self, ep_infos, obs, critic_obs, **kwargs):
        ...

    def save(self, path, infos=None):
        ...

    def load(self, path, load_optimizer=True):
        ...

    def get_inference_policy(self, device=None):
        ...

    def process_env_step(self, *args, **kwargs):
        pass

    @timeit("data_process_time")
    def process_rollout(
        self, rollout_datas: 
            List[Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            ]]
        ):
        L = len(rollout_datas); obs_list, cri_list, act_list, rew_list, done_list, extra_list = zip(*rollout_datas)
        # rollout_obs  = torch.stack(obs_list,  dim=1)
        # rollout_cri  = torch.stack(cri_list,  dim=1)
        rollout_act  = torch.stack(act_list,  dim=1)
        rollout_rew  = torch.stack(rew_list,  dim=1).unsqueeze(-1)
        # rollout_done = torch.stack(done_list, dim=1).unsqueeze(-1)
        rollout_timeout = torch.stack([i["timeout"] for i in extra_list], dim=1).unsqueeze(-1)
        rollout_termination = torch.stack([i["termination"] for i in extra_list], dim=1).unsqueeze(-1)
        
        # rollout_extra = stack_dict(extra_list)
        return self.analyze_rollout_distribution(
            actions = rollout_act,
            rewards = rollout_rew,
            timeout = rollout_timeout,
            termination = rollout_termination
        )


    @staticmethod
    @torch.no_grad()
    def analyze_rollout_distribution(
        actions: torch.Tensor,        # [B, T, act_dim]
        rewards: torch.Tensor,        # [B, T, 1]
        timeout: torch.Tensor,        # [B, T, 1]
        termination: torch.Tensor,    # [B, T, 1]
        tanh_sat_threshold: float = None # 0.95,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze rollout sampling distribution for SAC-style algorithms.

        This function is intentionally stateless and side-effect free.
        """

        # ---------------------------------------------------------
        # Action statistics
        # ---------------------------------------------------------
        act_abs = actions.abs()

        act_stats = {
            "act_abs_mean": act_abs.mean(),
            "act_abs_max": act_abs.max(),
            "act_std": actions.std(dim=(0, 1)).mean(),
        }
        act_stats_tanh = {
            # tanh saturation: policy collapsing to boundary actions
            "act_tanh_sat_ratio": (act_abs > tanh_sat_threshold).float().mean(),
        } if tanh_sat_threshold is not None else {}

        # ---------------------------------------------------------
        # Reward statistics
        # ---------------------------------------------------------
        rew_stats = {
            "rew_min": rewards.min(),
            "rew_max":  rewards.max(),
            "rew_mean": rewards.mean(),
            "rew_std": rewards.std(),
            "rew_nonzero_ratio": (rewards.abs() > 1e-5).float().mean(),
            "rew_positive_ratio": (rewards > 0).float().mean(),
        }

        # ---------------------------------------------------------
        # Termination / timeout statistics
        # ---------------------------------------------------------
        done_stats = {
            "timeout_ratio": timeout.float().mean(),
            "termination_ratio": termination.float().mean(),
            # should be strictly zero
            "both_done_ratio": (
                timeout.bool() & termination.bool()
            ).float().mean(),
        }

        return {
            **act_stats,
            **act_stats_tanh,
            **rew_stats,
            **done_stats,
        }

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()

@configclass
class BaseRunnerCfg(ModuleBaseCfg):
    class_type: type[BaseRunner] = BaseRunner
    
    seed: int = 42
    """The seed for the experiment. Default is 42."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""

    resume: bool = False
    """Whether to resume. Default is False."""

    load_checkpoint: str = "model_.*.pt"
    
    logger_cfg: LoggerBaseCfg = LoggerBaseCfg()
    obs_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    critic_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
