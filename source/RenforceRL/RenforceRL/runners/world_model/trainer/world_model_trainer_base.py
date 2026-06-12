from __future__ import annotations

import torch
import tqdm
from typing import List, Union, Dict, Literal, Tuple
from dataclasses import MISSING
from RenforceRL import configclass

from RenforceRL.components.world_models.world_model_base import WorldModelBase, WorldModelBaseCfg
from RenforceRL.utils.mapping import tranverse_dict_value, transform_dict_mapping, check_mapping_valid
from RenforceRL.utils.logging import timeit
from RenforceRL.runners.world_model.trainer.evaluators import BaseEvaluator, BaseEvaluatorCfg
from RenforceRL.utils.template import ClassTemplateBase

from RenforceRL.buffer import DataPipeline


class WorldModelTrainerBase(ClassTemplateBase):
    """
    Manage replay buffer and 
    """
    cfg: WorldModelTrainerBaseCfg
    def __init__(
        self, 
        cfg:WorldModelTrainerBaseCfg, 
        world_model:WorldModelBase, 
        device,
        **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.world_model: WorldModelBase = world_model
        self.world_model.to(self.device)
            
    @timeit("update_time_total")
    def update(self, *args, **kwargs):
        raise NotImplementedError("Unimplemented API.")   
            
    def process_step(self, obs, action, reward, done, infos, process_step_method="traj"):
        if process_step_method == "traj":
            self._process_traj(obs, action, reward, done, infos)
        elif process_step_method == "transition":
            self._process_step(obs, action, reward, done, infos)
        elif process_step_method == "chunk":
            self._process_chunk(obs, action, reward, done, infos)
        else:
            raise NotImplementedError(f"process_step_method: {process_step_method} unsupport.")
            
    def _process_traj(self, obs, action, reward, done, infos):
        raise NotImplementedError("Unimplemented API.")
            
    def _process_step(self, obs, action, reward, done, infos):
        samples = {
            "timeout":      infos["timeout"],
            "termination":  infos["termination"],
            "policy":       infos["observations"]["policy"],
            "critic":       infos["observations"].get("critic", None),
            "dynamic":      infos["observations"].get("dynamic", None),
            "action":       action,
            "reward":       reward,
            "rewards":      infos["rewards"],
        }
        # move to device
        samples = transform_dict_mapping(samples, self.cfg.data_pipeline.MAPPING_PROCESS_ENV_SAMPLE_2_REPLAYBUFFER_DATA)
        samples = tranverse_dict_value(samples, lambda v: v.to(self.device) if v is not None else None)
        return samples

    def _process_chunk(self, obs, action, reward, done, infos):
        raise NotImplementedError("Unimplemented API.")
            
    # ----------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------
    def setup_evaluators(self, evaluators: List[Union[BaseEvaluator, BaseEvaluatorCfg]]):
        ret = []
        for evaluator in evaluators:
            if isinstance(evaluator, BaseEvaluator): ret.append(BaseEvaluator)
            elif isinstance(evaluator, BaseEvaluatorCfg): 
                ret.append(BaseEvaluator.construct_from_cfg(evaluator))
        self.evaluators = ret
        return ret
    
    planner = None
    evaluators: List[BaseEvaluator] = []
    @torch.no_grad()
    def evaluate(self, **kwargs):
        total_tb = {}
        for evaluator in self.evaluators:
            _eval_tb = evaluator.evaluate(world_model=self.world_model, planner=self.planner, **kwargs)
            total_tb.update(_eval_tb)
        return total_tb
            
    def act(self, obs, **kwargs):
        return None
            
    def save_ckpt(self, path, infos=None):
        save_dict = {
            "infos": infos,
            "world_model_dict": self.world_model.state_dict(),
            "world_optim_state_dict": {
                "optim": self.world_model.optimizer.state_dict(),
                "scaler": self.world_model.scaler.state_dict()
            }
        }
        torch.save(save_dict, path)
            
@configclass
class WorldModelTrainerBaseCfg:
    class_type      : type[WorldModelTrainerBase] = WorldModelTrainerBase
    data_pipeline   : DataPipeline = MISSING
