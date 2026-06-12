from __future__ import annotations
from .base_evaluator import BaseEvaluator, BaseEvaluatorCfg
from typing import Dict, List
from collections import defaultdict
import torch
from RenforceRL.utils.logging import timeit

from RenforceRL import configclass
from RenforceRL.utils.isaaclab.trajectory import load_hdf5_trajectories

from dataclasses import MISSING

class DatasetEvaluator(BaseEvaluator):
    cfg: DatasetEvaluatorCfg
    def __init__(self, cfg: DatasetEvaluatorCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.dataset_path = cfg.dataset_path
        self.load_dataset()

    # ----------------------------------------------------------------------
    # Dataset loading
    # ----------------------------------------------------------------------
    def load_dataset(self) -> Dict[str, torch.Tensor]:
        """
        Dataset example:
        {
            "reward": (T,),
            "termination": (T,),
            "timeout": (T,),
            "action": (T, A),
            "policy": (T, ObsDim),
            "dynamic": (T, LatentDim?) optional,
        }
        """
        generator = load_hdf5_trajectories(self.dataset_path)
        self.dataset: List[Dict[str, torch.Tensor]] = [i for idx, i in enumerate(generator) if idx < self.cfg.max_traj_load]
    
@configclass
class DatasetEvaluatorCfg(BaseEvaluatorCfg):
    class_type: type[DatasetEvaluator] = DatasetEvaluator
    # Add dataset-specific configuration parameters here if needed
    dataset_path: str = MISSING  # Path to the dataset for evaluation
    max_traj_load: int = MISSING