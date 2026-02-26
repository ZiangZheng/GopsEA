from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from GopsEA import configclass
from dataclasses import MISSING

class GroupedOptimizer:
    optimizers  : Dict[str, torch.optim.Optimizer]
    schedulers  : Dict[str, torch.optim.lr_scheduler._LRScheduler]
    cfg         : GroupedOptimizerCfg
    
    def __init__(self, 
                 optimizers: Dict[str, torch.optim.Optimizer], 
                 cfg: GroupedOptimizerCfg,
                 schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = None):
        self.optimizers = optimizers
        self.schedulers = schedulers or {}
        self.grad_cp = cfg.grad_clip_norm
        self.grad_cps = cfg.grad_clip_norms

    def zero_grad(self, *args, **kwargs):
        for opt in self.optimizers.values():
            opt.zero_grad(*args, **kwargs)

    def step(self):
        for opt in self.optimizers.values():
            opt.step()

    def scheduler_step(self):
        """Call LR schedule step for every registered scheduler."""
        for sch in self.schedulers.values():
            sch.step()

    def clip_grad_norm_(self):
        if self.grad_cps:
            for name, opt in self.optimizers.items():
                params = self._get_params(opt)
                if name in self.grad_cps:
                    nn.utils.clip_grad_norm_(params, self.grad_cps[name])
        else:
            all_params = []
            for opt in self.optimizers.values():
                all_params.extend(self._get_params(opt))
            nn.utils.clip_grad_norm_(all_params, self.grad_cp)

    @staticmethod
    def _get_params(opt):
        params = []
        for group in opt.param_groups:
            params.extend(group["params"])
        return params

    # -------------------------
    #   Checkpoint support
    # -------------------------
    def state_dict(self):
        return {
            "optimizers": {
                name: opt.state_dict()
                for name, opt in self.optimizers.items()
            },
            "schedulers": {
                name: sch.state_dict()
                for name, sch in self.schedulers.items()
            }
        }

    def load_state_dict(self, state_dict):
        for name, opt in self.optimizers.items():
            if name in state_dict["optimizers"]:
                opt.load_state_dict(state_dict["optimizers"][name])
        for name, sch in self.schedulers.items():
            if name in state_dict["schedulers"]:
                sch.load_state_dict(state_dict["schedulers"][name])

    def set_learning_rate(self, lr_dict: Dict[str, float]):
        for name, lr in lr_dict.items():
            opt: torch.optim.Optimizer = self.optimizers[name]
            for param_group in opt.param_groups:
                param_group["lr"] = lr
    
@configclass
class GroupedOptimizerCfg:
    # ------- Grad Clip -------
    grad_clip_norm: float = 10.0
    grad_clip_norms: Dict[str, float] = None
