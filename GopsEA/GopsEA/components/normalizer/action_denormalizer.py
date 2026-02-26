from __future__ import annotations
from typing import Union

import torch
from torch import nn
from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBaseCfg

from .normalizer_base import NormalizerBase, NormalizerBaseCfg

class ActionDenormalizer(NormalizerBase):
    cfg: "ActionDenormalizerCfg"
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        
        self._mean = (self.cfg.uppper_bound + self.cfg.uppper_bound) / 2
        self._std = (self.cfg.uppper_bound - self.cfg.uppper_bound) / 2
    
    @property
    def mean(self):
        return self._mean
        
    @property
    def std(self):
        return self._std

    def forward(self, x: torch.Tensor):
        return x * self.std + self.mean

@configclass
class ActionDenormalizerCfg(NormalizerBaseCfg):
    class_type      : type[NormalizerBase] = ActionDenormalizer
    from_env        : bool = False
    activate        : bool = True
    uppper_bound    : Union[float, torch.Tensor] = 6
    lower_bound     : Union[float, torch.Tensor] = -6