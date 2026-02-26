from __future__ import annotations

import torch
from torch import nn
from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBaseCfg, ModuleBase


class NormalizerBase(ModuleBase):
    @property
    def mean(self):
        ...
        
    @property
    def std(self):
        ...

    def forward(self, x):
        ...
    
@configclass
class NormalizerBaseCfg(ModuleBaseCfg):
    class_type: type[nn.Module] = nn.Identity

    def construct_from_cfg(self, *args, **kwargs):
        """Construct a normalizer instance.

        For the default case where `class_type` is `nn.Identity`, we cannot pass
        the config object into the constructor (since `nn.Identity` expects only
        `inplace`), so we special-case it here. For all other normalizers we
        fall back to the standard ModuleBaseCfg behaviour.
        """
        if self.class_type is nn.Identity:
            return nn.Identity()
        return super().construct_from_cfg(*args, **kwargs)