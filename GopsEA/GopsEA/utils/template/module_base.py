from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import MISSING
from GopsEA import configclass

from GopsEA.utils.template import ClassTemplateBaseCfg
from contextlib import contextmanager

class ModuleBase(nn.Module):
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
    
    def defreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)

    @contextmanager
    def frozen(self):
        try:
            self.freeze()
            yield self
        finally:
            self.defreeze()

@configclass
class ModuleBaseCfg(ClassTemplateBaseCfg):
    class_type: type[nn.Module] = MISSING