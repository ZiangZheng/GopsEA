from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from typing import List, Union, Dict, Literal, Tuple
from dataclasses import MISSING
from RenforceRL import configclass
from RenforceRL.utils.template.module_base import ModuleBaseCfg
from RenforceRL.utils.template import ClassTemplateBase
from abc import abstractmethod

class WorldModelBase(nn.Module, ClassTemplateBase):
    def __init__(
            self, cfg: WorldModelBaseCfg, 
            *args, **kwargs
        ):
        super().__init__()
        self.cfg = cfg
        self.use_amp = cfg.use_amp
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32

        self.init_components()
        self.init_loss_fn()
        self.init_optimizers()

    # Helper function for masked mean loss
    @staticmethod
    def masked_mean_loss(loss_fn, prediction, target, mask):
        """Calculates the weighted average loss: sum(Loss * Mask) / sum(Mask)"""
        loss:torch.Tensor = loss_fn(prediction, target)
        while loss.dim() < mask.dim():
            loss = loss.unsqueeze(-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1e-5)

    @abstractmethod
    def init_components(self):
        ...

    def init_loss_fn(self):
        ...

    def init_optimizers(self):
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    @abstractmethod
    def update(
            self, obs: torch.Tensor, action: torch.Tensor, 
            reward: torch.Tensor, next_obs: torch.Tensor, 
            termination: torch.Tensor, is_valid: torch.Tensor=None
        ) -> Dict[str, float]:
        ...

    @abstractmethod
    def predict_next(self, state, action):
        ...

    @abstractmethod
    def evaluate(
        self, is_valid: torch.Tensor, **kwargs
    ) -> Dict[str, float]:
        ...

    @property
    @abstractmethod
    def comp_dim(self) -> Dict[str, int]:
        ...

    def act(self):
        """Act with world model component. """
        raise NotImplementedError("Can not act for this instance")

    def save_world_model(self, path, infos=None):
        saved_dict = {
            "infos": infos,
            "world_model_dict": self.state_dict(),
            "world_optim_state_dict": {
                "optim": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict()
            }
        }
        torch.save(saved_dict, path)

    def load_world_model(self, path, load_optim=False):
        loaded_dict = torch.load(path)
        self.load_state_dict(loaded_dict["world_model_state_dict"])
        if load_optim:
            self.optimizer.load_state_dict(loaded_dict["world_optim_state_dict"]["optim"])
            self.scaler.load_state_dict(loaded_dict["world_optim_state_dict"]["scaler"])
        return loaded_dict["infos"]

@configclass
class WorldModelBaseCfg(ModuleBaseCfg):
    class_type      : type[nn.Module] = WorldModelBase
    use_amp         : bool = True
    # learning_rate: float = 1e-4