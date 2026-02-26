from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple
from dataclasses import MISSING

from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg
from GopsEA.networks.mlp import MLP, MLPCfg

class VNetwork(ModuleBase):

    def __init__(self, cfg: VNetworkCfg, state_dim: int, out_feature: int = 1):
        """
        Args:
            cfg (VNetworkCfg): Configuration for backbone MLP.
            state_dim (int): Input state dimension.
            action_dim (int): Input action dimension.
            out_feature (int): Size of Q output. Usually 1 (scalar Q-value).
        """
        super().__init__()

        self.cfg = cfg
        self.state_dim = state_dim
        self.in_feature = state_dim
        self.out_feature = out_feature

        # MLP backbone: concatenated state–action → Q
        self.backbone: MLP = cfg.backbone_cfg.class_type(
            in_feature=self.in_feature,
            out_feature=out_feature,
            cfg=cfg.backbone_cfg
        )

    def forward(
        self, 
        state: torch.Tensor
    ) -> torch.Tensor:
        return self.backbone(state)
    
    def reset(self, *args, **kwargs):
        pass

@configclass
class VNetworkCfg(ModuleBaseCfg):
    """Configuration for the Q-network."""
    class_type: type[nn.Module] = VNetwork
    backbone_cfg: MLPCfg = MLPCfg(
        hidden_features=[256],
        activations=[
            [('LayerNorm', {}), ('ReLU', {})],
            [('LayerNorm', {}), ('ReLU', {})] # prefer
        ]
    )

    def construct_from_cfg(self, *args, dim_params: dict=None, **kwargs):
        if dim_params is None:
            return super().construct_from_cfg(*args, **kwargs)
        return VNetwork(self, state_dim=dim_params["critic_dim"])
    