from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple
from dataclasses import MISSING

from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg
from GopsEA.networks.mlp import MLP, MLPCfg

class QNetwork(ModuleBase):

    def __init__(self, cfg: QNetworkCfg, state_dim: int, action_dim: int, out_feature: int = 1):
        """
        Args:
            cfg (QNetworkCfg): Configuration for backbone MLP.
            state_dim (int): Input state dimension.
            action_dim (int): Input action dimension.
            out_feature (int): Size of Q output. Usually 1 (scalar Q-value).
        """
        super().__init__()

        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.in_feature = state_dim + action_dim
        self.out_feature = out_feature

        # MLP backbone: concatenated state–action → Q
        self.backbone: MLP = cfg.backbone_cfg.class_type(
            in_feature=self.in_feature,
            out_feature=out_feature,
            cfg=cfg.backbone_cfg
        )

    def forward(
        self, 
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        return self.backbone(torch.concat([state, action], dim=-1))

@configclass
class QNetworkCfg(ModuleBaseCfg):
    """Configuration for the Q-network."""
    class_type: type[nn.Module] = QNetwork
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
        return QNetwork(self, state_dim=dim_params["critic_dim"], action_dim=dim_params["action_dim"])
    