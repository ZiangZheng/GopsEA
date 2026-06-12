from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple
from dataclasses import MISSING

from RenforceRL import configclass
from RenforceRL.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RenforceRL.networks.mlp import MLP, MLPCfg

class GaussianQNetwork(ModuleBase):

    def __init__(self, cfg: GaussianQNetworkCfg, state_dim: int, action_dim: int, out_feature: int = 2):
        """
        Args:
            cfg (GaussianQNetworkCfg): Configuration for backbone MLP.
            state_dim (int): Input state dimension.
            action_dim (int): Input action dimension.
            out_feature (int): Size of Q output. Usually 2 (Gaussian Q-value).
        """
        super().__init__()

        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.in_feature = state_dim + action_dim
        self.out_feature = out_feature

        # MLP backbone: concatenated state–action → q_mean, q_std
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
        input = torch.concat([state, action], dim=-1)
        output = self.backbone(input)
        q_mean, q_std = torch.chunk(output, 2, dim=-1)
        q_std = torch.nn.functional.softplus(q_std)
        return q_mean, q_std

@configclass
class GaussianQNetworkCfg(ModuleBaseCfg):
    """Configuration for the Q-network."""
    class_type: type[nn.Module] = GaussianQNetwork
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
        return GaussianQNetwork(self, state_dim=dim_params["critic_dim"], action_dim=dim_params["action_dim"])
    