from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBaseCfg
from GopsEA.networks.mlp import MLP, MLPCfg
from GopsEA.components.actor.actor_base import ActorBase
from dataclasses import MISSING

class PiNetwork(ActorBase):
    """
    Policy network π(s): state → action or action distribution parameters.

    Supports state inputs of shape:
        - (B, C_s)      → (B, action_dim) or (B, action_dim*2)
        - (B, L, C_s)   → (B, L, action_dim) or (B, L, action_dim*2)

    If cfg.stochastic = True:
        output = concat([mean, log_std], dim=-1), with shape (..., 2 * action_dim)

    If cfg.stochastic = False:
        output = deterministic action vector, shape (..., action_dim)
    """

    def __init__(
        self,
        cfg: PiNetworkCfg,
        state_dim: int,
        action_dim: int,
    ):
        """
        Args:
            cfg (PiNetworkCfg): Configuration for backbone MLP.
            state_dim (int): Input state feature dimension.
            action_dim (int): Action dimension.
        """
        super().__init__(state_dim=state_dim, action_dim=action_dim)

        self.cfg = cfg
        self.stochastic = cfg.stochastic

        if self.stochastic:
            self.out_feature = action_dim * 2   # mean + log_std
        else:
            self.out_feature = action_dim       # deterministic action

        # Backbone: maps state → policy parameters
        self.backbone: MLP = cfg.backbone_cfg.class_type(
            in_feature=state_dim,
            out_feature=self.out_feature,
            cfg=cfg.backbone_cfg
        )

        # Optional log_std clamping (standard practice)
        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes π(s).

        Args:
            state: Input tensor (B, C_s) or (B, L, C_s)

        Returns:
            If stochastic: Tuple[mean, log_std]
            If deterministic: (B, action_dim) or (B, L, action_dim)
        """

        logits = self.backbone(state)
        if self.stochastic:
            mean, log_std = torch.split(logits, self.action_dim, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
        else:
            return logits

    @torch.no_grad()
    def act_inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for inference/play.
        For deterministic mode, returns the action directly.
        For stochastic mode, returns the mean.
        """
        result = self(state)
        if self.stochastic:
            # Return mean for stochastic mode
            mean, _ = result
            return mean
        else:
            # Return deterministic action
            return result


@configclass
class PiNetworkCfg(ModuleBaseCfg):
    """Configuration for the policy network."""
    class_type: type[nn.Module] = PiNetwork
    backbone_cfg: MLPCfg = MISSING

    # Whether π(s) is Gaussian (mean + log_std)
    stochastic: bool = False

    # Log std range (only for stochastic mode)
    log_std_min: float = -10.0
    log_std_max: float = 2.0
