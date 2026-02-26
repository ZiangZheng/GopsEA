from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as D

from GopsEA import configclass
from GopsEA.networks.mlp import MLPCfg
from GopsEA.utils.template.module_base import ModuleBaseCfg
from GopsEA.components.actor.actor_base import ActorBase
from dataclasses import MISSING


class SACActor(ActorBase):
    """
    SAC-style Gaussian actor with tanh-squash and state-dependent std.

    π(a|s) = tanh( N(μ(s), o(s)) )
    """

    def __init__(
        self,
        cfg: SACActorCfg,
        state_dim: int,
        action_dim: int,
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)

        self.cfg = cfg
        self.eps = 1e-6

        # ------------------------------------------------------------ #
        # Backbone
        # ------------------------------------------------------------ #
        self.backbone = cfg.backbone_cfg.class_type(
            in_feature=state_dim,
            out_feature=cfg.hidden_dim,
            cfg=cfg.backbone_cfg,
        )

        # Mean / log-std heads
        self.mean_head = nn.Linear(cfg.hidden_dim, action_dim)
        self.log_std_head = nn.Linear(cfg.hidden_dim, action_dim)

        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max

        self.act_dist: D.Normal | None = None
        
        self.init_action_scale()


    action_scale: torch.Tensor
    action_bias: torch.Tensor
    def init_action_scale(self):
        if self.cfg.action_scale is not None:
            self.register_buffer("action_scale", torch.tensor(self.cfg.action_scale))
        else:
            self.register_buffer("action_scale", torch.ones(self.action_dim))

        if self.cfg.action_bias is not None:
            self.register_buffer("action_bias", torch.tensor(self.cfg.action_bias))
        else:
            self.register_buffer("action_bias", torch.zeros(self.action_dim))

    # ------------------------------------------------------------ #
    # Forward: compute mean, std
    # ------------------------------------------------------------ #
    def forward(self, state: torch.Tensor):
        h = self.backbone(state)
        # h1, h2 = torch.chunk(h, 2, dim=-1)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats
        
        std = log_std.exp()
        return mean, std

    def _update_dist(self, state: torch.Tensor):
        mean, std = self(state)
        self.act_dist = D.Normal(mean, std)
        return self.act_dist

    # ------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------ #
    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action   : tanh-squashed action in [-1, 1]
            log_prob : log π(a|s), shape [B]
        """
        dist = self._update_dist(state)

        if deterministic:
            u = dist.mean
        else:
            u = dist.rsample()   # reparameterization

        if self.cfg.use_tanh:
            a = torch.tanh(u)
            # ---- log prob ----
            log_prob_u = dist.log_prob(u)
            # tanh Jacobian correction
            log_prob = log_prob_u - torch.log(1 - a.pow(2) + 1e-6)
            log_prob -= torch.log(self.action_scale + 1e-6)
            
            a = a * self.action_scale
        else:
            # Non-tanh case
            a = u
            log_prob = dist.log_prob(a)

        log_prob = log_prob.sum(dim=-1)
        return a, log_prob

    # ------------------------------------------------------------ #
    # Log-prob with tanh correction
    # ------------------------------------------------------------ #
    def get_actions_log_prob(
        self,
        action: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute log π(a|s) where a is already tanh-squashed.
        """
        if state is not None:
            self._update_dist(state)

        assert self.act_dist is not None, "Distribution not initialized"

        # Inverse tanh: a -> u
        if self.cfg.use_tanh:
            eps = 1e-6
            a = torch.clamp(action, -1 + eps, 1 - eps)
            u = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh

            # log N(u | mean, std)
            log_prob_u = self.act_dist.log_prob(u).sum(dim=-1)
            # tanh Jacobian correction
            log_prob_u -= torch.sum(
                torch.log(1 - a.pow(2) + eps),
                dim=-1
            )
            log_prob_u -= torch.log(self.action_scale + 1e-6)
            return log_prob_u
        else:
            return self.act_dist.log_prob(action).sum(dim=-1)

    # ------------------------------------------------------------ #
    # Deterministic action (eval)
    # ------------------------------------------------------------ #
    def act(self, state: torch.Tensor) -> torch.Tensor:
        self._update_dist(state)
        return torch.tanh(self.act_dist.mean)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.act_dist.mean

    def reset(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def act_inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for inference/play.
        Returns tanh-squashed mean action.
        """
        dist = self._update_dist(state)
        if self.cfg.use_tanh:
            return torch.tanh(dist.mean) * self.action_scale + self.action_bias
        else:
            return dist.mean



@configclass
class SACActorCfg(ModuleBaseCfg):
    """Configuration for SACActor."""

    class_type: type[nn.Module] = SACActor

    # backbone outputs hidden features, not action dim
    backbone_cfg: MLPCfg = MLPCfg(
        hidden_features=[512, 256, 128],
        activations=[
            [('SiLU', {})],
        ] * 4,
    )

    action_scale: float = 1
    action_bias: float = 0
    use_tanh: bool = True

    hidden_dim: int = 32

    # log-std clamp (SAC-stable range)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    def construct_from_cfg(self, *args, dim_params: dict = None, **kwargs):
        if dim_params is None:
            return super().construct_from_cfg(*args, **kwargs)

        return SACActor(
            self,
            dim_params["policy_dim"],
            dim_params["action_dim"],
        )
