from __future__ import annotations
import math
from typing import Any

import torch
import torch.nn as nn
import torch.distributions as D

from RenforceRL import configclass
from RenforceRL.networks.mlp import MLPCfg
from RenforceRL.utils.template.module_base import ModuleBaseCfg
from RenforceRL.components.actor.actor_base import ActorBase
from dataclasses import MISSING

beyondmimic_dof_pos_lower_limit_list=[
        -2.5307,
        -0.5236,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.5307,
        -2.9671,
        -2.7576,
        -0.087267,
        -0.87267,
        -0.2618,
        -2.618,
        -0.52,
        -0.52,
        -3.0892,
        -1.5882,
        -2.618,
        -1.0472,
        -1.972222054,
        -1.61443,
        -1.61443,
        -3.0892,
        -2.2515,
        -2.618,
        -1.0472,
        -1.972222054,
        -1.61443,
        -1.61443,
    ]

beyondmimic_dof_pos_upper_limit_list=[
        2.8798,
        2.9671,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.8798,
        0.5236,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.618,
        0.52,
        0.52,
        2.6704,
        2.2515,
        2.618,
        2.0944,
        1.972222054,
        1.61443,
        1.61443,
        2.6704,
        1.5882,
        2.618,
        2.0944,
        1.972222054,
        1.61443,
        1.61443,
    ]

beyondmimic_default_joint_angles={
            "left_hip_pitch_joint": -0.312,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.669,
            "left_ankle_pitch_joint": -0.363,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.312,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.669,
            "right_ankle_pitch_joint": -0.363,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.6,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.6,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

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
    action_scaling_factors: torch.Tensor
    action_bias_factors: torch.Tensor

    def _build_action_scaling_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-DOF action scaling factors using preset default values:
            action_scaling_factors = max_range / action_scale
        """
        # Preset joint limits / default angles (can be replaced by robot config later)
        dof_pos_lower_limits = torch.tensor(beyondmimic_dof_pos_lower_limit_list)
        dof_pos_upper_limits = torch.tensor(beyondmimic_dof_pos_upper_limit_list)
        default_angles = torch.tensor(list(beyondmimic_default_joint_angles.values()))

        # Keep a scalar environment action scale for compatibility
        action_scale = 1.0

        range_to_lower = torch.abs(dof_pos_lower_limits - default_angles)
        range_to_upper = torch.abs(dof_pos_upper_limits - default_angles)
        max_range = torch.maximum(range_to_lower, range_to_upper)

        action_scaling_factors = max_range / action_scale

        # range_to_lower = dof_pos_lower_limits - default_angles
        # range_to_upper = dof_pos_upper_limits - default_angles
        # action_scaling_factors = 0.5 * (range_to_upper - range_to_lower)
        # action_bias_factors = 0.5 * (range_to_upper + range_to_lower)

        action_bias_factors = torch.zeros(29)
        return action_scaling_factors, action_bias_factors

    def init_action_scale(self):
        action_scaling_factors, action_bias_factors = self._build_action_scaling_factors()
        self.register_buffer("action_scaling_factors", action_scaling_factors)
        self.register_buffer("action_bias_factors", action_bias_factors)

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
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
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
            action   : scaled (and biased) action in env action space
            log_prob : log π(a|s), shape [B]
        """
        dist = self._update_dist(state)

        if deterministic:
            u = dist.mean
        else:
            u = dist.rsample()   # reparameterization

        if self.cfg.use_tanh:
            # raw squashed action in [-1, 1]
            a_tanh = torch.tanh(u)
            a = a_tanh * self.action_scaling_factors

            # ---- log prob ----
            log_prob_u = dist.log_prob(u)
            # tanh Jacobian correction
            log_prob = log_prob_u - torch.log(1 - a_tanh.pow(2) + 1e-6)
            # scale correction uses per-DOF action scaling factors
            log_prob -= torch.log(self.action_scaling_factors + 1e-6)

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
            log_prob_u -= torch.log(self.action_scaling_factors + 1e-6)
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
            return torch.tanh(dist.mean) * self.action_scaling_factors + self.action_bias_factors
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
