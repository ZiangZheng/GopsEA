from __future__ import annotations
import copy
import torch
from torch.nn import functional as F
from typing import Optional, Tuple, Dict
from dataclasses import MISSING

from RenforceRL import configclass
from RenforceRL.components.world_models.tdmpcs.latent_dynamics_base import (
    LatentDynamicsBase, LatentDynamicsBaseCfg
)

from RenforceRL.networks.optimizer import GroupedOptimizer, GroupedOptimizerCfg
from RenforceRL.utils.logging import timeit

import RenforceRL.utils.math as math

class TDMPC2TaskLatentDynamics(LatentDynamicsBase):
    cfg: TDMPC2TaskLatentDynamicsCfg

    def __init__(self, cfg: TDMPC2TaskLatentDynamicsCfg, obs_dim, action_dim, task_dim, **kwargs):
        super().__init__(cfg, obs_dim, action_dim)

    def init_components(self):
        super().init_components()
        
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def init_optimizers(self):
        # dynamics optimizer — updates encoder, transition, reward_head, term_head
        dyn_params = []
        # dyn_params += list(self.encoder.parameters())
        dyn_params += list(self.transition.parameters())
        dyn_params += list(self.reward_head.parameters())
        dyn_params += list(self.term_head.parameters())

        self.optim_enc = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.dyn_learning_rate,
        )

        self.optim_dyn = torch.optim.Adam(
            dyn_params,
            lr=self.cfg.dyn_learning_rate,
            weight_decay=self.cfg.dyn_weight_decay
        )

        # critic optimizer — updates q_network only
        self.optim_q = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.cfg.q_learning_rate,
            weight_decay=self.cfg.q_weight_decay
        )

        # AMP scaler (optional)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        self.optimizer = GroupedOptimizer(
            {
                "optim_enc": self.optim_enc,
                "optim_dyn": self.optim_dyn,
                "optim_q": self.optim_q
            },
            self.cfg.optimizers_cfg
        )  # for compatibility

    @timeit("update_seq_time")
    def update_seq(self, obs, action, reward, termination, timeout, **kwargs):
        device          = obs.device
        B, T1           = obs.shape[:2]; T = T1 - 1   # number of transitions
        obs             = obs.permute(1, 0, 2)            # [T+1,B,obs_dim]
        action          = action.permute(1, 0, 2)      # [T,B,act_dim]
        reward          = reward.permute(1, 0, 2)      # [T,B,1]
        terminated      = termination.permute(1, 0, 2).float()
        timeout         = timeout.permute(1, 0, 2).float()

        valid_mask = 1.0 - ((terminated + timeout) > 0.5).float()
        valid_mask_bin = valid_mask > 0.5
        z_enc = self.encoder(obs)

        with torch.no_grad():
            z_enc_target = self.target_encoder(obs)
            td_target = reward[:-1] + self.cfg.gamma * (1-terminated[:-1]) * self.target_q_network(z_enc_target[1:], action[1:])
        
        z_pred = torch.empty(T+1, B, self.cfg.latent_dim, device=device)
        z_pred[0], z = z_enc[0], z_enc[0]

        consistency_loss = 0.0
        for t in range(T):
            z = self.transition(torch.concat([z, action[t]], dim=-1))
            z_pred[t+1] = z
            consistency_loss += (
                F.mse_loss(z, z_enc[t+1], reduction='none') * valid_mask[t]
            ).mean() * (self.cfg.rho ** t)
            z = torch.where(valid_mask_bin[t], z, z_enc[t+1])
        consistency_loss /= T

        qs              = self.q_network(z_pred[:-1], action[:-1])  # [num_q,T,B,1]
        reward_preds    = self.reward_head(torch.concat([z_pred[:-1], action[:-1]], dim=-1))      # [T,B,1]
        term_preds      = self.term_head(z_pred[1:])  # [T,B,1]

        rho_weights     = self.cfg.rho ** torch.arange(T, device=reward_preds.device)
        reward_loss     = (((reward_preds - reward[:-1]) ** 2).mean(dim=1) * rho_weights).mean()
        value_loss      = (((qs - td_target) ** 2).mean(dim=1) * rho_weights).mean()
        term_loss       = (F.binary_cross_entropy_with_logits(term_preds, terminated[:-1], reduction='none').mean(dim=1) * rho_weights).mean()

        total_loss = (
            self.cfg.loss_latent_coef * consistency_loss +
            self.cfg.loss_reward_coef * reward_loss +
            self.cfg.loss_q_coef * value_loss +
            self.cfg.loss_term_coef * term_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.clip_grad_norm_()
        self.optimizer.step()
        
        # self.update_pi()
        self._ema_update(self.q_network, self.target_q_network, self.cfg.ema_m)
        self._ema_update(self.encoder, self.target_encoder, self.cfg.ema_m)

        return {
            "consistency_loss": consistency_loss.item(),
            "reward_loss": reward_loss.item(),
            "value_loss": value_loss.item(),
            "termination_loss": term_loss.item(),
            "total_loss": total_loss.item(),
        }

@configclass
class TDMPC2TaskLatentDynamicsCfg(LatentDynamicsBaseCfg):
    class_type: type[TDMPC2TaskLatentDynamics] = TDMPC2TaskLatentDynamics

    loss_latent_coef    : float = 1.0
    loss_reward_coef    : float = 1.0
    loss_term_coef      : float = 1.0
    loss_q_coef         : float = 1.0

    rho         : float = 0.8
    ema_m               : float = 0.995
    gamma               : float = 0.99
