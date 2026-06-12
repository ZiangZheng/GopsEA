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
from RenforceRL.components.actor import GaussianActor, GaussianActorCfg

from RenforceRL.networks.optimizer import GroupedOptimizer, GroupedOptimizerCfg
from RenforceRL.utils.logging import timeit

import RenforceRL.utils.math as math

class TDMPC2OffLatentDynamics(LatentDynamicsBase):
    cfg: TDMPC2OffLatentDynamicsCfg
    pi_network: GaussianActor
    
    def __init__(self, cfg: TDMPC2OffLatentDynamicsCfg, obs_dim, action_dim, **kwargs):
        super().__init__(cfg, obs_dim, action_dim)

    def init_components(self):
        super().init_components()
        
        self.pi_network  : GaussianActor = \
            GaussianActor(self.cfg.actor_cfg, state_dim=self.latent_dim, action_dim=self.action_dim)

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
        
        self.optim_pi = torch.optim.Adam(
            self.pi_network.parameters(),
            lr=self.cfg.pi_learning_rate,
            weight_decay=self.cfg.pi_weight_decay
        )

    def update_pi(self, z_seq: torch.Tensor):
        T, B, _ = z_seq.shape
        action_dist = self.pi_network.forward(z_seq)
        action_seq = action_dist.rsample()
        action_entropy = action_dist.entropy()
        with self.q_network.frozen():
            q_seq = self.q_network.forward(z_seq, action_seq)
        rho_weights = self.cfg.rho ** torch.arange(T, device=action_seq.device)
        pi_loss = (-(self.cfg.entropy_coef * action_entropy + q_seq).mean(dim=(1,2)) * rho_weights).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.pi_network.parameters(), self.cfg.pi_grad_clip_norm)
        self.optim_pi.step()
        self.optim_pi.zero_grad()
        return pi_loss

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
            pi_action = self.pi_network.sample(z_enc, deterministic=True)
            td_target = reward[:-1] + self.cfg.gamma * (1-terminated[:-1]) * self.target_q_network(z_enc[1:], pi_action[1:])
        
        z_pred = torch.empty(T+1, B, self.cfg.latent_dim, device=device)
        z_pred[0], z = z_enc[0], z_enc[0]

        consistency_loss = 0.0
        for t in range(T):
            z = self.transition(torch.concat([z, action[t]], dim=-1))
            z_pred[t+1] = z
            # consistency_loss += (
            #     F.mse_loss(z, z_enc[t+1], reduction='none') * valid_mask[t]
            # ).mean() * (self.cfg.rho ** t)
            # z = torch.where(valid_mask_bin[t], z, z_enc[t+1])
            consistency_loss += (
                F.mse_loss(z, z_enc[t+1], reduction='none')
            ).mean() * (self.cfg.rho ** t)
            
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
        
        self._ema_update(self.q_network, self.target_q_network, self.cfg.ema_m)
        # pi_loss = self.update_pi(z_pred.detach())
        pi_loss = self.update_pi(z_enc.detach())
        
        return {
            "consistency_loss": consistency_loss.item(),
            "reward_loss": reward_loss.item(),
            "value_loss": value_loss.item(),
            "termination_loss": term_loss.item(),
            "total_loss": total_loss.item(),
            "pi_loss": pi_loss.item(),
        }
        
    def act(self, obs):
        action_dist = self.pi_network(self.encoder(obs))
        return action_dist.rsample()

@configclass
class TDMPC2OffLatentDynamicsCfg(LatentDynamicsBaseCfg):
    class_type: type[TDMPC2OffLatentDynamics] = TDMPC2OffLatentDynamics

    actor_cfg           : GaussianActorCfg = MISSING

    loss_latent_coef    : float = 1.0
    loss_reward_coef    : float = 1.0
    loss_term_coef      : float = 1.0
    loss_q_coef         : float = 1.0

    rho                 : float = 0.8
    ema_m               : float = 0.995
    gamma               : float = 0.99
    entropy_coef        : float = 0.5

    # ------- Optimizers -------
    dyn_learning_rate            : float = 1e-4           # dynamics optimizer
    dyn_weight_decay             : float = 0.05
    q_learning_rate              : float = 1e-4         # critic optimizer
    q_weight_decay               : float = 0.05
    pi_learning_rate             : float = 1e-4         # actor optimizer
    pi_weight_decay              : float = 0.05
    
    optimizers_cfg               : GroupedOptimizerCfg = GroupedOptimizerCfg(
        grad_clip_norm=10.0,
        grad_clip_norms={
            "optim_dyn": 10.0,
            "optim_q": 1.0
        }
    )
    
    pi_grad_clip_norm: int = 5.0