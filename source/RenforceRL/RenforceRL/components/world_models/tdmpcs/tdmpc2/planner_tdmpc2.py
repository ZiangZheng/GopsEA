from __future__ import annotations
from typing import Optional, Callable, Literal, Tuple
import torch
import torch.nn as nn

from RenforceRL import configclass
from RenforceRL.components.world_models.planner_base import PlannerBase, PlannerBaseCfg
from RenforceRL.utils.logging import timeit
from .tdmpc2_latent_dynamics import TDMPC2LatentDynamics
from .jit_utils import _update_mean_std

class PlannerTDMPC2(PlannerBase):
    def __init__(self, cfg: PlannerTDMPC2Cfg, action_dim: int):
        self.cfg = cfg
        self.action_dim = action_dim
        self._mean = None
        self._std = None
        self._prior_sequences = None
        self._prior_mode = cfg.prior_mode
        self.prior_mean_action_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self._scratch = {}

    def set_prior_mean_action_fn(self, fn: Optional[Callable[[torch.Tensor], torch.Tensor]]):
        self.prior_mean_action_fn = fn

    @torch.no_grad()
    def _warm_start(self, latent: torch.Tensor, obs: torch.Tensor):
        """Return (mean, std) initialized from previous mean/std or fresh."""
        B = latent.shape[0]
        H = self.cfg.horizon
        A = self.action_dim
        dev = latent.device

        if not (self.cfg.warm_start and self._mean is not None and self._std is not None) or self._mean.shape[0] != B:
            return self._fresh_init(B, H, A, dev)

        mean = self._mean.to(dev)
        std = self._std.to(dev)

        mean = torch.roll(mean, shifts=-1, dims=1)
        std = torch.roll(std, shifts=-1, dims=1)

        if self.prior_mean_action_fn is not None:
            last = self.prior_mean_action_fn(obs)
            if last.ndim == 1:
                last = last.unsqueeze(0)
            mean[:, -1] = last.to(dev)

        std[:, -1].clamp_(min=self.cfg.min_std, max=self.cfg.max_std)
        return mean, std

    def _fresh_init(self, B, H, A, device):
        mean = torch.zeros((B, H, A), device=device)
        std = torch.full((B, H, A), self.cfg.init_std, device=device)
        self._prior_sequences = None
        return mean, std

    @torch.no_grad()
    def _inject_prior(self, seq, obs, device):
        """Inject prior action sequences into seq[:, :prior_count]."""
        B, P, H, A = seq.shape
        prior_count = int(self.cfg.prior_fraction * P)

        if prior_count <= 0:
            return

        if self._prior_mode == "policy" and self._prior_sequences is not None:
            K = min(prior_count, self._prior_sequences.shape[1])
            seq[:, :K].copy_(self._prior_sequences[:, :K].to(device))

            extra = prior_count - K
            if extra > 0 and self.prior_mean_action_fn is not None:
                prior = self.prior_mean_action_fn(obs).view(B,1,1,A).expand(B,extra,H,A).to(device)
                seq[:, K:prior_count].copy_(prior)
            return

        if self.prior_mean_action_fn is not None:
            prior = self.prior_mean_action_fn(obs).view(B,1,1,A).expand(B,prior_count,H,A).to(device)
            seq[:, :prior_count].copy_(prior)

    @torch.no_grad()
    def _rollout(self, world_model, seq, cur_z, gamma, scratch):
        B, P, H, A = seq.shape
        dev = seq.device

        reward_acc = scratch['reward_acc']
        alive = scratch['alive']
        reward_acc.fill_(0.0)
        alive.fill_(1.0)

        for t in range(H):
            a_t = seq[:, :, t].reshape(B * P, A)
            next_z, r_hat, term_logits, _ = world_model.predict_next(cur_z, a_t)

            reward_acc.add_(r_hat.reshape(-1) * alive)
            dead = (torch.sigmoid(term_logits.reshape(-1)) > self.cfg.term_threshold).to(alive.dtype)
            alive.mul_(1.0 - dead)
            cur_z = next_z.reshape(B * P, -1)

        last_actions = seq[:, :, -1].reshape(B * P, A)
        q_term = world_model.q_network(cur_z, last_actions).reshape(B * P)
        q_term.mul_(alive)

        returns = reward_acc + (gamma ** H) * q_term
        return returns.view(B, P), q_term.view(B, P)

    @torch.no_grad()
    def _mppi_iteration(
        self, mean, std, flat_z0, world_model, obs, scratch, gamma
    ):
        B, H, A = mean.shape
        P = self.cfg.popsize
        dev = mean.device

        noise = scratch["noise"]
        noise.copy_(torch.randn(noise.shape, device=dev, dtype=noise.dtype))

        seq = scratch["seq"]
        seq.copy_(mean.unsqueeze(1))
        seq.add_(std.unsqueeze(1) * noise)
        seq = self._clamp_actions(seq, self.cfg.action_min, self.cfg.action_max)

        self._inject_prior(seq, obs, dev)
        returns, q_term = self._rollout(world_model, seq, flat_z0.clone(), gamma, scratch)
        scores = returns  # TODO if you add prior_reg -> modify here
        new_mean, new_std, w = _update_mean_std(
            seq, scores, float(self.cfg.mppi_beta),
            float(self.cfg.eps), float(self.cfg.min_std), float(self.cfg.max_std)
        )
        best_idx = scores.argmax(dim=1)
        ar = torch.arange(B, device=dev)
        diagnostics = {
            "best_return_mean": returns[ar, best_idx].mean().item(),
            "weights_mean": w.mean().item(),
            "q_term_mean": q_term.mean().item(),
        }
        return new_mean, new_std, diagnostics

    @timeit("planning_time", -1)
    @torch.no_grad()
    def plan(self, obs, world_model, latent=None, exec_sample=True, **kw):
        device = next(world_model.parameters()).device

        obs = obs.to(device)
        latent = world_model.encode(obs) if latent is None else latent.to(device)
        if latent.ndim == 3:
            latent = latent[:, -1]

        B = latent.shape[0]
        H = self.cfg.horizon
        A = self.action_dim
        P = self.cfg.popsize
        gamma = getattr(world_model, "gamma", 0.99)

        mean, std = self._warm_start(latent, obs)

        # flat latent for rollout
        flat_z0 = latent.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)

        # scratch
        key = (device, B, H, A, P)
        if key not in self._scratch:
            s = {}
            s['noise'] = torch.empty((B, P, H, A), device=device)
            s['seq'] = torch.empty((B, P, H, A), device=device)
            s['reward_acc'] = torch.empty((B*P,), device=device)
            s['alive'] = torch.empty((B*P,), device=device)
            s['prior_log'] = torch.zeros((B*P,), device=device)
            self._scratch[key] = s
        scratch = self._scratch[key]

        diag_list = []
        for _ in range(self.cfg.mppi_iters):
            mean, std, di = self._mppi_iteration(
                mean, std, flat_z0, world_model, obs, scratch, gamma
            )
            diag_list.append(di)

        self._mean, self._std  = mean, std

        if exec_sample:
            a0 = mean[:, 0] + std[:, 0] * torch.randn((B, A), device=device)
        else:
            a0 = mean[:, 0]

        a0 = self._clamp_actions(a0, self.cfg.action_min, self.cfg.action_max)
        info = {"mean": self._mean, "std": self._std, "diagnostics": diag_list}
        return a0, info

    @staticmethod
    def _predict_q(world_model: TDMPC2LatentDynamics, latent, action):
        return world_model.q_network(latent, action)

@configclass
class PlannerTDMPC2Cfg(PlannerBaseCfg):
    class_type              : type[PlannerBase] = PlannerTDMPC2
    horizon                 : int = 12
    popsize                 : int = 512
    mppi_iters              : int = 5
    mppi_beta               : float = 1.0
    prior_fraction          : float = 0.25
    prior_reg_coef          : float = 1.0
    warm_start              : bool = True
    init_std                : float = 1.0
    min_std                 : float = 1e-4
    max_std                 : float = 10.0
    eps                     : float = 1e-6
    action_min              : Optional[float] = -20.0
    action_max              : Optional[float] = 20.0
    term_threshold          : float = 0.5
    prior_mode              : Literal['none', 'policy', 'mean', 'random'] = 'policy'
