import torch
from GopsEA import configclass
from typing import Generator, Dict, Tuple
from GopsEA.buffer import PipeBufferTransition
from .sac_base import SAC, SACCfg

class SACTrans(SAC):
    
    def update(self, generator: Generator[PipeBufferTransition.TransitionData, None, None]):
        self.ptr_update = 0
        critic_losses, q_means, target_q_means, actor_losses, alpha_losses, alphas = [], [], [], [], [], []
        for minib in generator:
            critic_loss, q_mean, target_q_mean, actor_loss, alpha_loss, alpha = self.update_trans(
                obs=minib.obs,
                critic_obs=minib.critic_obs,
                actions=minib.actions,
                rewards=minib.rewards,
                next_obs=minib.next_obs,
                next_critic_obs=minib.next_critic_obs,
                termination=minib.termination,
                timeout=minib.timeout
            )
            critic_losses.append(critic_loss)
            q_means.append(q_mean)
            target_q_means.append(target_q_mean)
            if actor_loss is not None: actor_losses.append(actor_loss)
            if alpha_loss is not None: alpha_losses.append(alpha_loss)
            alphas.append(alpha)
            self.ptr_update += 1
        return {
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "q_mean": sum(q_means) / len(q_means),
            "target_q_mean": sum(target_q_means) / len(target_q_means),
            "alpha_loss": sum(alpha_losses) / len(alpha_losses),
            "alpha": sum(alphas) / len(alphas),
            "mini_batch_num": self.ptr_update
        }
    
    
@configclass
class SACTransCfg(SACCfg):
    class_type: type[SAC] = SACTrans