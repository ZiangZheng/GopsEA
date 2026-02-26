import torch
from GopsEA import configclass
from GopsEA.utils.template.module_base import ModuleBase, ModuleBaseCfg

class ActorCritic(ModuleBase):
    def __init__(self, cfg: "ActorCriticPackCfg", dim_params: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.actor = cfg.actor_cfg.construct_from_cfg(dim_params=dim_params)
        self.critic = cfg.critic_cfg.construct_from_cfg(dim_params=dim_params)
        
    @torch.no_grad()
    def act_inference(self, obs):
        return self.actor.act_inference(obs)

@configclass
class ActorCriticPackCfg(ModuleBaseCfg):
    class_type: type[ActorCritic] = ActorCritic
    
    actor_cfg: ModuleBaseCfg = None
    critic_cfg: ModuleBaseCfg = None
    