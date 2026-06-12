from typing import Callable, Optional, Tuple
import torch
from RenforceRL import configclass
from RenforceRL.utils.template import ClassTemplateBase, ClassTemplateBaseCfg

class PlannerBase(ClassTemplateBase):
    @staticmethod
    def _clamp_actions(acts: torch.Tensor, a_min: Optional[float], a_max: Optional[float]) -> torch.Tensor:
        """Clamp action(s). Accepts tensor of shape (..., A)."""
        if a_min is None and a_max is None:
            return acts
        if a_min is None:
            return torch.minimum(acts, torch.tensor(a_max, device=acts.device, dtype=acts.dtype))
        if a_max is None:
            return torch.maximum(acts, torch.tensor(a_min, device=acts.device, dtype=acts.dtype))
        return torch.clamp(acts, a_min, a_max)
    
    def plan(self, obs, world_model, *arg, **kwargs) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError("Not implemented yet.")
    
    def act(self, *arg, **kwargs):
        raise NotImplementedError("Not implemented yet.")

@configclass
class PlannerBaseCfg(ClassTemplateBaseCfg):
    class_type: callable = None
