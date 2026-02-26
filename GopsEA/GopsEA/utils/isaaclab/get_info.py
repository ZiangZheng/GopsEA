from __future__ import annotations

import torch
from isaaclab.assets import Articulation

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

def get_joint_pos_limits(env: ManagerBasedRLEnv, name="robot") -> torch.Tensor:
    asset:Articulation = env.scene["name"]
    return \
        asset.data.soft_joint_pos_limits[0, None, 0], \
        asset.data.soft_joint_pos_limits[0, None, 1]
        
def get_reward_weights(env: ManagerBasedRLEnv):
    weights = \
        [env.reward_manager.get_term_cfg(name).weight for name in env.reward_manager.active_terms]
    weights = torch.tensor(weights, dtype=torch.float32).reshape(1, -1)
    return weights