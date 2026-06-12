import pickle
from typing import Dict, Tuple
from RenforceRL import configclass

from dataclasses import MISSING

from RenforceRL.utils.env_wrapper.lab_wrapper import RenforceRLEnvWrapper

from RenforceRL.components.world_models import  WorldModelBaseCfg, WorldModelBase, PlannerBaseCfg, PlannerBase
from RenforceRL.runners.world_model.trainer import WorldModelTrainerBaseCfg, WorldModelTrainerBase
from RenforceRL.buffer import ReplayBufferBaseCfg, ReplayBufferBase
from RenforceRL.utils.mapping import transform_dict_mapping

@configclass
class WorldModelBundleCfg:
    world_pretrain_times:       int = int(1e3)
    world_model_cfg:            WorldModelBaseCfg = MISSING
    world_model_trainer_cfg:    WorldModelTrainerBaseCfg = MISSING
    replay_buffer_cfg:          ReplayBufferBaseCfg = MISSING
    planner_cfg:                PlannerBaseCfg = None
    
    def save_as_pkl(self, path, dim_params=None):
        saved_dict = {
            "world_model_bundle_cfg": self,
            "dim_params": dim_params
        }
        with open(path, "wb") as f:
            pickle.dump(saved_dict, f)

def construct_world_model_components(
        world_cfg: WorldModelBundleCfg, num_envs, dim_params: Dict[str, int], device: str
    ) -> Tuple[WorldModelTrainerBase, WorldModelBase, ReplayBufferBase, PlannerBase]:
    
    data_pipeline = world_cfg.world_model_trainer_cfg.data_pipeline
    
    replay_buffer = ReplayBufferBase.construct_from_cfg(
        cfg = world_cfg.replay_buffer_cfg,
        dim_params = transform_dict_mapping(dim_params, data_pipeline.MAPPING_CONSTRUCTION_ENV_DIM_2_REPLAYBUFFER_DIM),
        num_envs = num_envs, device = device
    )
    print(replay_buffer.pretty_report())
    
    world_model = WorldModelBase.construct_from_cfg(
        cfg=world_cfg.world_model_cfg,
        **replay_buffer.dim_params
    )
    print(f"World Model: \n{world_model}")
    
    world_trainer = WorldModelTrainerBase.construct_from_cfg(
        world_cfg.world_model_trainer_cfg, world_model = world_model, 
        num_envs = num_envs,
        replay_buffer = replay_buffer, device = device
    )
    
    planner = PlannerBase.construct_from_cfg(
        cfg = world_cfg.planner_cfg,
        action_dim = dim_params["action_dim"]
    ) if world_cfg.planner_cfg is not None else None
    
    return world_trainer, world_model, replay_buffer, planner

def construct_world_model_components_with_env(
        world_cfg: WorldModelBundleCfg, env: RenforceRLEnvWrapper, device: str
    ) -> Tuple[WorldModelTrainerBase, WorldModelBase, ReplayBufferBase, PlannerBase]:
    return construct_world_model_components(world_cfg, env.num_envs, env.dim_params, device)


def construct_world_model_components_from_pkl(path, device, num_envs=16):
    with open(path, "rb") as f:
        saved_dict = pickle.load(f)
    world_cfg: WorldModelBundleCfg = saved_dict["world_model_bundle_cfg"]
    dim_params: Dict[str, int] = saved_dict["dim_params"]
    return construct_world_model_components(world_cfg, num_envs, dim_params, device=device)