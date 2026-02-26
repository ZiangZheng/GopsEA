from dataclasses import dataclass, MISSING
from typing import Tuple, TYPE_CHECKING
from GopsEA import configclass
from GopsEA.utils.template import ClassTemplateBase, ClassTemplateBaseCfg
from .pipeline_based.data_pipeline import DataPipeline, DataPipelineCfg, ComponentCfg
from .replay_buffer_base import ReplayBufferBase, ReplayBufferBaseCfg

if TYPE_CHECKING:
    from GopsEA.buffer import ReplayBufferBaseCfg, ReplayBufferBase, PipeBufferBase

@dataclass
class ShadowEnv:
    dim_params: dict
    num_env: int

@configclass
class ReplayBundle:
    replay_buffer_cfg: "ReplayBufferBaseCfg" = MISSING
    replay_num_epoches: int = None
    replay_mini_batch_size: int = None
    replay_num_batch_per_epoch: int = None
    
    def construct_replay_buffer(self, env, device) -> ReplayBufferBase:
        ...

@configclass
class ReplayPipelineBundle(ReplayBundle):
    """
    pipeline construct replay buffer. Default for the offline RL transition based replay.
    """
    data_pipeline_cfg: DataPipelineCfg = DataPipelineCfg(
        component_cfg=ComponentCfg(
            comp_names=["obs", "critic_obs", "action", "reward", "termination", "timeout"],
            comp_shape=[-1, -1, -1, 1, 1, 1],
            comp_dtype=["float32", "float32", "float32", "float32", "float32", "float32"]
        ),
        replay_buffer_dim_names = ["obs", "critic_obs", None, "action", None],
        replay_buffer_data_names= ["timeout", "termination", "obs", "critic_obs", None, "action", "reward", None]
    )
    
    def construct_replay_buffer(self, env, device) -> Tuple["ReplayBufferBase", "DataPipeline"]:
        data_pipeline: "DataPipeline" = self.data_pipeline_cfg.construct_from_cfg()
        replay_buffer: "PipeBufferBase" = self.replay_buffer_cfg.construct_from_cfg(
            data_pipeline.REPLAY_BUFFER_COMP,
            dim_params = data_pipeline.buffer_dim_params(env.dim_params),
            env_dim_params = env.dim_params,
            device = device,
            num_envs = env.num_envs
        )
        return replay_buffer # , data_pipeline
