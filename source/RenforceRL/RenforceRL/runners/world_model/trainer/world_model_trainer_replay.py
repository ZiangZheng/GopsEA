from __future__ import annotations

import torch
import tqdm
from typing import List, Union, Dict, Literal, Tuple
from dataclasses import MISSING
from RenforceRL import configclass

from RenforceRL.components.world_models.world_model_base import WorldModelBase, WorldModelBaseCfg
from RenforceRL.buffer.replay_buffer_base import ReplayBufferBase, ReplayBufferBaseCfg
from RenforceRL.utils.mapping import tranverse_dict_value, transform_dict_mapping, check_mapping_valid
from RenforceRL.utils.logging import timeit

from .world_model_trainer_base import WorldModelTrainerBase, WorldModelTrainerBaseCfg

class WorldModelTrainerReplay(WorldModelTrainerBase):
    """
    Manage replay buffer and 
    """
    cfg: WorldModelTrainerReplayCfg
    def __init__(
        self, 
        cfg:WorldModelTrainerReplayCfg, 
        world_model:WorldModelBase, 
        num_envs:int,
        replay_buffer,
        device,
    ):
        super().__init__(cfg, world_model=world_model, num_envs=num_envs, device=device)
        self.mini_batch_size = self.cfg.mini_batch_size
        self.num_epoches = self.cfg.num_epoches
        self.replay_buffer = replay_buffer
        self.comp_names = self.replay_buffer.cfg.component_cfg.comp_names
        self.traj_buffer = {c: [[] for _ in range(num_envs)] for c in self.comp_names}
    
    def set_replay_buffer(self, buffer: ReplayBufferBase):
        self.replay_buffer = buffer
    
    @timeit("trainer_update_time_total")
    def update(self, buffer:ReplayBufferBase=None, show_pbar=False, num_epoches=None, mini_batch_size=None):
        res = {}
        self.world_model.train()
        if buffer is None: buffer = self.replay_buffer
        if num_epoches is None: num_epoches = self.num_epoches
        if mini_batch_size is None: mini_batch_size = self.mini_batch_size

        if buffer.is_warmingup():
            # print(buffer.pretty_report())
            return res
            
        generator = buffer.mini_batch_generator(num_epoches, mini_batch_size, max_batches_per_epoch=self.cfg.max_batches_per_epoch)
        if show_pbar:
            generator = tqdm.tqdm(generator, desc="update epoch")

        for minib in generator:
            # Detach and clone tensors for gradient tracking
            params_dict = minib
            params_dict = tranverse_dict_value(params_dict, lambda obj: obj.to(torch.float32).to(self.device))

            # Ensure the correct tensors are used for gradient tracking
            world_model_tb_dict = self.world_model.update(
                **params_dict,
            )
        
        # wm_tb_dict = self.world_model.evaluate(**params_dict)
        return world_model_tb_dict

    def _process_traj(self, obs, action, reward, done, infos):
        """
        Flat process, where the trajs will be flaten and add.
        """
        samples = {
            "timeout":      infos["timeout"],
            "termination":  infos["termination"],
            "policy":       infos["observations"]["policy"],
            "critic":       infos["observations"].get("critic", None),
            "dynamic":      infos["observations"].get("dynamic", None),
            "action":       action,
            "reward":       reward,
            "rewards":      infos["rewards"],
        }
        # move to device
        samples = tranverse_dict_value(samples, lambda v: v.to(self.device) if v is not None else None)
        done = done.to(self.device)
        for cname, v in samples.items():
            if v is None: 
                continue
            li = self.traj_buffer[cname]
            for env_id, item in enumerate(v): li[env_id].append(item)
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        
        sample_to_add = {}
        for cname in self.comp_names:
            trajs = []
            for env_id in done_ids.tolist():
                buf = self.traj_buffer[cname][env_id]
                if not len(buf) == 0:
                    trajs.append(torch.stack(buf, dim=0))  # [T, ...]
            if len(trajs) > 0:
                sample_to_add[cname] = torch.concat(trajs, dim=0).unsqueeze(0)
            else:
                break

        for env_id in done_ids.tolist():
            for cname in self.comp_names:
                self.traj_buffer[cname][env_id].clear()
        if sample_to_add:
            self.replay_buffer.add_traj(sample_to_add)

    def _process_step(self, obs, action, reward, done, infos):
        sample = super()._process_step(obs, action, reward, done, infos)
        self.replay_buffer.add_steps(samples=sample)

    def _process_chunk(self, obs, action, reward, done, infos):
        sample = super()._process_step(obs, action, reward, done, infos)
        self.replay_buffer.add_chunk(samples=sample)

@configclass
class WorldModelTrainerReplayCfg(WorldModelTrainerBaseCfg):
    class_type              : type[WorldModelTrainerReplay] = WorldModelTrainerReplay
    mini_batch_size         : int = MISSING
    num_epoches             : int = MISSING
    max_batches_per_epoch   : int = None