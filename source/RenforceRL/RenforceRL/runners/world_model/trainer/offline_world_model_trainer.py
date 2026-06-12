from __future__ import annotations

import os
import time
from typing import List
import torch
import tqdm
from collections import defaultdict
import torch.nn as nn

from RenforceRL import configclass
from dataclasses import MISSING


from RenforceRL.runners.world_model.trainer.world_model_trainer_replay import \
    WorldModelTrainerReplay, WorldModelTrainerReplayCfg
from RenforceRL.components.world_models.world_model_base import WorldModelBase, WorldModelBaseCfg
from RenforceRL.buffer.replay_buffer_base import ReplayBufferBase, ReplayBufferBaseCfg

from RenforceRL.utils.mapping import tranverse_dict_value, transform_dict_mapping
from RenforceRL.utils.isaaclab.trajectory import load_hdf5_trajectories, make_next_obs, DATA_TRAJ_MAPPING
from RenforceRL.runners.world_model.trainer.evaluators import BaseEvaluator

class OfflineWorldModelTrainer:
    """
    Offline world model trainer.
    Trains a world model purely from offline replay buffers:
        - train_replay_buffer
        - val_replay_buffer
    """

    def __init__(
        self,
        cfg: OfflineWorldModelTrainerCfg,
        device,
        log_dir=None,
        *args,
        trainer: WorldModelTrainerReplay = None,
        **kwargs
    ):
        self.cfg:OfflineWorldModelTrainerCfg = cfg
        num_envs = kwargs.get("num_envs", 32)
        
        self.train_data_loader = load_hdf5_trajectories(self.cfg.train_data_path)
        first_sample = next(self.train_data_loader)

        dim_params = {
            "dynamic_obs_dim"   : first_sample["dynamic"].shape[-1],
            "policy_obs_dim"    : first_sample["policy"].shape[-1],
            "action_dim"        : first_sample["action"].shape[-1],
            "rewards_dim"       : first_sample["rewards"].shape[-1],
        }
        
        dim_params_wm = transform_dict_mapping(dim_params, self.cfg.mapping_dataset_to_worldmodel_construct)
        dim_params_rp = transform_dict_mapping(dim_params, self.cfg.mapping_dataset_to_replaybuffer_construct)
        
        replay_buffer = ReplayBufferBase.construct_from_cfg(
            self.cfg.replay_buffer_cfg, dim_params=dim_params_rp, device=device
        )
        
        if trainer is None:
            world_model = self.cfg.world_model_cfg.class_type(
                self.cfg.world_model_cfg, **dim_params_wm
            )
            trainer = WorldModelTrainerReplay(
                self.cfg.world_model_trainer_cfg, world_model, 
                replay_buffer=replay_buffer,
                num_envs=num_envs, device=device
            )
        self.trainer:WorldModelTrainerReplay = trainer

        self.device = trainer.device
        self.world_model:WorldModelBase = trainer.world_model
        self.train_replay_buffer:ReplayBufferBase = trainer.replay_buffer

        self.load_data()

        self.log_dir = log_dir
        self.writer = None
        if self.log_dir is not None:
            self.init_logger()

        os.makedirs(self.log_dir, exist_ok=True)

        print("==== Offline World Model Trainer Initialized ====")
        print(self.world_model)
        print("=================================================")

    def load_data(self):
        self.train_data_loader = load_hdf5_trajectories(self.cfg.train_data_path)
        self.val_data_loader = load_hdf5_trajectories(self.cfg.val_data_path)
        type_func = lambda obj: tranverse_dict_value(
            obj, lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0))
        
        train_buffer_pbar = tqdm.tqdm(total=self.train_replay_buffer.max_steps)
        for sample in self.train_data_loader:
            sample = type_func(sample)
            self.trainer.replay_buffer.add_traj(sample)
            train_buffer_pbar.update(sample["policy"].shape[1])
            if self.trainer.replay_buffer.is_full():
                train_buffer_pbar.close()
                print(self.trainer.replay_buffer.pretty_report())
                break
        
    # ----------------------------------------------------------------------
    # Logging initialization
    # ----------------------------------------------------------------------
    def init_logger(self):
        from RenforceRL.runners.on_policy.on_policy_runner import OnPolicyRunner
        OnPolicyRunner.init_logger(self)

    # ----------------------------------------------------------------------
    # Core training loop
    # ----------------------------------------------------------------------
    def train(self):
        pbar = tqdm.tqdm(range(self.cfg.max_iters), desc="Offline World Model Training")
        for it in pbar:
            # --------------------------------------------------------------
            # 2. Train step
            # --------------------------------------------------------------
            tb_dict = self.trainer.update()

            # --------------------------------------------------------------
            # 3. Logging
            # --------------------------------------------------------------
            if self.writer is not None:
                for key, val in tb_dict.items():
                    self.writer.add_scalar("wm_train/" + key, val, it)

            # --------------------------------------------------------------
            # 4. Periodic Evaluation
            # --------------------------------------------------------------
            if it % self.cfg.eval_interval == 0:
                eval_tb = self.evaluate(it)
                if self.writer is not None:
                    for key, val in eval_tb.items():
                        self.writer.add_scalar("wm_eval/" + key, val, it)
                print("="*50)
                for key, val in eval_tb.items():
                    print(f"wm_eval/{key}:\t{val}")
                print("="*50)

            # --------------------------------------------------------------
            # 5. Save Checkpoint
            # --------------------------------------------------------------
            if it % self.cfg.save_interval == 0 and it > 0:
                self.world_model.save_world_model(os.path.join(self.log_dir, f"model_{it}.pt"))

    # ----------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------
    evaluators: List[BaseEvaluator]
    @torch.no_grad()
    def evaluate(self, it, verbose=True, writer_log=True, **kwargs):
        res_eval_tb = {}
        for evaluator in self.evaluators:
            eval_tbd = evaluator.evaluate(world_model=self.world_model, planner=self.planner, **kwargs)
            res_eval_tb.update(eval_tbd)
            
        if writer_log and self.writer is not None:
            for key, val in res_eval_tb.items():
                self.writer.add_scalar("wm_eval/" + key, val, it)
        if verbose:
            print("="*50)
            for key, val in res_eval_tb.items():
                print(f"wm_eval/{key}:\t{val}")
            print("="*50)
        return res_eval_tb

@configclass
class OfflineWorldModelTrainerCfg:
    class_type = OfflineWorldModelTrainer
    max_iters:                  int = MISSING
    eval_interval:              int = MISSING
    save_interval:              int = MISSING
    logger:                     str = "tensorboard"

    world_model_cfg:            WorldModelBaseCfg = MISSING
    world_model_trainer_cfg:    WorldModelTrainerReplayCfg = MISSING
    replay_buffer_cfg:      ReplayBufferBaseCfg = MISSING
    
    train_data_path:            str = MISSING
    val_data_path:              str = MISSING
    
    # None for not use
    mapping_dataset_to_worldmodel_construct:  dict = None
    mapping_dataset_to_replaybuffer_construct:  dict = None