from __future__ import annotations

import os
import time
from collections import deque


import torch
from dataclasses import MISSING

from RenforceRL.utils.logging import timeit
from RenforceRL import configclass
from RenforceRL.components.actor_critic_pack import ActorCriticPackCfg
from RenforceRL.components.normalizer import NormalizerBaseCfg
from RenforceRL.runners.logger import LoggerBaseCfg
from RenforceRL.runners.on_policy.on_policy_runner import (
    OnPolicyRunner,
    OnPolicyRunnerCfg,
)
from RenforceRL.algorithms.on_policy.mbpo.mbpo import MBPO, MBPOCfg
from RenforceRL.utils.template.module_base import ModuleBaseCfg

from RenforceRL.utils.env_wrapper import lab_wrapper


class MBPOOnPolicyRunner(OnPolicyRunner):
    """On-policy runner with model-based rollouts (MBPO).

    This runner mirrors `OnPolicyRunner` but:
    - maintains a learned system dynamics model via `MBPO.update_system_dynamics`;
    - optionally performs imagination rollouts and mixes them into PPO updates.
    """

    alg: MBPO
    cfg: "MBPOOnPolicyRunnerCfg"
    env: "lab_wrapper.RFImagineEnvWrapper"

    def __init__(
        self,
        train_cfg: "MBPOOnPolicyRunnerCfg",
        env: "lab_wrapper.RFImagineEnvWrapper",
        log_dir=None,
        device: str = "cpu",
    ):
        # Override algorithm cfg type so OnPolicyRunner sees correct subtype.
        self.alg_cfg: MBPOCfg = train_cfg.algorithm
        self.policy_cfg = train_cfg.policy
        super().__init__(train_cfg=train_cfg, env=env, log_dir=log_dir, device=device)
        
    def init_components(self):
        # Actor-critic and PPO-style storage as in OnPolicyRunner.
        num_obs, num_critic_obs = (
            self.env.dim_params["policy_dim"],
            self.env.dim_params["critic_dim"],
        )
        self.actor_critic = self.policy_cfg.construct_from_cfg(
            dim_params=self.env.dim_params
        )

        # System dynamics and state/action normalizers are constructed from cfg.
        system_dynamics = self.cfg.system_dynamics_cfg.construct_from_cfg(
            dim_params=self.env.dim_params, device=self.device
        )
        state_normalizer = self.cfg.system_state_normalize_cfg.construct_from_cfg()
        action_normalizer = self.cfg.system_action_normalize_cfg.construct_from_cfg()

        self.env.set_system_dynamics(system_dynamics)

        self.alg = self.alg_cfg.construct_from_cfg(
            actor_critic=self.actor_critic,
            device=self.device,
            system_dynamics=system_dynamics,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
        )

        # Initialise PPO (real) storage.
        self.alg.init_storage(
            self.env.num_envs,
            self.cfg.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Optionally initialise imagination storage.
        if (
            self.cfg.imagination_num_envs > 0
            and self.cfg.imagination_num_steps_per_env > 0
        ):
            self.alg.init_imagination_storage(
                num_envs=self.cfg.imagination_num_envs,
                num_transitions_per_env=self.cfg.imagination_num_steps_per_env,
                actor_obs_shape=[num_obs],
                critic_obs_shape=[num_critic_obs],
                action_shape=[self.env.num_actions],
            )

        # Inform the algorithm about system-dynamics input dimensionality if
        # the env exposes it (otherwise user must call manually).
        self.alg.init_system_replay_buffer(
            state_dim           = self.env.dim_params["dynamic_dim"],
            action_dim          = self.env.dim_params["action_dim"],
            extension_dim       = self.env.dim_params.get("extension_dim", 0),
            contact_dim         = self.env.dim_params.get("contact_dim", 0),
            termination_dim     = self.env.dim_params.get("termination_dim", 0),
        )

        # Normalisers for policy/critic inputs.
        self.obs_normalizer = self.cfg.obs_normalize_cfg.construct_from_cfg(
            shape=self.env.dim_params["policy_dim"]
        )
        self.critic_normalizer = self.cfg.critic_normalize_cfg.construct_from_cfg(
            shape=self.env.dim_params["critic_dim"]
        )
        self.obs_normalizer.to(self.device)
        self.critic_normalizer.to(self.device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self.logger.init_logger()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        self._learn(num_learning_iterations=num_learning_iterations)

    def _learn(self, num_learning_iterations: int):
        ep_infos = []
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()

        for it in range(start_iter, tot_iter):
            # 1) Collect real rollouts
            sample_infos, obs, critic_obs = self._sample_rollout_mbpo(
                ep_infos, obs, critic_obs, it, start_iter
            )
            collection_time = sample_infos["collection_time"]

            # 2) Update system dynamics
            sys_losses = self.alg.update_system_dynamics()

            # 3) Policy update (with or without imagination)
            start = time.time()
            if it >= start_iter + self.cfg.system_dynamics_warmup_iterations:
                if (
                    self.cfg.imagination_num_envs > 0
                    and self.cfg.imagination_num_steps_per_env > 0
                ):
                    # Generate imagination rollouts and fill imagination storage
                    if it == start_iter + self.cfg.system_dynamics_warmup_iterations:
                        # Initialize state/action history for imagination
                        self.state_history, self.action_history = self.alg.prepare_imagination()
                    # Generate virtual trajectories using system dynamics
                    imagine_infos = self._imagine()
                    sample_infos.update(imagine_infos)
                    alg_update_infos = self.alg.update(use_imagination=True)
                else:
                    alg_update_infos = self.alg.update(use_imagination=False)
            else:
                alg_update_infos = self.alg.update(use_imagination=False)
            alg_update_infos.update(sys_losses)
            stop = time.time()
            learn_time = stop - start

            self.current_learning_iteration = it
            if self.logger.log_dir is not None:
                # Merge infos for logging
                log_locals = locals()
                self.logger.log(self, log_locals)
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    @timeit("collection_time")
    def _sample_rollout_mbpo(self, ep_infos, obs, critic_obs, it, start_iter):
        """Extended rollout sampling that also fills system-dynamics replay."""
        rollout_datas = []
        with torch.inference_mode():
            for i in range(self.cfg.num_steps_per_env):
                actions = self.alg.act(obs, critic_obs)
                obs, reward, done, infos = self.env.step(actions.to(self.env.device))
                critic_obs = infos["observations"].get("critic", obs)
                obs, critic_obs, reward, done = (
                    obs.to(self.device),
                    critic_obs.to(self.device),
                    reward.to(self.device),
                    done.to(self.device),
                )
                obs = self.obs_normalizer(obs)
                critic_obs = self.critic_normalizer(critic_obs)

                # Fill system-dynamics replay buffer if env provides system obs.
                (
                    system_state,
                    system_action,
                    system_extension,
                    system_contact,
                    system_termination,
                ) = self.env.get_system_observation()
                
                self.alg.fill_history_buffer(
                    system_state,
                    system_action,
                    system_extension,
                    system_contact,
                    system_termination,
                )

                # Only after warmup do we feed data into PPO storage.
                if it >= start_iter + self.cfg.system_dynamics_warmup_iterations:
                    self.process_env_step(reward, done, infos)

                rollout_datas.append((obs, critic_obs, actions, reward, done, infos))

                if self.logger.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    self.cur_reward_sum += reward
                    self.cur_episode_length += 1
                    new_ids = (done > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(
                        self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.lenbuffer.extend(
                        self.cur_episode_length[new_ids][:, 0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            # compute returns for real trajectories only after warmup
            if it >= start_iter + self.cfg.system_dynamics_warmup_iterations:
                self.alg.compute_returns(critic_obs)

            process_infos = self.process_rollout(rollout_datas)
            sample_infos = {
                "cur_reward_sum": self.cur_reward_sum,
                "cur_episode_length": self.cur_episode_length,
            }
            sample_infos.update(process_infos)
            return sample_infos, obs, critic_obs

    @timeit("imagination_collection_time")
    def _imagine(self):
        """Generate imagination rollouts using system dynamics model.
        
        This method generates virtual trajectories by:
        1. Getting observations from state/action history
        2. Using current policy to select actions
        3. Using system dynamics to predict next states and rewards
        4. Storing results in imagination_storage
        """

        # Clear imagination storage for new rollouts
        if self.alg.imagination_storage is not None:
            self.alg.imagination_storage.clear()

        with torch.inference_mode():
            for i in range(self.cfg.imagination_num_steps_per_env):
                # Get observation from state/action history
                imagination_obs = self.env.get_imagination_observation(
                    self.state_history, self.action_history
                )
                critic_obs = imagination_obs  # Assume same for now, can be extended
                
                # Normalize observations
                imagination_obs = self.obs_normalizer(imagination_obs)
                critic_obs = self.critic_normalizer(critic_obs)
                
                # Use current policy to select actions
                imagination_actions = self.alg.act_imagination(imagination_obs, critic_obs)
                
                (
                    imagination_obs_next,
                    imagination_rewards,
                    imagination_dones,
                    imagination_extras,
                    self.state_history,
                    self.action_history,
                ) = self.env.imagination_step(
                    imagination_actions, self.state_history, self.action_history
                )
                
                # Handle resets: sample new initial states from replay buffer
                reset_env_ids = (imagination_dones > 0).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_env_ids) > 0:
                    # Sample new initial histories for reset environments
                    history_horizon = (
                        self.alg.mbpo_cfg.system_dynamics_history_horizon
                        + self.alg.mbpo_cfg.system_dynamics_forecast_horizon
                    )
                    imagination_generator = self.alg.system_replay_buffer.mini_batch_generator(
                        sequence_length=history_horizon,
                        num_mini_batches=1,
                        mini_batch_size=len(reset_env_ids),
                    )
                    imagination_state_history, imagination_action_history = next(imagination_generator)[:2]
                    self.state_history[reset_env_ids] = imagination_state_history[
                        :, -self.state_history.shape[1] :
                    ]
                    self.action_history[reset_env_ids] = imagination_action_history[
                        :, -self.action_history.shape[1] :
                    ]
                
                # Normalize next observation
                imagination_obs_next_norm = self.obs_normalizer(imagination_obs_next)
                critic_obs_next = imagination_extras.get("observations", {}).get("critic", imagination_obs_next)
                critic_obs_next_norm = self.critic_normalizer(critic_obs_next)
                
                # Store transition in imagination storage
                # Note: The current step's obs/action are already in imagination_transition from act_imagination
                # We need to update the observations to next step's obs for the transition (obs_t, a_t, r_t, obs_{t+1})
                # But actually, in RolloutStorage, we store obs_t at step t, so we should keep current obs
                # The next obs will be stored in the next iteration
                self.alg.process_env_step(
                    imagination_rewards,
                    imagination_dones,
                    imagination_extras,
                    imagination=True,
                )
                
                # Update observation for next iteration (already normalized)
                # Note: These will be used as the "current" obs in the next iteration
                imagination_obs = imagination_obs_next_norm
                critic_obs = critic_obs_next_norm

            # Compute returns for imagination trajectories
            if self.alg.imagination_storage is not None:
                self.alg.compute_imagination_returns(critic_obs)


@configclass
class MBPOOnPolicyRunnerCfg(OnPolicyRunnerCfg):
    class_type: type[MBPOOnPolicyRunner] = MBPOOnPolicyRunner

    # Algorithm and policy
    policy: ActorCriticPackCfg = MISSING
    algorithm: MBPOCfg = MISSING

    # Normalisation
    obs_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    critic_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()

    # System dynamics & its input normalizers
    system_dynamics_cfg: ModuleBaseCfg = None
    system_state_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    system_action_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()

    # System dynamics warmup: number of PPO iterations before using rollouts
    system_dynamics_warmup_iterations: int = 0

    # Imagination configuration (algorithm and env must cooperate to use it)
    imagination_num_envs: int = 0
    imagination_num_steps_per_env: int = 0

    logger_cfg: LoggerBaseCfg = LoggerBaseCfg()

