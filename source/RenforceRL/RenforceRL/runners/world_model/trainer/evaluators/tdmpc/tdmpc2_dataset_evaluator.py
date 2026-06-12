from __future__ import annotations
from typing import Dict, Any, List
import torch

from RenforceRL.utils.logging import timeit
from RenforceRL import configclass
from RenforceRL.utils.isaaclab.trajectory import load_hdf5_trajectories
import random

from RenforceRL.runners.world_model.trainer.evaluators.dataset_evaluator import (
    DatasetEvaluator,
    DatasetEvaluatorCfg,
)
from RenforceRL.components.world_models.tdmpcs.tdmpc2 import (
    TDMPC2LatentDynamics, PlannerTDMPC2
)

class TDMPC2DatasetEvaluator(DatasetEvaluator):
    """
    Offline dataset evaluator for TDMPC2.
    It rolls latent dynamics forward using ground-truth actions
    and periodically invokes MPC planner for evaluation.
    """

    cfg: TDMPC2DatasetEvaluatorCfg

    def __init__(self, cfg: "TDMPC2DatasetEvaluatorCfg", **kwargs):
        super().__init__(cfg, **kwargs)

    # ----------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------
    @timeit("TDMPC2DatasetEvaluator/evaluate")
    @torch.no_grad()
    def evaluate(
        self,
        world_model: TDMPC2LatentDynamics,
        planner: PlannerTDMPC2,
        **kwargs,
    ) -> Dict[str, Any]:

        device = planner.device
        cfg = self.cfg

        # Randomly select one trajectory from the dataset
        dataset = random.choice(self.dataset)

        # Observations and actions (GT)
        obs = torch.as_tensor(dataset["policy"], device=device)
        act = torch.as_tensor(dataset["action"], device=device)
        reward = torch.as_tensor(dataset["reward"], device=device)
        termination = torch.as_tensor(dataset["termination"], device=device).to(torch.float)

        T = obs.shape[0]

        # ---------- Metrics ----------
        action_errors = []
        reward_errors = []
        termination_errors = []

        # ------------------------------------------------------------
        #  Warm start latent with encoder
        # ------------------------------------------------------------
        z = world_model.encode(obs[0].unsqueeze(0))  # [1, Z]

        for t in range(1, T, cfg.eval_stride):

            # --------------------------------------------------------
            # 1. Ground-truth action → rollout world model
            # --------------------------------------------------------
            act_t = act[t - 1].unsqueeze(0)  # [1, A]

            z_pred, r_pred, t_pred, _ = world_model.predict_next(z, act_t)  # 1-step latent
            z = z_pred
            
            reward_errors.append(torch.norm(r_pred - reward[t].unsqueeze(0), p=2).item())
            termination_errors.append(torch.norm(t_pred.to(torch.float) - termination[t].unsqueeze(0), p=2).item())

            # --------------------------------------------------------
            # 3. Invoke MPC planner every eval_stride steps
            # --------------------------------------------------------
            if t % cfg.mpc_interval == 0:
                mp_action, _ = planner.plan(z, world_model=world_model)

                # Evaluate action reconstruction (simple baseline)
                action_err = torch.norm(mp_action - act[t].unsqueeze(0), p=2).item()
                action_errors.append(action_err)

        # ------------------------------------------------------------
        #  Summary
        # ------------------------------------------------------------
        return {
            "T": T,
            "action_error_mean": float(torch.tensor(action_errors).mean()) if action_errors else 0.0,
            "predicted_reward_mean": float(torch.tensor(reward_errors).mean()) if reward_errors else 0.0,
            "num_mpc_calls": len(action_errors),
        }

# ============================================================================
# Config
# ============================================================================

@configclass
class TDMPC2DatasetEvaluatorCfg(DatasetEvaluatorCfg):
    """
    Additional config for TDMPC2 offline evaluation.
    """
    class_type: type[TDMPC2DatasetEvaluator] = TDMPC2DatasetEvaluator
    # how often to call MPC
    mpc_interval: int = 5

    # step interval for evaluation loop
    eval_stride: int = 1

    # more parameters can be added later:
    # warmup_steps: int = 5
    # use_groundtruth_latent: bool = False
    # compute_worldmodel_nll: bool = False
