from __future__ import annotations

from RenforceRL import configclass

from RenforceRL.runners.world_model.trainer.evaluators.env_inter_evaluator import EnvInterEvaluator, EnvInterEvaluatorCfg
from RenforceRL.runners.world_model.trainer.world_model_trainer_replay import WorldModelTrainerReplay, WorldModelTrainerReplayCfg
from RenforceRL.components import world_model_bundle

class TDMPC2EnvInterEvaluator(EnvInterEvaluator):
    env: world_model_bundle.RenforceRLEnvWrapper
    trainer: world_model_bundle.WorldModelTrainerBase
    world_model: world_model_bundle.WorldModelBase
    planner: world_model_bundle.PlannerBase
    def __init__(self, cfg, env, **kwargs):
        super().__init__(cfg, **kwargs)
        
    def evaluate(self, env, **kwargs):
        return super().evaluate(env, **kwargs)
    
@configclass
class TDMPC2EnvInterEvaluator(EnvInterEvaluatorCfg):
    class_type: type[EnvInterEvaluatorCfg] = EnvInterEvaluatorCfg
    