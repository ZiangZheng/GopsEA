from __future__ import annotations
from .base_evaluator import BaseEvaluator, BaseEvaluatorCfg
from typing import Dict, List
from collections import defaultdict
import torch
from RenforceRL.utils.logging import timeit
from RenforceRL import configclass
from dataclasses import MISSING

class EnvInterEvaluator(BaseEvaluator):
    cfg: EnvInterEvaluatorCfg
    def __init__(self, cfg: EnvInterEvaluatorCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
    def evaluate(self, env, **kwargs):
        return super().evaluate(**kwargs)


@configclass
class EnvInterEvaluatorCfg(BaseEvaluatorCfg):
    class_type: type[EnvInterEvaluator] = EnvInterEvaluator
