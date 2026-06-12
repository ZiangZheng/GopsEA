from __future__ import annotations
from typing import Dict
from RenforceRL import configclass
from RenforceRL.utils.logging import timeit
from RenforceRL.utils.template import ClassTemplateBase

class BaseEvaluator(ClassTemplateBase):
    """
    Base class for evaluators in reinforcement learning.
    Evaluators are responsible for assessing the performance of agents or models.
    """

    def __init__(self, cfg: BaseEvaluatorCfg, **kwargs):
        """
        Initialize the evaluator with the given configuration.

        Arguments:
            cfg (BaseEvaluatorCfg): Configuration for the evaluator.
        """
        super().__init__()
        self.cfg = cfg

        if kwargs:
            print(f"Warning: Unused kwargs: {list(kwargs.keys())}")  # Warning for any unused arguments

    def evaluate(self, **kwargs) -> Dict[str, float]:
        """
        Evaluate the performance based on provided data.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        raise NotImplementedError("The evaluate method must be implemented by subclasses.")
    
@configclass
class BaseEvaluatorCfg:
    class_type: type[BaseEvaluator] = BaseEvaluator
    # Add common configuration parameters for evaluators here if needed