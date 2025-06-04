from typing import Dict, Any, Callable, Type, Union
import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2, BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
from .metrics import dict_exact_match_metric

"""Optimization Manager for DSPy modules and pipelines

This module provides the OptimizationManager class which:
1. Configures optimization strategies (BootstrapFewShot, MIPROv2, or BootstrapFewShotWithRandomSearch)
2. Provides a default exact-match metric function for evaluation
3. Creates teleprompter instances
4. Optimizes DSPy modules/pipelines
"""

class OptimizationManager:
    """Manages optimization of DSPy modules and pipelines
    
    Configures and executes optimization strategies using teleprompters.
    Provides a default exact-match metric function for evaluation.
    """
    
    def __init__(self) -> None:
        self._config = {
            'strategy': 'bootstrap_few_shot',
            'metric': dict_exact_match_metric,
            'max_bootstrapped_demos': 4,
            'max_labeled_demos': 4
        }
        self._teleprompters = {
            'bootstrap_few_shot': BootstrapFewShot,
            'mipro': MIPROv2,
            'bootstrap_random': BootstrapFewShotWithRandomSearch
        }

    def configure(self, **kwargs):
        """Update optimization configuration"""
        self._config.update(kwargs)

    def get_teleprompter(self):
        """Get configured teleprompter instance"""
        strategy = self._config['strategy']
        if strategy not in self._teleprompters:
            raise KeyError(f"Unknown optimization strategy: {strategy}")
            
        # Different teleprompters have different parameter requirements
        if strategy == 'bootstrap_few_shot':
            return self._teleprompters[strategy](
                metric=self._config['metric'],
                max_bootstrapped_demos=self._config['max_bootstrapped_demos'],
                max_labeled_demos=self._config['max_labeled_demos']
            )
        elif strategy == 'mipro':
            return self._teleprompters[strategy](
                metric=self._config['metric']
            )
        elif strategy == 'bootstrap_random':
            return self._teleprompters[strategy](
                metric=self._config['metric'],
                max_bootstrapped_demos=self._config['max_bootstrapped_demos'],
                max_labeled_demos=self._config['max_labeled_demos']
            )
        else:
            return self._teleprompters[strategy](
                metric=self._config['metric']
            )

    def optimize(self, module: dspy.Module, trainset: list) -> dspy.Module:
        """
        Optimize a module or pipeline using the configured strategy
        
        Args:
            module: DSPy module or pipeline to optimize
            trainset: Training dataset for optimization (list of examples)
            
        Returns:
            Optimized DSPy module or pipeline
            
        Note: For pipeline modules, resets state after optimization
        """
        teleprompter = self.get_teleprompter()
        compiled = teleprompter.compile(module, trainset=trainset)
        
        # For pipelines, reset after optimization to avoid state carryover
        if hasattr(module, 'steps'):
            module.reset()
            
        return compiled
