from typing import Dict, Any, Callable, Type, Union
import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2
from dspy.evaluate import Evaluate

"""Optimization Manager for DSPy modules and pipelines

This module provides the OptimizationManager class which:
1. Configures optimization strategies (BootstrapFewShot or MIPROv2)
2. Provides a default exact-match metric function
3. Creates teleprompter instances
4. Optimizes DSPy modules/pipelines
"""

class OptimizationManager:
    """Manages optimization of DSPy modules and pipelines
    
    Configures and executes optimization strategies using teleprompters.
    Provides a default exact-match metric function for evaluation.
    """
    
    def __init__(self):
        self._config = {
            'strategy': 'bootstrap_few_shot',
            'metric': self.default_metric,
            'max_bootstrapped_demos': 4,
            'max_labeled_demos': 4
        }
        self._teleprompters = {
            'bootstrap_few_shot': BootstrapFewShot,
            'mipro': MIPROv2
        }
        
    def default_metric(self, example: Dict[str, Any], 
                      prediction: Union[Dict[str, Any], 
                      trace: Any = None) -> float:
        """Calculates exact match score between example and prediction
        
        Args:
            example: Ground truth dictionary
            prediction: Model prediction (dict, tuple, or single value)
            trace: Optional trace information (unused)
            
        Returns:
            Float score between 0.0 and 1.0
            
        Behavior:
        - Empty example: returns 1.0 if prediction empty, else 0.0
        - Non-dict predictions are normalized to dict format
        - Compares keys present in example dictionary
        - Scores: 1.0 for exact match, 0.0 for no match
        """
        # Handle empty example case
        if not example:
            return 1.0 if not prediction else 0.0
            
        # Normalize prediction to dict
        if not isinstance(prediction, dict):
            if isinstance(prediction, tuple):
                prediction = {f'output_{i}': val for i, val in enumerate(prediction)}
            else:
                prediction = {'output': prediction}
                
        # Calculate exact match score
        score = 0
        total = 0
        for key, value in example.items():
            total += 1
            if key in prediction and prediction[key] == value:
                score += 1
                
        return score / total if total > 0 else 0.0

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
            # MIPRO doesn't accept max_labeled_demos
            return self._teleprompters[strategy](
                metric=self._config['metric']
            )
        else:
            # Generic fallback
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
