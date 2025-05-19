from typing import Dict, Any, Callable
import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2
from dspy.evaluate import Evaluate

class OptimizationManager:
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
        
    def default_metric(self, example, prediction, trace=None):
        """Default metric function that checks exact match of predictions"""
        # Handle empty example case
        if not example or len(example) == 0:
            return 0.0 if prediction and len(prediction) > 0 else 1.0
            
        score = 0
        for key, value in example.items():
            if key in prediction and prediction[key] == value:
                score += 1
        return score / len(example)

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

    def optimize(self, module, trainset):
        """Optimize a module using the configured strategy"""
        teleprompter = self.get_teleprompter()
        return teleprompter.compile(module, trainset=trainset)
