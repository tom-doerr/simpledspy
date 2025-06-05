"""Module Caller for DSPy modules

Provides base classes for Predict and ChainOfThought function calls
"""

import dspy
from .pipeline_manager import PipelineManager
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
from .evaluator import Evaluator

class BaseCaller:
    """Base class for DSPy module callers"""
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
            instance.pipeline_manager = PipelineManager()
            instance.module_factory = ModuleFactory()
            instance.optimization_manager = OptimizationManager()
            instance.lm = dspy.LM(model="deepseek/deepseek-chat")
            instance.evaluator = Evaluator()
            dspy.configure(lm=instance.lm, cache=False)
        return cls._instances[cls]
    
    def _create_module(self, inputs: List[str], outputs: List[str], 
                     input_types: Dict[str, type] = None,
                     output_types: Dict[str, type] = None,
                     description: str = "") -> dspy.Module:
        return self.module_factory.create_module(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )

    def __call__(self, *args, inputs: List[str] = None, outputs: List[str] = None, description: str = None, lm_params: dict = None) -> Any:
        # Use custom input names if provided, otherwise generate meaningful defaults
        if inputs is None:
            input_names = [f"input_{i+1}" for i in range(len(args))]
        else:
            if len(inputs) != len(args):
                raise ValueError(f"Expected {len(args)} input names, got {len(inputs)}")
            input_names = inputs
        
        # Use custom output names if provided, otherwise use a single default
        output_names = outputs or ["output"]
        
        module = self._create_module(
            inputs=input_names, 
            outputs=output_names,
            description=description
        )
        
        input_dict = dict(zip(input_names, args))
        
        # Save original LM parameters
        original_params = {}
        if lm_params:
            for key, value in lm_params.items():
                if hasattr(self.lm, key):
                    original_params[key] = getattr(self.lm, key)
                    setattr(self.lm, key, value)
        
        try:
            prediction_result = module(**input_dict)
        finally:
            # Restore original LM parameters
            for key, value in original_params.items():
                setattr(self.lm, key, value)
        
        self.pipeline_manager.register_step(inputs=input_names, outputs=output_names, module=module)
        
        output_values = [getattr(prediction_result, name) for name in output_names]
        
        # Log with evaluation
        output_dict = dict(zip(output_names, output_values))
        self.evaluator.log_with_evaluation(
            module=self.FUNCTION_NAME,
            inputs=input_dict,
            outputs=output_dict,
            description=description
        )
        
        if len(output_values) == 1:
            return output_values[0]
        return tuple(output_values)

class Predict(BaseCaller):
    """Predict module caller - replaces pipe() function"""
    FUNCTION_NAME = 'predict'

class ChainOfThought(BaseCaller):
    """ChainOfThought module caller"""
    FUNCTION_NAME = 'chain_of_thought'
    
    def _create_module(self, inputs: List[str], outputs: List[str], 
                     input_types: Dict[str, type] = None,
                     output_types: Dict[str, type] = None,
                     description: str = "") -> dspy.Module:
        signature_class = self.module_factory.create_signature(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        return dspy.ChainOfThought(signature_class)
