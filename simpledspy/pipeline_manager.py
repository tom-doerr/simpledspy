"""Pipeline Manager for DSPy modules

Provides:
- Singleton access to pipeline manager
- Step registration with input/output specifications
- Pipeline assembly from registered steps
- State reset functionality
"""

from typing import Any, List, Tuple, Dict
import dspy
import threading

class PipelineManager:
    """Manages DSPy pipeline construction and execution
    
    Provides:
    - Singleton access to pipeline manager
    - Step registration with input/output specifications
    - Pipeline assembly from registered steps
    - State reset functionality
    """
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "PipelineManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._steps = []
        return cls._instance

    def register_step(self, inputs: List[str], outputs: List[str], module: Any) -> None:
        """Register a pipeline step with input/output specifications
        
        Args:
            inputs: List of input field names
            outputs: List of output field names
            module: DSPy module to execute for this step
        """
        self._steps.append((inputs, outputs, module))

    def reset(self) -> None:
        """Reset the pipeline steps and any module state"""
        self._steps = []

    def assemble_pipeline(self) -> dspy.Module:
        """Assembles and returns a DSPy pipeline from registered steps
        
        The pipeline is constructed as a DSPy Module that chains together
        the registered steps. Each step's output becomes available for
        subsequent steps as inputs.
        
        Returns:
            dspy.Module: The assembled pipeline module
            
        Raises:
            ValueError: If no steps are registered
        """
        if not self._steps:
            raise ValueError("Cannot assemble an empty pipeline")
        
        class Pipeline(dspy.Module):
            """DSPy pipeline module"""
            def __init__(self, steps: List[Tuple[List[str], List[str], Any]]) -> None:
                super().__init__()
                self.step_tuples = steps  # store the full step tuples
                for i, (_, _, module) in enumerate(steps):
                    setattr(self, f'step_{i}', module)
            
            def forward(self, **inputs: Dict[str, Any]) -> dspy.Prediction:
                """Execute the pipeline steps sequentially"""
                data = inputs.copy()
                all_outputs = {}
                
                for i, (input_names, output_names, _) in enumerate(self.step_tuples):
                    # Prepare step inputs
                    step_inputs = {}
                    for name in input_names:
                        if name in data:
                            step_inputs[name] = data[name]
                        else:
                            raise ValueError(f"Pipeline Step {i}: Missing input '{name}'")
                    
                    # Execute step
                    module = getattr(self, f'step_{i}')
                    prediction = module(**step_inputs)
                    
                    # Collect outputs
                    for name in output_names:
                        if hasattr(prediction, name):
                            value = getattr(prediction, name)
                        else:
                            raise ValueError(f"Pipeline Step {i}: Output field '{name}' not found")
                        
                        data[name] = value
                        all_outputs[name] = value
                
                return dspy.Prediction(**all_outputs)
        
        return Pipeline(self._steps)
