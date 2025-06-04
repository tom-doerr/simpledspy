from typing import Any, List, Tuple
import dspy

class PipelineManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._steps = []
        return cls._instance

    def register_step(self, inputs: List[str], outputs: List[str], module: Any):
        self._steps.append((inputs, outputs, module))

    def reset(self):
        """Reset the pipeline steps and any module state"""
        self._steps = []
        # Also reset any DSPy module state
        dspy.settings.configure(reset=True)

    def assemble_pipeline(self):
        """Assembles and returns a DSPy pipeline from registered steps
        
        The pipeline is constructed as a DSPy Module that chains together
        the registered steps. Each step's output becomes available for
        subsequent steps as inputs.
        
        Returns:
            dspy.Module: The assembled pipeline module
        """
        if not self._steps:
            raise ValueError("Cannot assemble an empty pipeline")
        
        class Pipeline(dspy.Module):
            def __init__(self, steps):
                super().__init__()
                self.step_tuples = steps  # store the full step tuples
                for i, (_, _, module) in enumerate(steps):
                    setattr(self, f'step_{i}', module)
            
            def forward(self, **inputs):
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
