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
        """Reset the pipeline steps"""
        self._steps = []

    def assemble_pipeline(self):
        class Pipeline(dspy.Module):
            def __init__(self, steps: List[Tuple[List[str], List[str], dspy.Module]]):
                super().__init__()
                self.steps = steps
                for i, (_, _, module) in enumerate(steps):
                    setattr(self, f'step_{i}', module)

            def forward(self, **inputs):
                # Start with the initial inputs
                data = inputs.copy()
                
                # We'll collect all outputs for the final result
                all_outputs = {}
                
                for i, (input_names, output_names, _) in enumerate(self.steps):
                    # Prepare inputs for this step
                    step_inputs = {}
                    for name in input_names:
                        if name not in data:
                            raise ValueError(f"Pipeline Step {i}: Missing input '{name}'")
                        step_inputs[name] = data[name]
                    
                    # Run the step
                    prediction = getattr(self, f'step_{i}')(**step_inputs)
                    
                    # Store outputs for next steps and final collection
                    for name in output_names:
                        if not hasattr(prediction, name):
                            raise ValueError(f"Pipeline Step {i}: Output field '{name}' not found")
                        value = getattr(prediction, name)
                        data[name] = value  # Make available for next steps
                        all_outputs[name] = value
                
                # Return all outputs as a dictionary
                return all_outputs

        return Pipeline(self._steps)
