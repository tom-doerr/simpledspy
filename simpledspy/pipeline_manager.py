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
        """Assembles and returns a DSPy pipeline from registered steps"""
        if not self._steps:
            raise ValueError("Cannot assemble an empty pipeline")
            
        class Pipeline(dspy.Module):
            def __init__(self, steps: List[Tuple[List[str], List[str], dspy.Module]]):
                super().__init__()
                self.steps = steps
                for i, (_, _, module) in enumerate(steps):
                    setattr(self, f'step_{i}', module)

            def forward(self, **inputs):
                """Executes the pipeline with given inputs"""
                # Start with the initial inputs
                data = inputs.copy()
                
                # We'll collect all outputs for the final result
                all_outputs = {}
                
                for i, (input_names, output_names, _) in enumerate(self.steps):
                    # Prepare inputs for this step
                    step_inputs = {}
                    for name in input_names:
                        if name not in data:
                            # Try to find matching input by prefix/suffix
                            matches = [k for k in data.keys() if name in k]
                            if matches:
                                step_inputs[name] = data[matches[0]]
                            else:
                                raise ValueError(f"Pipeline Step {i}: Missing input '{name}'")
                        else:
                            step_inputs[name] = data[name]
                    
                    # Run the step
                    prediction = getattr(self, f'step_{i}')(**step_inputs)
                    
                    # Store outputs for next steps and final collection
                    for name in output_names:
                        if isinstance(prediction, dict):
                            if name in prediction:
                                value = prediction[name]
                            else:
                                # Try to find matching output by prefix/suffix
                                matches = [k for k in prediction if name in k]
                                if matches:
                                    value = prediction[matches[0]]
                                else:
                                    raise ValueError(f"Pipeline Step {i}: Output field '{name}' not found")
                        else:
                            if hasattr(prediction, name):
                                value = getattr(prediction, name)
                            else:
                                # Try to find matching output by prefix/suffix
                                matches = [k for k in prediction.__dict__ if name in k]
                                if matches:
                                    value = getattr(prediction, matches[0])
                                else:
                                    raise ValueError(f"Pipeline Step {i}: Output field '{name}' not found")
                
                        data[name] = value  # Make available for next steps
                        all_outputs[name] = value
                
                # Return all outputs as a dictionary
                return dspy.Prediction(**all_outputs)

        return Pipeline(self._steps)
