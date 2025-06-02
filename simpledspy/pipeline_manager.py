from typing import List, Any, Tuple, Dict
import dspy

class PipelineManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Ensure _steps is initialized only once for the singleton instance
            cls._instance._steps = [] 
        return cls._instance

    def register_step(self, inputs: List[str], outputs: List[str], module: Any):
        # Stores a tuple: (list_of_input_names, list_of_output_names, module_instance)
        self._steps.append((inputs, outputs, module))

    def assemble_pipeline(self):
        """
        Assembles the registered steps into a DSPy pipeline module.
        
        Returns:
            dspy.Module: The assembled DSPy pipeline.
        """
        if not self._steps:
            raise ValueError("No pipeline steps registered. Please make pipe calls before assembling the pipeline.")

        class Pipeline(dspy.Module):
            def __init__(self, steps: List[Tuple[List[str], List[str], dspy.Module]]):
                super().__init__()
                self.steps = steps
                # Register each module instance
                for i, (_, _, module) in enumerate(steps):
                    setattr(self, f'step_{i}', module)

            def forward(self, **inputs):
                data = inputs.copy()
                
                for i, (input_names, output_names, _) in enumerate(self.steps):
                    step_inputs = {}
                    for name in input_names:
                        if name not in data:
                            raise ValueError(f"Pipeline Step {i}: Missing input '{name}'")
                        step_inputs[name] = data[name]
                    
                    # Execute the module
                    prediction = getattr(self, f'step_{i}')(**step_inputs)
                    
                    # Update data with outputs
                    for name in output_names:
                        if not hasattr(prediction, name):
                            raise ValueError(f"Pipeline Step {i}: Output field '{name}' not found")
                        data[name] = getattr(prediction, name)
                
                # Return final outputs
                if len(output_names) == 1:
                    return data[output_names[0]]
                return tuple(data[name] for name in output_names)

        return Pipeline(self._steps)
