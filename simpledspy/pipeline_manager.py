from typing import List, Tuple, Any
import dspy

class PipelineManager:
    def __init__(self):
        self._steps = []

    def register_step(self, inputs: List[str], outputs: List[str], module: Any):
        self._steps.append((inputs, outputs, module))

    def assemble_pipeline(self):
        if not self._steps:
            raise ValueError("No pipeline steps registered. Please make pipe calls before assembling the pipeline.")

        class Pipeline(dspy.Module):
            def __init__(self, steps: List[Tuple[List[str], List[str], dspy.Module]]):
                super().__init__()
                self.steps = steps
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
                    
                    prediction = getattr(self, f'step_{i}')(**step_inputs)
                    
                    for name in output_names:
                        if not hasattr(prediction, name):
                            raise ValueError(f"Pipeline Step {i}: Output field '{name}' not found")
                        data[name] = getattr(prediction, name)
                
                if len(output_names) == 1:
                    return data[output_names[0]]
                return tuple(data[name] for name in output_names)

        return Pipeline(self._steps)
