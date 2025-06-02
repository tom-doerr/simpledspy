from typing import Any, List
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

    def assemble_pipeline(self):
        class Pipeline(dspy.Module):
            def __init__(self, steps: List[Tuple[List[str], List[str], dspy.Module]]):
                super().__init__()
                self.steps = steps
                for i, (_, _, module) in enumerate(steps):
                    setattr(self, f'step_{i}', module)

            def forward(self, **inputs):
                data = inputs.copy()
                output_names = []
                
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
                
                if not output_names:
                    return None
                if len(output_names) == 1:
                    return data[output_names[0]]
                return tuple(data[name] for name in output_names)

        return Pipeline(self._steps)
