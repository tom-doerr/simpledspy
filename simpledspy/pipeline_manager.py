from typing import List, Any
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
            def __init__(self, steps):
                super().__init__()
                self.steps = steps
                for i, step in enumerate(steps):
                    setattr(self, f'step_{i}', step[2])

            def forward(self, *args):
                result = args
                for step in self.steps:
                    module = step[2]
                    result = module(*result)
                return result

        return Pipeline(self._steps)
