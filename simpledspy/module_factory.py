from typing import List
import dspy

class ModuleFactory:
    def create_module(self, inputs: List[str], outputs: List[str], description: str = "") -> dspy.Module:
        class DynamicModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.input_fields = inputs
                self.output_fields = outputs

            def forward(self, **kwargs):
                return {output: kwargs[input] for input, output in zip(self.input_fields, self.output_fields)}

        return DynamicModule()
