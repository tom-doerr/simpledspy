from typing import Any, Tuple, List
import dspy
from pipeline_manager import PipelineManager

class PipeFunction:
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.lm = None

    def configure_lm(self, model_name: str = "gpt-3.5-turbo"):
        """Configure the language model to use"""
        self.lm = dspy.OpenAI(model=model_name)
        dspy.configure(lm=self.lm)

    def __call__(self, *args, modules: List[Any]) -> Tuple[Any, ...]:
        """
        Executes modules immediately and registers the steps.
        
        Args:
            *args: Input arguments
            modules: List of DSPy modules to process the inputs
            
        Returns:
            Tuple containing the outputs
        """
        if not modules:
            raise ValueError("Modules parameter is required")
            
        # Configure default LM if not already configured
        if self.lm is None:
            self.configure_lm()
            
        # Get outputs from module signatures
        outputs = []
        for module in modules:
            outputs.extend(list(module.signature.output_fields.keys()))
            
        # Create input dict from args and module inputs
        results = []
        for module in modules:
            # Get the input field names for this module
            input_fields = list(module.signature.input_fields.keys())
            
            # Create input dict matching the module's signature
            if len(input_fields) != len(args):
                raise ValueError(f"Module expects {len(input_fields)} inputs but got {len(args)}")
                
            input_dict = {field: arg for field, arg in zip(input_fields, args)}
            
            # Execute module with proper keyword arguments
            result = module(**input_dict)
            results.append(result)
        
        # Register steps using actual input/output names from module signatures
        for module in modules:
            inputs = list(module.signature.input_fields.keys())
            module_outputs = list(module.signature.output_fields.keys())
            self.pipeline_manager.register_step(inputs=inputs, outputs=module_outputs, module=module)
            
        # Extract and return the output values
        return tuple(getattr(result, output) for result, output in zip(results, outputs))

# Instantiate the pipe function
pipe = PipeFunction()
