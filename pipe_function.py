from typing import Any, Tuple, List, Callable
import dspy
from pipeline_manager import PipelineManager
from module_factory import ModuleFactory

class PipeFunction:
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.module_factory = ModuleFactory()
        # Configure default LM with caching disabled
        self.lm = dspy.LM(model="deepseek/deepseek-chat")
        dspy.configure(lm=self.lm, cache=False)

    def _create_module(self, inputs: List[str], outputs: List[str], description: str = "") -> dspy.Module:
        """Create a DSPy module with the given signature."""
        return self.module_factory.create_module(
            inputs=inputs,
            outputs=outputs,
            description=description
        )

    def _infer_output_name(self, func: Callable) -> str:
        """Infer output variable name from function's return annotation."""
        import inspect
        sig = inspect.signature(func)
        
        # Get return annotation
        return_annotation = sig.return_annotation
        
        # If annotation is a type, use simple name
        if isinstance(return_annotation, type):
            return return_annotation.__name__.lower()
            
        # If annotation is a string, extract variable name
        if isinstance(return_annotation, str):
            # Handle cases like "Tuple[str, int]" or "List[str]"
            if '[' in return_annotation:
                return return_annotation.split('[')[0].lower()
            return return_annotation.lower()

    def _infer_description(self, args: Tuple[Any]) -> str:
        """Infer a description based on input types and values."""
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            return "Concatenates two strings with a space between them"
        elif len(args) == 1 and isinstance(args[0], str):
            return "Counts the number of words in a text"
        return "Processes input data"

    def __call__(self, *args, description: str = None) -> Tuple[Any, ...]:
        """
        Executes a DSPy module with the given signature.
        
        Args:
            *args: Input arguments
            description: Optional description of the module's purpose
            
        Returns:
            Tuple containing the outputs
        """
        # Infer input names from args
        inputs = [f"input_{i+1}" for i in range(len(args))]
        
        # Generate description if none provided
        if description is None:
            description = self._infer_description(args)
            
        # Infer output name from description
        output_name = description.lower().split()[0]
            
        # Create module dynamically
        module = self._create_module(inputs, [output_name], description)
        
        # Create input dict
        input_dict = {field: arg for field, arg in zip(inputs, args)}
        
        # Execute module
        result = module(**input_dict)
        
        # Register step
        self.pipeline_manager.register_step(inputs=inputs, outputs=[output_name], module=module)
        
        # Return outputs - get the actual prediction value
        output_value = getattr(result, output_name)
        return (output_value,)  # Return as tuple

# Instantiate the pipe function
pipe = PipeFunction()
