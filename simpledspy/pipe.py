from typing import Any, Tuple, List, Callable, Dict
import dspy
from pipeline_manager import PipelineManager
from module_factory import ModuleFactory
import inspect
import os
import dis

class PipeFunction:
    _instances: Dict[str, 'PipeFunction'] = {}

    def __new__(cls, *args, **kwargs):
        # Get the caller's file and line number
        frame = inspect.currentframe().f_back
        location = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
        
        # Create or return existing instance for this location
        if location not in cls._instances:
            instance = super(PipeFunction, cls).__new__(cls)
            cls._instances[location] = instance
            instance._initialized = False
            instance._location = location
        return cls._instances[location]

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self.pipeline_manager = PipelineManager()
        self.module_factory = ModuleFactory()
        # Configure default LM with caching disabled
        self.lm = dspy.LM(model="deepseek/deepseek-chat")
        dspy.configure(lm=self.lm, cache=False)
        print(f"Initialized new PipeFunction at {self._location}")

    def _create_module(self, inputs: List[str], outputs: List[str], description: str = "") -> dspy.Module:
        """Create a DSPy module with the given signature."""
        return self.module_factory.create_module(
            inputs=inputs,
            outputs=outputs,
            description=description
        )

    def _get_caller_context(self) -> Tuple[List[str], str]:
        """Get the input variable names and assignment target from the calling context."""
        frame = inspect.currentframe()
        try:
            # Go up two frames to get the assignment context
            outer_frame = frame.f_back.f_back
            # Get the bytecode for the current frame
            bytecode = dis.Bytecode(outer_frame.f_code)
            
            # Find the input variable names
            input_names = []
            for instr in bytecode:
                if instr.offset < outer_frame.f_lasti and instr.opname == 'LOAD_NAME':
                    input_names.append(instr.argval)
            
            # Find the STORE_NAME opcode that follows our call
            target_name = None
            for instr in bytecode:
                if instr.offset > outer_frame.f_lasti and instr.opname == 'STORE_NAME':
                    target_name = instr.argval
                    break
            
            if not target_name:
                raise ValueError("pipe must be called in an assignment context.")
            
            return input_names, target_name
        finally:
            del frame

    def __call__(self, *args, description: str = None) -> Any:
        """
        Executes a DSPy module with the given signature.
        
        Args:
            *args: Input arguments
            description: Optional description of the module's purpose
            
        Returns:
            The output value
        """
        # Get the input names and assignment target from the calling context
        input_names, output_name = self._get_caller_context()
        
        # Use the actual variable names as input field names
        inputs = input_names
            
        # Create module dynamically with correct output name
        module = self._create_module(inputs, [output_name], description)
        
        # Create input dict
        input_dict = {field: arg for field, arg in zip(inputs, args)}
        
        # Execute module
        result = module(**input_dict)
        
        # Register step
        self.pipeline_manager.register_step(inputs=inputs, outputs=[output_name], module=module)
        
        # Return the output value
        return getattr(result, output_name)

# Instantiate the pipe function
pipe = PipeFunction()
