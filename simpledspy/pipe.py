from typing import Any, Tuple, List, Callable, Dict, Optional, get_type_hints, get_origin, get_args
import dspy
from .pipeline_manager import PipelineManager
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
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
        self.optimization_manager = OptimizationManager()
        # Configure default LM with caching disabled
        self.lm = dspy.LM(model="deepseek/deepseek-chat")
        dspy.configure(lm=self.lm, cache=False)

    def _create_module(self, inputs: List[str], outputs: List[str], 
                     input_types: Dict[str, type] = None,
                     output_types: Dict[str, type] = None,
                     description: str = "") -> dspy.Module:
        """Create a DSPy module with the given signature."""
        return self.module_factory.create_module(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )

    def _get_caller_context(self, num_args: int) -> Tuple[List[str], List[str]]:
        """Get the names of variables being passed and assigned.
        
        Args:
            num_args: Number of arguments expected
            
        Returns:
            Tuple of (input_names, output_names)
        """
        frame = inspect.currentframe()
        try:
            # Go up two frames to get the assignment context
            outer_frame = frame.f_back.f_back
            # Get the bytecode for the current frame
            bytecode = dis.Bytecode(outer_frame.f_code)
            
            # Find the variable names being passed
            input_names = []
            seen_ops = set()
            
            # Walk backwards through instructions to find actual inputs
            for instr in reversed(list(bytecode)):
                if instr.offset >= outer_frame.f_lasti:
                    continue
                    
                # Skip if we've already seen this operation
                if instr.offset in seen_ops:
                    continue
                seen_ops.add(instr.offset)
                
                # Only include LOAD operations that are actual inputs
                if instr.opname in ('LOAD_NAME', 'LOAD_FAST'):
                    # Skip function names and other non-inputs
                    if instr.argval not in ('pipe', 'print', 'self'):
                        # Only add if it's one of our actual arguments
                        print("instr:", instr)
                        print("instr.argval:", instr.argval)
                        if len(input_names) < num_args:
                            input_names.append(instr.argval)
            
            # Reverse to maintain original order and ensure correct count
            input_names = list(reversed(input_names))
            
            # If we didn't get enough names, fill with generic ones
            if len(input_names) < num_args:
                input_names.extend(f"input_{i+1}" for i in range(len(input_names), num_args))
                
            # Find STORE_NAME/STORE_FAST opcodes only for the current line
            output_names = []
            current_line = outer_frame.f_lineno
            for instr in bytecode:
                # Only look at instructions after our call and on the same line
                if (instr.offset > outer_frame.f_lasti and 
                    instr.positions.lineno == current_line):
                    if instr.opname in ('STORE_NAME', 'STORE_FAST'):
                        output_names.append(instr.argval)
                        # Stop when we hit a different operation
                        if instr.opname not in ('STORE_NAME', 'STORE_FAST'):
                            break
            
            if not output_names:
                raise ValueError("pipe must be called in an assignment context.")
            
            # Remove duplicates and output names from inputs
            input_names = list(dict.fromkeys(input_names))
            for output_name in output_names:
                if output_name in input_names:
                    input_names.remove(output_name)
            
            return input_names, output_names
        finally:
            del frame

    def __call__(self, *args, description: str = None, metric: Callable = None) -> Any:
        """
        Executes a DSPy module with the given signature.
        
        Args:
            *args: Input arguments
            description: Optional description of the module's purpose
            metric: Optional metric function for optimization
            
        Returns:
            The output value
        """
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        
        # Get type hints from the caller's local scope
        type_hints = {}
        if '__annotations__' in frame.f_locals:
            type_hints = frame.f_locals['__annotations__']
        elif '__annotations__' in frame.f_globals:
            type_hints = frame.f_globals['__annotations__']
            
        # Extract output type from the assignment target
        output_types = {}
        for var_name, hint in type_hints.items():
            if var_name in frame.f_locals:
                if get_origin(hint) is tuple:
                    # Multiple return values
                    output_types = {f'output_{i}': t for i, t in enumerate(get_args(hint))}
                else:
                    # Single return value
                    output_types = {'output': hint}
                break
        # Configure metric if provided
        if metric is not None:
            self.optimization_manager.configure(metric=metric)
            
        # Get the input and output variable names
        input_names, output_names = self._get_caller_context(len(args))
        
        # Create module with type hints from reflection
        module = self._create_module(
            input_names, 
            output_names,
            output_types=output_types,
            description=description
        )
        
        # Use actual input names if we found them, otherwise fall back to generic names
        if len(input_names) != len(args):
            input_names = [f"input_{i+1}" for i in range(len(args))]
        module = self._create_module(input_names, output_names, description)
        
        # Create input dict
        input_dict = {field: arg for field, arg in zip(input_names, args)}
        
        # Execute module
        result = module(**input_dict)
        
        # Register step
        self.pipeline_manager.register_step(inputs=input_names, outputs=output_names, module=module)
        
        # Handle outputs based on the output_names list with type conversion
        outputs = []
        for i, field in enumerate(output_names):
            value = getattr(result, field)
            
            # Get the output type for this field
            output_type = None
            print("output_types:", output_types)
            if output_types:
                # Try to match by field name first
                if field in output_types:
                    output_type = output_types[field]
                # Fall back to positional index for tuple returns
                elif f'output_{i}' in output_types:
                    output_type = output_types[f'output_{i}']
                elif 'output' in output_types and len(output_names) == 1:
                    output_type = output_types['output']
            
            # Perform type conversion if we have a type
            if output_type:
                # Handle string numbers with commas/spaces
                if isinstance(value, str) and output_type in (int, float):
                    value = value.replace(',', '').strip()
                
                # Convert to target type
                if output_type is int:
                    value = int(float(value))  # Convert via float first to handle decimal strings
                elif output_type is float:
                    value = float(value)
                elif output_type is bool:
                    value = bool(value)
                elif output_type is str:
                    value = str(value)
            
            outputs.append(value)
            
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

# Instantiate the pipe function
pipe = PipeFunction()
