from typing import Any, Tuple, List, Callable, Dict, Optional, get_type_hints, get_origin, get_args
import dspy
from .pipeline_manager import PipelineManager
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
import inspect
import os
import dis

class PipeFunction:
    _instance: Optional['PipeFunction'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            instance = super(PipeFunction, cls).__new__(cls)
            cls._instance = instance
            instance._initialized = False # Ensure __init__ runs only once for the true instance
        return cls._instance

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
                    if instr.argval not in ('pipe', 'print', 'self'): # 'self' could be a valid var name in some contexts
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
                        # Stop when we hit a different operation (might be too restrictive for tuple unpacking to multiple statements)
                        # However, for 'a, b = pipe()', it should capture 'a' and 'b' if they are stored sequentially.
                        # If it's 'tmp = pipe(); a=tmp[0]; b=tmp[1]', this only gets 'tmp'.
                    elif output_names: # If we started collecting output_names and hit something else
                        break 
            
            if not output_names:
                raise ValueError("pipe must be called in an assignment context.")
            
            # Remove duplicates and output names from inputs (if accidentally picked up)
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
        # Prioritize f_locals over f_globals for annotations in the immediate calling scope
        if hasattr(frame, 'f_locals') and '__annotations__' in frame.f_locals:
            type_hints.update(frame.f_locals['__annotations__'])
        if hasattr(frame, 'f_globals') and '__annotations__' in frame.f_globals:
            # Merge global annotations, giving precedence to local ones if names clash
            for k, v in frame.f_globals['__annotations__'].items():
                if k not in type_hints:
                    type_hints[k] = v
            
        # Get the input and output variable names
        input_names, output_names = self._get_caller_context(len(args))

        # Extract output types based on determined output_names and available type_hints
        output_types_for_module = {}
        for name in output_names:
            hint = type_hints.get(name)
            if hint is not None:
                if get_origin(hint) is tuple and get_args(hint): # e.g. result: Tuple[int, str]
                    # If a single assignment target is a Tuple, this implies multiple conceptual outputs.
                    # The module signature should reflect these individual fields.
                    # This part is complex: ModuleFactory expects output_types to map to names in 'outputs' list.
                    # If 'outputs' is ['result'] but hint is Tuple[int, str], ModuleFactory needs adjustment
                    # or 'outputs' list itself needs to be ['result_0', 'result_1'].
                    # For now, we'll pass the type hint for the variable name itself.
                    # The ModuleFactory will use this to describe the field 'result'.
                    # Actual parsing of tuple string from LLM and conversion is a deeper issue.
                    output_types_for_module[name] = hint 
                else:
                    output_types_for_module[name] = hint
            
        # Configure metric if provided
        if metric is not None:
            self.optimization_manager.configure(metric=metric)
            
        # Create module with type hints from reflection
        module = self._create_module(
            inputs=input_names, 
            outputs=output_names,
            # input_types can be similarly inferred if needed, but not currently done
            output_types=output_types_for_module,
            description=description
        )
        
        # Use actual input names if we found them, otherwise fall back to generic names
        # This check ensures that if _get_caller_context's input name inference was incomplete/failed,
        # we use generic names and recreate the module with that generic signature.
        if len(input_names) != len(args) or not all(isinstance(name, str) for name in input_names):
            input_names = [f"input_{i+1}" for i in range(len(args))]
            module = self._create_module(
                inputs=input_names, 
                outputs=output_names,
                output_types=output_types_for_module,
                description=description
            )
        
        # Create input dict
        input_dict = {field: arg for field, arg in zip(input_names, args)}
        
        # Execute module
        prediction_result = module(**input_dict) # This is a dspy.Prediction object
        
        # Register step
        self.pipeline_manager.register_step(inputs=input_names, outputs=output_names, module=module)
        
        # Handle outputs based on the output_names list with type conversion
        processed_outputs = []
        for i, field_name in enumerate(output_names):
            value = getattr(prediction_result, field_name)
            
            # Get the output type for this specific field_name from the caller's annotations
            output_type = type_hints.get(field_name)
            
            # Perform type conversion if we have a type
            if output_type:
                # Handle string numbers with commas/spaces before specific type conversion
                if isinstance(value, str) and (get_origin(output_type) is None and output_type in (int, float)):
                    value = value.replace(',', '').strip()
                
                # Convert to target type
                # TODO: Add more robust handling for complex types like Tuple[int, str]
                # For now, basic types are handled.
                if output_type is int:
                    try:
                        value = int(float(value))  # Convert via float first to handle decimal strings like "30.0"
                    except (ValueError, TypeError):
                        pass # Keep original value if conversion fails
                elif output_type is float:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        pass 
                elif output_type is bool:
                    if isinstance(value, str):
                        value = value.lower() == 'true'
                    else:
                        value = bool(value)
                elif output_type is str:
                    value = str(value)
            
            processed_outputs.append(value)
            
        if len(processed_outputs) == 1:
            return processed_outputs[0]
        return tuple(processed_outputs)
