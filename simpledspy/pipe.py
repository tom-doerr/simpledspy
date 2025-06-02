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
            # Initialize the instance
            instance.pipeline_manager = PipelineManager()
            instance.module_factory = ModuleFactory()
            instance.optimization_manager = OptimizationManager()
            # Configure default LM with caching disabled
            instance.lm = dspy.LM(model="deepseek/deepseek-chat")
            dspy.configure(lm=instance.lm, cache=False)
        return cls._instance

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
        """Get the names of variables being passed and assigned using frame inspection."""
        frame = inspect.currentframe()
        try:
            # Go up two frames to get the assignment context
            outer_frame = frame.f_back.f_back
            if outer_frame is None:
                # Fallback to generic names if we can't get the outer frame
                input_names = [f"input_{i+1}" for i in range(num_args)]
                output_names = [f"output_{i+1}" for i in range(1)]  # Default to one output
                return input_names, output_names
            
            # Get input variable names from the call
            call_line = outer_frame.f_lineno
            source_lines = inspect.getsource(outer_frame.f_code).splitlines()
            if call_line > len(source_lines):
                # Fallback to generic names if the call line is beyond the source
                input_names = [f"input_{i+1}" for i in range(num_args)]
                output_names = [f"output_{i+1}" for i in range(1)]  # Default to one output
                return input_names, output_names
                
            call_line_source = source_lines[call_line - 1].strip()
            
            # Extract output names (left side of assignment)
            output_names = []
            if '=' in call_line_source:
                assignment_target = call_line_source.split('=')[0].strip()
                # Handle tuple unpacking
                if ',' in assignment_target:
                    output_names = [name.strip() for name in assignment_target.split(',')]
                else:
                    output_names = [assignment_target]
            
            # Extract input names from the call arguments
            input_names = []
            # Check if the call to pipe is present
            if 'pipe(' in call_line_source:
                call_args = call_line_source.split('pipe(')[1].split(')')[0].split(',')
                for arg in call_args:
                    arg = arg.strip()
                    if arg and not arg.startswith('description='):
                        input_names.append(arg)
            
            # If we didn't get enough input names, use generics
            if len(input_names) < num_args:
                input_names.extend(f"input_{i+1}" for i in range(len(input_names), num_args))
            
            return input_names, output_names
        except Exception as e:
            # Fallback to generic names on any error
            input_names = [f"input_{i+1}" for i in range(num_args)]
            output_names = [f"output_{i+1}" for i in range(1)]  # Default to one output
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
                # For basic types, try to convert
                try:
                    if output_type is int:
                        value = int(float(value.replace(',', '').strip()))
                    elif output_type is float:
                        value = float(value.replace(',', '').strip())
                    elif output_type is bool:
                        if isinstance(value, str):
                            value = value.lower() == 'true'
                        else:
                            value = bool(value)
                    elif output_type is str:
                        value = str(value)
                except (ValueError, TypeError):
                    pass  # Keep original value if conversion fails
            
            processed_outputs.append(value)
            
        if len(processed_outputs) == 1:
            return processed_outputs[0]
        return tuple(processed_outputs)
