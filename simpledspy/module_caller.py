import os
import dis
from typing import Any, Tuple, List, Callable, Dict, Optional, get_type_hints, get_origin, get_args
import dspy
from .pipeline_manager import PipelineManager
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
import inspect

class BaseCaller:
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
            instance.pipeline_manager = PipelineManager()
            instance.module_factory = ModuleFactory()
            instance.optimization_manager = OptimizationManager()
            instance.lm = dspy.LM(model="deepseek/deepseek-chat")
            dspy.configure(lm=instance.lm, cache=False)
            
            # Store creation location
            frame = inspect.currentframe()
            while frame.f_back:
                frame = frame.f_back
            instance._location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        return cls._instances[cls]
    
    def _create_module(self, inputs: List[str], outputs: List[str], 
                     input_types: Dict[str, type] = None,
                     output_types: Dict[str, type] = None,
                     description: str = "") -> dspy.Module:
        return self.module_factory.create_module(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )

    def _get_caller_context(self, num_args: int) -> Tuple[List[str], List[str]]:
        """Get input and output names with improved robustness"""
        try:
            frame = inspect.currentframe().f_back.f_back
            if frame is None:
                return self._generic_names(num_args)
                
            call_line = frame.f_lineno
            source_lines = inspect.getsource(frame.f_code).splitlines()
            if call_line > len(source_lines):
                return self._generic_names(num_args)
                
            call_line_source = source_lines[call_line-1].strip()
            output_names = self._extract_output_names(call_line_source)
            input_names = self._extract_input_names(call_line_source, num_args)
            
            return input_names, output_names
        except Exception as e:
            # Fallback to generic names on any error
            return self._generic_names(num_args)

    def _generic_names(self, num_args: int) -> Tuple[List[str], List[str]]:
        """Fallback to generic names"""
        input_names = [f"input_{i+1}" for i in range(num_args)]
        output_names = [f"output_{i+1}" for i in range(1)]  # Default to one output
        return input_names, output_names

    def _extract_output_names(self, source: str) -> List[str]:
        """Extract output variable names from assignment"""
        if '=' not in source:
            return []
            
        assignment = source.split('=')[0].strip()
        if ',' in assignment:
            return [name.strip() for name in assignment.split(',')]
        return [assignment]

    def _extract_input_names(self, source: str, num_args: int) -> List[str]:
        """Extract input variable names from function call"""
        # Use the actual function name from the class
        func_name = self.FUNCTION_NAME
        if not func_name:
            return self._generic_names(num_args)[0]
            
        if f'{func_name}(' not in source:
            return self._generic_names(num_args)[0]
            
        args_str = source.split(f'{func_name}(')[1].split(')')[0]
        args_list = [arg.strip() for arg in args_str.split(',') if arg.strip()]
        
        # Filter out description keyword arguments
        input_names = []
        for arg in args_list:
            if not arg.startswith('description='):
                input_names.append(arg.split('=')[0] if '=' in arg else arg)
        
        # Ensure we have enough names
        if len(input_names) < num_args:
            input_names += [f"input_{i+1}" for i in range(len(input_names), num_args)]
            
        return input_names

    def __call__(self, *args, description: str = None, metric: Callable = None) -> Any:
        frame = inspect.currentframe().f_back
        type_hints = {}
        if hasattr(frame, 'f_locals') and '__annotations__' in frame.f_locals:
            type_hints.update(frame.f_locals['__annotations__'])
        if hasattr(frame, 'f_globals') and '__annotations__' in frame.f_globals:
            for k, v in frame.f_globals['__annotations__'].items():
                if k not in type_hints:
                    type_hints[k] = v
                    
        input_names, output_names = self._get_caller_context(len(args))
        input_types = {}
        output_types = {}
        
        # Get type hints for inputs and outputs
        for name in input_names:
            if name in type_hints:
                input_types[name] = type_hints[name]
        for name in output_names:
            if name in type_hints:
                output_types[name] = type_hints[name]
        
        module = self._create_module(
            inputs=input_names, 
            outputs=output_names,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        
        input_dict = {}
        for i, arg in enumerate(args):
            input_dict[input_names[i]] = arg
            
        prediction_result = module(**input_dict)
        self.pipeline_manager.register_step(inputs=input_names, outputs=output_names, module=module)
        
        processed_outputs = []
        for name in output_names:
            value = getattr(prediction_result, name)
            
            # Apply type conversion if specified
            if name in output_types:
                target_type = output_types[name]
                try:
                    if target_type is int:
                        value = int(float(value.replace(',', '').strip()))
                    elif target_type is float:
                        value = float(value.replace(',', '').strip())
                    elif target_type is bool:
                        value = value.lower() == 'true' if isinstance(value, str) else bool(value)
                    elif target_type is str:
                        value = str(value)
                except (ValueError, TypeError) as e:
                    # Preserve original value but add warning
                    value = f"CONVERSION ERROR: {str(e)} - Original: {value}"
            processed_outputs.append(value)
            
        if len(processed_outputs) == 1:
            return processed_outputs[0]
        return tuple(processed_outputs)

class Predict(BaseCaller):
    """Predict module caller - replaces pipe() function"""
    FUNCTION_NAME = 'predict'

class ChainOfThought(BaseCaller):
    """ChainOfThought module caller"""
    FUNCTION_NAME = 'chain_of_thought'
    
    def _create_module(self, inputs: List[str], outputs: List[str], 
                     input_types: Dict[str, type] = None,
                     output_types: Dict[str, type] = None,
                     description: str = "") -> dspy.Module:
        signature = self.module_factory.create_module(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        return dspy.ChainOfThought(signature)
