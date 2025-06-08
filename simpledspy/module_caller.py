"""Module Caller for DSPy modules

Provides base classes for Predict and ChainOfThought function calls
"""

import inspect
from typing import List, Dict, Any, Tuple
import dspy
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
from .logger import Logger

class BaseCaller:
    """Base class for DSPy module callers"""
    _instances = {}
    
    def __new__(cls):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
            instance.module_factory = ModuleFactory()
            instance.optimization_manager = OptimizationManager()
            instance.lm = dspy.LM(model="deepseek/deepseek-chat")
            instance.logger = Logger()
            dspy.configure(lm=instance.lm, cache=False)
        return cls._instances[cls]
    
    # pylint: disable=too-many-arguments,too-many-positional-arguments
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

    def _infer_output_names(self, frame: Any) -> List[str]:
        """Infer output names based on assignment context"""
        if frame is None:
            return ["output"]
        
        # Remove parentheses from the line
        try:
            lines = inspect.getframeinfo(frame).code_context
            if not lines:
                return ["output"]
            line = lines[0].strip().replace('(', ' ').replace(')', ' ')
        except (AttributeError, IndexError, TypeError):
            return ["output"]
        
        # Handle single assignment without commas: names = predict(...)
        if '=' in line:
            lhs = line.split('=')[0].strip()
            # Handle multi-assignment: name, another = predict(...)
            if ',' in lhs:
                output_names = [name.strip() for name in lhs.split(',')]
            # Handle single assignment: result = predict(...)
            else:
                output_names = [lhs]
        # Handle no assignment: predict(...)
        else:
            output_names = ["output"]
            
        return output_names

    def _process_return_annotation(self, return_ann, output_names, output_types):
        """Helper to process return annotation types"""
        if return_ann == inspect.Signature.empty:
            return
            
        if len(output_names) == 1:
            output_types[output_names[0]] = return_ann
        elif hasattr(return_ann, '__args__') and \
                len(return_ann.__args__) == len(output_names):
            tuple_types = return_ann.__args__
            for i, t in enumerate(tuple_types):
                if i < len(output_names):
                    output_types[output_names[i]] = t

    # pylint: disable=too-many-locals,too-many-branches
    def _get_call_types_from_signature(self, frame: Any, 
            input_names: List[str], output_names: List[str]) -> Tuple[
                Dict[str, type], Dict[str, type]]:
        """Get input/output types from function signature"""
        input_types = {}
        output_types = {}
        if frame is None:
            return input_types, output_types

        caller_frame = frame
        caller_locals = caller_frame.f_locals
        caller_globals = caller_frame.f_globals
        # Get the function where the call happened
        func_name = caller_frame.f_code.co_name
        try:
            # First try to get function from locals
            func = caller_locals.get(func_name, None)
            if not func:
                # Then try globals
                func = caller_globals.get(func_name, None)
            if not func and hasattr(caller_frame, 'f_back'):
                # Try to get from outer frames
                outer_frame = caller_frame.f_back
                while outer_frame and not func:
                    func = outer_frame.f_locals.get(
                        func_name, None) or outer_frame.f_globals.get(
                        func_name, None)
                    outer_frame = outer_frame.f_back
            # If we found something that isn't callable, set to None
            if not callable(func):
                func = None
        except (AttributeError, KeyError, TypeError):
            func = None
            
        if func:
            try:
                signature = inspect.signature(func, follow_wrapped=True)
                # Get the type hints for parameters in the calling function
                for param_name in signature.parameters:
                    if param_name in input_names:
                        param = signature.parameters[param_name]
                        if param.annotation != inspect.Parameter.empty:
                            input_types[param_name] = param.annotation
                # Get the type hints for return value
                return_ann = signature.return_annotation
                self._process_return_annotation(
                    return_ann, output_names, output_types)
            except (ValueError, TypeError):
                # Skip signature issues in nested functions
                pass
        return input_types, output_types

    def _infer_input_names(self, args) -> List[str]:
        """Infer input variable names using frame inspection"""
        try:
            frame = inspect.currentframe().f_back.f_back
            if frame is None:
                return [f"arg{i}" for i in range(len(args))]
            
            # Get all local and global variables from the calling frame
            all_vars = {**frame.f_globals, **frame.f_locals}
            
            # Build a mapping from value id to list of variable names
            value_to_names = {}
            for name, value in all_vars.items():
                vid = id(value)
                if vid not in value_to_names:
                    value_to_names[vid] = []
                value_to_names[vid].append(name)
            
            # Map each argument to a unique variable name
            used_names = set()
            arg_names = []
            for arg in args:
                vid = id(arg)
                candidate_names = value_to_names.get(vid, [])
                # Filter out reserved names and names we've already used
                candidate_names = [
                    name for name in candidate_names 
                    if name not in ['args', 'kwargs', 'self'] and name not in used_names
                ]
                if candidate_names:
                    # Sort to ensure deterministic order
                    candidate_names.sort()
                    chosen_name = candidate_names[0]
                    arg_names.append(chosen_name)
                    used_names.add(chosen_name)
                else:
                    arg_names.append(f"arg{len(arg_names)}")
            
            return arg_names
        except (AttributeError, ValueError, IndexError, TypeError):
            return [f"arg{i}" for i in range(len(args))]
    
    def _run_module(self, module, input_dict, lm_params):
        """Run the module with optional LM parameter overrides"""
        if lm_params:
            original_params = {}
            for key, value in lm_params.items():
                if hasattr(self.lm, key):
                    original_params[key] = getattr(self.lm, key)
                    setattr(self.lm, key, value)
            try:
                return module(**input_dict)
            finally:
                for key, value in original_params.items():
                    setattr(self.lm, key, value)
        else:
            return module(**input_dict)
    
    def _log_results(self, input_dict, input_names, output_names, output_values, description):
        """Log module inputs and outputs with meaningful names"""
        # Create input structure with both names and values
        inputs_data = []
        for name in input_names:
            if name in input_dict:
                inputs_data.append({
                    'name': name,
                    'value': input_dict[name]
                })
        
        # Create output structure with both names and values
        outputs_data = []
        for i, name in enumerate(output_names):
            outputs_data.append({
                'name': name,
                'value': output_values[i]
            })
        
        self.logger.log({
            'module': self.__class__.__name__.lower(),
            'inputs': inputs_data,
            'outputs': outputs_data,
            'description': description
        })
    
    def __call__(self, *args, inputs: List[str] = None, 
            outputs: List[str] = None, description: str = None, 
            lm_params: dict = None) -> Any:
        # Inspect variables if names needed
        captured_vars = {}
        if inputs is None:
            try:
                # Capture possible local/global variables
                frame = inspect.currentframe().f_back
                captured_vars = {**frame.f_locals, **frame.f_globals} if frame else {}
            except (AttributeError, ValueError, IndexError, TypeError):
                captured_vars = {}
        
        # Infer input names if not provided
        if inputs is None:
            try:
                input_names = self._infer_input_names(args)
                # Map from values to names for preserving original names
                value_to_name = {id(v): name for name, v in captured_vars.items()}
                # Try to map each argument to its original name
                reserved = ['args', 'kwargs', 'self']
                for idx, arg in enumerate(args):
                    arg_id = id(arg)
                    if arg_id in value_to_name and value_to_name[arg_id] not in reserved:
                        input_names[idx] = value_to_name[arg_id]
            except (AttributeError, ValueError, IndexError, TypeError):
                input_names = [f"arg{i}" for i in range(len(args))]
        else:
            if len(inputs) != len(args):
                raise ValueError(f"Expected {len(args)} input names, got {len(inputs)}")
            input_names = inputs
        
        # Infer output names if not provided
        if outputs is None:
            try:
                frame = inspect.currentframe().f_back
                output_names = self._infer_output_names(frame)
            except (AttributeError, ValueError, IndexError, TypeError):
                output_names = ["output"]
        else:
            output_names = outputs
        
        # Get type hints from caller context
        input_types = {}
        output_types = {}
        try:
            frame = inspect.currentframe().f_back
        except (AttributeError, ValueError, IndexError, TypeError):
            frame = None
        input_types, output_types = self._get_call_types_from_signature(
            frame, input_names, output_names)

        # Create and run the module
        module = self._create_module(
            inputs=input_names, 
            outputs=output_names,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        input_dict = dict(zip(input_names, args))
        prediction_result = self._run_module(module, input_dict, lm_params)
        
        # Process and validate results
        for name in output_names:
            if not hasattr(prediction_result, name):
                raise AttributeError(f"Output field '{name}' not found in prediction result")
        output_values = [getattr(prediction_result, name) for name in output_names]
        
        # Log results with meaningful names
        self._log_results(input_dict, input_names, output_names, output_values, description)
        
        # Return single value or tuple
        return output_values[0] if len(output_values) == 1 else tuple(output_values)

# pylint: disable=too-few-public-methods
class Predict(BaseCaller):
    """Predict module caller - replaces pipe() function"""

class ChainOfThought(BaseCaller):
    """ChainOfThought module caller"""
    
    # pylint: disable=too-many-arguments, R0917
    def _create_module(self, inputs: List[str], outputs: List[str], 
                     input_types: Dict[str, type] = None,
                     output_types: Dict[str, type] = None,
                     description: str = "") -> dspy.Module:
        signature_class = self.module_factory.create_signature(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        return dspy.ChainOfThought(signature_class)
