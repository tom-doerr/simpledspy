"""Module Caller for DSPy modules

Provides base classes for Predict and ChainOfThought function calls
"""

import dis
import inspect
from typing import List, Dict, Any, Tuple
import dspy
from .pipeline_manager import PipelineManager
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

    # pylint: disable=too-many-locals,too-many-branches
    def _get_call_types_from_signature(self, frame: Any, 
            input_names: List[str], output_names: List[str]) -> Tuple[Dict[str, type], Dict[str, type]]:
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
            if func_name in caller_globals and isinstance(
                    caller_globals[func_name], type):
                # It's a class, get the method
                if func_name in caller_locals:
                    func = caller_locals[func_name]
                else:
                    func = caller_globals[func_name]
            else:
                func = caller_globals.get(func_name, None)
            if not (callable(func) or (isinstance(func, type) and 
                    hasattr(func, func_name))):
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
                if return_ann != inspect.Signature.empty:
                    # For single output, set type for the output name
                    if len(output_names) == 1:
                        output_types[output_names[0]] = return_ann
                    # For multiple outputs with Tuple type hints
                    elif hasattr(return_ann, '__args__') and len(return_ann.__args__) == len(output_names):
                        tuple_types = return_ann.__args__
                        for i, t in enumerate(tuple_types):
                            if i < len(output_names):
                                output_types[output_names[i]] = t
            except (ValueError, TypeError):
                # Skip signature issues in nested functions
                pass
        return input_types, output_types

    def _infer_input_names(self, args) -> List[str]:
        """Infer input variable names using bytecode analysis"""
        try:
            frame = inspect.currentframe()
            if frame is None:
                return [f"arg{i}" for i in range(len(args))]
            frame = frame.f_back
            if frame is None:
                return [f"arg{i}" for i in range(len(args))]
            code = frame.f_code
            call_index = frame.f_lasti
            instructions = list(dis.get_instructions(code))
            current_instruction = None
            for i, inst in enumerate(instructions):
                if inst.offset == call_index:
                    current_instruction = inst
                    break
            if current_instruction and current_instruction.opname == 'CALL_FUNCTION':
                arg_names = []
                # Start from the instruction before the call and go backwards
                start_index = i - 1
                # We'll collect the argument instructions in reverse order
                for j in range(len(args)):
                    idx = start_index - j
                    if idx < 0:
                        break
                    inst = instructions[idx]
                    if inst.opname in ['LOAD_NAME', 'LOAD_FAST', 'LOAD_GLOBAL', 'LOAD_DEREF']:
                        arg_names.append(inst.argval)
                    else:
                        break
                # Reverse to get the correct left-to-right order
                arg_names.reverse()
                if len(arg_names) < len(args):
                    # If we didn't get enough names, fill the rest with fallbacks
                    arg_names += [f"arg{k}" for k in range(len(arg_names), len(args))]
                return arg_names
            return [f"arg{i}" for i in range(len(args))]
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
    
    def _log_results(self, input_dict, output_names, output_values, description):
        """Log module inputs and outputs"""
        self.logger.log({
            'module': self.__class__.__name__.lower(),
            'inputs': input_dict,
            'outputs': dict(zip(output_names, output_values)),
            'description': description
        })
    
    def __call__(self, *args, inputs: List[str] = None, 
            outputs: List[str] = None, description: str = None, 
            lm_params: dict = None) -> Any:
        # Infer input names if not provided
        if inputs is None:
            try:
                input_names = self._infer_input_names(args)
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
        input_types, output_types = self._get_call_types_from_signature(frame, input_names, output_names)

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
        
        # Log results
        self._log_results(input_dict, output_names, output_values, description)
        
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
