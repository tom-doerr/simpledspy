"""Module Caller for DSPy modules

Provides base classes for Predict and ChainOfThought function calls
"""

import dis
import dspy
from typing import List, Dict, Any, Tuple
from .pipeline_manager import PipelineManager
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
from .logger import Logger

class BaseCaller:
    """Base class for DSPy module callers"""
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
            instance.pipeline_manager = PipelineManager()
            instance.module_factory = ModuleFactory()
            instance.optimization_manager = OptimizationManager()
            instance.lm = dspy.LM(model="deepseek/deepseek-chat")
            instance.logger = Logger()
            dspy.configure(lm=instance.lm, cache=False)
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

    def _infer_output_names(self, frame: Any) -> List[str]:
        """Infer output names based on assignment context"""
        import inspect
        
        if frame is None:
            return ["output"]
        
        # Remove parentheses from the line
        try:
            lines = inspect.getframeinfo(frame).code_context
            if not lines:
                return ["output"]
            line = lines[0].strip().replace('(', ' ').replace(')', ' ')
        except Exception:
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

    def _get_call_types_from_signature(self, frame: Any, input_names: List[str]) -> Tuple[Dict[str, type], Dict[str, type]]:
        """Get input/output types from function signature"""
        import inspect
        
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
            if func_name in caller_globals and isinstance(caller_globals[func_name], type):
                # It's a class, get the method
                if func_name in caller_locals:
                    func = caller_locals[func_name]
                else:
                    func = caller_globals[func_name]
            else:
                func = caller_globals.get(func_name, None)
            if not (callable(func) or (isinstance(func, type) and hasattr(func, func_name))):
                func = None
        except (AttributeError, KeyError, TypeError):
            func = None
            
        if func:
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
                # For single output, set type for output0
                if len(input_names) == 1:
                    output_types[input_names[0]] = return_ann
                # For multiple outputs with Tuple type hints
                elif hasattr(return_ann, '__tuple_params__'):
                    tuple_types = return_ann.__tuple_params__
                    for i, t in enumerate(tuple_types):
                        if i < len(input_names):
                            output_types[input_names[i]] = t
                # For multiple outputs without hint - ignore
        return input_types, output_types

    def __call__(self, *args, inputs: List[str] = None, outputs: List[str] = None, description: str = None, lm_params: dict = None) -> Any:
        if not hasattr(self, 'FUNCTION_NAME'):
            self.FUNCTION_NAME = 'base_caller'
        # Use custom input names if provided, otherwise generate meaningful defaults
        if inputs is None:
            # Try to get the names of the variables passed as arguments
            try:
                import inspect
                frame = inspect.currentframe().f_back
                code = frame.f_code
                call_index = frame.f_lasti
                instructions = list(dis.get_instructions(code))
                current_instruction = None
                for i, inst in enumerate(instructions):
                    if inst.offset == call_index:
                        current_instruction = inst
                        break
                if current_instruction and current_instruction.opname == 'CALL_FUNCTION':
                    # The names are the variable names in the stack
                    # This is a simplified approach: we use the names from the locals
                    # that were used in the call
                    # We get the names from the bytecode
                    # This might not work in all cases, but it's a best effort
                    arg_names = []
                    for i in range(len(args)):
                        # Look for LOAD_NAME or LOAD_FAST instructions before the call
                        # We go backwards from the current instruction
                        for j in range(i+1):
                            prev_inst = instructions[i - j]
                            if prev_inst.opname in ['LOAD_NAME', 'LOAD_FAST', 'LOAD_GLOBAL', 'LOAD_DEREF']:
                                arg_names.append(prev_inst.argval)
                                break
                        else:
                            arg_names.append(f"arg{i}")
                    input_names = arg_names
                else:
                    input_names = [f"arg{i}" for i in range(len(args))]
            except Exception:
                input_names = [f"arg{i}" for i in range(len(args))]
        else:
            if len(inputs) != len(args):
                raise ValueError(f"Expected {len(args)} input names, got {len(inputs)}")
            input_names = inputs
        
        # Use custom output names if provided, otherwise try to infer from the assignment
        if outputs is None:
            try:
                import inspect
                frame = inspect.currentframe().f_back
                output_names = self._infer_output_names(frame)
            except Exception:
                output_names = ["output"]
        else:
            output_names = outputs
        
        # Get type hints from the caller's function if available
        input_types = {}
        output_types = {}
        frame = None
        try:
            import inspect
            frame = inspect.currentframe().f_back
        except Exception:
            pass
        input_types, output_types = self._get_call_types_from_signature(frame, input_names)

        module = self._create_module(
            inputs=input_names, 
            outputs=output_names,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        input_dict = dict(zip(input_names, args))
        
        # Save and update LM parameters if needed
        if lm_params:
            original_params = {}
            for key, value in lm_params.items():
                if hasattr(self.lm, key):
                    original_params[key] = getattr(self.lm, key)
                    setattr(self.lm, key, value)
            try:
                prediction_result = module(**input_dict)
            finally:
                for key, value in original_params.items():
                    setattr(self.lm, key, value)
        else:
            prediction_result = module(**input_dict)
        
        self.pipeline_manager.register_step(inputs=input_names, outputs=output_names, module=module)
        
        # Check that the module returned the expected outputs
        for name in output_names:
            if not hasattr(prediction_result, name):
                raise AttributeError(f"Output field '{name}' not found in prediction result")
                
        output_values = [getattr(prediction_result, name) for name in output_names]
        
        # Log inputs and outputs
        self.logger.log({
            'module': self.FUNCTION_NAME,
            'inputs': input_dict,
            'outputs': dict(zip(output_names, output_values)),
            'description': description
        })
        
        # Return single value for one output, tuple for multiple
        if len(output_values) == 1:
            return output_values[0]
        return tuple(output_values)

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
        signature_class = self.module_factory.create_signature(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        return dspy.ChainOfThought(signature_class)
