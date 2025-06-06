"""Module Caller for DSPy modules

Provides base classes for Predict and ChainOfThought function calls
"""

import dis
import dspy
from typing import List, Dict, Any
from .pipeline_manager import PipelineManager
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
from .evaluator import Evaluator

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
            instance.evaluator = Evaluator()
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

    def __call__(self, *args, inputs: List[str] = None, outputs: List[str] = None, description: str = None, lm_params: dict = None, evaluation_instructions: List[str] = None) -> Any:
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
                # Get the line of code that called this function
                lines = inspect.getframeinfo(frame).code_context
                if lines:
                    line = lines[0].strip()
                    # Check if it's an assignment
                    if '=' in line:
                        lhs = line.split('=')[0].strip()
                        # Count the number of variables on the left-hand side
                        if ',' in lhs:
                            count = len(lhs.split(','))
                        else:
                            count = 1
                        output_names = [f"output{i}" for i in range(count)]
                    else:
                        output_names = ["output"]
                else:
                    output_names = ["output"]
            except Exception:
                output_names = ["output"]
        else:
            output_names = outputs
        
        # Get type hints from the caller's function if available
        input_types = {}
        output_types = {}
        try:
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_locals = caller_frame.f_locals
            caller_globals = caller_frame.f_globals
            # Get the function where the call happened
            func_name = caller_frame.f_code.co_name
            if func_name in caller_globals and isinstance(caller_globals[func_name], type):
                # It's a class, get the method
                if func_name in caller_locals:
                    func = caller_locals[func_name]
                else:
                    func = caller_globals[func_name]
            else:
                func = caller_globals.get(func_name, None)
            if func and callable(func):
                signature = inspect.signature(func)
                # Get the type hints for the parameters
                for param_name, param in signature.parameters.items():
                    if param.annotation != inspect.Parameter.empty:
                        input_types[param_name] = param.annotation
                # Get the return type
                if signature.return_annotation != inspect.Signature.empty:
                    if signature.return_annotation in (tuple, list):
                        # We don't know the types of the elements, so skip
                        pass
                    else:
                        # For single return, we set the type for the first output
                        if output_names:
                            output_types[output_names[0]] = signature.return_annotation
        except Exception:
            pass
        
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
        
        # Log with evaluation
        output_dict = dict(zip(output_names, output_values))
        self.evaluator.log_with_evaluation(
            module=self.FUNCTION_NAME,
            inputs=input_dict,
            outputs=output_dict,
            description=description,
            evaluation_instructions=evaluation_instructions
        )
        
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
