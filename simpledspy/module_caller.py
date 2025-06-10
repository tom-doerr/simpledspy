"""Module Caller for DSPy modules

Provides base classes for Predict and ChainOfThought function calls
"""

import inspect
import ast
from typing import List, Dict, Any, Tuple
import dspy
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
from .logger import Logger
from .settings import settings as global_settings

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
        
        try:
            # Get the code context lines
            lines = inspect.getframeinfo(frame).code_context
            if not lines:
                return ["output"]
            line = lines[0].strip()
            
            # Handle multi-line assignment
            if '=' in line:
                lhs = line.split('=')[0].strip()
                # Handle tuple assignment: name1, name2 = ...
                if ',' in lhs:
                    output_names = [name.strip() for name in lhs.split(',')]
                # Handle single assignment: name = ...
                else:
                    output_names = [lhs]
            # Handle no assignment (standalone call)
            else:
                output_names = ["output"]
                
            return output_names
        except (AttributeError, IndexError, TypeError):
            return ["output"]

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
        input_types: Dict[str, type] = {}
        output_types: Dict[str, type] = {}
        if frame is None:
            return input_types, output_types

        try:
            # Get the function object from the frame hierarchy
            func_name = frame.f_code.co_name
            func = None
            
            # Check current frame's locals
            func = frame.f_locals.get(func_name, None)
            if func is None:
                # Check current frame's globals
                func = frame.f_globals.get(func_name, None)
                
            # If not found, check the outer frame
            if func is None and frame.f_back:
                outer_frame = frame.f_back
                func = outer_frame.f_locals.get(func_name, 
                      outer_frame.f_globals.get(func_name, None))
            
            if func and callable(func):
                try:
                    signature = inspect.signature(func, follow_wrapped=True)
                    
                    # Get input parameter types
                    for param_name, param in signature.parameters.items():
                        if (param_name in input_names 
                                and param.annotation != inspect.Parameter.empty):
                            input_types[param_name] = param.annotation
                    
                    # Get return type hints
                    return_ann = signature.return_annotation
                    self._process_return_annotation(return_ann, output_names, output_types)
                except (ValueError, TypeError):
                    # Skip if signature inspection fails
                    pass
        except Exception:  # pylint: disable=broad-except
            # Skip on any exception to prevent test failures
            pass
            
        return input_types, output_types

    def _infer_input_names(self, args) -> List[str]:
        """Infer input variable names using frame inspection"""
        try:
            frame = inspect.currentframe().f_back.f_back
            if frame is None:
                return [f"arg{i}" for i in range(len(args))]
            
            # Get the code context of the call
            context_lines = inspect.getframeinfo(frame).code_context
            if not context_lines:
                return [f"arg{i}" for i in range(len(args))]
                
            call_line = context_lines[0].strip()
            
            # Try the AST method
            try:
                tree = ast.parse(call_line)
                call_node = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        call_node = node
                        break
                if call_node is None:
                    return [f"arg{i}" for i in range(len(args))]
                    
                arg_names = []
                for arg in call_node.args:
                    if isinstance(arg, ast.Name):
                        name = arg.id
                        arg_names.append(name)
                    elif (isinstance(arg, ast.Attribute) and 
                          isinstance(arg.value, ast.Name) and 
                          arg.value.id == 'self'):
                        name = arg.attr
                        arg_names.append(name)
                    else:
                        # For any other type of node, assign argX name
                        arg_names.append(f"arg{len(arg_names)}")
                
                # Sanitize reserved words
                reserved = ['args', 'kwargs', 'self']
                for i, name in enumerate(arg_names):
                    if name in reserved:
                        arg_names[i] = f'arg{i}'
                
                return arg_names
            except (SyntaxError, TypeError, ValueError):
                # If AST method fails, fall back to arg0, arg1, ...
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
    
    def _log_results(
            self, 
            module_name, 
            input_dict, 
            input_names, 
            output_names, 
            output_values, 
            description
    ):
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
        
        # Create logger for this specific module
        logger = Logger(module_name=module_name)
        logger.log({
            'module': module_name,
            'inputs': inputs_data,
            'outputs': outputs_data,
            'description': description
        })
    
    def __call__(self, *args, inputs: List[str] = None, 
            outputs: List[str] = None, description: str = None, 
            lm_params: dict = None, name: str = None, 
            trainset: list = None) -> Any:
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

        # Sanitize input names to avoid reserved words
        reserved = ['args', 'kwargs', 'self']
        for i, var_name in enumerate(input_names):
            if var_name in reserved:
                input_names[i] = f'arg{i}'
        
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
        
        # First priority: use explicitly passed trainset
        if trainset is not None:
            demos = []
            for example in trainset:
                if isinstance(example, dict):
                    demos.append(dspy.Example(**example))
                else:
                    demos.append(example)
            module.demos = demos
        # Second priority: load from training file
        elif name is not None:
            logger = Logger(module_name=name)
            training_examples = logger.load_training_data()
            if training_examples:
                demos = []
                for example in training_examples:
                    if isinstance(example, dict):
                        try:
                            # Convert to dspy.Example
                            demos.append(dspy.Example(**example))
                        except TypeError:
                            demos.append(example)
                    else:
                        demos.append(example)
                module.demos = demos
        
        input_dict = dict(zip(input_names, args))
        prediction_result = self._run_module(module, input_dict, lm_params)
        
        # Process and validate results
        for output_name in output_names:
            if not hasattr(prediction_result, output_name):
                raise AttributeError(f"Output field '{output_name}' not found in prediction result")
        output_values = [getattr(prediction_result, output_name) for output_name in output_names]
        
        # Generate module name if not provided
        if name is None:
            output_part = '_'.join(output_names)
            module_type = self.__class__.__name__.lower()
            input_part = '_'.join(input_names)
            name = f"{output_part}__{module_type}__{input_part}"
        
        # Check if logging is enabled globally or via lm_params
        logging_enabled = global_settings.logging_enabled
        if lm_params and 'logging_enabled' in lm_params:
            logging_enabled = lm_params['logging_enabled']
        
        # Log results if enabled
        if logging_enabled:
            self._log_results(
                name, 
                input_dict, 
                input_names, 
                output_names, 
                output_values, 
                description
            )
        
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
