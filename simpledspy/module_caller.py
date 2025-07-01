"""Module Caller for DSPy modules

Provides base classes for Predict and ChainOfThought function calls
"""

import inspect
import threading
from typing import List, Dict, Any, Tuple
import dspy
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
from .settings import settings as global_settings
from .exceptions import ValidationError
from .retry import RetryConfig, with_retry
from .inference_utils import InferenceUtils
from .training_utils import TrainingUtils
from .logging_utils import LoggingUtils


class BaseCaller:
    """Base class for DSPy module callers"""

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[cls] = instance
                instance.module_factory = ModuleFactory()
                instance.optimization_manager = OptimizationManager()
                # Use default LM from settings if available, otherwise use the global lm
                if global_settings.default_lm:
                    instance.lm = dspy.LM(model=global_settings.default_lm)
                elif global_settings.lm:
                    instance.lm = global_settings.lm
                else:
                    instance.lm = dspy.LM(model="openai/gpt-3.5-turbo")
                dspy.configure(lm=instance.lm, cache=False)
                
                # Initialize retry config
                retry_attempts = getattr(global_settings, 'retry_attempts', 3)
                retry_delay = getattr(global_settings, 'retry_delay', 1.0)
                instance.retry_config = RetryConfig(
                    max_attempts=retry_attempts,
                    initial_delay=retry_delay
                )
            return cls._instances[cls]

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _create_module(
        self,
        inputs: List[str],
        outputs: List[str],
        input_types: Dict[str, type] = None,
        output_types: Dict[str, type] = None,
        description: str = "",
    ) -> dspy.Module:
        return self.module_factory.create_module(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description,
        )

    def _infer_output_names(self, frame: Any) -> List[str]:
        """Infer output names based on assignment context"""
        return InferenceUtils.infer_output_names(frame)


    def _get_call_types_from_signature(
        self, frame: Any, input_names: List[str], output_names: List[str]
    ) -> Tuple[Dict[str, type], Dict[str, type]]:
        """Get input/output types from function signature"""
        return InferenceUtils.get_type_hints_from_signature(frame, input_names, output_names)

    
    def _infer_input_names(self, args) -> List[str]:
        """Infer input variable names using frame inspection"""
        try:
            # Need to go back 3 frames now due to refactoring
            frame = inspect.currentframe().f_back.f_back.f_back
            return InferenceUtils.infer_input_names(args, frame)
        except (AttributeError, ValueError, IndexError, TypeError):
            return [f"arg{i}" for i in range(len(args))]

    def _run_module(self, module, input_dict, lm_params):
        """Run the module with optional LM parameter overrides and retry logic"""
        @with_retry(self.retry_config)
        def _execute():
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
        
        return _execute()

    def _log_results(
        self,
        module_name,
        input_dict,
        input_names,
        output_names,
        output_values,
        description,
    ):
        """Log module inputs and outputs with meaningful names"""
        LoggingUtils.log_results(
            module_name, input_dict, input_names, output_names, output_values, description
        )


    def _prepare_input_names(self, args: tuple, inputs: List[str] = None) -> List[str]:
        """Prepare and sanitize input names"""
        if inputs is None:
            # Capture variables and infer names
            captured_vars = {}
            try:
                # Need to go back 3 frames due to refactoring
                frame = inspect.currentframe().f_back.f_back.f_back
                captured_vars = {**frame.f_locals, **frame.f_globals} if frame else {}
            except (AttributeError, ValueError, IndexError, TypeError):
                captured_vars = {}

            try:
                input_names = self._infer_input_names(args)
                # Map from values to names for preserving original names
                value_to_name = {id(v): name for name, v in captured_vars.items()}
                # Try to map each argument to its original name
                reserved = ["args", "kwargs", "self"]
                for idx, arg in enumerate(args):
                    arg_id = id(arg)
                    if (
                        arg_id in value_to_name
                        and value_to_name[arg_id] not in reserved
                    ):
                        input_names[idx] = value_to_name[arg_id]
            except (AttributeError, ValueError, IndexError, TypeError):
                input_names = [f"arg{i}" for i in range(len(args))]
        else:
            if len(inputs) != len(args):
                raise ValueError(f"Expected {len(args)} input names, got {len(inputs)}")
            input_names = inputs

        # Sanitize input names to avoid reserved words
        reserved = ["args", "kwargs", "self"]
        sanitized_names = []
        for i, var_name in enumerate(input_names):
            if var_name in reserved:
                sanitized_names.append(f"arg{i}")
            else:
                sanitized_names.append(var_name)
        input_names = sanitized_names

        return input_names

    def _prepare_output_names(self, outputs: List[str] = None) -> List[str]:
        """Prepare output names"""
        if outputs is None:
            try:
                # Need to go back 3 frames due to refactoring
                frame = inspect.currentframe().f_back.f_back.f_back
                output_names = self._infer_output_names(frame)
            except (AttributeError, ValueError, IndexError, TypeError):
                output_names = ["output"]
        else:
            output_names = outputs
        return output_names

    def _prepare_module(
        self,
        input_names: List[str],
        output_names: List[str],
        description: str = None,
    ) -> Tuple[dspy.Module, Dict[str, type], Dict[str, type]]:
        """Prepare module with type hints"""
        # Get type hints from caller context
        input_types = {}
        output_types = {}
        try:
            # Need to go back 3 frames due to refactoring
            frame = inspect.currentframe().f_back.f_back.f_back
        except (AttributeError, ValueError, IndexError, TypeError):
            frame = None
        input_types, output_types = self._get_call_types_from_signature(
            frame, input_names, output_names
        )

        # Create the module
        module = self._create_module(
            inputs=input_names,
            outputs=output_names,
            input_types=input_types,
            output_types=output_types,
            description=description,
        )

        return module, input_types, output_types

    def _apply_training_data(
        self, module: dspy.Module, trainset: list = None, name: str = None
    ):
        """Apply training data to module"""
        TrainingUtils.apply_training_data(module, trainset, name)

    def _execute_and_log(
        self,
        module: dspy.Module,
        args: tuple,
        input_names: List[str],
        output_names: List[str],
        name: str,
        description: str,
        lm_params: dict = None,
    ) -> Any:
        """Execute module and optionally log results"""
        # Prepare input dictionary
        input_dict = dict(zip(input_names, args))

        # Run module with optional parameters
        prediction_result = self._run_module(module, input_dict, lm_params)

        # Process and validate results
        for output_name in output_names:
            if not hasattr(prediction_result, output_name):
                raise AttributeError(
                    f"Output field '{output_name}' not found in prediction result"
                )
        output_values = [
            getattr(prediction_result, output_name) for output_name in output_names
        ]

        # Check if logging is enabled
        logging_enabled = global_settings.logging_enabled
        if lm_params and "logging_enabled" in lm_params:
            logging_enabled = lm_params["logging_enabled"]

        # Log results if enabled
        if logging_enabled:
            self._log_results(
                name, input_dict, input_names, output_names, output_values, description
            )

        # Return single value or tuple
        return output_values[0] if len(output_values) == 1 else tuple(output_values)

    def __call__(
        self,
        *args,
        inputs: List[str] = None,
        outputs: List[str] = None,
        description: str = None,
        lm_params: dict = None,
        name: str = None,
        trainset: list = None,
    ) -> Any:
        """Execute the module call with all parameters"""
        # Validate inputs
        if not args:
            raise ValidationError("At least one argument is required")
        
        if lm_params is not None and not isinstance(lm_params, dict):
            raise ValidationError("lm_params must be a dictionary")
        
        if inputs is not None and not isinstance(inputs, list):
            raise ValidationError("inputs must be a list of strings")
        
        if outputs is not None and not isinstance(outputs, list):
            raise ValidationError("outputs must be a list of strings")
        
        if trainset is not None and not isinstance(trainset, list):
            raise ValidationError("trainset must be a list")
        
        # Prepare input and output names
        input_names = self._prepare_input_names(args, inputs)
        output_names = self._prepare_output_names(outputs)

        # Generate module name if not provided
        if name is None:
            output_part = "_".join(output_names)
            module_type = self.__class__.__name__.lower()
            input_part = "_".join(input_names)
            name = f"{output_part}__{module_type}__{input_part}"

        # Create and configure module
        module, _, _ = self._prepare_module(
            input_names, output_names, description
        )

        # Apply training data
        self._apply_training_data(module, trainset, name)

        # Execute and return results
        return self._execute_and_log(
            module, args, input_names, output_names, name, description, lm_params
        )


# pylint: disable=too-few-public-methods
class Predict(BaseCaller):
    """Predict module caller - replaces pipe() function"""


class ChainOfThought(BaseCaller):
    """ChainOfThought module caller"""

    # pylint: disable=too-many-arguments, R0917
    def _create_module(
        self,
        inputs: List[str],
        outputs: List[str],
        input_types: Dict[str, type] = None,
        output_types: Dict[str, type] = None,
        description: str = "",
    ) -> dspy.Module:
        signature_class = self.module_factory.create_signature(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description,
        )
        return dspy.ChainOfThought(signature_class)
