"""SimpleDSPy: Simplified interface for building DSPy pipelines

Provides:
- predict: Function for simple predictions
- chain_of_thought: Function for multi-step reasoning
- PipelineManager: For building complex pipelines
- ModuleFactory: For creating custom DSPy modules
- configure: Function to set global settings
"""

import sys as _sys

# Import our own modules
from .module_caller import Predict, ChainOfThought
from .pipeline_manager import PipelineManager
from .module_factory import ModuleFactory
from .optimization_manager import OptimizationManager
from .settings import settings as global_settings
from .exceptions import (
    SimpleDSPyError,
    ConfigurationError,
    ValidationError,
    PipelineError,
    ModuleError,
    SecurityError
)

# Create the function instances
predict = Predict()
chain_of_thought = ChainOfThought()

# Check if the module is being imported incorrectly
if not _sys.modules["simpledspy"].__name__.startswith("simpledspy"):
    import warnings as _warnings

    _warnings.warn(
        "It looks like you might be importing 'simpledspy' incorrectly. "
        "Please use 'import simpledspy' instead of 'import dspy'.",
        ImportWarning,
    )


def configure(**kwargs):
    """Set global configuration settings for SimpleDSPy.

    Example:
        configure(lm=dspy.LM(model="deepseek/deepseek-chat"), temperature=0.7, max_tokens=100)
    """
    for key, value in kwargs.items():
        setattr(global_settings, key, value)


# Package version
__version__ = "0.3.1"

__all__ = [
    "predict",
    "chain_of_thought",
    "PipelineManager",
    "ModuleFactory",
    "OptimizationManager",
    "configure",
    "__version__",
    # Exceptions
    "SimpleDSPyError",
    "ConfigurationError",
    "ValidationError",
    "PipelineError",
    "ModuleError",
    "SecurityError",
]
