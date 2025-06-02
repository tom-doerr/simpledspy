# Import our own modules
from .module_caller import Predict, ChainOfThought
from .pipeline_manager import PipelineManager
from .module_factory import ModuleFactory

# Create the function instances
predict = Predict()
chain_of_thought = ChainOfThought()

# Check if the module is being imported incorrectly
import sys as _sys
if __name__ != "simpledspy" and "simpledspy" in _sys.modules:
    import warnings as _warnings
    _warnings.warn(
        "It looks like you might be importing 'simpledspy' incorrectly. "
        "Please use 'import simpledspy' instead of 'import dspy'.",
        ImportWarning
    )

__all__ = ['predict', 'chain_of_thought', 'PipelineManager', 'ModuleFactory']
