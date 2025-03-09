# Import DSPy explicitly to avoid confusion
import dspy as _dspy

# Import our own modules
from .pipe import PipeFunction
from .pipeline_manager import PipelineManager
from .module_factory import ModuleFactory

# Create the pipe function instance
pipe = PipeFunction()

# Check if the module is being imported incorrectly
import sys as _sys
if __name__ != "simpledspy" and "simpledspy" in _sys.modules:
    import warnings as _warnings
    _warnings.warn(
        "It looks like you might be importing 'simpledspy' incorrectly. "
        "Please use 'import simpledspy' instead of 'import dspy'.",
        ImportWarning
    )

__all__ = ['pipe', 'PipelineManager', 'ModuleFactory', 'PipeFunction']
