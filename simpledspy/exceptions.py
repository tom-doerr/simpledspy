"""Custom exceptions for SimpleDSPy"""


class SimpleDSPyError(Exception):
    """Base exception for all SimpleDSPy errors"""


class ConfigurationError(SimpleDSPyError):
    """Raised when there's an error in configuration"""


class ValidationError(SimpleDSPyError):
    """Raised when input validation fails"""


class PipelineError(SimpleDSPyError):
    """Raised when there's an error in pipeline execution"""


class ModuleError(SimpleDSPyError):
    """Raised when there's an error in module execution"""


class SecurityError(SimpleDSPyError):
    """Raised when a security violation is detected"""
