"""Tests for custom exceptions"""

import pytest
from simpledspy.exceptions import (
    SimpleDSPyError,
    ConfigurationError,
    ValidationError,
    PipelineError,
    ModuleError,
    SecurityError
)


def test_exception_hierarchy():
    """Test that all exceptions inherit from SimpleDSPyError"""
    assert issubclass(ConfigurationError, SimpleDSPyError)
    assert issubclass(ValidationError, SimpleDSPyError)
    assert issubclass(PipelineError, SimpleDSPyError)
    assert issubclass(ModuleError, SimpleDSPyError)
    assert issubclass(SecurityError, SimpleDSPyError)


def test_exception_messages():
    """Test exception messages"""
    msg = "Test error message"
    
    with pytest.raises(SimpleDSPyError) as exc_info:
        raise SimpleDSPyError(msg)
    assert str(exc_info.value) == msg
    
    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError(msg)
    assert str(exc_info.value) == msg
    
    with pytest.raises(ValidationError) as exc_info:
        raise ValidationError(msg)
    assert str(exc_info.value) == msg