"""Tests for base_caller.py"""

import os
import sys
from unittest.mock import MagicMock, patch

import dspy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simpledspy.module_caller import BaseCaller


def test_base_caller_singleton():
    """Test BaseCaller follows singleton pattern"""
    caller1 = BaseCaller()
    caller2 = BaseCaller()
    assert caller1 is caller2


def test_base_caller_module_creation():
    """Test module creation with type hints"""
    caller = BaseCaller()
    mock_factory = MagicMock()
    mock_module = MagicMock(spec=dspy.Module)
    mock_factory.create_module.return_value = mock_module
    caller.module_factory = mock_factory

    # Create module with type hints
    module = caller._create_module(  # pylint: disable=protected-access
        inputs=["text"],
        outputs=["result"],
        input_types={"text": str},
        output_types={"result": int},
        description="Test module",
    )

    assert isinstance(module, dspy.Module)
    mock_factory.create_module.assert_called_once()


@patch("inspect.signature")
@patch("simpledspy.logging_utils.Logger")
def test_input_name_inference_in_function_scope(_mock_logger, _mock_signature):
    """Test input name inference in function scope"""
    caller = BaseCaller()
    mock_factory = MagicMock()
    mock_module = MagicMock()
    mock_module.return_value = dspy.Prediction(output="test output")
    mock_factory.create_module.return_value = mock_module
    caller.module_factory = mock_factory

    # Define a function to test variable capture
    def test_function():
        local_var1 = "value1"
        local_var2 = "value2"
        caller(local_var1, local_var2)

    test_function()

    # Verify create_module was called with correct names
    call_kwargs = mock_factory.create_module.call_args[1]
    assert call_kwargs["inputs"] == ["local_var1", "local_var2"]


def test_base_caller_module_execution():
    """Test module execution with inputs/outputs"""
    with patch("simpledspy.module_caller.BaseCaller._create_module") as mock_create:
        caller = BaseCaller()

        # Create mock module
        mock_module = MagicMock()
        mock_module.return_value = MagicMock(output1="value1", output2="value2")
        mock_create.return_value = mock_module

        # Call with inputs/outputs specification
        result1, result2 = caller(
            "input1", "input2", inputs=["in1", "in2"], outputs=["output1", "output2"]
        )

        assert result1 == "value1"
        assert result2 == "value2"
        mock_module.assert_called_once_with(in1="input1", in2="input2")
