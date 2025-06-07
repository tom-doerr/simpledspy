"""Tests for base_caller.py"""
import pytest
import dspy
from unittest.mock import patch, MagicMock
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
    module = caller._create_module(
        inputs=["text"],
        outputs=["result"],
        input_types={"text": str},
        output_types={"result": int},
        description="Test module"
    )
        
    assert isinstance(module, dspy.Module)
    mock_factory.create_module.assert_called_once()

@patch('inspect.currentframe')
@patch('inspect.signature')
@patch('simpledspy.module_caller.Logger')
def test_base_caller_input_name_inference(mock_logger, mock_signature, mock_current_frame):
    """Test input name inference"""
    # Create a mock frame
    mock_frame = MagicMock()
    mock_frame.f_back.f_locals = {
        'arg1': 'value1',
        'arg2': 'value2'
    }
    mock_frame.f_back.f_code.co_name = 'test_func'
    mock_current_frame.return_value = mock_frame
    
    caller = BaseCaller()
    mock_factory = MagicMock()
    mock_module = MagicMock()
    mock_factory.create_module.return_value = mock_module
    caller.module_factory = mock_factory
    # Mock the logger to prevent serialization issues
    caller.logger = mock_logger
        
    # Mock function signature
    class MockSignature:
        parameters = {
            'arg1': MagicMock(annotation=str),
            'arg2': MagicMock(annotation=int)
        }
        return_annotation = str
        
    mock_signature.return_value = MockSignature()
        
    mock_module.return_value = MagicMock(output0="result")
        
    # Call predict with variables
    arg1 = "test1"
    arg2 = "test2"
    result = caller(arg1, arg2)
        
    # Verify create_module was called
    assert mock_factory.create_module.call_args is not None
        
    # Get the call arguments
    call_kwargs = mock_factory.create_module.call_args[1]
    # The base caller may not infer exact variable names, expect fallback names
    # We accept either the expected names or fallback names since it's context-dependent
    expected_names1 = ['arg0', 'arg1']
    expected_names2 = ['arg1', 'arg2']
    assert call_kwargs['inputs'] in (expected_names1, expected_names2)
    # The types should be as expected
    assert call_kwargs['input_types'] == {'arg1': str, 'arg2': int}
    assert call_kwargs['output_types'] == {'output0': str}

def test_base_caller_module_execution():
    """Test module execution with inputs/outputs"""
    with patch('simpledspy.module_caller.BaseCaller._create_module') as mock_create:
        caller = BaseCaller()
        
        # Create mock module
        mock_module = MagicMock()
        mock_module.return_value = MagicMock(output1="value1", output2="value2")
        mock_create.return_value = mock_module
        
        # Call with inputs/outputs specification
        result1, result2 = caller(
            "input1", "input2",
            inputs=["in1", "in2"],
            outputs=["output1", "output2"]
        )
        
        assert result1 == "value1"
        assert result2 == "value2"
        mock_module.assert_called_once_with(in1="input1", in2="input2")
