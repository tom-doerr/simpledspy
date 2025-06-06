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
    with patch('simpledspy.module_caller.ModuleFactory') as mock_factory:
        caller = BaseCaller()
        mock_module = MagicMock()
        mock_factory.return_value.create_module.return_value = mock_module
        
        # Create module with type hints
        module = caller._create_module(
            inputs=["text"],
            outputs=["result"],
            input_types={"text": str},
            output_types={"result": int},
            description="Test module"
        )
        
        assert module is mock_module
        mock_factory.return_value.create_module.assert_called_once()

@patch('simpledspy.module_caller.inspect')
def test_base_caller_input_name_inference(mock_inspect):
    """Test input name inference"""
    mock_inspect.current_frame.return_value.f_back.f_locals = {
        'arg1': 'value1',
        'arg2': 'value2'
    }
    mock_inspect.current_frame.return_value.f_back.f_code.co_name = 'test_func'
    
    with patch('simpledspy.module_caller.BaseCaller._create_module') as mock_create:
        caller = BaseCaller()
        
        # Mock function signature
        class MockSignature:
            parameters = {
                'arg1': MagicMock(annotation=str),
                'arg2': MagicMock(annotation=int)
            }
            return_annotation = str
        
        mock_inspect.signature.return_value = MockSignature()
        
        mock_module = MagicMock()
        mock_module.return_value = MagicMock(output0="result")
        mock_create.return_value = mock_module
        
        # Call predict with variables
        arg1 = "test1"
        arg2 = "test2"
        result = caller(arg1, arg2)
        
        # Check input names used
        call_args = mock_create.call_args[1]
        assert call_args['inputs'] == ['arg1', 'arg2']
        assert call_args['input_types'] == {'arg1': str, 'arg2': int}
        assert call_args['output_types'] == {'output0': str}

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
