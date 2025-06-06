import pytest
from simpledspy import predict, chain_of_thought
from simpledspy.module_caller import Predict
import dspy
from unittest.mock import patch, MagicMock

def test_basic_string_output():
    """Test basic string output"""
    # Save original LM and reset after test
    original_lm = dspy.settings.lm
    try:
        # Mock LM configuration
        mock_lm = MagicMock()
        dspy.settings.lm = mock_lm
        
        # Mock module response
        with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
            mock_module = MagicMock()
            mock_module.return_value = MagicMock(output="Mocked Hello")
            mock_create.return_value = mock_module
            
            text = "Hello, world!"
            result = predict(text)
            assert result == "Mocked Hello"
    finally:
        dspy.settings.lm = original_lm

def test_chain_of_thought():
    """Test chain_of_thought function"""
    with patch('simpledspy.module_caller.ChainOfThought._create_module') as mock_create:
        # Create a mock module that properly handles forward calls
        class MockModule(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(output="result")
            
        mock_create.return_value = MockModule()
            
        text = "What is the capital of France?"
        result = chain_of_thought(text, description="Reason step by step")
        assert result == "result"

def test_custom_input_output_names():
    """Test predict function with custom input/output names"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create a mock module
        class MockModule(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(full_name="John Doe", age=30)
            
        mock_create.return_value = MockModule()
            
        first = "John"
        last = "Doe"
        full_name, age = predict(
            first, 
            last,
            inputs=["first_name", "last_name"],
            outputs=["full_name", "age"],
            description="Combine names and guess age"
        )
        assert full_name == "John Doe"
        assert age == 30

def test_predict_multiple_outputs():
    """Test predict function with multiple outputs"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create a mock module that returns multiple outputs
        class MockModule(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(
                    a_reversed='lkj', 
                    b_repeated='abcabc'
                )
            
        mock_create.return_value = MockModule()
            
        a = 'jkl'
        b = 'abc'
        a_reversed, b_repeated = predict(
            b, 
            a,
            inputs=["b", "a"],
            outputs=["a_reversed", "b_repeated"],
            description="Reverse a and repeat b"
        )
        assert a_reversed == 'lkj'
        assert b_repeated == 'abcabc'
        
def test_predict_unpack_error():
    """Test error when unpacking wrong number of outputs"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create mock module that returns single output
        class MockModule(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(output="single value")
        
        mock_create.return_value = MockModule()
        
        # Use without specifying outputs (default is single output)
        # But try to unpack to two variables
        with pytest.raises(ValueError) as exc_info:
            a, b = predict("input1", "input2")
        
        assert "not enough values to unpack" in str(exc_info.value) or "too many values to unpack" in str(exc_info.value)
