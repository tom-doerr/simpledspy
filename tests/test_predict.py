import pytest
from simpledspy import predict, chain_of_thought
from simpledspy.module_caller import Predict
import dspy
from unittest.mock import patch, MagicMock

@patch('dspy.settings.configure')
def test_basic_string_output(mock_configure):
    """Test basic string output"""
    # Mock LM configuration
    mock_lm = MagicMock()
    mock_configure.return_value = None
    dspy.settings.lm = mock_lm
    
    # Mock module response
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        mock_module = MagicMock()
        mock_module.return_value = MagicMock(output="Mocked Hello")
        mock_create.return_value = mock_module
        
        text = "Hello, world!"
        result = predict(text)
        assert result == "Mocked Hello"

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

# ... (rest of the tests remain unchanged) ...
