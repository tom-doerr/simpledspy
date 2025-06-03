import pytest
from simpledspy import predict, chain_of_thought
from simpledspy.module_caller import Predict
import dspy
from unittest.mock import patch

def test_basic_string_output():
    """Test basic string output"""
    text = "Hello, world!"
    result = predict(text)
    assert isinstance(result, str)
    assert "Hello" in result

def test_chain_of_thought():
    """Test chain_of_thought function"""
    with patch('simpledspy.module_caller.BaseCaller._create_module') as mock_create:
        # Create a proper signature for ChainOfThought
        class TestSignature(dspy.Signature):
            text = dspy.InputField()
            output = dspy.OutputField()
            
        mock_module = dspy.ChainOfThought(TestSignature)
        mock_module.forward = lambda **kwargs: dspy.Prediction(output="result")
        mock_create.return_value = mock_module
        
        text = "What is the capital of France?"
        result = chain_of_thought(text, description="Reason step by step")
        assert result == "result"

# ... (rest of the tests remain unchanged) ...
