import pytest
from simpledspy import predict, chain_of_thought
from simpledspy.module_caller import Predict  # Added import
import inspect
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
        mock_module = dspy.ChainOfThought(lambda text: dspy.Prediction(output="result"))
        mock_create.return_value = mock_module
        
        text = "What is the capital of France?"
        result = chain_of_thought(text, description="Reason step by step")
        assert result == "result"

def test_multiple_outputs():
    """Test handling multiple outputs"""
    with patch('simpledspy.module_caller.BaseCaller._create_module') as mock_create:
        mock_module = dspy.Predict(lambda text: dspy.Prediction(name="John Doe", age="30"))
        mock_create.return_value = mock_module
        
        text = "John Doe, 30 years old"
        result = predict(text, description="Extract name and age")
        assert isinstance(result, tuple)
        name, age = result
        assert isinstance(name, str)
        assert "John" in name
        assert isinstance(age, str)
        assert "30" in age

def test_empty_input():
    """Test handling of empty input strings"""
    text = ""
    result = predict(text)
    assert isinstance(result, str)

def test_whitespace_input():
    """Test handling of whitespace input strings"""
    text = "   "
    result = predict(text)
    assert isinstance(result, str)

def test_special_characters():
    """Test handling of special characters"""
    text = "!@#$%^&*()"
    result = predict(text)
    assert isinstance(result, str)
    assert result == "!@#$%^&*()"

def test_unicode_characters():
    """Test handling of Unicode characters"""
    text = "こんにちは世界"  # Hello world in Japanese
    result = predict(text, description="Return the input text")
    assert isinstance(result, str)
    assert result == "こんにちは世界"

def test_multiple_inputs():
    """Test handling multiple input arguments"""
    text1 = "Hello"
    text2 = "World"
    result = predict(text1, text2)
    assert isinstance(result, str)
    assert "Hello" in result or "World" in result

def test_multiple_outputs_tuple():
    """Test returning multiple outputs as a tuple"""
    with patch('simpledspy.module_caller.BaseCaller._create_module') as mock_create:
        mock_module = dspy.Predict(lambda text: dspy.Prediction(name="John Doe", age="30"))
        mock_create.return_value = mock_module
        
        text = "John Doe, 30 years old"
        result = predict(text)
        assert isinstance(result, tuple)
        assert len(result) == 2
        name, age = result
        assert isinstance(name, str)
        assert isinstance(age, str)
        assert "John" in name
        assert "30" in age

def test_description_parameter():
    """Test using the description parameter"""
    with patch('simpledspy.module_caller.BaseCaller._create_module') as mock_create:
        mock_module = dspy.Predict(lambda text: dspy.Prediction(output="563"))
        mock_create.return_value = mock_module
        
        text = "54 563 125"
        result1 = predict(text, description="Get the biggest number")
        assert "563" in result1
        result2 = predict(text)
        assert "563" in result2

def test_nested_pipe_calls():
    """Test nested pipe calls"""
    with patch('simpledspy.module_caller.BaseCaller._create_module') as mock_create:
        mock_module = dspy.Predict(lambda text: dspy.Prediction(output="72.5"))
        mock_create.return_value = mock_module
        
        text = "The temperature is 72.5 degrees Fahrenheit"
        temp_str = predict(text)
        assert isinstance(temp_str, str)
        assert "72.5" in temp_str

def test_custom_module_creation():
    """Test creating a custom module with pipe"""
    # Create a test module
    class TestModule(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction(output_1="test output")
    
    # Mock the module creation
    original_create_module = Predict._create_module
    try:
        def mock_create_module(self, *args, **kwargs):
            return TestModule()
        Predict._create_module = mock_create_module
    
        result = predict("test input")
        assert result == "test output"
    finally:
        Predict._create_module = original_create_module

def test_large_input():
    """Test handling large input strings"""
    large_text = "word " * 1000
    result = predict(large_text)
    assert isinstance(result, str)
    assert "word" in result
