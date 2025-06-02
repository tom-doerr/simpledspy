import pytest
from simpledspy import pipe, PipeFunction
import inspect
import dspy


def test_basic_string_output():
    """Test basic string output"""
    text = "Hello, world!"
    result = pipe(text)
    assert isinstance(result, str)
    assert "Hello" in result

def test_multiple_outputs():
    """Test handling multiple outputs"""
    text = "John Doe, 30 years old"
    
    # Get both name and age in one call
    result = pipe(text, description="Extract name and age")
    assert isinstance(result, tuple)
    name, age = result
    assert isinstance(name, str)
    assert "John" in name
    assert isinstance(age, str)
    assert "30" in age

def test_empty_input():
    """Test handling of empty input strings"""
    text = ""
    result = pipe(text)
    assert isinstance(result, str)

def test_whitespace_input():
    """Test handling of whitespace input strings"""
    text = "   "
    result = pipe(text)
    assert isinstance(result, str)

def test_special_characters():
    """Test handling of special characters"""
    text = "!@#$%^&*()"
    result = pipe(text)
    assert isinstance(result, str)
    assert result == "!@#$%^&*()"

def test_unicode_characters():
    """Test handling of Unicode characters"""
    text = "こんにちは世界"  # Hello world in Japanese
    result = pipe(text, description="Return the input text")
    assert isinstance(result, str)
    assert result == "こんにちは世界"

def test_multiple_inputs():
    """Test handling multiple input arguments"""
    text1 = "Hello"
    text2 = "World"
    
    result = pipe(text1, text2)
    assert isinstance(result, str)
    assert "Hello" in result or "World" in result

def test_multiple_outputs_tuple():
    """Test returning multiple outputs as a tuple"""
    text = "John Doe, 30 years old"
    
    # Get both name and age at once
    result = pipe(text)
    assert isinstance(result, tuple)
    assert len(result) == 2
    name, age = result
    assert isinstance(name, str)
    assert isinstance(age, str)
    assert "John" in name
    assert "30" in age

def test_description_parameter():
    """Test using the description parameter"""
    text = "54 563 125"
    
    # With description
    result1 = pipe(text, description="Get the biggest number")
    assert "563" in result1
    
    # Without description
    result2 = pipe(text)
    assert "563" in result2  # LLM often returns the number as string

def test_nested_pipe_calls():
    """Test nested pipe calls"""
    text = "The temperature is 72.5 degrees Fahrenheit"
    
    # Extract temperature as string
    temp_str = pipe(text)
    assert isinstance(temp_str, str)
    assert "72.5" in temp_str

def test_custom_module_creation():
    """Test creating a custom module with pipe"""
    # Create a test module
    class TestModule(dspy.Module):
        def forward(self, **kwargs):
            # Simple module that returns the input
            return dspy.Prediction(output_1="test output")
    
    # Mock the module creation
    original_create_module = PipeFunction._create_module
    try:
        # Replace with our test module
        def mock_create_module(self, *args, **kwargs):
            return TestModule()
    
        PipeFunction._create_module = mock_create_module
    
        # Test the pipe with our mock module
        result = pipe("test input")
        assert result == "test output"
    finally:
        # Restore original method
        PipeFunction._create_module = original_create_module

def test_large_input():
    """Test handling large input strings"""
    # Create a large input string
    large_text = "word " * 1000
    
    # Extract first word
    result = pipe(large_text)
    assert isinstance(result, str)
    assert "word" in result
