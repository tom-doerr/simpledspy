import pytest
from simpledspy import pipe, PipeFunction
import inspect
import dspy

def test_instance_caching():
    """Test that PipeFunction instances are cached based on call location"""
    # Get the current location
    frame = inspect.currentframe()
    location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    
    # Create two instances at the same location
    instance1 = PipeFunction()
    instance2 = PipeFunction()
    
    # They should be the same object
    assert instance1 is instance2
    
    # Their location should match our calculated location
    assert instance1._location == location

def test_type_conversion_str():
    """Test string type conversion"""
    text = "Hello, world!"
    result: str = pipe(text)
    assert isinstance(result, str)
    assert result == "Hello, world!"

def test_type_conversion_int():
    """Test integer type conversion"""
    text = "42"
    result: int = pipe(text)
    assert isinstance(result, int)
    assert result == 42
    
    # Test with commas
    text = "1,234"
    result: int = pipe(text)
    assert isinstance(result, int)
    assert result == 1234

def test_type_conversion_float():
    """Test float type conversion"""
    text = "3.14"
    result: float = pipe(text)
    assert isinstance(result, float)
    assert result == 3.14
    
    # Test with commas
    text = "1,234.56"
    result: float = pipe(text)
    assert isinstance(result, float)
    assert result == 1234.56

def test_type_conversion_bool():
    """Test boolean type conversion"""
    text = "True"
    result: bool = pipe(text)
    assert isinstance(result, bool)
    assert result is True
    
    text = "False"
    result: bool = pipe(text)
    assert isinstance(result, bool)
    assert result is False

def test_multiple_outputs_with_types():
    """Test handling multiple outputs with type hints"""
    text = "John Doe, 30 years old"
    
    # Get both name and age in one call
    name, age = pipe(text)
    assert isinstance(name, str)
    assert name == "John Doe"
    assert isinstance(age, str)
    assert age == "30"  # LLM returns string by default

def test_error_handling():
    """Test error handling for invalid inputs"""
    # Test with non-convertible string to int
    text = "not a number"
    with pytest.raises(ValueError):
        result: int = pipe(text)
    
    # Test with non-convertible string to float
    with pytest.raises(ValueError):
        result: float = pipe(text)

def test_empty_input():
    """Test handling of empty input strings"""
    text = ""
    result: str = pipe(text)
    assert isinstance(result, str)
    
    # Empty input to int should raise error
    with pytest.raises(ValueError):
        result: int = pipe(text)

def test_whitespace_input():
    """Test handling of whitespace input strings"""
    text = "   "
    result: str = pipe(text)
    assert isinstance(result, str)
    
    # Whitespace input to int should raise error
    with pytest.raises(ValueError):
        result: int = pipe(text)

def test_special_characters():
    """Test handling of special characters"""
    text = "!@#$%^&*()"
    result: str = pipe(text)
    assert isinstance(result, str)
    assert result == "!@#$%^&*()"

def test_unicode_characters():
    """Test handling of Unicode characters"""
    text = "こんにちは世界"  # Hello world in Japanese
    result: str = pipe(text)
    assert isinstance(result, str)
    assert result == "こんにちは世界"

def test_multiple_inputs():
    """Test handling multiple input arguments"""
    text1 = "Hello"
    text2 = "World"
    
    result: str = pipe(text1, text2)
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
    assert name == "John Doe"
    assert age == "30"

def test_description_parameter():
    """Test using the description parameter"""
    text = "54 563 125"
    
    # With description
    result1: int = pipe(text, description="Get the biggest number")
    assert isinstance(result1, int)
    assert result1 == 563
    
    # Without description
    result2: int = pipe(text)
    assert isinstance(result2, int)

def test_nested_pipe_calls():
    """Test nested pipe calls"""
    text = "The temperature is 72.5 degrees Fahrenheit"
    
    # Extract temperature as string
    temp_str: str = pipe(text)
    assert isinstance(temp_str, str)
    assert "72.5" in temp_str
    
    # Convert to float
    temp: float = pipe(temp_str)
    assert isinstance(temp, float)
    assert temp == 72.5

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
    result: str = pipe(large_text)
    assert isinstance(result, str)
    assert "word" in result
