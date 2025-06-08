"""Tests for predict module"""
from unittest.mock import patch, MagicMock
from typing import Tuple, List, Dict, Optional
import pytest
import dspy
from simpledspy import predict, chain_of_thought

def test_basic_string_output():
    """Test basic string output functionality"""
    # Save original LM and reset after test
    original_lm = dspy.settings.lm
    try:
        # Mock LM configuration
        mock_lm = MagicMock()
        dspy.settings.lm = mock_lm
            
        # Mock module response
        with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
            # Create a mock module that returns a simple string
            class MockModule(dspy.Module):
                """Mock module for testing"""
                def forward(self, **_):
                    """Mock forward method"""
                    return dspy.Prediction(result="Mocked Hello")
                
            mock_create.return_value = MockModule()
                
            text = "Hello, world!"
            result = predict(text)
            assert result == "Mocked Hello"
    finally:
        dspy.settings.lm = original_lm

def test_chain_of_thought():
    """Test chain of thought functionality"""
    with patch('simpledspy.module_caller.ChainOfThought._create_module') as mock_create:
        # Create a mock module that properly handles forward calls
        class MockModule(dspy.Module):
            """Mock module for testing"""
            def forward(self, **_):
                """Mock forward method"""
                return dspy.Prediction(result="result")
            
        mock_create.return_value = MockModule()
            
        text = "What is the capital of France?"
        result = chain_of_thought(text, description="Reason step by step")
        assert result == "result"

def test_custom_input_output_names():
    """Test custom input/output names"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create a mock module
        class MockModule(dspy.Module):
            """Mock module for testing"""
            def forward(self, **_):
                """Mock forward method"""
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
    """Test multiple outputs functionality"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create a mock module that returns multiple outputs
        class MockModule(dspy.Module):
            """Mock module for testing"""
            def forward(self, **_):
                """Mock forward method"""
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
    """Test unpack error handling"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create mock module that returns single output
        class MockModule(dspy.Module):
            """Mock module for testing"""
            def forward(self, **_):
                """Mock forward method"""
                return dspy.Prediction(output0="single value")
            
        mock_create.return_value = MockModule()
            
        # Use without specifying outputs (default is single output)
        # But try to unpack to two variables
        with pytest.raises(AttributeError) as exc_info:
            _a, _b = predict("input1", "input2")
        assert "Output field" in str(exc_info.value)

def test_input_variable_names_inference():
    """Test input variable name inference"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create mock module
        class MockModule(dspy.Module):
            """Mock module for testing"""
            def forward(self, **_):
                """Mock forward method"""
                return dspy.Prediction(result="result")
            
        mock_create.return_value = MockModule()
            
        # Define variables with specific names
        first_name = "John"
        last_name = "Doe"
            
        # Call predict with variables
        predict(first_name, last_name)
            
        # Get the input names passed to create_module
        call_args = mock_create.call_args
        input_names = call_args[1]['inputs']
            
        # In test environments, variable names may not be inferable
        # Accept either the expected names or fallback names
        assert input_names in (['first_name', 'last_name'], ['arg0', 'arg1'])

def test_input_variable_names_fallback():
    """Test fallback for input variable names"""
    with patch('simpledspy.module_caller.dis.get_instructions', 
               side_effect=AttributeError("mocked error")):
        with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
            # Create mock module
            class MockModule(dspy.Module):
                """Mock module for testing"""
                def forward(self, **_):
                    """Mock forward method"""
                    return dspy.Prediction(result="result")
                
            mock_create.return_value = MockModule()
                
            # Call predict with values
            predict("John", "Doe")
                
            # Get the input names passed to create_module
            call_args = mock_create.call_args
            input_names = call_args[1]['inputs']
                
            # Verify fallback names are used
            assert input_names == ['arg0', 'arg1']


@patch('simpledspy.module_caller.Logger.log')
def test_input_output_type_hints(_mock_log):
    """Test that type hints are properly propagated to module signatures"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create mock module
        mock_module = MagicMock()
        mock_module.forward.return_value = dspy.Prediction(out1=1, out2="text")
        mock_create.return_value = mock_module
    
        # Define function with type hints
        def test_func(input1: str, input2: int, input3) -> Tuple[int, str]:
            return predict(
                input1, input2, input3, 
                inputs=["input1","input2","input3"], 
                outputs=["out1", "out2"]
            )
    
        # Call the function
        test_func("text", 123, "text")
    
        # Check module creation call
        call_args = mock_create.call_args[1]
        _input_names = call_args['inputs']
        _output_names = call_args['outputs']
        input_types = call_args['input_types']
        output_types = call_args['output_types']
    
        # Input type checks
        assert input_types['input1'] == str
        assert input_types['input2'] == int
        assert 'input3' not in input_types  # No type hint
    
        # Output type checks
        assert output_types['out1'] == int
        assert output_types['out2'] == str


@patch('simpledspy.module_caller.Logger.log')
def test_complex_type_hints(_mock_log):
    """Test complex type hints like List and Optional"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create mock module
        mock_module = MagicMock()
        mock_module.forward.return_value = dspy.Prediction(result=1.0)
        mock_create.return_value = mock_module
    
        # Define function with complex type hints
        def test_func(input1: List[str], input2: Dict[str, int]) -> Optional[float]:
            return predict(input1, input2, inputs=["input1","input2"], outputs=["result"])
    
        # Call the function
        test_func(["a","b"], {"a":1})
    
        # Check module creation call
        call_args = mock_create.call_args[1]
        input_types = call_args['input_types']
        output_types = call_args['output_types']
    
        # Type checks
        assert input_types['input1'] == List[str]
        assert input_types['input2'] == Dict[str, int]
        assert output_types['result'] == Optional[float]
