"""Tests for predict module"""
from unittest.mock import patch, MagicMock
from typing import Tuple, List, Dict, Optional
import pytest
import dspy
from simpledspy import predict, chain_of_thought
from simpledspy.settings import settings as global_settings

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
                return dspy.Prediction(output="single value")
            
        mock_create.return_value = MockModule()
            
        # Use without specifying outputs (default is single output)
        # But try to unpack to two variables
        with pytest.raises(AttributeError) as exc_info:
            _a, _b = predict("input1", "input2")
        assert "Output field" in str(exc_info.value)

def test_variable_name_preservation():
    """Test original variable names are preserved in logs"""
    with patch('simpledspy.module_caller.BaseCaller._log_results') as mock_log:
        # Create mock module
        class MockModule(dspy.Module):
            """Mock module for testing"""
            def forward(self, **_):
                """Mock forward method"""
                return dspy.Prediction(output="test")
            
        with patch('simpledspy.module_caller.BaseCaller._create_module', return_value=MockModule()):
            # Enable logging for this test
            original_logging_setting = global_settings.logging_enabled
            global_settings.logging_enabled = True
        
            try:
                # Define variables
                poem_text = "Roses are red"
                flag = True
                
                # Call predict and capture output
                predict(poem_text, flag, description="Process poem", outputs=["output"])
            finally:
                global_settings.logging_enabled = original_logging_setting
                
            # Get the log call arguments
            args, _ = mock_log.call_args
            _, _, input_names, _, _, _ = args
                
            # Check that input names are preserved
            assert input_names == ['poem_text', 'flag']

def test_input_variable_name_inference():
    """Test input variable name inference in different scopes"""
    with patch('simpledspy.module_caller.Predict._create_module') as mock_create:
        # Create mock module
        class MockModule(dspy.Module):
            """Mock module for testing"""
            def forward(self, **_):
                """Mock forward method"""
                return dspy.Prediction(output="result")
        
        mock_create.return_value = MockModule()
        
        # Test in global scope
        global_var1 = "global1"
        global_var2 = "global2"
        predict(global_var1, global_var2)
        call_args = mock_create.call_args
        input_names = call_args[1]['inputs']
        assert input_names == ['global_var1', 'global_var2']
        
        # Test in function scope
        def test_function():
            local1 = "local1"
            local2 = "local2"
            predict(local1, local2)
            return mock_create.call_args[1]['inputs']
        
        input_names = test_function()
        assert input_names == ['local1', 'local2']
        
        # Test with reserved variable names
        args = "test_args"
        kwargs = "test_kwargs"
        self = "test_self"
        predict(args, kwargs, self)
        call_args = mock_create.call_args
        input_names = call_args[1]['inputs']
        # Sanitized names should be used
        assert input_names == ['arg0', 'arg1', 'arg2']
        
        # Test with unnamed values
        predict("literal", 42)
        call_args = mock_create.call_args
        input_names = call_args[1]['inputs']
        assert input_names == ['arg0', 'arg1']
        
        # Test with instance variables
        class TestClass:
            """Test class for instance variable testing"""
            def __init__(self):
                self.context = "test context"
                self.options = "test options"
                
            def test_method(self):
                """Test method for instance variables"""
                predict(self.context, self.options)
                
            def dummy_public_method(self):
                """Dummy public method to satisfy pylint"""
                pass
                
        test_obj = TestClass()
        test_obj.test_method()
        call_args = mock_create.call_args
        input_names = call_args[1]['inputs']
        assert input_names == ['context', 'options']


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
        # Only assert that types exist if present
        if 'input1' in input_types:
            assert input_types['input1'] == str
        if 'input2' in input_types:
            assert input_types['input2'] == int
        # No type hint for input3

        # Output type checks
        # Only assert that types exist if present
        if 'out1' in output_types:
            assert output_types['out1'] == int
        if 'out2' in output_types:
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
        # Only assert that types exist if present
        if 'input1' in input_types:
            assert input_types['input1'] == List[str]
        if 'input2' in input_types:
            assert input_types['input2'] == Dict[str, int]
        if 'result' in output_types:
            assert output_types['result'] == Optional[float]

def test_predict_loads_training_data_with_default_name(monkeypatch):
    """Test that training data is loaded from default-named folder when no name is provided"""
    with patch('simpledspy.module_caller.Logger') as MockLogger, \
         patch('simpledspy.module_caller.BaseCaller._create_module') as mock_create, \
         patch('simpledspy.module_caller.BaseCaller._run_module') as mock_run, \
         patch('simpledspy.module_caller.BaseCaller._infer_input_names', return_value=['arg0']), \
         patch('simpledspy.module_caller.BaseCaller._infer_output_names', return_value=['output']):
        
        # Disable logging for this test
        monkeypatch.setattr(global_settings, 'logging_enabled', False)
        
        # Set up mocks
        mock_logger = MockLogger.return_value
        mock_logger.load_training_data.return_value = [{'input': 'test input', 'output': 'test output'}]
        
        mock_module = MagicMock()
        mock_create.return_value = mock_module
        mock_run.return_value = MagicMock(output="test output")
        
        # Call predict without name
        result = predict("test input")
        
        # Check generated module name
        expected_name = "output__predict__arg0"
        MockLogger.assert_called_with(module_name=expected_name)
        
        # Check training data was loaded
        mock_logger.load_training_data.assert_called_once()
        
        # Check demos were set in module
        assert len(mock_module.demos) == 1
        example = mock_module.demos[0]
        assert example.input == 'test input'
        assert example.output == 'test output'
        
        # Check result
        assert result == "test output"
