"""Comprehensive tests for module_caller.py"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import inspect
import dspy
from simpledspy.module_caller import BaseCaller, Predict, ChainOfThought
from simpledspy.exceptions import ValidationError, SecurityError
from simpledspy.settings import settings


class TestBaseCaller:
    """Test BaseCaller class functionality"""
    
    def test_singleton_pattern(self):
        """Test that BaseCaller implements singleton pattern correctly"""
        # Create two instances
        caller1 = Predict()
        caller2 = Predict()
        
        # They should be the same instance
        assert caller1 is caller2
        
        # Different subclasses should have different instances
        cot = ChainOfThought()
        assert cot is not caller1
    
    def test_thread_safe_singleton(self):
        """Test thread safety of singleton pattern"""
        instances = []
        
        def create_instance():
            instances.append(Predict())
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=create_instance)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)
    
    @patch('simpledspy.module_caller.dspy')
    def test_lm_configuration(self, mock_dspy):
        """Test LM configuration from settings"""
        # Test with default_lm in settings
        with patch.object(settings, 'default_lm', 'openai/gpt-4'):
            with patch.object(settings, 'lm', None):
                # Force new instance
                Predict._instances.clear()
                caller = Predict()
                
                # Should use default_lm
                mock_dspy.LM.assert_called_with(model='openai/gpt-4')
        
        # Test with lm object in settings
        mock_lm = Mock()
        with patch.object(settings, 'default_lm', None):
            with patch.object(settings, 'lm', mock_lm):
                # Force new instance
                Predict._instances.clear()
                caller = Predict()
                
                # Should use the lm object
                assert caller.lm == mock_lm
        
        # Test fallback to default
        with patch.object(settings, 'default_lm', None):
            with patch.object(settings, 'lm', None):
                # Force new instance
                Predict._instances.clear()
                caller = Predict()
                
                # Should fallback to gpt-3.5-turbo
                mock_dspy.LM.assert_called_with(model='openai/gpt-3.5-turbo')
    
    def test_retry_config_initialization(self):
        """Test retry configuration initialization"""
        with patch.object(settings, 'retry_attempts', 5):
            with patch.object(settings, 'retry_delay', 2.0):
                # Force new instance
                Predict._instances.clear()
                caller = Predict()
                
                assert caller.retry_config.max_attempts == 5
                assert caller.retry_config.initial_delay == 2.0
    
    def test_prepare_input_names_with_explicit_names(self):
        """Test _prepare_input_names with explicit input names"""
        caller = Predict()
        
        args = ("value1", "value2")
        inputs = ["input1", "input2"]
        
        result = caller._prepare_input_names(args, inputs)
        assert result == ["input1", "input2"]
    
    def test_prepare_input_names_wrong_count(self):
        """Test _prepare_input_names with wrong number of names"""
        caller = Predict()
        
        args = ("value1", "value2")
        inputs = ["input1"]  # Only one name for two args
        
        with pytest.raises(ValueError) as exc_info:
            caller._prepare_input_names(args, inputs)
        
        assert "Expected 2 input names, got 1" in str(exc_info.value)
    
    def test_prepare_input_names_sanitization(self):
        """Test that reserved words are sanitized"""
        caller = Predict()
        
        args = ("value1", "value2", "value3")
        inputs = ["args", "kwargs", "self"]  # Reserved words
        
        result = caller._prepare_input_names(args, inputs)
        assert result == ["arg0", "arg1", "arg2"]
    
    @patch('inspect.currentframe')
    def test_prepare_input_names_inference(self, mock_frame):
        """Test input name inference from frame"""
        caller = Predict()
        
        # Mock the frame chain
        mock_frame.return_value = Mock(
            f_back=Mock(f_back=Mock(f_back=Mock(
                f_locals={'test_var': 'value1'},
                f_globals={}
            )))
        )
        
        with patch.object(caller, '_infer_input_names', return_value=['test_var']):
            result = caller._prepare_input_names(('value1',))
            assert result == ['test_var']
    
    def test_prepare_output_names_explicit(self):
        """Test _prepare_output_names with explicit names"""
        caller = Predict()
        
        result = caller._prepare_output_names(["output1", "output2"])
        assert result == ["output1", "output2"]
    
    @patch('inspect.currentframe')
    def test_prepare_output_names_inference(self, mock_frame):
        """Test output name inference"""
        caller = Predict()
        
        with patch.object(caller, '_infer_output_names', return_value=['result']):
            result = caller._prepare_output_names()
            assert result == ['result']
    
    def test_execute_and_log(self):
        """Test _execute_and_log method"""
        caller = Predict()
        
        # Mock module
        mock_module = Mock()
        mock_result = Mock(output="test result")
        mock_module.return_value = mock_result
        
        # Mock logging - enable it in settings
        with patch.object(settings, 'logging_enabled', True):
            with patch('simpledspy.module_caller.LoggingUtils.log_results') as mock_log:
                result = caller._execute_and_log(
                    module=mock_module,
                    args=("input",),
                    input_names=["arg0"],
                    output_names=["output"],
                    name="test_module",
                    description="Test description",
                    lm_params=None
                )
                
                assert result == "test result"
                
                # Check logging was called
                mock_log.assert_called_once()
    
    def test_execute_and_log_multiple_outputs(self):
        """Test _execute_and_log with multiple outputs"""
        caller = Predict()
        
        # Mock module with multiple outputs
        mock_module = Mock()
        mock_result = Mock(out1="result1", out2="result2")
        mock_module.return_value = mock_result
        
        with patch('simpledspy.module_caller.LoggingUtils.log_results'):
            result = caller._execute_and_log(
                module=mock_module,
                args=("input",),
                input_names=["arg0"],
                output_names=["out1", "out2"],
                name="test_module",
                description=None,
                lm_params=None
            )
            
            assert result == ("result1", "result2")
    
    def test_execute_and_log_missing_output(self):
        """Test _execute_and_log with missing output field"""
        caller = Predict()
        
        # Mock module that doesn't return expected field
        mock_module = Mock()
        mock_result = Mock(spec=['other_field'])
        mock_module.return_value = mock_result
        
        with pytest.raises(AttributeError) as exc_info:
            caller._execute_and_log(
                module=mock_module,
                args=("input",),
                input_names=["arg0"],
                output_names=["missing_field"],
                name="test_module",
                description=None,
                lm_params=None
            )
        
        assert "Output field 'missing_field' not found" in str(exc_info.value)
    
    def test_execute_and_log_with_lm_params(self):
        """Test _execute_and_log with LM parameter overrides"""
        caller = Predict()
        
        # Mock module and LM
        mock_module = Mock()
        mock_result = Mock(output="result")
        mock_module.return_value = mock_result
        
        caller.lm = Mock(temperature=0.7, max_tokens=100)
        
        lm_params = {"temperature": 0.9, "max_tokens": 200}
        
        with patch('simpledspy.module_caller.LoggingUtils.log_results'):
            result = caller._execute_and_log(
                module=mock_module,
                args=("input",),
                input_names=["arg0"],
                output_names=["output"],
                name="test_module",
                description=None,
                lm_params=lm_params
            )
            
            # Check that parameters were temporarily changed
            assert result == "result"
    
    def test_execute_and_log_logging_disabled(self):
        """Test that logging respects settings"""
        caller = Predict()
        
        mock_module = Mock()
        mock_result = Mock(output="result")
        mock_module.return_value = mock_result
        
        with patch.object(settings, 'logging_enabled', False):
            with patch('simpledspy.module_caller.LoggingUtils.log_results') as mock_log:
                caller._execute_and_log(
                    module=mock_module,
                    args=("input",),
                    input_names=["arg0"],
                    output_names=["output"],
                    name="test_module",
                    description=None,
                    lm_params=None
                )
                
                # Logging should not be called
                mock_log.assert_not_called()
    
    def test_call_validation_errors(self):
        """Test validation in __call__ method"""
        caller = Predict()
        
        # No arguments
        with pytest.raises(ValidationError) as exc_info:
            caller()
        assert "At least one argument is required" in str(exc_info.value)
        
        # Invalid lm_params
        with pytest.raises(ValidationError) as exc_info:
            caller("test", lm_params="not a dict")
        assert "lm_params must be a dictionary" in str(exc_info.value)
        
        # Invalid inputs
        with pytest.raises(ValidationError) as exc_info:
            caller("test", inputs="not a list")
        assert "inputs must be a list of strings" in str(exc_info.value)
        
        # Invalid outputs
        with pytest.raises(ValidationError) as exc_info:
            caller("test", outputs="not a list")
        assert "outputs must be a list of strings" in str(exc_info.value)
        
        # Invalid trainset
        with pytest.raises(ValidationError) as exc_info:
            caller("test", trainset="not a list")
        assert "trainset must be a list" in str(exc_info.value)
    
    @patch('simpledspy.module_caller.dspy')
    def test_call_complete_flow(self, mock_dspy):
        """Test complete flow of __call__ method"""
        caller = Predict()
        
        # Mock module creation and execution
        mock_module = Mock()
        mock_result = Mock(output="test result")
        mock_module.return_value = mock_result
        
        with patch.object(caller.module_factory, 'create_module', return_value=mock_module):
            with patch('simpledspy.module_caller.TrainingUtils.apply_training_data'):
                with patch('simpledspy.module_caller.LoggingUtils.log_results'):
                    result = caller(
                        "test input",
                        inputs=["text"],
                        outputs=["output"],
                        description="Test",
                        name="test_module",
                        trainset=[{"text": "example", "output": "result"}]
                    )
                    
                    assert result == "test result"
    
    def test_call_auto_name_generation(self):
        """Test automatic name generation"""
        caller = Predict()
        
        mock_module = Mock()
        mock_result = Mock(result="value")
        mock_module.return_value = mock_result
        
        with patch.object(caller.module_factory, 'create_module', return_value=mock_module):
            with patch('simpledspy.module_caller.TrainingUtils.apply_training_data') as mock_train:
                with patch('simpledspy.module_caller.LoggingUtils.log_results'):
                    caller("input", inputs=["text"], outputs=["result"])
                    
                    # Check the generated name
                    args, kwargs = mock_train.call_args
                    # apply_training_data is called with positional args
                    assert len(args) >= 3
                    assert "result__predict__text" in args[2]
    
    def test_run_module_with_retry(self):
        """Test _run_module with retry logic"""
        caller = Predict()
        
        # Mock module that fails twice then succeeds
        mock_module = Mock()
        mock_module.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            Mock(output="success")
        ]
        
        # Set retry config
        caller.retry_config.max_attempts = 3
        
        result = caller._run_module(mock_module, {"input": "test"}, None)
        assert result.output == "success"
        assert mock_module.call_count == 3
    
    def test_run_module_retry_exhausted(self):
        """Test _run_module when retries are exhausted"""
        caller = Predict()
        
        # Mock module that always fails
        mock_module = Mock()
        mock_module.side_effect = Exception("Always fails")
        
        # Set retry config
        caller.retry_config.max_attempts = 2
        
        with pytest.raises(Exception) as exc_info:
            caller._run_module(mock_module, {"input": "test"}, None)
        
        assert "Always fails" in str(exc_info.value)
        assert mock_module.call_count == 2


class TestPredict:
    """Test Predict class specific functionality"""
    
    def test_predict_inherits_base_caller(self):
        """Test that Predict properly inherits from BaseCaller"""
        predict = Predict()
        assert isinstance(predict, BaseCaller)
    
    def test_predict_uses_base_create_module(self):
        """Test that Predict uses base implementation"""
        predict = Predict()
        
        with patch.object(predict.module_factory, 'create_module') as mock_create:
            predict._create_module(
                inputs=["input"],
                outputs=["output"],
                description="Test"
            )
            
            mock_create.assert_called_once_with(
                inputs=["input"],
                outputs=["output"],
                input_types=None,
                output_types=None,
                description="Test"
            )


class TestChainOfThought:
    """Test ChainOfThought class specific functionality"""
    
    def test_chain_of_thought_inherits_base_caller(self):
        """Test that ChainOfThought properly inherits from BaseCaller"""
        cot = ChainOfThought()
        assert isinstance(cot, BaseCaller)
    
    @patch('simpledspy.module_caller.dspy.ChainOfThought')
    def test_chain_of_thought_create_module(self, mock_cot_class):
        """Test ChainOfThought creates proper DSPy module"""
        cot = ChainOfThought()
        
        mock_signature = Mock()
        with patch.object(cot.module_factory, 'create_signature', return_value=mock_signature):
            result = cot._create_module(
                inputs=["question"],
                outputs=["answer"],
                input_types={"question": str},
                output_types={"answer": str},
                description="Q&A"
            )
            
            # Should create signature
            cot.module_factory.create_signature.assert_called_once_with(
                inputs=["question"],
                outputs=["answer"],
                input_types={"question": str},
                output_types={"answer": str},
                description="Q&A"
            )
            
            # Should create ChainOfThought with signature
            mock_cot_class.assert_called_once_with(mock_signature)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_frame_inspection_failure(self):
        """Test graceful handling of frame inspection failures"""
        caller = Predict()
        
        with patch('inspect.currentframe', return_value=None):
            # Should fall back to default names
            names = caller._prepare_input_names(("arg1", "arg2"))
            assert names == ["arg0", "arg1"]
    
    def test_inference_with_no_frame(self):
        """Test inference methods with no frame"""
        caller = Predict()
        
        # Test with None frame
        names = caller._infer_output_names(None)
        assert names == ["output"]
    
    def test_type_hints_extraction_failure(self):
        """Test type hints extraction when it fails"""
        caller = Predict()
        
        # Create a mock frame with no function
        mock_frame = Mock()
        mock_frame.f_code.co_name = "nonexistent"
        mock_frame.f_locals = {}
        mock_frame.f_globals = {}
        mock_frame.f_back = None
        
        input_types, output_types = caller._get_call_types_from_signature(
            mock_frame, ["input"], ["output"]
        )
        
        # Should return empty dicts
        assert input_types == {}
        assert output_types == {}
    
    def test_module_execution_with_exception(self):
        """Test module execution when exception is raised"""
        caller = Predict()
        
        # Mock module that raises exception
        mock_module = Mock()
        mock_module.side_effect = RuntimeError("LLM Error")
        
        # Even with retries, should eventually raise
        caller.retry_config.max_attempts = 1
        
        # The retry wrapper converts to ModuleError
        from simpledspy.exceptions import ModuleError
        with pytest.raises(ModuleError) as exc_info:
            caller._run_module(mock_module, {"input": "test"}, None)
        
        assert "Failed after 1 attempts: LLM Error" in str(exc_info.value)