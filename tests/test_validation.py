"""Tests for input validation in module_caller and pipeline_manager"""

import pytest
from unittest.mock import Mock, patch
from simpledspy.module_caller import Predict, ChainOfThought
from simpledspy.pipeline_manager import PipelineManager
from simpledspy.exceptions import ValidationError, ConfigurationError


class TestModuleCallerValidation:
    """Test validation in module_caller"""
    
    def test_predict_no_args(self):
        """Test that Predict requires at least one argument"""
        predict = Predict()
        
        with pytest.raises(ValidationError) as exc_info:
            predict()
        
        assert "At least one argument is required" in str(exc_info.value)
    
    def test_predict_invalid_lm_params(self):
        """Test that lm_params must be a dictionary"""
        predict = Predict()
        
        with pytest.raises(ValidationError) as exc_info:
            predict("test", lm_params="not a dict")
        
        assert "lm_params must be a dictionary" in str(exc_info.value)
    
    def test_predict_invalid_inputs_type(self):
        """Test that inputs must be a list"""
        predict = Predict()
        
        with pytest.raises(ValidationError) as exc_info:
            predict("test", inputs="not a list")
        
        assert "inputs must be a list of strings" in str(exc_info.value)
    
    def test_predict_invalid_outputs_type(self):
        """Test that outputs must be a list"""
        predict = Predict()
        
        with pytest.raises(ValidationError) as exc_info:
            predict("test", outputs="not a list")
        
        assert "outputs must be a list of strings" in str(exc_info.value)
    
    def test_predict_invalid_trainset_type(self):
        """Test that trainset must be a list"""
        predict = Predict()
        
        with pytest.raises(ValidationError) as exc_info:
            predict("test", trainset="not a list")
        
        assert "trainset must be a list" in str(exc_info.value)
    
    def test_chain_of_thought_validation(self):
        """Test validation in ChainOfThought"""
        cot = ChainOfThought()
        
        # Should have same validation as Predict
        with pytest.raises(ValidationError) as exc_info:
            cot()
        
        assert "At least one argument is required" in str(exc_info.value)
    
    @patch('simpledspy.module_caller.dspy')
    def test_valid_inputs(self, mock_dspy):
        """Test that valid inputs pass validation"""
        predict = Predict()
        
        # Mock the module creation and execution
        mock_module = Mock()
        mock_module.return_value = Mock(output="test result")
        predict.module_factory.create_module = Mock(return_value=mock_module)
        
        # These should not raise any validation errors
        result = predict("test input", 
                        inputs=["arg"],
                        outputs=["output"],
                        lm_params={"temperature": 0.7},
                        trainset=[{"input": "example", "output": "result"}])
        
        assert result == "test result"


class TestPipelineManagerValidation:
    """Test validation in PipelineManager"""
    
    def test_register_step_invalid_inputs_type(self):
        """Test that inputs must be a list"""
        pm = PipelineManager()
        mock_module = Mock()
        
        with pytest.raises(ValidationError) as exc_info:
            pm.register_step("not a list", ["output"], mock_module)
        
        assert "inputs must be a list of strings" in str(exc_info.value)
    
    def test_register_step_invalid_inputs_content(self):
        """Test that inputs must contain only strings"""
        pm = PipelineManager()
        mock_module = Mock()
        
        with pytest.raises(ValidationError) as exc_info:
            pm.register_step([1, 2, 3], ["output"], mock_module)
        
        assert "inputs must be a list of strings" in str(exc_info.value)
    
    def test_register_step_invalid_outputs_type(self):
        """Test that outputs must be a list"""
        pm = PipelineManager()
        mock_module = Mock()
        
        with pytest.raises(ValidationError) as exc_info:
            pm.register_step(["input"], "not a list", mock_module)
        
        assert "outputs must be a list of strings" in str(exc_info.value)
    
    def test_register_step_invalid_outputs_content(self):
        """Test that outputs must contain only strings"""
        pm = PipelineManager()
        mock_module = Mock()
        
        with pytest.raises(ValidationError) as exc_info:
            pm.register_step(["input"], [1, 2, 3], mock_module)
        
        assert "outputs must be a list of strings" in str(exc_info.value)
    
    def test_register_step_invalid_module(self):
        """Test that module must be callable"""
        pm = PipelineManager()
        
        with pytest.raises(ValidationError) as exc_info:
            pm.register_step(["input"], ["output"], "not callable")
        
        assert "module must be callable" in str(exc_info.value)
    
    def test_assemble_empty_pipeline(self):
        """Test assembling an empty pipeline"""
        from simpledspy.exceptions import PipelineError
        pm = PipelineManager()
        pm.reset()  # Ensure it's empty
        
        with pytest.raises(PipelineError) as exc_info:
            pm.assemble_pipeline()
        
        assert "No steps in pipeline" in str(exc_info.value)
    
    def test_valid_pipeline(self):
        """Test that valid pipeline passes validation"""
        pm = PipelineManager()
        pm.reset()  # Ensure clean state
        
        # Create mock modules that return predictions
        module1 = Mock()
        module1.return_value = Mock(output1=10)
        
        module2 = Mock()
        module2.return_value = Mock(output2=11)
        
        # Register valid steps - fixed input names
        pm.register_step(["input1"], ["output1"], module1)
        pm.register_step(["output1"], ["output2"], module2)
        
        # Assemble and run pipeline
        pipeline = pm.assemble_pipeline()
        result = pipeline(input1=5)
        
        # Check that modules were called correctly
        module1.assert_called_once_with(input1=5)
        module2.assert_called_once_with(output1=10)
        
        # Check final output
        assert result.output2 == 11


class TestSecurityValidation:
    """Test security-related validation"""
    
    def test_ast_parsing_code_too_long(self):
        """Test that overly long code is rejected"""
        predict = Predict()
        
        # Create a very long variable name
        long_var = "x" * 2000
        
        # Mock the frame to return long code
        with patch('inspect.currentframe') as mock_frame:
            mock_frame.return_value.f_back.f_back.f_back.f_locals = {long_var: "value"}
            mock_frame.return_value.f_back.f_back.f_back.code_context = [f"{long_var} = predict('test')"]
            
            # Should handle gracefully without security error
            with patch('simpledspy.inference_utils.InferenceUtils.safe_parse_ast', side_effect=Exception("Code too long")):
                # Should fall back to default names
                names = predict._infer_input_names(("test",))
                assert names == ["arg0"]
    
    def test_ast_parsing_max_depth(self):
        """Test that deeply nested AST is rejected"""
        from simpledspy.exceptions import SecurityError
        predict = Predict()
        
        # Create deeply nested code - this should trigger the SecurityError
        nested_code = "f(" * 20 + "'test'" + ")" * 20
        
        # Mock the frame inspection to return our nested code
        mock_frame = Mock()
        mock_frame.f_back.f_back.f_back = Mock()
        mock_frame.f_back.f_back.f_back.f_locals = {}
        mock_frame.f_back.f_back.f_back.f_globals = {}
        
        with patch('inspect.currentframe', return_value=mock_frame):
            with patch('inspect.getframeinfo') as mock_info:
                mock_info.return_value.code_context = [nested_code]
                
                # The deeply nested AST should raise SecurityError
                with pytest.raises(SecurityError) as exc_info:
                    predict._infer_input_names(("test",))
                
                assert "AST too deep" in str(exc_info.value)