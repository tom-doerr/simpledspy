"""Tests for pipeline_manager.py"""
"""Tests for pipeline_manager.py"""
import pytest
import dspy
from unittest.mock import MagicMock, patch
from simpledspy.pipeline_manager import PipelineManager

class MockModule(dspy.Module):
    """Mock module for pipeline testing"""
    def __init__(self, output_value=None, **outputs):
        super().__init__()
        if output_value:
            outputs['output'] = output_value
        self.outputs = outputs
    
    def forward(self, **kwargs):  # pylint: disable=unused-argument
        return dspy.Prediction(**self.outputs)

def test_singleton_pattern():
    """Test that PipelineManager follows the singleton pattern"""
    manager1 = PipelineManager()
    manager2 = PipelineManager()
    assert manager1 is manager2
    manager1.reset()
    # pylint: disable=protected-access
    assert not manager2._steps  # Check empty list using boolean context

def test_empty_pipeline():
    """Test assembling an empty pipeline"""
    manager = PipelineManager()
    manager.reset()
    with pytest.raises(ValueError):
        _ = manager.assemble_pipeline()  # unused variable

def test_pipeline_reset():
    """Test resetting the pipeline"""
    manager = PipelineManager()
    manager.reset()
    
    module = MockModule("test output")
    manager.register_step(
        inputs=["input1"],
        outputs=["output1"],
        module=module
    )
    
    assert len(manager._steps) == 1
    manager.reset()
    assert len(manager._steps) == 0
    with pytest.raises(ValueError):
        _ = manager.assemble_pipeline()  # unused variable

def test_pipeline_assembly():
    """Test assembling a pipeline with multiple steps"""
    manager = PipelineManager()
    manager.reset()
    
    module1 = MockModule(output1="step1 output")
    module2 = MockModule(output2="step2 output")
    
    manager.register_step(
        inputs=["input1"],
        outputs=["output1"],
        module=module1
    )
    manager.register_step(
        inputs=["output1"],
        outputs=["output2"],
        module=module2
    )
    
    pipeline = manager.assemble_pipeline()
    assert pipeline is not None
    
    # Test pipeline execution
    result = pipeline(input1="test input")
    assert result.output1 == "step1 output"
    assert result.output2 == "step2 output"

def test_missing_input():
    """Test missing input in pipeline execution"""
    manager = PipelineManager()
    manager.reset()
    
    module = MockModule("test output")
    manager.register_step(
        inputs=["input1"],
        outputs=["output1"],
        module=module
    )
    
    pipeline = manager.assemble_pipeline()
    
    with pytest.raises(ValueError):
        _ = pipeline()  # Missing input (unused variable)

def test_missing_output():
    """Test missing output in pipeline execution"""
    manager = PipelineManager()
    manager.reset()
    
    class BadModule(dspy.Module):
        """Bad module that returns no output"""
        def forward(self, **kwargs):  # pylint: disable=unused-argument
            return {}  # No output
    
    module = BadModule()
    manager.register_step(
        inputs=["input1"],
        outputs=["output1"],
        module=module
    )
    
    pipeline = manager.assemble_pipeline()
    
    with pytest.raises(ValueError):
        _ = pipeline(input1="test input")  # unused variable
