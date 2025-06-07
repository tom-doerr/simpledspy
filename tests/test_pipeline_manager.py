import pytest
from simpledspy.pipeline_manager import PipelineManager
import dspy
from unittest.mock import MagicMock, patch

class MockModule(dspy.Module):
    def __init__(self, output_value=None, **outputs):
        super().__init__()
        if output_value:
            outputs['output'] = output_value
        self.outputs = outputs
    
    def forward(self, **kwargs):
        return dspy.Prediction(**self.outputs)

def test_singleton_pattern():
    """Test that PipelineManager follows the singleton pattern"""
    manager1 = PipelineManager()
    manager2 = PipelineManager()
    assert manager1 is manager2
    manager1.reset()
    assert manager2._steps == []

def test_empty_pipeline():
    """Test assembling an empty pipeline"""
    manager = PipelineManager()
    manager.reset()
    with pytest.raises(ValueError):
        pipeline = manager.assemble_pipeline()

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
        pipeline = manager.assemble_pipeline()

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
        result = pipeline()  # Missing input

def test_missing_output():
    """Test missing output in pipeline execution"""
    manager = PipelineManager()
    manager.reset()
    
    class BadModule(dspy.Module):
        def forward(self, **kwargs):
            return {}  # No output
    
    module = BadModule()
    manager.register_step(
        inputs=["input1"],
        outputs=["output1"],
        module=module
    )
    
    pipeline = manager.assemble_pipeline()
    
    with pytest.raises(ValueError):
        result = pipeline(input1="test input")
