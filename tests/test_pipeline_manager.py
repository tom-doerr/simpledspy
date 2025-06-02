import pytest
from simpledspy.pipeline_manager import PipelineManager
import dspy
from unittest.mock import MagicMock

class MockModule(dspy.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args):
        return args

def test_singleton_pattern():
    """Test that PipelineManager follows the singleton pattern"""
    manager1 = PipelineManager()
    manager2 = PipelineManager()
    assert manager1 is manager2
    manager1._steps = []
    assert manager2._steps == []

def test_empty_pipeline():
    """Test assembling an empty pipeline"""
    manager = PipelineManager()
    manager._steps = []
    pipeline = manager.assemble_pipeline()
    assert pipeline() is None

def test_pipeline_reset():
    """Test resetting the pipeline"""
    manager = PipelineManager()
    manager._steps = []
    
    module = MockModule()
    manager.register_step(
        inputs=["input1"],
        outputs=["output1"],
        module=module
    )
    
    assert len(manager._steps) == 1
    manager._steps = []
    assert len(manager._steps) == 0
    pipeline = manager.assemble_pipeline()
    assert pipeline() is None
