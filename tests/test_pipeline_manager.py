"""Tests for pipeline_manager.py"""

import os
import sys
import dspy
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simpledspy.pipeline_manager import PipelineManager


class MockModule(dspy.Module):
    """Mock module for pipeline testing"""

    def __init__(self, output_value=None, **outputs):
        super().__init__()
        if output_value:
            outputs["output"] = output_value
        self.outputs = outputs

    def forward(self, **kwargs):  # pylint: disable=unused-argument
        """Mock forward pass."""
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
    from simpledspy.exceptions import PipelineError
    manager = PipelineManager()
    manager.reset()
    with pytest.raises(PipelineError):
        _ = manager.assemble_pipeline()  # unused variable


def test_pipeline_reset():
    """Test resetting the pipeline"""
    manager = PipelineManager()
    manager.reset()

    module = MockModule("test output")
    manager.register_step(inputs=["input1"], outputs=["output1"], module=module)

    # pylint: disable=protected-access
    assert len(manager._steps) == 1
    manager.reset()
    # pylint: disable=protected-access
    assert len(manager._steps) == 0
    from simpledspy.exceptions import PipelineError
    with pytest.raises(PipelineError):
        _ = manager.assemble_pipeline()  # unused variable


def test_pipeline_assembly():
    """Test assembling a pipeline with multiple steps"""
    manager = PipelineManager()
    manager.reset()

    module1 = MockModule(output1="step1 output")
    module2 = MockModule(output2="step2 output")

    manager.register_step(inputs=["input1"], outputs=["output1"], module=module1)
    manager.register_step(inputs=["output1"], outputs=["output2"], module=module2)

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
    manager.register_step(inputs=["input1"], outputs=["output1"], module=module)

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
            """Mock forward pass that returns nothing."""
            return {}  # No output

    module = BadModule()
    manager.register_step(inputs=["input1"], outputs=["output1"], module=module)

    pipeline = manager.assemble_pipeline()

    with pytest.raises(ValueError):
        _ = pipeline(input1="test input")  # unused variable
