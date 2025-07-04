"""Tests for optimization_manager.py"""

import dspy
import pytest
import os
import sys
from dspy.teleprompt import BootstrapFewShot, MIPROv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simpledspy.metrics import dict_exact_match_metric
from simpledspy.optimization_manager import OptimizationManager


def test_default_config() -> None:
    """Test default configuration"""
    manager = OptimizationManager()

    # Check default values
    # pylint: disable=protected-access
    assert manager._config["strategy"] == "bootstrap_few_shot"
    assert manager._config["max_bootstrapped_demos"] == 4
    assert manager._config["max_labeled_demos"] == 4
    assert callable(manager._config["metric"])


def test_configure() -> None:
    """Test configuration update"""
    manager = OptimizationManager()

    # Update configuration
    manager.configure(strategy="mipro", max_bootstrapped_demos=6, max_labeled_demos=8)

    # Check updated values
    # pylint: disable=protected-access
    assert manager._config["strategy"] == "mipro"
    assert manager._config["max_bootstrapped_demos"] == 6
    assert manager._config["max_labeled_demos"] == 8


def test_dict_exact_match_metric() -> None:
    """Test the dict_exact_match_metric function"""
    # Test with exact match
    example = {"name": "John", "age": 30}
    prediction = {"name": "John", "age": 30}
    score = dict_exact_match_metric(example, prediction)
    assert score == 1.0

    # Test with partial match
    prediction = {"name": "John", "age": 25}
    score = dict_exact_match_metric(example, prediction)
    assert score == 0.5

    # Test with no match
    prediction = {"name": "Jane", "age": 25}
    score = dict_exact_match_metric(example, prediction)
    assert score == 0.0

    # Test with missing keys
    prediction = {"name": "John"}
    score = dict_exact_match_metric(example, prediction)
    assert score == 0.5


def test_get_teleprompter() -> None:
    """Test teleprompter creation"""
    manager = OptimizationManager()

    # Test bootstrap_few_shot teleprompter
    manager.configure(strategy="bootstrap_few_shot")
    teleprompter = manager.get_teleprompter()
    assert isinstance(teleprompter, BootstrapFewShot)

    # Test mipro teleprompter
    manager.configure(strategy="mipro")
    teleprompter = manager.get_teleprompter()
    assert isinstance(teleprompter, MIPROv2)


def test_dict_exact_match_metric_empty() -> None:
    """Test the dict_exact_match_metric function with empty inputs"""
    # Test with empty example and prediction
    example = {}
    prediction = {}

    # Should return 1.0 for empty example and empty prediction
    score = dict_exact_match_metric(example, prediction)
    assert score == 1.0

    # Test with empty example and non-empty prediction
    prediction = {"name": "John"}
    score = dict_exact_match_metric(example, prediction)
    assert score == 0.0


def test_dict_exact_match_metric_none_values() -> None:
    """Test the dict_exact_match_metric function with None values"""
    # Test with None values
    example = {"name": None, "age": 30}
    prediction = {"name": None, "age": 30}
    score = dict_exact_match_metric(example, prediction)
    assert score == 1.0

    # Test with mismatched None values
    prediction = {"name": "John", "age": None}
    score = dict_exact_match_metric(example, prediction)
    assert score == 0.0


def test_invalid_strategy() -> None:
    """Test error handling for invalid strategy"""
    manager = OptimizationManager()

    # Configure with invalid strategy
    manager.configure(strategy="invalid_strategy")

    # Should raise KeyError
    with pytest.raises(KeyError):
        manager.get_teleprompter()


def test_optimize_module() -> None:
    """Test module optimization"""
    manager = OptimizationManager()

    # Create a simple module
    class TestModule(dspy.Module):
        """Test module for optimization"""

        def __init__(self):
            super().__init__()

        def forward(self, x):
            """Mock forward method"""
            return {"y": x["x"] * 2}

    module = TestModule()

    # Create a simple trainset
    trainset = [{"x": 1, "y": 2}, {"x": 2, "y": 4}]

    # Mock the teleprompter compile method
    original_get_teleprompter = manager.get_teleprompter

    class MockTeleprompter:  # pylint: disable=too-few-public-methods
        """Mock teleprompter for testing"""

        def compile(self, module, trainset):  # pylint: disable=unused-argument
            """Mock compile method."""
            return module

    manager.get_teleprompter = MockTeleprompter

    # Test optimization
    optimized_module = manager.optimize(module, trainset)
    assert optimized_module is module

    # Restore original method
    manager.get_teleprompter = original_get_teleprompter


def test_configure_multiple_times() -> None:
    """Test multiple configuration updates"""
    manager = OptimizationManager()

    # Update configuration multiple times
    manager.configure(strategy="mipro")
    manager.configure(max_bootstrapped_demos=6)
    manager.configure(max_labeled_demos=8)

    # Check final values
    # pylint: disable=protected-access
    assert manager._config["strategy"] == "mipro"
    assert manager._config["max_bootstrapped_demos"] == 6
    assert manager._config["max_labeled_demos"] == 8


def test_configure_with_invalid_values() -> None:
    """Test configuration with invalid values"""
    manager = OptimizationManager()

    # Update with invalid values
    manager.configure(unknown_param="value", another_unknown=123)

    # Unknown parameters should be added to config
    # pylint: disable=protected-access
    assert manager._config["unknown_param"] == "value"
    assert manager._config["another_unknown"] == 123


def test_dict_exact_match_metric_with_trace() -> None:
    """Test dict_exact_match_metric function with trace parameter"""
    # Create example, prediction and trace
    example = {"name": "John"}
    prediction = {"name": "John"}
    trace = {"some": "trace", "data": 123}

    # Metric should work with trace
    score = dict_exact_match_metric(example, prediction, trace)
    assert score == 1.0
