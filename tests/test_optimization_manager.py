import pytest
from simpledspy.optimization_manager import OptimizationManager
from dspy.teleprompt import BootstrapFewShot, MIPROv2
import dspy

def test_default_config() -> None:
    """Test default configuration"""
    manager = OptimizationManager()
    
    # Check default values
    assert manager._config['strategy'] == 'bootstrap_few_shot'
    assert manager._config['max_bootstrapped_demos'] == 4
    assert manager._config['max_labeled_demos'] == 4
    assert callable(manager._config['metric'])

def test_configure() -> None:
    """Test configuration update"""
    manager = OptimizationManager()
    
    # Update configuration
    manager.configure(
        strategy='mipro',
        max_bootstrapped_demos=6,
        max_labeled_demos=8
    )
    
    # Check updated values
    assert manager._config['strategy'] == 'mipro'
    assert manager._config['max_bootstrapped_demos'] == 6
    assert manager._config['max_labeled_demos'] == 8

def test_default_metric() -> None:
    """Test the default metric function"""
    manager = OptimizationManager()
    
    # Test with exact match
    example = {'name': 'John', 'age': 30}
    prediction = {'name': 'John', 'age': 30}
    score = manager.default_metric(example, prediction)
    assert score == 1.0
    
    # Test with partial match
    prediction = {'name': 'John', 'age': 25}
    score = manager.default_metric(example, prediction)
    assert score == 0.5
    
    # Test with no match
    prediction = {'name': 'Jane', 'age': 25}
    score = manager.default_metric(example, prediction)
    assert score == 0.0
    
    # Test with missing keys
    prediction = {'name': 'John'}
    score = manager.default_metric(example, prediction)
    assert score == 0.5

def test_get_teleprompter() -> None:
    """Test teleprompter creation"""
    manager = OptimizationManager()
    
    # Test bootstrap_few_shot teleprompter
    manager.configure(strategy='bootstrap_few_shot')
    teleprompter = manager.get_teleprompter()
    assert isinstance(teleprompter, BootstrapFewShot)
    
    # Test mipro teleprompter
    manager.configure(strategy='mipro')
    teleprompter = manager.get_teleprompter()
    assert isinstance(teleprompter, MIPROv2)

def test_default_metric_empty() -> None:
    """Test the default metric function with empty inputs"""
    manager = OptimizationManager()
    
    # Test with empty example and prediction
    example = {}
    prediction = {}
    
    # Should return 1.0 for empty example and empty prediction
    score = manager.default_metric(example, prediction)
    assert score == 1.0
    
    # Test with empty example and non-empty prediction
    prediction = {'name': 'John'}
    score = manager.default_metric(example, prediction)
    assert score == 0.0

def test_default_metric_none_values() -> None:
    """Test the default metric function with None values"""
    manager = OptimizationManager()
    
    # Test with None values
    example = {'name': None, 'age': 30}
    prediction = {'name': None, 'age': 30}
    score = manager.default_metric(example, prediction)
    assert score == 1.0
    
    # Test with mismatched None values
    prediction = {'name': 'John', 'age': None}
    score = manager.default_metric(example, prediction)
    assert score == 0.0

def test_invalid_strategy() -> None:
    """Test error handling for invalid strategy"""
    manager = OptimizationManager()
    
    # Configure with invalid strategy
    manager.configure(strategy='invalid_strategy')
    
    # Should raise KeyError
    with pytest.raises(KeyError):
        manager.get_teleprompter()

def test_optimize_module() -> None:
    """Test module optimization"""
    manager = OptimizationManager()
    
    # Create a simple module
    class TestModule(dspy.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return {'y': x['x'] * 2}
    
    module = TestModule()
    
    # Create a simple trainset
    trainset = [{'x': 1, 'y': 2}, {'x': 2, 'y': 4}]
    
    # Mock the teleprompter compile method
    original_get_teleprompter = manager.get_teleprompter
    
    class MockTeleprompter:
        def compile(self, module, trainset):
            return module
    
    manager.get_teleprompter = lambda: MockTeleprompter()
    
    # Test optimization
    optimized_module = manager.optimize(module, trainset)
    assert optimized_module is module
    
    # Restore original method
    manager.get_teleprompter = original_get_teleprompter

def test_configure_multiple_times() -> None:
    """Test multiple configuration updates"""
    manager = OptimizationManager()
    
    # Update configuration multiple times
    manager.configure(strategy='mipro')
    manager.configure(max_bootstrapped_demos=6)
    manager.configure(max_labeled_demos=8)
    
    # Check final values
    assert manager._config['strategy'] == 'mipro'
    assert manager._config['max_bootstrapped_demos'] == 6
    assert manager._config['max_labeled_demos'] == 8

def test_configure_with_invalid_values() -> None:
    """Test configuration with invalid values"""
    manager = OptimizationManager()
    
    # Update with invalid values
    manager.configure(
        unknown_param='value',
        another_unknown=123
    )
    
    # Unknown parameters should be added to config
    assert manager._config['unknown_param'] == 'value'
    assert manager._config['another_unknown'] == 123

def test_metric_with_trace() -> None:
    """Test metric function with trace parameter"""
    manager = OptimizationManager()
    
    # Create example, prediction and trace
    example = {'name': 'John'}
    prediction = {'name': 'John'}
    trace = {'some': 'trace', 'data': 123}
    
    # Metric should work with trace
    score = manager.default_metric(example, prediction, trace)
    assert score == 1.0
