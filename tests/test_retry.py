"""Tests for retry logic"""

import time
import pytest
from unittest.mock import Mock, call
from simpledspy.retry import RetryConfig, with_retry, retry_with_config
from simpledspy.exceptions import ModuleError


def test_retry_config_defaults():
    """Test default retry configuration"""
    config = RetryConfig()
    
    assert config.max_attempts == 3
    assert config.initial_delay == 1.0
    assert config.max_delay == 60.0
    assert config.exponential_base == 2.0
    assert config.jitter is True
    assert config.exceptions == (Exception,)


def test_retry_config_custom():
    """Test custom retry configuration"""
    config = RetryConfig(
        max_attempts=5,
        initial_delay=0.5,
        max_delay=30.0,
        exponential_base=3.0,
        jitter=False,
        exceptions=(ValueError, TypeError)
    )
    
    assert config.max_attempts == 5
    assert config.initial_delay == 0.5
    assert config.max_delay == 30.0
    assert config.exponential_base == 3.0
    assert config.jitter is False
    assert config.exceptions == (ValueError, TypeError)


def test_calculate_delay_exponential():
    """Test exponential backoff calculation"""
    config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)
    
    assert config.calculate_delay(1) == 1.0  # 1 * 2^0
    assert config.calculate_delay(2) == 2.0  # 1 * 2^1
    assert config.calculate_delay(3) == 4.0  # 1 * 2^2
    assert config.calculate_delay(4) == 8.0  # 1 * 2^3


def test_calculate_delay_max_cap():
    """Test that delay is capped at max_delay"""
    config = RetryConfig(
        initial_delay=1.0,
        max_delay=5.0,
        exponential_base=2.0,
        jitter=False
    )
    
    assert config.calculate_delay(1) == 1.0
    assert config.calculate_delay(2) == 2.0
    assert config.calculate_delay(3) == 4.0
    assert config.calculate_delay(4) == 5.0  # Capped at max_delay
    assert config.calculate_delay(5) == 5.0  # Still capped


def test_calculate_delay_with_jitter():
    """Test delay calculation with jitter"""
    config = RetryConfig(initial_delay=1.0, jitter=True)
    
    # With jitter, delay should be slightly different each time
    delays = [config.calculate_delay(2) for _ in range(10)]
    
    # All should be around 2.0 (base delay) but with variation
    assert all(2.0 <= d <= 2.2 for d in delays)  # Max jitter is 0.1 * delay
    # Not all delays should be exactly the same
    assert len(set(delays)) > 1


def test_with_retry_success():
    """Test successful function call without retry"""
    mock_func = Mock(return_value="success")
    
    @with_retry()
    def test_func():
        return mock_func()
    
    result = test_func()
    
    assert result == "success"
    assert mock_func.call_count == 1


def test_with_retry_eventual_success():
    """Test function that fails then succeeds"""
    mock_func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])
    
    config = RetryConfig(max_attempts=3, initial_delay=0.01)
    
    @with_retry(config)
    def test_func():
        return mock_func()
    
    result = test_func()
    
    assert result == "success"
    assert mock_func.call_count == 3


def test_with_retry_max_attempts_exceeded():
    """Test that ModuleError is raised after max attempts"""
    mock_func = Mock(side_effect=ValueError("always fails"))
    
    config = RetryConfig(max_attempts=3, initial_delay=0.01)
    
    @with_retry(config)
    def test_func():
        return mock_func()
    
    with pytest.raises(ModuleError) as exc_info:
        test_func()
    
    assert "Failed after 3 attempts" in str(exc_info.value)
    assert mock_func.call_count == 3


def test_with_retry_specific_exceptions():
    """Test retry only on specific exceptions"""
    mock_func = Mock(side_effect=[ValueError("retry this"), TypeError("don't retry")])
    
    config = RetryConfig(exceptions=(ValueError,), initial_delay=0.01)
    
    @with_retry(config)
    def test_func():
        return mock_func()
    
    # Should retry on ValueError but not on TypeError
    with pytest.raises(TypeError):
        test_func()
    
    assert mock_func.call_count == 2


def test_with_retry_timing():
    """Test that retry delays are applied"""
    mock_func = Mock(side_effect=[ValueError("fail"), "success"])
    
    config = RetryConfig(max_attempts=2, initial_delay=0.1, jitter=False)
    
    @with_retry(config)
    def test_func():
        return mock_func()
    
    start_time = time.time()
    result = test_func()
    elapsed = time.time() - start_time
    
    assert result == "success"
    # Should have waited approximately 0.1 seconds
    assert 0.08 < elapsed < 0.15


def test_retry_with_config_function():
    """Test retry_with_config helper function"""
    mock_func = Mock(side_effect=[ValueError("fail"), "success"])
    
    config = RetryConfig(max_attempts=2, initial_delay=0.01)
    
    result = retry_with_config(mock_func, config=config)
    
    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_config_args_kwargs():
    """Test retry_with_config with function arguments"""
    def test_func(a, b, c=None):
        if test_func.call_count < 2:
            test_func.call_count += 1
            raise ValueError("fail")
        return f"{a}-{b}-{c}"
    
    test_func.call_count = 0
    
    config = RetryConfig(max_attempts=3, initial_delay=0.01)
    
    result = retry_with_config(test_func, "arg1", "arg2", config=config, c="arg3")
    
    assert result == "arg1-arg2-arg3"
    assert test_func.call_count == 2


def test_retry_preserves_function_metadata():
    """Test that decorator preserves function metadata"""
    @with_retry()
    def example_func(x, y):
        """Example function docstring"""
        return x + y
    
    assert example_func.__name__ == "example_func"
    assert example_func.__doc__ == "Example function docstring"


def test_retry_nested_exceptions():
    """Test that original exception is preserved in chain"""
    original_error = ValueError("original error")
    mock_func = Mock(side_effect=original_error)
    
    config = RetryConfig(max_attempts=1, initial_delay=0.01)
    
    @with_retry(config)
    def test_func():
        return mock_func()
    
    with pytest.raises(ModuleError) as exc_info:
        test_func()
    
    # Check that original exception is in the chain
    assert exc_info.value.__cause__ is original_error