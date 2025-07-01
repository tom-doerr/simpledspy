"""Retry logic for LLM calls"""

import time
import random
from typing import Callable, Any, Optional, Tuple, Type
from functools import wraps
from .exceptions import ModuleError


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """
        Initialize retry configuration
        
        Args:
            max_attempts: Maximum number of attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            exceptions: Tuple of exceptions to retry on
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number"""
        delay = min(
            self.initial_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay
        )
        
        if self.jitter:
            # Add jitter: random value between 0 and 0.1 * delay
            delay += random.uniform(0, 0.1 * delay)
        
        return delay


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator to add retry logic to a function
    
    Args:
        config: Retry configuration (uses defaults if None)
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        # Last attempt failed
                        raise ModuleError(
                            f"Failed after {config.max_attempts} attempts: {str(e)}"
                        ) from e
                    
                    # Calculate delay and wait
                    delay = config.calculate_delay(attempt)
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise ModuleError("Unexpected retry error")
        
        return wrapper
    return decorator


def retry_with_config(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Execute a function with retry logic
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of func
    """
    if config is None:
        config = RetryConfig()
    
    decorated_func = with_retry(config)(func)
    return decorated_func(*args, **kwargs)
