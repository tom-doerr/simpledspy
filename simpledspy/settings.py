"""Global settings for SimpleDSPy"""


class Settings:  # pylint: disable=too-few-public-methods
    """Holds global settings for SimpleDSPy"""

    def __init__(self):
        self.lm = None
        self.temperature = None
        self.max_tokens = None
        self.logging_enabled = False
        self.log_dir = ".simpledspy"  # Default logging directory
        self.default_lm = None  # Default LM for module_caller
        self.evaluator_lm = None  # Default LM for evaluator
        # Rate limiting settings
        self.rate_limit_calls = 100  # Max calls per window
        self.rate_limit_window = 60  # Window size in seconds
        # Retry settings
        self.retry_attempts = 3  # Max retry attempts
        self.retry_delay = 1.0  # Initial retry delay in seconds

    def __str__(self):
        return (
            f"Settings(lm={self.lm}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, logging_enabled={self.logging_enabled}, "
            f"log_dir='{self.log_dir}')"
        )


# Global settings instance
settings = Settings()
