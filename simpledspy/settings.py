"""Global settings for SimpleDSPy

This module holds global configuration settings for the SimpleDSPy library.
"""

class Settings:
    """Holds global settings for SimpleDSPy"""
    def __init__(self):
        self.lm = None
        self.temperature = None
        self.max_tokens = None
        # Add other settings as needed

# Global settings instance
settings = Settings()
