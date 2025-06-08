"""Global settings for SimpleDSPy

This module holds global configuration settings for the SimpleDSPy library.
"""

class Settings:
    """Holds global settings for SimpleDSPy"""
    def __init__(self):
        self.lm = None
        self.temperature = None
        self.max_tokens = None
        self.logging_enabled = False  # Disabled by default
        # Add other settings as needed
        
    def __str__(self):
        return (f"Settings(lm={self.lm}, temperature={self.temperature}, "
                f"max_tokens={self.max_tokens}, logging_enabled={self.logging_enabled})")

# Global settings instance
settings = Settings()
