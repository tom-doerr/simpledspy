"""Global settings for SimpleDSPy"""

class Settings:  # pylint: disable=too-few-public-methods
    """Holds global settings for SimpleDSPy"""
    def __init__(self):
        self.lm = None
        self.temperature = None
        self.max_tokens = None
        self.logging_enabled = False
        self.log_dir = ".simpledspy"  # Default logging directory
        
    def __str__(self):
        return (f"Settings(lm={self.lm}, temperature={self.temperature}, "
                f"max_tokens={self.max_tokens}, logging_enabled={self.logging_enabled}, "
                f"log_dir='{self.log_dir}')")

# Global settings instance
settings = Settings()
