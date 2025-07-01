"""Tests for configure.py"""

from simpledspy import configure
from simpledspy.settings import settings as global_settings


def test_configure_settings():
    """Test configure updates global settings"""
    # Save original settings
    original_lm = global_settings.lm
    original_temp = global_settings.temperature

    try:
        # Configure new settings
        configure(lm="test_lm", temperature=0.7, max_tokens=100)

        # Check settings updated
        assert global_settings.lm == "test_lm"
        assert global_settings.temperature == 0.7
        assert global_settings.max_tokens == 100

        # Configure partial update
        configure(temperature=0.9)
        assert global_settings.temperature == 0.9
        assert global_settings.lm == "test_lm"  # unchanged
    finally:
        # Restore original settings
        global_settings.lm = original_lm
        global_settings.temperature = original_temp
        if hasattr(global_settings, "max_tokens"):
            delattr(global_settings, "max_tokens")
