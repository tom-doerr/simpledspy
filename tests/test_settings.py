"""Tests for settings module"""

import pytest
from simpledspy.settings import Settings, settings


class TestSettings:
    """Test cases for Settings class"""

    def test_settings_initialization(self):
        """Test default initialization of Settings"""
        s = Settings()
        assert s.lm is None
        assert s.temperature is None
        assert s.max_tokens is None
        assert s.logging_enabled is False
        assert s.log_dir == ".simpledspy"
        assert s.default_lm is None
        assert s.evaluator_lm is None

    def test_settings_str_representation(self):
        """Test string representation of Settings"""
        s = Settings()
        expected = (
            "Settings(lm=None, temperature=None, max_tokens=None, "
            "logging_enabled=False, log_dir='.simpledspy')"
        )
        assert str(s) == expected

    def test_settings_str_with_values(self):
        """Test string representation with values set"""
        s = Settings()
        s.lm = "test-model"
        s.temperature = 0.7
        s.max_tokens = 100
        s.logging_enabled = True
        expected = (
            "Settings(lm=test-model, temperature=0.7, max_tokens=100, "
            "logging_enabled=True, log_dir='.simpledspy')"
        )
        assert str(s) == expected

    def test_global_settings_instance(self):
        """Test that global settings is a Settings instance"""
        assert isinstance(settings, Settings)

    def test_settings_mutable_attributes(self):
        """Test that settings attributes are mutable"""
        s = Settings()
        s.lm = "new-model"
        s.temperature = 0.5
        s.max_tokens = 200
        s.logging_enabled = True
        s.log_dir = "/tmp/logs"
        s.default_lm = "default-model"
        s.evaluator_lm = "eval-model"

        assert s.lm == "new-model"
        assert s.temperature == 0.5
        assert s.max_tokens == 200
        assert s.logging_enabled is True
        assert s.log_dir == "/tmp/logs"
        assert s.default_lm == "default-model"
        assert s.evaluator_lm == "eval-model"

    def test_settings_independent_instances(self):
        """Test that multiple Settings instances are independent"""
        s1 = Settings()
        s2 = Settings()

        s1.lm = "model1"
        s2.lm = "model2"

        assert s1.lm == "model1"
        assert s2.lm == "model2"

    def test_settings_with_custom_log_dir(self):
        """Test settings with custom log directory"""
        s = Settings()
        s.log_dir = "/custom/path"

        expected = (
            "Settings(lm=None, temperature=None, max_tokens=None, "
            "logging_enabled=False, log_dir='/custom/path')"
        )
        assert str(s) == expected

    def test_settings_attribute_types(self):
        """Test that settings can handle various attribute types"""
        s = Settings()

        # Test with various types
        s.lm = {"model": "test", "api_key": "key"}
        s.temperature = 0.0
        s.max_tokens = 0
        s.logging_enabled = False
        s.log_dir = ""

        assert isinstance(s.lm, dict)
        assert s.temperature == 0.0
        assert s.max_tokens == 0
        assert s.logging_enabled is False
        assert s.log_dir == ""

    def test_settings_none_values(self):
        """Test that None values are handled correctly"""
        s = Settings()
        s.lm = "model"
        s.lm = None  # Reset to None

        assert s.lm is None

    def test_global_settings_modification(self):
        """Test modifying the global settings instance"""
        # Store original values
        original_lm = settings.lm
        original_logging = settings.logging_enabled

        try:
            # Modify global settings
            settings.lm = "test-global-model"
            settings.logging_enabled = True

            assert settings.lm == "test-global-model"
            assert settings.logging_enabled is True

        finally:
            # Restore original values
            settings.lm = original_lm
            settings.logging_enabled = original_logging
