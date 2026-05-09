"""Tests for configure.py"""

from unittest.mock import Mock, patch

from simpledspy import configure
from simpledspy.settings import settings as global_settings
from simpledspy.module_caller import ChainOfThought, Predict


def test_configure_settings():
    """Test configure updates global settings"""
    predict_caller = Predict()
    cot_caller = ChainOfThought()

    # Save original settings
    original_lm = global_settings.lm
    original_temp = global_settings.temperature
    original_max_tokens = global_settings.max_tokens
    original_predict_lm = predict_caller.lm
    original_cot_lm = cot_caller.lm

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
        global_settings.max_tokens = original_max_tokens
        predict_caller.lm = original_predict_lm
        cot_caller.lm = original_cot_lm


def test_configure_refreshes_existing_callers():
    """Test configure updates already-created caller instances."""
    predict_caller = Predict()
    cot_caller = ChainOfThought()
    original_lm = global_settings.lm
    original_default_lm = global_settings.default_lm
    original_predict_lm = predict_caller.lm
    original_cot_lm = cot_caller.lm

    try:
        lm = Mock()
        with patch("simpledspy.module_caller.dspy.configure") as mock_configure:
            configure(lm=lm, default_lm=None)

        assert predict_caller.lm is lm
        assert cot_caller.lm is lm
        assert mock_configure.called
    finally:
        global_settings.lm = original_lm
        global_settings.default_lm = original_default_lm
        predict_caller.lm = original_predict_lm
        cot_caller.lm = original_cot_lm
