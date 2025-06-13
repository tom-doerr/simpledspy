"""Tests for training data loading functionality"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
import pytest

from simpledspy.logger import Logger
from simpledspy.module_caller import Predict, BaseCaller
from simpledspy.settings import settings as global_settings


# Mock LM to prevent "No LM is loaded" errors
@pytest.fixture(autouse=True)
def mock_lm(monkeypatch):
    """Mocks the dspy.LM for all tests."""
    mock_lm_instance = MagicMock()
    monkeypatch.setattr(dspy, 'configure', MagicMock())
    monkeypatch.setattr(dspy, 'LM', MagicMock(return_value=mock_lm_instance))


def test_training_data_loading():
    """Test that training data is properly loaded and used in modules"""
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up test data
        module_name = "test_module"
        base_dir = os.path.join(tmpdir, ".simpledspy")
        logger = Logger(module_name, base_dir)
        
        # Create training data in both formats
        for data in [
            # New format
            {
                'inputs': [{'name': 'input1', 'value': 'test input'}],
                'outputs': [{'name': 'output1', 'value': 'test output'}],
                'section': 'training'
            },
            # Old format
            {
                'input1': 'test input',
                'output1': 'test output',
                'section': 'training'
            }
        ]:
            logger.log_to_section(data, "training")
        
        # Create logged data (should be ignored)
        logged_data = {
            'inputs': [{'name': 'input1', 'value': 'bad input'}],
            'outputs': [{'name': 'output1', 'value': 'bad output'}],
            'section': 'logged'
        }
        logger.log_to_section(logged_data, "logged")
        
        # Monkeypatch global_settings.log_dir to use tmpdir
        original_log_dir = global_settings.log_dir
        global_settings.log_dir = base_dir # The .simpledspy folder within tmpdir

        try:
            with patch.object(BaseCaller, '_create_module') as mock_create_module:
                mock_module_instance = MagicMock(spec=dspy.Module)
                mock_module_instance.demos = [] # Initialize demos attribute
                mock_create_module.return_value = mock_module_instance

                # Instantiate Predict. Note: Predict is a singleton, reset for clean test state if necessary
                # For this test, assuming a fresh state or that prior state doesn't interfere.
                # If Predict was a singleton managed by BaseCaller._instances, resetting might be:
                # if Predict in BaseCaller._instances: del BaseCaller._instances[Predict]
                predict_instance = Predict()

                # Call predict to trigger demo loading. Module name must match logger's.
                # Inputs/outputs for the call don't strictly matter for demo loading test, but must be valid.
                try:
                    predict_instance("dummy_input", inputs=["input1"], outputs=["output1"], name=module_name)
                except Exception as e:
                    # The call might fail if LM is not configured or if dummy input is bad,
                    # but demo loading should happen before LM interaction for this test's purpose.
                    # We are interested in mock_module_instance.demos set by _load_and_prepare_demos.
                    pass # Allow progression to demo assertion

                demos = mock_module_instance.demos
                assert len(demos) == 2  # Ensure only training data is loaded
        
                # Check content of loaded demos (both formats should be handled)
                # The _format_example method converts to dspy.Example
                # Example structure: dspy.Example(input1='test input', output1='test output')
                expected_demo_content = [{'input1': 'test input', 'output1': 'test output'}]
                
                assert len(demos) == 2
                for demo in demos:
                    assert isinstance(demo, dspy.Example)
                    # Check if the demo content matches one of the expected structures
                    # Stripping away dspy.Example's internal fields for comparison
                    demo_dict = {k: v for k, v in demo.items() if k in ['input1', 'output1']}
                    assert demo_dict in expected_demo_content
        finally:
            global_settings.log_dir = original_log_dir


def test_malformed_training_data():
    """Test that malformed training data is skipped"""
    with tempfile.TemporaryDirectory() as tmpdir:
        module_name = "test_module"
        base_dir = os.path.join(tmpdir, ".simpledspy")
        logger = Logger(module_name, base_dir)
        
        # Create invalid training data (missing 'outputs')
        invalid_data = {
            'inputs': [{'name': 'input1', 'value': 'test input'}],
            'section': 'training'
        }
        logger.log_to_section(invalid_data, "training")
        
        # Monkeypatch global_settings.log_dir to use tmpdir
        original_log_dir = global_settings.log_dir
        global_settings.log_dir = base_dir

        try:
            with patch.object(BaseCaller, '_create_module') as mock_create_module:
                mock_module_instance = MagicMock(spec=dspy.Module)
                mock_module_instance.demos = []
                mock_create_module.return_value = mock_module_instance

                predict_instance = Predict()
                try:
                    # Name is important for logger to find the file.
                    predict_instance("dummy", inputs=["input1"], outputs=["output1"], name=module_name) 
                except Exception:
                    pass # Interested in demos, not necessarily successful run

                # Verify no demos loaded
                assert not mock_module_instance.demos  # No valid training data should be loaded
        finally:
            global_settings.log_dir = original_log_dir


def test_training_data_formatting():
    """Test training data formatting for DSPy compatibility"""
    with tempfile.TemporaryDirectory() as tmpdir:
        module_name = "test_module"
        base_dir = os.path.join(tmpdir, ".simpledspy")
        logger = Logger(module_name, base_dir)
        
        # Create training data in both formats
        for data in [
            # New format
            {
                'inputs': [
                    {'name': 'input1', 'value': 'value1'},
                    {'name': 'input2', 'value': 42}
                ],
                'outputs': [
                    {'name': 'output1', 'value': True},
                    {'name': 'output2', 'value': 3.14}
                ],
                'section': 'training'
            },
            # Old format
            {
                'input1': 'value1',
                'input2': 42,
                'output1': True,
                'output2': 3.14,
                'section': 'training'
            }
        ]:
            logger.log_to_section(data, "training")
        
        # Create Predict module
        predict = Predict()
        # pylint: disable=protected-access
        demos = predict._load_training_data(module_name)
        assert len(demos) == 2
        
        # Verify both demos
        for demo in demos:
            assert demo.input1 == "value1"
            assert demo.input2 == 42
            assert demo.output1 is True
            assert demo.output2 == 3.14
