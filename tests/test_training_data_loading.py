"""Tests for training data loading functionality"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
import pytest

from simpledspy.logger import Logger
from simpledspy.module_caller import Predict


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
        
        # Create Predict module
        predict = Predict()
        
        # Verify training data was loaded
        # pylint: disable=protected-access
        demos = predict._load_training_data(module_name)
        assert len(demos) == 2
        
        # Check both demos
        for demo in demos:
            assert demo.input1 == "test input"
            assert demo.output1 == "test output"

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
        
        # Verify no demos loaded
        predict = Predict()
        # pylint: disable=protected-access
        demos = predict._load_training_data(module_name)
        assert len(demos) == 0

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
