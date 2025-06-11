"""Tests for training data loading functionality"""
import os
import json
import tempfile
import pytest
import dspy
from simpledspy.module_caller import Predict
from simpledspy.logger import Logger

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
        
        # Create and call Predict module
        predict = Predict()
        result = predict("test input", name=module_name)
        
        # Verify training data was loaded
        module = predict._create_module(["input1"], ["output1"])
        assert len(module.demos) == 1
        demo = module.demos[0]
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
        
        # Create module and verify no demos loaded
        predict = Predict()
        module = predict._create_module(["input1"], ["output1"])
        assert not hasattr(module, 'demos') or len(module.demos) == 0

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
        
        # Create and call Predict module
        predict = Predict()
        module = predict._create_module(
            ["input1", "input2"], 
            ["output1", "output2"]
        )
        
        # Verify data was formatted correctly
        assert len(module.demos) == 1
        demo = module.demos[0]
        assert demo.input1 == "value1"
        assert demo.input2 == 42
        assert demo.output1 is True
        assert demo.output2 == 3.14
