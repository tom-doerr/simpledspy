"""Tests for logger.py"""
import pytest
import os
import json
import tempfile
import time
from simpledspy.logger import Logger

def test_logger_init_creates_file():
    """Test that Logger creates the log file on init"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "test.log")
        logger = Logger(log_file)
        assert os.path.exists(log_file)

def test_logger_appends_data():
    """Test that log() appends JSON lines to file"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "test.log")
        logger = Logger(log_file)
        
        # Add first log entry
        data1 = {"test": "data1"}
        logger.log(data1)
        
        # Add second log entry
        data2 = {"test": "data2"}
        logger.log(data2)
        
        # Read log file
        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            
            entry0 = json.loads(lines[0])
            entry1 = json.loads(lines[1])
            
            assert entry0["test"] == "data1"
            assert entry1["test"] == "data2"
            assert isinstance(entry0["timestamp"], float)

def test_get_training_data_with_reward_group():
    """Test get_training_data() returns proper reward groups and scores"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "test.log")
        logger = Logger(log_file)
        
        # Add entries with different scores and groups
        logger.log({'score': 9.0, 'reward_group': 'group1', 'inputs': {}, 'outputs': {}})
        logger.log({'score': 7.0, 'reward_group': 'group1', 'inputs': {}, 'outputs': {}})
        logger.log({'score': 9.0, 'reward_group': 'group2', 'inputs': {}, 'outputs': {}})
        logger.log({'score': 8.0, 'reward_group': 'default', 'inputs': {}, 'outputs': {}})
        
        # Test group filtering
        group1_data = logger.get_training_data(min_score=8.0, reward_group='group1')
        assert len(group1_data) == 1
        
        # Test default group
        default_data = logger.get_training_data(min_score=8.0)
        assert len(default_data) == 1

def test_get_training_data_missing_file():
    """Test get_training_data() returns empty list when file missing"""
    logger = Logger("non_existent_file.log")
    training_data = logger.get_training_data()
    assert training_data == []

def test_get_training_data_invalid_json():
    """Test get_training_data() skips invalid JSON lines"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "test.log")
            
        # Create file with invalid JSON and then a valid log entry
        with open(log_file, "w") as f:
            f.write("not json" + "\n")
            f.write('{"inputs": {}, "outputs": {}, "score": 9.0, "reward_group": "default"}' + "\n")
            
        logger = Logger(log_file)
        training_data = logger.get_training_data()
        assert len(training_data) == 1
        # Check that the training data has the expected structure
        assert training_data[0]["inputs"] == {}
        assert training_data[0]["outputs"] == {}
        assert training_data[0].get("instruction") == ''  # not set in the log entry
