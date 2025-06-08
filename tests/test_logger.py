"""Tests for logger.py"""
import os
import json
import tempfile
import time
import pytest
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
