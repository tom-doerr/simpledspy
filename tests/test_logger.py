"""Tests for logger.py"""
import os
import json
import re
import tempfile
from simpledspy.logger import Logger

def test_logger_init_creates_file():
    """Test that Logger creates the log file on init"""
    with tempfile.TemporaryDirectory() as tmpdir:
        module_name = "test_module"
        logger = Logger(module_name, base_dir=tmpdir)
        assert os.path.exists(logger.logged_file)
        assert os.path.exists(logger.training_file)

def test_logger_appends_data():
    """Test that log() appends JSON lines to file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        module_name = "test_module"
        logger = Logger(module_name, base_dir=tmpdir)
        
        # Add first log entry
        data1 = {"test": "data1"}
        logger.log(data1)
        
        # Add second log entry
        data2 = {"test": "data2"}
        logger.log(data2)
        
        # Read log file
        with open(logger.logged_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
            
            entry0 = json.loads(lines[0])
            entry1 = json.loads(lines[1])
            
            assert entry0["test"] == "data1"
            assert entry1["test"] == "data2"
            # Check timestamp format is ISO 8601 with 'T'
            assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", entry0["timestamp"])

def test_logger_handles_invalid_data():
    """Test that logger skips empty lines and invalid JSON when loading training data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        module_name = "test_module"
        logger = Logger(module_name, base_dir=tmpdir)
        
        # Write invalid data to training file
        with open(logger.training_file, "w", encoding="utf-8") as f:
            f.write("\n")  # empty line
            f.write("{invalid json}\n")  # invalid JSON
            f.write(json.dumps({"valid": "data"}) + "\n")  # valid JSON
        
        # Load training data
        examples = logger.load_training_data()
        
        # Should only have the valid entry
        assert len(examples) == 1
        assert examples[0]["valid"] == "data"

def test_timestamp_format():
    """Test that timestamps use ISO 8601 format with T separator"""
    with tempfile.TemporaryDirectory() as tmpdir:
        module_name = "test_module"
        logger = Logger(module_name, base_dir=tmpdir)
        
        # Log sample data
        data = {"key": "value"}
        logger.log(data)
        
        # Read log file
        with open(logger.logged_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            entry = json.loads(line)
            
            # Check timestamp format matches YYYY-MM-DDTHH:MM:SS.microsecondsZ
            assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$", entry["timestamp"])
