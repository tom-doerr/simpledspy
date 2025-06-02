import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock
from simpledspy.cli import main

def test_cli_direct_input():
    """Test CLI with direct input argument"""
    # Run the CLI command
    result = subprocess.run(
        [sys.executable, '-m', 'simpledspy.cli', 'Hello, world!', '-d', 'extract the greeting'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Hello" in result.stdout

def test_cli_multiple_inputs():
    """Test CLI with multiple input arguments"""
    # Run the CLI command
    result = subprocess.run(
        [sys.executable, '-m', 'simpledspy.cli', 'apple banana cherry', 'orange grape', '-d', 'extract the first word from each list'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "apple" in result.stdout
    assert "orange" in result.stdout

def test_cli_stdin():
    """Test CLI with stdin input"""
    # Run the CLI command with stdin
    result = subprocess.run(
        [sys.executable, '-m', 'simpledspy.cli', '-d', 'extract the first word'],
        input='Hello, world!',
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Hello" in result.stdout

def test_cli_help():
    """Test CLI help output"""
    # Run the CLI command with --help
    result = subprocess.run(
        [sys.executable, '-m', 'simpledspy.cli', '--help'],
        capture_output=True,
        text=True
    )
    
    # Check the output
    assert result.returncode == 0
    assert "SimpleDSPy command line interface" in result.stdout
    assert "--optimize" in result.stdout
    assert "--strategy" in result.stdout
    assert "--max-demos" in result.stdout
