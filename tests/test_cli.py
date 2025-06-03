import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock
from simpledspy.cli import main

def test_cli_direct_input(capsys):
    """Test CLI with direct input argument"""
    # Mock the arguments
    with patch('sys.argv', ['cli.py', 'Hello, world!', '-d', 'extract the greeting']):
        # Mock the ModuleFactory and module
        with patch('simpledspy.cli.ModuleFactory') as MockFactory:
            mock_factory = MockFactory.return_value
            mock_module = MagicMock()
            mock_factory.create_module.return_value = mock_module
            mock_module.return_value = MagicMock(output="Hello")
            
            # Call the main function
            main()
            
            # Capture the output
            captured = capsys.readouterr()
            assert "Hello" in captured.out

def test_cli_multiple_inputs(capsys):
    """Test CLI with multiple input arguments"""
    # Mock the arguments
    with patch('sys.argv', ['cli.py', 'apple banana cherry', 'orange grape', '-d', 'extract the first word from each list']):
        # Mock the ModuleFactory and module
        with patch('simpledspy.cli.ModuleFactory') as MockFactory:
            mock_factory = MockFactory.return_value
            mock_module = MagicMock()
            mock_factory.create_module.return_value = mock_module
            mock_module.return_value = MagicMock(output="apple\norange")
            
            # Call the main function
            main()
            
            # Capture the output
            captured = capsys.readouterr()
            assert "apple" in captured.out
            assert "orange" in captured.out

def test_cli_stdin(capsys, monkeypatch):
    """Test CLI with stdin input"""
    # We cannot easily mock stdin by opening a file, so instead we mock sys.stdin.isatty and sys.stdin.read
    # The CLI uses sys.stdin.isatty() to check if there's data, and then reads sys.stdin.read()
    # We mock both
    with patch('sys.stdin.isatty', return_value=False):
        with patch('sys.stdin.read', return_value='Hello, world!'):
            with patch('sys.argv', ['cli.py', '-d', 'extract the first word']):
                # Mock the ModuleFactory and module
                with patch('simpledspy.cli.ModuleFactory') as MockFactory:
                    mock_factory = MockFactory.return_value
                    mock_module = MagicMock()
                    mock_factory.create_module.return_value = mock_module
                    mock_module.return_value = MagicMock(output="Hello")
                    
                    # Call the main function
                    main()
                    
                    # Capture the output
                    captured = capsys.readouterr()
                    assert "Hello" in captured.out

def test_cli_help(capsys):
    """Test CLI help output"""
    # Mock the arguments
    with patch('sys.argv', ['cli.py', '--help']):
        try:
            main()
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "SimpleDSPy command line interface" in captured.out
        assert "--optimize" in captured.out
        assert "--strategy" in captured.out
        assert "--max-demos" in captured.out

def test_cli_json_output(capsys):
    """Test CLI with JSON output"""
    # Mock the arguments
    with patch('sys.argv', ['cli.py', 'Hello, world!', '-d', 'extract the greeting', '--json']):
        # Mock the ModuleFactory and module
        with patch('simpledspy.cli.ModuleFactory') as MockFactory:
            mock_factory = MockFactory.return_value
            mock_module = MagicMock()
            mock_factory.create_module.return_value = mock_module
            mock_module.return_value = MagicMock(output="Hello")
            
            # Call the main function
            main()
            
            # Capture the output
            captured = capsys.readouterr()
            assert "output" in captured.out
            assert "Hello" in captured.out
            assert "{" in captured.out  # Check for JSON structure
