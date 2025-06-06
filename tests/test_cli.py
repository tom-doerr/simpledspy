import pytest
import subprocess
import sys
import dspy
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

def test_cli_optimization(capsys):
    """Test CLI optimization flag"""
    # Mock the arguments
    with patch('sys.argv', ['cli.py', 'Hello, world!', '-d', 'extract the greeting', '--optimize']):
        # Mock dependencies
        with patch('simpledspy.cli.ModuleFactory') as MockFactory, \
             patch('simpledspy.optimization_manager.OptimizationManager') as MockOptManager:
            
            # Setup mock module
            mock_factory = MockFactory.return_value
            mock_module = MagicMock()
            mock_module._compiled = False
            mock_module.reset_copy = lambda: mock_module
            mock_factory.create_module.return_value = mock_module
            mock_module.return_value = MagicMock(output="Optimized Hello")
                
            # Setup optimization manager
            mock_manager = MockOptManager.return_value
            mock_manager.optimize.return_value = mock_module
                
            # Call the main function
            main()
                
            # Capture the output
            captured = capsys.readouterr()
            assert "Optimized Hello" in captured.out
                
def test_cli_pipeline(capsys):
    """Test CLI pipeline execution"""
    # Mock the arguments
    with patch('sys.argv', ['cli.py', 'Hello, world!', '--pipeline', 
                           'Step 1 description', 'Step 2 description']):
        # Mock dependencies
        with patch('simpledspy.cli.ModuleFactory') as MockFactory, \
             patch('simpledspy.pipeline_manager.PipelineManager') as MockPipelineManager:
    
            # Setup mock module factory
            mock_factory = MockFactory.return_value
    
            # Setup mock pipeline
            mock_manager = MockPipelineManager.return_value
            mock_pipeline = MagicMock()
            mock_manager.assemble_pipeline.return_value = mock_pipeline
    
            # Create a proper pipeline output object
            output_value = "Pipeline Output"
            output_name = "output_2"

            # Create a simple object to simulate the prediction
            class SimplePrediction:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            prediction = SimplePrediction(**{output_name: output_value})

            # Set the pipeline mock to return the prediction
            mock_pipeline.return_value = prediction

            # Call the main function
            main()

            # Capture the output
            captured = capsys.readouterr()
            # The output should contain the expected value
            assert output_value in captured.out
            # Also check that the mock pipeline was called with the input
            mock_pipeline.assert_called_once_with(input_1="Hello, world!")
            # Also check that the mock pipeline was called with the input
            mock_pipeline.assert_called_once_with(text="Hello, world!")
                
            # Verify pipeline was created and executed
            mock_manager.register_step.assert_called()
            mock_manager.assemble_pipeline.assert_called_once()
            mock_pipeline.assert_called_once()
