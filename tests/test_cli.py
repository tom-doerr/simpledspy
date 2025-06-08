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
             patch('simpledspy.cli.PipelineManager') as MockPipelineManager:
    
            # Setup mock module factory
            mock_factory = MockFactory.return_value
            # Create mock modules that return actual Prediction objects
            class MockPrediction:
                def __init__(self, result_dict):
                    for key, value in result_dict.items():
                        setattr(self, key, value)
                
            # Define outputs for the pipeline steps
            step0_output = "step0 output"
            step1_output = "Pipeline Output"
                
            def mock_step_module_forward(*args, **kwargs):
                # For step0: returns output_1; step1: returns output_2
                if hasattr(kwargs.get('input_1', None), '__len__'):
                    return MockPrediction({'output_1': step0_output})
                elif hasattr(kwargs.get('output_1', None), '__len__'):
                    return MockPrediction({'output_2': step1_output})
                return MockPrediction({})
                
            # Create a mock module with proper forward behavior
            mock_module = MagicMock()
            mock_module.forward.side_effect = mock_step_module_forward
            mock_factory.create_module.return_value = mock_module
    
            # Setup mock pipeline and manager
            mock_manager = MockPipelineManager.return_value
            mock_manager._steps = []   # reset steps
            # Create a MagicMock that returns the output value directly
            output_value = "Pipeline Output"
                
            # Assign the mock pipeline to the manager
            # Create a function that returns the SimpleResult directly
            class SimpleResult:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
                            
            def pipeline_function(**kwargs):
                return SimpleResult(output_2=output_value)
            mock_manager.assemble_pipeline.return_value = pipeline_function
        
            # Call the main function
            main()
        
            # Capture the output
            captured = capsys.readouterr()
            # Strip newlines for exact matching
            output_lines = captured.out.strip().split('\n')
            # For single output, the output should be exactly the string
            assert output_value in output_lines
            # Also check it didn't output MagicMock representation
            for line in output_lines:
                assert "MagicMock" not in line
            # Verify pipeline was created and executed
            mock_manager.register_step.assert_called()
            mock_manager.assemble_pipeline.assert_called_once()
