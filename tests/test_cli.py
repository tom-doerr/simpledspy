"""Tests for the CLI interface"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simpledspy.cli import main


def test_cli_direct_input(capsys):
    """Test CLI with direct input argument"""
    # Mock the arguments
    with patch("sys.argv", ["cli.py", "Hello, world!", "-d", "extract the greeting"]):
        # Mock the ModuleFactory and module
        with patch("simpledspy.cli.ModuleFactory") as mock_factory_class:
            mock_factory = mock_factory_class.return_value
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
    with patch(
        "sys.argv",
        [
            "cli.py",
            "apple banana cherry",
            "orange grape",
            "-d",
            "extract the first word from each list",
        ],
    ):
        # Mock the ModuleFactory and module
        with patch("simpledspy.cli.ModuleFactory") as mock_factory_class:
            mock_factory = mock_factory_class.return_value
            mock_module = MagicMock()
            mock_factory.create_module.return_value = mock_module
            mock_module.return_value = MagicMock(output="apple\norange")

            # Call the main function
            main()

            # Capture the output
            captured = capsys.readouterr()
            assert "apple" in captured.out
            assert "orange" in captured.out


def test_cli_stdin(capsys):
    """Test CLI with stdin input"""
    # We mock sys.stdin.isatty and sys.stdin.read since we cannot easily mock stdin.
    with patch("sys.stdin.isatty", return_value=False):
        with patch("sys.stdin.read", return_value="Hello, world!"):
            with patch("sys.argv", ["cli.py", "-d", "extract the first word"]):
                # Mock the ModuleFactory and module
                with patch("simpledspy.cli.ModuleFactory") as mock_factory_class:
                    mock_factory = mock_factory_class.return_value
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
    with patch("sys.argv", ["cli.py", "--help"]):
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
    with patch(
        "sys.argv",
        ["cli.py", "Hello, world!", "-d", "extract the greeting", "--optimize"],
    ):
        # Mock dependencies
        with patch("simpledspy.cli.ModuleFactory") as mock_factory_class, patch(
            "simpledspy.optimization_manager.OptimizationManager"
        ) as mock_opt_manager_class:

            # Setup mock module
            mock_factory = mock_factory_class.return_value
            mock_module = MagicMock()
            mock_module._compiled = False  # pylint: disable=protected-access
            mock_module.reset_copy = lambda: mock_module
            mock_factory.create_module.return_value = mock_module
            mock_module.return_value = MagicMock(output="Optimized Hello")

            # Setup optimization manager
            mock_manager = mock_opt_manager_class.return_value
            mock_manager.optimize.return_value = mock_module

            # Call the main function
            main()

            # Capture the output
            captured = capsys.readouterr()
            assert "Optimized Hello" in captured.out


# Helpers for pipeline test
class _MockPrediction:  # pylint: disable=too-few-public-methods
    """Mock prediction object for pipeline testing."""

    def __init__(self, result_dict):
        for key, value in result_dict.items():
            setattr(self, key, value)


def _mock_step_module_forward(**kwargs):
    """Mock forward pass for pipeline steps."""
    if "input_1" in kwargs:
        return _MockPrediction({"output_1": "step0 output"})
    if "output_1" in kwargs:
        return _MockPrediction({"output_2": "Pipeline Output"})
    return _MockPrediction({})


class _SimpleResult:  # pylint: disable=too-few-public-methods
    """A simple result object for mock pipeline."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _pipeline_function(**_):
    """Mock pipeline function."""
    return _SimpleResult(output_2="Pipeline Output")


def test_cli_pipeline(capsys):
    """Test CLI pipeline execution"""
    argv = [
        "cli.py",
        "Hello, world!",
        "--pipeline",
        "Step 1 description",
        "Step 2 description",
    ]
    with patch("sys.argv", argv), patch(
        "simpledspy.cli.ModuleFactory"
    ) as mock_factory_class, patch(
        "simpledspy.cli.PipelineManager"
    ) as mock_pipeline_manager_class:

        # Setup mock module factory
        mock_factory = mock_factory_class.return_value
        mock_module = MagicMock()
        mock_module.forward.side_effect = _mock_step_module_forward
        mock_factory.create_module.return_value = mock_module

        # Setup mock pipeline and manager
        mock_manager = mock_pipeline_manager_class.return_value
        mock_manager.reset_mock()
        mock_manager.assemble_pipeline.return_value = _pipeline_function

        # Call the main function
        main()

        # Capture and assert output
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")
        assert "Pipeline Output" in output_lines
        for line in output_lines:
            assert "MagicMock" not in line

        # Verify pipeline was created and executed
        mock_manager.register_step.assert_called()
        mock_manager.assemble_pipeline.assert_called_once()
