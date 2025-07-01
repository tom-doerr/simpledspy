"""Test for training data loading functionality"""

import os
import tempfile
from unittest.mock import MagicMock, patch
import dspy
from simpledspy.logger import Logger
from simpledspy.module_caller import BaseCaller, Predict
from simpledspy.settings import settings


def test_training_data_loading(monkeypatch):
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
                "inputs": [{"name": "input1", "value": "test input"}],
                "outputs": [{"name": "output1", "value": "test output"}],
                "section": "training",
            },
            # Old format
            {"input1": "test input", "output1": "test output", "section": "training"},
        ]:
            logger.log_to_section(data, "training")

        # Create logged data (should be ignored)
        logged_data = {
            "inputs": [{"name": "input1", "value": "bad input"}],
            "outputs": [{"name": "output1", "value": "bad output"}],
            "section": "logged",
        }
        logger.log_to_section(logged_data, "logged")

        # Monkeypatch global_settings.log_dir to use tmpdir
        monkeypatch.setattr(settings, "log_dir", base_dir)

        # Track what demos are loaded
        loaded_demos = []

        def track_apply_training_data(self, module, trainset=None, name=None):
            """Track when _apply_training_data is called and capture demos"""
            # Call the original method first
            original_apply(self, module, trainset, name)
            # Then capture what was set
            if hasattr(module, "demos"):
                loaded_demos.extend(module.demos)

        # Store original method before patching
        original_apply = BaseCaller._apply_training_data

        # Patch both _create_module and _apply_training_data
        with patch.object(BaseCaller, "_create_module") as mock_create_module:
            mock_module_instance = MagicMock(spec=dspy.Module)
            mock_module_instance.demos = []
            mock_create_module.return_value = mock_module_instance

            with patch.object(
                BaseCaller, "_apply_training_data", track_apply_training_data
            ):
                # Clear singleton instance if it exists
                if Predict in BaseCaller._instances:
                    del BaseCaller._instances[Predict]

                # Instantiate Predict
                predict_instance = Predict()

                # Call predict to trigger demo loading. Module name must match logger's.
                # Inputs/outputs for the call don't strictly matter for demo loading test, but must be valid.
                try:
                    predict_instance(
                        "dummy_input",
                        inputs=["input1"],
                        outputs=["output1"],
                        name=module_name,
                    )
                except Exception as e:
                    # The call might fail if LM is not configured or if dummy input is bad,
                    # but demo loading should happen before LM interaction for this test's purpose.
                    # We are interested in loaded_demos captured by track_apply_training_data.
                    pass  # Allow progression to demo assertion

                # Check loaded demos
                assert len(loaded_demos) == 2  # Ensure only training data is loaded

                # Check content of loaded demos (both formats should be handled)
                # The _format_example method converts to dspy.Example
                # Example structure: dspy.Example(input1='test input', output1='test output')
                for demo in loaded_demos:
                    assert isinstance(demo, dspy.Example)
                    assert hasattr(demo, "input1")
                    assert hasattr(demo, "output1")
                    assert demo.input1 == "test input"
                    assert demo.output1 == "test output"
