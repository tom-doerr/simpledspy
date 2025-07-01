"""Utilities for handling training data and demonstrations"""

from typing import List, Dict, Any
import dspy
from .logger import Logger
from .settings import settings as global_settings


class TrainingUtils:
    """Utilities for loading and preparing training data"""
    
    @staticmethod
    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a logged example into a DSPy Example compatible dictionary."""
        formatted = {}
        # New format with 'inputs' and 'outputs' keys
        if "inputs" in example and "outputs" in example:
            for item in example.get("inputs", []):
                formatted[item["name"]] = item["value"]
            for item in example.get("outputs", []):
                formatted[item["name"]] = item["value"]
        # Old format with top-level keys
        else:
            reserved = ["section", "timestamp", "module", "description"]
            for key, value in example.items():
                if key not in reserved:
                    formatted[key] = value
        return formatted
    
    @staticmethod
    def load_and_prepare_demos(name: str) -> List[dspy.Example]:
        """Load and prepare training demonstrations from a log file."""
        # Use log_dir from settings if available
        base_dir = global_settings.log_dir if global_settings.log_dir else ".simpledspy"
        logger = Logger(module_name=name, base_dir=base_dir)
        training_examples = logger.load_training_data()
        if not training_examples:
            return []

        demos = []
        for example in training_examples:
            try:
                formatted = TrainingUtils.format_example(example)
                if formatted:
                    demos.append(dspy.Example(**formatted))
            except (TypeError, KeyError):
                continue
        return demos
    
    @staticmethod
    def apply_training_data(
        module: dspy.Module, trainset: list = None, name: str = None
    ):
        """Apply training data to module"""
        # First priority: use explicitly passed trainset
        if trainset is not None:
            demos = []
            for example in trainset:
                if isinstance(example, dict):
                    demos.append(dspy.Example(**example))
                else:
                    demos.append(example)
            module.demos = demos
        # Second priority: load from training file
        elif name is not None:
            demos = TrainingUtils.load_and_prepare_demos(name)
            if demos:
                module.demos = demos
