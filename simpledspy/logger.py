"""Logging system for DSPy inputs and outputs

Features:
- Logs inputs and outputs to JSONL files organized by module
- Supports easy data collection for optimization datasets
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts non-serializable objects to strings"""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class Logger:
    """Handles logging of DSPy inputs and outputs per module"""
    
    def __init__(self, module_name: str) -> None:
        # Create base directory
        self.base_dir = Path(".simpledspy") / "modules" / module_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create files for training and logged data
        self.training_file = self.base_dir / "training.jsonl"
        self.logged_file = self.base_dir / "logged.jsonl"
        
        # Create files if they don't exist
        for file in [self.training_file, self.logged_file]:
            if not file.exists():
                file.touch()

    def log_to_section(self, data: Dict[str, Any], section: str = "logged") -> None:
        """Log a dictionary to the specified section
        
        Args:
            data: Dictionary of data to log
            section: Either 'training' or 'logged'
        """
        data['timestamp'] = time.time()
        target_file = self.training_file if section == "training" else self.logged_file
        
        with open(target_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, cls=CustomJSONEncoder) + "\n")

    def log(self, data: Dict[str, Any]) -> None:
        """Log data to the 'logged' section by default"""
        self.log_to_section(data, section="logged")
