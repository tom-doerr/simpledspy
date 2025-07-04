"""Logging system for DSPy inputs and outputs

Features:
- Logs inputs and outputs to JSONL files organized by module
- Supports easy data collection for optimization datasets
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts non-serializable objects to strings"""

    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            return str(o)


class Logger:
    """Handles logging of DSPy inputs and outputs per module"""

    def __init__(self, module_name: str, base_dir: str = ".simpledspy") -> None:
        # Sanitize module_name to prevent path traversal
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', module_name)
        safe_name = safe_name.replace('..', '_')
        
        # Create base directory
        self.base_dir = Path(base_dir) / "modules" / safe_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create files for training and logged data
        self.training_file = self.base_dir / "training.jsonl"
        self.logged_file = self.base_dir / "logged.jsonl"

        # Create files if they don't exist
        for file in [self.training_file, self.logged_file]:
            if not file.exists():
                file.touch()

    def load_training_data(self) -> list:
        """Load training data from the training file, skipping empty lines and invalid JSON"""
        examples = []
        if self.training_file.exists():
            with open(self.training_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Include any valid JSON in training file
                        examples.append(data)
                    except json.JSONDecodeError as e:
                        # Log the error but continue processing other lines
                        print(
                            f"Warning: Skipping invalid JSON line in {self.training_file}: {e}"
                        )
                        continue
        return examples

    def log_to_section(self, data: Dict[str, Any], section: str = "logged") -> None:
        """Log a dictionary to the specified section

        Args:
            data: Dictionary of data to log
            section: Either 'training' or 'logged'
        """
        # Add section marker
        data["section"] = section
        # Use ISO 8601 format with 'T' separator
        data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        target_file = self.training_file if section == "training" else self.logged_file

        with open(target_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, cls=CustomJSONEncoder) + "\n")

    def log(self, data: Dict[str, Any]) -> None:
        """Log data to the 'logged' section by default"""
        self.log_to_section(data, section="logged")
