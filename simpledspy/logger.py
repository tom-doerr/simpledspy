"""Logging system for DSPy inputs and outputs

Features:
- Logs inputs and outputs to JSONL files organized by module
- Supports easy data collection for optimization datasets
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts non-serializable objects to strings"""
    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            return str(o)

class Logger:
    """Handles logging of DSPy inputs and outputs."""

    def __init__(self, module_name: Optional[str] = None,
                 base_dir: str = ".simpledspy",
                 log_file: Optional[str] = None) -> None:
        """Create a new logger.

        Args:
            module_name: Name of the module to create a logging directory for.
                Used when ``log_file`` is not provided.
            base_dir: Base directory for module logs.
            log_file: Optional direct path to a JSONL file. When provided, the
                logger will write to this file instead of creating the
                ``training.jsonl``/``logged.jsonl`` pair inside ``base_dir``.
        """

        if log_file:
            # Logging directly to a specified file. Only ``logged_file`` is
            # created and ``training_file`` is unused.
            self.logged_file = Path(log_file)
            self.logged_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.logged_file.exists():
                self.logged_file.touch()
            self.training_file = None
        else:
            # Default behaviour: create module specific directory with training
            # and logged files.
            if module_name is None:
                raise ValueError("module_name is required when log_file is not set")
            self.base_dir = Path(base_dir) / "modules" / module_name
            self.base_dir.mkdir(parents=True, exist_ok=True)

            self.training_file = self.base_dir / "training.jsonl"
            self.logged_file = self.base_dir / "logged.jsonl"

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
