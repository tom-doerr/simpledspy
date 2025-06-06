"""Logging system for DSPy inputs and outputs

Features:
- Logs inputs and outputs to JSONL files
- Supports easy data collection for optimization datasets
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

class Logger:
    """Handles logging of DSPy inputs and outputs"""
    
    def __init__(self, log_file: str = "dspy_logs.jsonl") -> None:
        self.log_file = Path(log_file)
        # Create parent directories if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Create the log file if it doesn't exist
        if not self.log_file.exists():
            self.log_file.touch()

    def log(self, data: Dict[str, Any]) -> None:
        """Log a dictionary as a JSON line
        
        Args:
            data: Dictionary of data to log
        """
        data['timestamp'] = time.time()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
