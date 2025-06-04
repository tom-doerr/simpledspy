"""Logging system for DSPy inputs, outputs and evaluations

Provides:
- Logger: Handles writing log entries to JSONL files
- get_training_data: Extracts high-scoring examples for training
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List

class Logger:
    """Handles logging of DSPy inputs, outputs, and evaluations"""
    
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

    def get_training_data(self, min_score: float = 8.0) -> List[Dict[str, Any]]:
        """Extract training data from logs with high scores
        
        Args:
            min_score: Minimum score to include in training data
            
        Returns:
            List of training examples meeting score threshold
        """
        training_data = []
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if 'score' in entry and entry['score'] >= min_score:
                        training_data.append({
                            'inputs': entry['inputs'],
                            'outputs': entry['outputs'],
                            'instruction': entry.get('instruction', '')
                        })
        except FileNotFoundError:
            pass
        return training_data
