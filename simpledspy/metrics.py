"""Metrics for DSPy modules"""

from typing import Any

def dict_exact_match_metric(example: dict, prediction: dict, _: Any = None) -> float:
    """Calculates exact match score between example and prediction"""
    if not example:
        return 1.0 if not prediction else 0.0
        
    if not isinstance(prediction, dict):
        if isinstance(prediction, tuple):
            prediction = {f'output_{i}': val for i, val in enumerate(prediction)}
        else:
            prediction = {'output': prediction}
            
    score = 0
    total = 0
    for key, value in example.items():
        total += 1
        if key in prediction and prediction[key] == value:
            score += 1
            
    return score / total if total > 0 else 0.0
