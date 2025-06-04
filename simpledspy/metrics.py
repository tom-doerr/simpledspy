from typing import List, Any, Dict, Union, Tuple


def dict_exact_match_metric(example: Dict[str, Any], 
                           prediction: Union[Dict[str, Any], Tuple, Any], 
                           trace: Any = None) -> float:
    """Calculates exact match score between example and prediction
    
    Args:
        example: Ground truth dictionary
        prediction: Model prediction (dict, tuple, or single value)
        trace: Optional trace information (unused)
        
    Returns:
        Float score between 0.0 and 1.0
        
    Behavior:
    - Empty example: returns 1.0 if prediction empty, else 0.0
    - Non-dict predictions are normalized to dict format
    - Compares keys present in example dictionary
    - Scores: 1.0 for exact match, 0.0 for no match
    """
    # Handle empty example case
    if not example:
        return 1.0 if not prediction else 0.0
        
    # Normalize prediction to dict
    if not isinstance(prediction, dict):
        if isinstance(prediction, tuple):
            prediction = {f'output_{i}': val for i, val in enumerate(prediction)}
        else:
            prediction = {'output': prediction}
            
    # Calculate exact match score
    score = 0
    total = 0
    for key, value in example.items():
        total += 1
        if key in prediction and prediction[key] == value:
            score += 1
            
    return score / total if total > 0 else 0.0
