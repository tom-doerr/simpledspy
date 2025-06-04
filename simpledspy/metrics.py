from typing import List, Any, Dict, Union, Tuple

def exact_match_metric(gold: List[Any], pred: List[Any], trace=None) -> float:
    """
    Calculates the exact match accuracy between gold and predicted outputs.
    
    Args:
        gold (List[Any]): List of ground truth values.
        pred (List[Any]): List of predicted values.
        trace: Additional trace information (unused).
    
    Returns:
        float: Exact match accuracy.
    """
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return correct / len(gold)

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
