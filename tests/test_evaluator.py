"""Tests for evaluator.py"""
import pytest
import dspy
from unittest.mock import patch, MagicMock
from simpledspy.evaluator import Evaluator

@patch('dspy.LM')
def test_evaluator_init(mock_lm):
    """Test Evaluator initializes correctly"""
    evaluator = Evaluator(
        evaluation_instruction="Test instruction",
        log_file="test.log"
    )
        
    assert evaluator.evaluation_instruction == "Test instruction"
    assert mock_lm.called

def test_evaluate_no_instruction():
    """Test evaluate() returns 0 without instruction"""
    evaluator = Evaluator()
    score = evaluator.evaluate({}, {})
    assert score == 0

@patch('dspy.LM')
def test_evaluate_successful(mock_lm):
    """Test evaluate() returns proper score from LM response"""
    mock_instance = mock_lm.return_value
    mock_instance.return_value = ["10"]
    
    evaluator = Evaluator("Test instruction")
    score = evaluator.evaluate(
        {"question": "What is 2+2?"},
        {"answer": "4"}
    )
    assert score == 10

@patch('dspy.LM')
def test_evaluate_invalid_response(mock_lm):
    """Test evaluate() returns 0 for invalid responses"""
    mock_instance = mock_lm.return_value
    mock_instance.return_value = ["invalid value"]
    
    evaluator = Evaluator("Test instruction")
    score = evaluator.evaluate({}, {})
    assert score == 0

@patch('dspy.LM')
@patch('simpledspy.evaluator.Logger')
def test_log_with_evaluation(mock_logger, mock_lm):
    """Test log_with_evaluation() calls get_score and logs properly"""
    mock_lm.return_value.return_value = ["9"]
        
    evaluator = Evaluator("Test instruction")
    evaluator.log_with_evaluation(
        module="test_module",
        inputs={"in": "test"},
        outputs={"out": "test"},
        description="test_desc"
    )
        
    # Check log data contains proper fields
    args, kwargs = mock_logger.return_value.log.call_args
    log_data = args[0]
        
    assert log_data["module"] == "test_module"
    assert log_data["inputs"] == {"in": "test"}
    assert log_data["outputs"] == {"out": "test"}
    assert log_data["description"] == "test_desc"
    assert log_data["score"] == 9
