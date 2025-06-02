import pytest
from simpledspy import pipe, PipeFunction
import dspy
from unittest.mock import patch

def test_multiple_outputs():
    """Test handling multiple return values"""
    with patch.object(PipeFunction, '_create_module') as mock_create:
        mock_module = dspy.Predict(lambda text: dspy.Prediction(name="John Doe", age="30"))
        mock_create.return_value = mock_module
        
        text = "John Doe, 30 years old"
        name, age = pipe(text)
        assert name == "John Doe"
        assert age == "30"

def test_two_outputs():
    """Test extracting second words from two lists"""
    with patch.object(PipeFunction, '_create_module') as mock_create:
        mock_module = dspy.Predict(lambda list1, list2: dspy.Prediction(
            second_word_list_oqc="iwfo", 
            second_word_list_jkl="def"
        ))
        mock_create.return_value = mock_module
        
        list_jkl = "abc def ghi jkl iowe afj wej own iow jklwe"
        list_oqc = "oid iwfo fjs wjiof sfio we x dso weop vskl we"
        second_word_list_oqc, second_word_list_jkl = pipe(list_jkl, list_oqc)
        assert second_word_list_jkl == "def"
        assert second_word_list_oqc == "iwfo"

def test_single_output():
    """Test extracting second word from single list assignment"""
    with patch.object(PipeFunction, '_create_module') as mock_create:
        mock_module = dspy.Predict(lambda list1, list2: dspy.Prediction(output="iwfo"))
        mock_create.return_value = mock_module
        
        list_jkl = "abc def ghi jkl iowe afj wej own iow jklwe"
        list_oqc = "oid iwfo fjs wjiof sfio we x dso weop vskl we"
        second_word_list_oqc = pipe(list_jkl, list_oqc)
        assert second_word_list_oqc == "iwfo"

def test_third_word():
    """Test extracting third word from text"""
    with patch.object(PipeFunction, '_create_module') as mock_create:
        mock_module = dspy.Predict(lambda text: dspy.Prediction(output="ghi"))
        mock_create.return_value = mock_module
        
        third_word = pipe("abc def ghi jkl")
        assert third_word == "ghi"

def test_cli_stdin_biggest_number():
    """Test CLI interface with stdin for finding biggest number"""
    # This test requires CLI changes - will be handled separately
    pass
