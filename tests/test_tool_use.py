"""Tests for tool_use.py"""
import pytest
from simpledspy.tool_use import ToolUseModule
import json
from unittest.mock import MagicMock, patch, ANY
import dspy

@patch('simpledspy.tool_use.dspy.Predict')
def test_tool_use_success(mock_predict):
    """Test successful tool use"""
    # Mock DSPy Predict call
    mock_predict().return_value = MagicMock(
        tool_name='test_tool',
        arguments='{"arg1":5}'
    )
    
    def test_tool(arg1):
        return arg1 * 2
        
    module = ToolUseModule(test_tool)
    result = module("Run test_tool with arg1=5")
    assert result.result == 10

@patch('simpledspy.tool_use.dspy.Predict')
def test_tool_use_retry(mock_predict):
    """Test tool use with retry"""
    # Mock DSPy Predict call
    mock_predict().return_value = MagicMock(
        tool_name='test_tool',
        arguments='{"arg1":5}'
    )
    
    mock_tool = MagicMock()
    mock_tool.side_effect = [ValueError, "success"]
    mock_tool.__name__ = "test_tool"
    
    module = ToolUseModule(mock_tool, max_retries=3)
    result = module("Run test_tool with arg1=5")
    assert result.result == "success"
    assert mock_tool.call_count == 2

@patch('simpledspy.tool_use.dspy.Predict')
def test_tool_not_found(mock_predict):
    """Test handling of unknown tool"""
    # Mock DSPy Predict call
    mock_predict().return_value = MagicMock(
        tool_name='unknown_tool',
        arguments='{}'
    )
    
    module = ToolUseModule([])
    with pytest.raises(ValueError):
        module("Run unknown_tool with some arguments")
