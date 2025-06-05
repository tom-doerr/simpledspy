"""Tests for tool_use.py"""
import pytest
from simpledspy.tool_use import ToolUseModule
import json
from unittest.mock import MagicMock
import dspy

def test_tool_use_success():
    """Test successful tool use"""
    # Configure mock LM
    mock_lm = MagicMock()
    dspy.configure(lm=mock_lm)
    
    def test_tool(arg1):
        return arg1 * 2
        
    module = ToolUseModule(test_tool)
    result = module("Run test_tool with arg1=5")
    assert result.result == 10

def test_tool_use_retry():
    """Test tool use with retry"""
    # Configure mock LM
    mock_lm = MagicMock()
    dspy.configure(lm=mock_lm)
    
    mock_tool = MagicMock()
    mock_tool.side_effect = [ValueError, "success"]
    mock_tool.__name__ = "test_tool"
    
    module = ToolUseModule(mock_tool, max_retries=3)
    result = module("Run test_tool with arg1=5")
    assert result.result == "success"
    assert mock_tool.call_count == 2

def test_tool_not_found():
    """Test handling of unknown tool"""
    # Configure mock LM
    mock_lm = MagicMock()
    dspy.configure(lm=mock_lm)
    
    module = ToolUseModule([])
    with pytest.raises(ValueError):
        module("Run unknown_tool with some arguments")
