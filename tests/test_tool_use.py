"""Tests for tool_use.py"""
import pytest
from simpledspy.tool_use import ToolUseModule
import json
from unittest.mock import MagicMock

def test_tool_use_success():
    """Test successful tool use"""
    def test_tool(arg1):
        return arg1 * 2
        
    module = ToolUseModule(test_tool)
    result = module("test_tool", '{"arg1": 5}')
    assert result.result == 10

def test_tool_use_retry():
    """Test tool use with retry"""
    mock_tool = MagicMock()
    mock_tool.side_effect = [ValueError, "success"]
    mock_tool.__name__ = "test_tool"
    
    module = ToolUseModule(mock_tool, max_retries=3)
    result = module("test_tool", '{"arg1": 5}')
    assert result.result == "success"
    assert mock_tool.call_count == 2

def test_tool_not_found():
    """Test handling of unknown tool"""
    module = ToolUseModule([])
    with pytest.raises(ValueError):
        module("unknown_tool", "{}")
