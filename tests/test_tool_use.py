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

@patch('simpledspy.tool_use.dspy.Predict')
def test_tool_use_with_multiple_tools(mock_predict):
    """Test tool selection from multiple options"""
    # Mock DSPy Predict call
    mock_predict().return_value = MagicMock(
        tool_name='tool2',
        arguments='{"param":"value"}'
    )
    
    def tool1():
        return "wrong tool"
        
    def tool2(param):
        return f"called with {param}"
    
    module = ToolUseModule([tool1, tool2])
    result = module("Use tool2 with param=value")
    assert result.result == "called with value"

@patch('simpledspy.tool_use.dspy.Predict')
def test_tool_use_max_retries_exceeded(mock_predict):
    """Test tool use failure after max retries"""
    # Mock DSPy Predict call
    mock_predict().return_value = MagicMock(
        tool_name='failing_tool',
        arguments='{"arg":1}'
    )
    
    def failing_tool(arg):
        raise ValueError("Simulated failure")
    
    module = ToolUseModule(failing_tool, max_retries=2)
    result = module("Use failing tool")
    assert hasattr(result, 'error')
    assert "Simulated failure" in result.error

def test_tool_use_signature_creation():
    """Test signature creation with tool descriptions"""
    def tool1():
        """Tool1 does something"""
        pass
        
    def tool2(param: str):
        """Tool2 does something else"""
        pass
        
    module = ToolUseModule([tool1, tool2])
    signature = module.signature
    assert "Tool1 does something" in signature.__doc__
    assert "Tool2 does something else" in signature.__doc__
