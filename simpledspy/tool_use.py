"""Tool Use for DSPy modules

Enables DSPy modules to call Python functions and return their results as Python objects.
"""

import dspy
import json
import traceback
from typing import Callable, List, Dict, Any, Union

class ToolUseModule(dspy.Module):
    """DSPy module for using Python functions as tools"""
    def __init__(self, tools: Union[Callable, List[Callable]], max_retries: int = 3, reward_group: str = "tool_use"):
        super().__init__()
        if callable(tools):
            self.tools = [tools]
        else:
            self.tools = tools
        self.max_retries = max_retries
        self.reward_group = reward_group
        self.signature = self._create_signature()

    def _create_signature(self):
        """Create a DSPy signature for the tool use"""
        # Describe the tools for the prompt
        tool_descs = []
        for i, tool in enumerate(self.tools):
            tool_descs.append(f"{i+1}. {tool.__name__}: {tool.__doc__ or 'No description'}")
        tool_desc = "\n".join(tool_descs)

        class ToolSignature(dspy.Signature):
            instructions = f"Use one of the available tools to solve the problem. Available tools:\n{tool_desc}"
            problem = dspy.InputField(desc="The problem to solve")
            tool_name = dspy.OutputField(desc="The name of the tool to use")
            arguments = dspy.OutputField(desc="Arguments as a JSON string")

        return ToolSignature

    def forward(self, problem: str) -> Any:
        """Use tools to solve the problem"""
        # Predict which tool to use and with what arguments
        prediction = dspy.Predict(self.signature)(problem=problem)
        tool_name = prediction.tool_name
        arguments = prediction.arguments

        # Find the tool by name
        tool = None
        for t in self.tools:
            if t.__name__ == tool_name:
                tool = t
                break
        if tool is None:
            raise ValueError(f"Tool {tool_name} not found")

        # Try to parse the arguments and call the tool
        for attempt in range(self.max_retries):
            try:
                args_dict = json.loads(arguments)
                # If the tool expects a single argument, we pass it directly
                # Otherwise, we pass as keyword arguments
                result = tool(**args_dict)
                # If successful, return the result
                return dspy.Prediction(result=result)
            except Exception as e:
                # On failure, try to self-correct
                error = str(e)
                problem = f"Previous error: {error}. Problem: {problem}. Try again."
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return the error
                    return dspy.Prediction(error=error)

        # This should not be reached
        return dspy.Prediction(error="Tool use failed")
