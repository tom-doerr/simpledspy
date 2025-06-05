"""Advice Generator for DSPy modules

Generates optimization advice from positive/negative examples
"""

import dspy
from typing import List, Dict, Any

class AdviceGenerator(dspy.Module):
    """Generates advice from positive/negative examples"""
    def __init__(self):
        super().__init__()
        self.generate_advice = dspy.ChainOfThought("examples, example_types, impacts -> advice")
        
    def forward(self, examples: List[Dict[str, Any]]) -> str:
        """Generate advice from examples
        
        Args:
            examples: List of example dictionaries with 'type', 'impact', and 'example'
            
        Returns:
            Generated advice string
        """
        # Format examples for the prompt
        formatted_examples = []
        for ex in examples:
            formatted_examples.append(
                f"- Example ({ex['type']}, impact={ex['impact']:.2f}): {ex['example']}"
            )
        
        # Generate advice
        result = self.generate_advice(
            examples="\n".join(formatted_examples),
            example_types=", ".join(set(ex['type'] for ex in examples)),
            impacts=", ".join(f"{ex['impact']:.2f}" for ex in examples)
        )
        return result.advice
