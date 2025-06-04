"""Evaluation system with 1-10 scoring

Features:
- Scores outputs using LLM evaluation with custom instructions
- Logs inputs/outputs with scores
- Provides a reward function that can be called anywhere
- Supports cumulative reward tracking for optimization
"""
import dspy
from typing import Dict, Any
from .logger import Logger

class Evaluator:
    def __init__(self, evaluation_instruction: str = "", log_file: str = "dspy_logs.jsonl"):
        self.evaluation_instruction = evaluation_instruction
        self.logger = Logger(log_file)
        self.evaluator_lm = dspy.LM(model="deepseek/deepseek-reasoner")
        dspy.configure(lm=self.evaluator_lm)

    def evaluate(self, inputs: Dict, outputs: Dict) -> int:
        """Evaluate outputs on a scale of 1-10"""
        if not self.evaluation_instruction:
            return 0  # No evaluation without instruction
        
        # Construct evaluation prompt
        prompt = f"{self.evaluation_instruction}\n\nInputs: {inputs}\nOutputs: {outputs}"
        completions = self.evaluator_lm(prompt)
        if not completions:
            return 0
            
        # Take the first completion
        response = completions[0]
        
        try:
            # Extract numerical score from response
            score = int(response.strip().split()[0])
            return max(1, min(10, score))
        except (ValueError, IndexError):
            return 0

    def log_with_evaluation(self, module: str, inputs: Dict, outputs: Dict, description: str = ""):
        """Log inputs/outputs with evaluation score"""
        score = self.evaluate(inputs, outputs)
        self.logger.log({
            'module': module,
            'inputs': inputs,
            'outputs': outputs,
            'description': description,
            'instruction': self.evaluation_instruction,
            'score': score
        })
