"""Evaluation system with 1-10 scoring

Features:
- Scores outputs using LLM evaluation with custom instructions
- Logs inputs/outputs with scores
"""

import re
import time
import dspy
from typing import Dict, List
from .logger import Logger

class Evaluator:
    """Evaluates DSPy module outputs"""
    def __init__(self, evaluation_instruction: str = "", log_file: str = "dspy_logs.jsonl"):
        self.evaluation_instruction = evaluation_instruction
        self.logger = Logger(log_file)
        self.evaluator_lm = dspy.LM(model="deepseek/deepseek-reasoner")
        dspy.configure(lm=self.evaluator_lm)

    def evaluate(self, inputs: Dict, outputs: Dict, evaluation_instructions: List[str] = None) -> float:
        """Evaluate outputs on a scale of 1-10 using multiple instructions"""
        if evaluation_instructions is None:
            evaluation_instructions = []
        
        if not evaluation_instructions and not self.evaluation_instruction:
            return 0.0  # No evaluation without instructions
        
        # Use instance instruction if no specific instructions provided
        if not evaluation_instructions:
            evaluation_instructions = [self.evaluation_instruction]
            
        scores = []
        for instruction in evaluation_instructions:
            # Construct evaluation prompt
            prompt = f"{instruction}\n\nInputs: {inputs}\nOutputs: {outputs}"
            completions = self.evaluator_lm(prompt)
            if not completions:
                continue
                
            # Take the first completion
            response = completions[0]
            
            # Extract scores from response
            matches = re.findall(r'\b(\d+\.?\d*)\b', response)
            for match in matches:
                try:
                    score = float(match)
                    if 1 <= score <= 10:
                        scores.append(score)
                        break  # take first valid score
                except ValueError:
                    continue
                    
        return sum(scores) / len(scores) if scores else 0.0

    def log_with_evaluation(self, module: str, inputs: Dict, outputs: Dict, description: str = "", evaluation_instructions: List[str] = None):
        """Log inputs/outputs with evaluation score"""
        score = self.evaluate(inputs, outputs, evaluation_instructions)
        
        # Store individual instructions if provided
        if evaluation_instructions:
            instructions = evaluation_instructions
        else:
            instructions = [self.evaluation_instruction] if self.evaluation_instruction else []
        
        # Get current timestamp
        timestamp = time.time()
        
        # Log to file
        self.logger.log({
            'module': module,
            'inputs': inputs,
            'outputs': outputs,
            'description': description,
            'instructions': instructions,
            'score': score,
            'timestamp': timestamp
        })
