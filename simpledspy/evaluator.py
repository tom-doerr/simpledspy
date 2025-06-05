"""Evaluation system with 1-10 scoring

Features:
- Scores outputs using LLM evaluation with custom instructions
- Logs inputs/outputs with scores
- Provides a reward function that can be called anywhere
- Supports cumulative reward tracking for optimization
"""

import time
import dspy
from typing import Dict, List
from .logger import Logger
from .reward_tracker import RewardTracker
from .advice_generator import AdviceGenerator

class Evaluator:
    """Evaluates DSPy module outputs and tracks rewards"""
    def __init__(self, evaluation_instruction: str = "", reward_group: str = "default", log_file: str = "dspy_logs.jsonl"):
        self.evaluation_instruction = evaluation_instruction
        self.reward_group = reward_group
        self.logger = Logger(log_file)
        self.reward_tracker = RewardTracker()
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
            
            try:
                # Extract numerical score from response
                score = int(response.strip().split()[0])
                scores.append(max(1, min(10, score)))
            except (ValueError, IndexError):
                continue
                
        return sum(scores) / len(scores) if scores else 0.0
        
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

    def log_with_evaluation(self, module: str, inputs: Dict, outputs: Dict, description: str = "", reward_group: str = None, evaluation_instructions: List[str] = None):
        """Log inputs/outputs with evaluation score from multiple instructions"""
        score = self.evaluate(inputs, outputs, evaluation_instructions)
        group = reward_group or self.reward_group
        
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
            'reward_group': group,
            'timestamp': timestamp
        })
        
        # Track reward
        self.reward_tracker.add_reward(group, score, timestamp)

    def get_cumulative_reward(self, reward_group: str = None) -> float:
        """Get cumulative discounted reward for a group"""
        group = reward_group or self.reward_group
        return self.reward_tracker.get_cumulative_reward(group)
    
    def get_advice(self, reward_group: str = None) -> str:
        """Generate advice from reward history"""
        group = reward_group or self.reward_group
        examples = self.reward_tracker.get_advice_examples(group)
        generator = AdviceGenerator()
        return generator(examples)
