"""Evaluation system with 1-10 scoring

Features:
- Scores outputs using LLM evaluation with custom instructions
- Logs inputs/outputs with scores
"""

import re
import time
from typing import Dict, List
import dspy
from .logger import Logger
from .settings import settings as global_settings


class Evaluator:
    """Evaluates DSPy module outputs

    Attributes:
        evaluation_instruction: Instruction for evaluation
        logger: Logger instance for recording evaluations
        evaluator_lm: Language model for evaluation
    """

    def __init__(
        self, evaluation_instruction: str = "", log_file: str = "dspy_logs.jsonl"
    ):
        self.evaluation_instruction = evaluation_instruction
        self.logger = Logger(log_file)
        # Use evaluator LM from settings if available, otherwise use default
        if global_settings.evaluator_lm:
            self.evaluator_lm = dspy.LM(model=global_settings.evaluator_lm)
        elif global_settings.lm:
            self.evaluator_lm = global_settings.lm
        else:
            self.evaluator_lm = dspy.LM(model="openai/gpt-3.5-turbo")
        dspy.configure(lm=self.evaluator_lm)

    def evaluate(
        self, inputs: Dict, outputs: Dict, evaluation_instructions: List[str] = None
    ) -> float:
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

            # Extract scores from response with improved pattern
            # Look for patterns like "score: 8", "8/10", "rating: 8", or just "8"
            score_patterns = [
                r"(?:score|rating)[:\s]*(\d+(?:\.\d+)?)",  # score: 8 or rating: 8
                r"(\d+(?:\.\d+)?)\s*/\s*10",  # 8/10 format
                r"\b(\d+(?:\.\d+)?)\b",  # standalone number
            ]

            score_found = False
            for pattern in score_patterns:
                if score_found:
                    break
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    try:
                        score = float(match)
                        if 1 <= score <= 10:
                            scores.append(score)
                            score_found = True
                            break  # take first valid score
                    except ValueError:
                        continue

        return sum(scores) / len(scores) if scores else 0.0

    def log_with_evaluation(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        module: str,
        inputs: Dict,
        outputs: Dict,
        description: str = "",
        evaluation_instructions: List[str] = None,
    ):
        """Log inputs/outputs with evaluation score"""
        score = self.evaluate(inputs, outputs, evaluation_instructions)

        # Store individual instructions if provided
        if evaluation_instructions:
            instructions = evaluation_instructions
        else:
            instructions = (
                [self.evaluation_instruction] if self.evaluation_instruction else []
            )

        # Get current timestamp
        timestamp = time.time()

        # Log to file - store as is (the logger will handle serialization)
        log_data = {
            "module": module,
            "inputs": inputs,
            "outputs": outputs,
            "description": description,
            "instructions": instructions,
            "score": score,
            "timestamp": timestamp,
        }
        self.logger.log(log_data)
