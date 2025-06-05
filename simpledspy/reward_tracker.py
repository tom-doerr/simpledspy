import numpy as np
from typing import List, Dict, Any

class RewardTracker:
    """Tracks cumulative discounted rewards over time"""
    def __init__(self, discount_factor: float = 0.9):
        self.discount_factor = discount_factor
        self.reward_history = {}
        
    def add_reward(self, reward_group: str, score: float, timestamp: float):
        """Add a new reward to the history"""
        if reward_group not in self.reward_history:
            self.reward_history[reward_group] = []
        self.reward_history[reward_group].append((score, timestamp))
        
    def get_cumulative_reward(self, reward_group: str) -> float:
        """Calculate cumulative discounted reward for a group"""
        if reward_group not in self.reward_history:
            return 0.0
            
        rewards = self.reward_history[reward_group]
        if not rewards:
            return 0.0
            
        # Sort rewards by timestamp
        rewards.sort(key=lambda x: x[1])
        
        # Calculate discounted cumulative reward
        cumulative = 0.0
        for i, (score, _) in enumerate(rewards):
            cumulative += score * (self.discount_factor ** i)
            
        return cumulative

    def get_advice_examples(self, reward_group: str, n_positive: int = 3, n_negative: int = 2) -> List[Dict[str, Any]]:
        """Get examples for advice generation"""
        if reward_group not in self.reward_history:
            return []
            
        # Sort examples by impact (score * time discount)
        sorted_examples = sorted(
            self.reward_history[reward_group],
            key=lambda x: x[0] * (self.discount_factor ** x[1]),
            reverse=True
        )
        
        # Get top positive and bottom negative examples
        positive = sorted_examples[:n_positive]
        negative = sorted_examples[-n_negative:] if len(sorted_examples) > n_negative else []
        
        return [
            {"example": ex, "type": "positive", "impact": ex[0] * (self.discount_factor ** ex[1])}
            for ex in positive
        ] + [
            {"example": ex, "type": "negative", "impact": ex[0] * (self.discount_factor ** ex[1])}
            for ex in negative
        ]
