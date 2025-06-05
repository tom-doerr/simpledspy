"""Reward Tracker for DSPy modules

Tracks cumulative discounted rewards over time and handles episode cutoffs
"""

from typing import List, Dict, Any

class RewardTracker:
    """Tracks cumulative discounted rewards over time"""
    def __init__(self, discount_factor: float = 0.9):
        self.discount_factor = discount_factor
        self.reward_history = {}
        self.episode_cutoffs = {}
        
    def add_reward(self, reward_group: str, score: float, timestamp: float):
        """Add a new reward to the history"""
        if reward_group not in self.reward_history:
            self.reward_history[reward_group] = []
        self.reward_history[reward_group].append((score, timestamp))
        
    def end_episode(self, reward_group: str):
        """Mark the end of an episode for a reward group"""
        self.episode_cutoffs[reward_group] = len(self.reward_history.get(reward_group, []))
        
    def get_cumulative_reward(self, reward_group: str) -> float:
        """Calculate cumulative discounted reward for a group"""
        if reward_group not in self.reward_history:
            return 0.0
            
        rewards = self.reward_history[reward_group]
        if not rewards:
            return 0.0
            
        # Get episode cutoff if set
        cutoff = self.episode_cutoffs.get(reward_group, len(rewards))
        episode_rewards = rewards[cutoff:]
        
        # Sort rewards by timestamp (oldest first)
        episode_rewards.sort(key=lambda x: x[1])
        
        # Calculate discounted cumulative reward from oldest to newest
        cumulative = 0.0
        for i, (score, _) in enumerate(reversed(episode_rewards)):
            cumulative += score * (self.discount_factor ** i)
            
        return cumulative

    def get_advice_examples(self, reward_group: str, n_positive: int = 3, n_negative: int = 2) -> List[Dict[str, Any]]:
        """Get examples for advice generation"""
        if reward_group not in self.reward_history:
            return []
            
        rewards = self.reward_history[reward_group]
        # Get episode cutoff if set
        cutoff = self.episode_cutoffs.get(reward_group, 0)
        episode_rewards = rewards[cutoff:]
        
        # Sort by timestamp (oldest first)
        episode_rewards.sort(key=lambda x: x[1])
        
        # Calculate impact as score * discount^(steps from end)
        n = len(episode_rewards)
        examples_with_impact = []
        for i, (score, timestamp) in enumerate(episode_rewards):
            # i=0 is oldest, i=n-1 is newest
            steps_from_end = n - 1 - i
            impact = score * (self.discount_factor ** steps_from_end)
            examples_with_impact.append((score, timestamp, impact))
            
        # Sort by impact descending
        examples_with_impact.sort(key=lambda x: x[2], reverse=True)
        
        # Get top positive and bottom negative examples
        positive = examples_with_impact[:n_positive]
        negative = examples_with_impact[-n_negative:] if n_negative > 0 else []
        
        return [
            {
                "example": f"Inputs: {entry[0]}, Outputs: {entry[1]}",
                "type": "positive",
                "impact": impact
            }
            for (score, timestamp, impact), entry in zip(positive, positive)
        ] + [
            {
                "example": f"Inputs: {entry[0]}, Outputs: {entry[1]}",
                "type": "negative",
                "impact": impact
            }
            for (score, timestamp, impact), entry in zip(negative, negative)
        ]
