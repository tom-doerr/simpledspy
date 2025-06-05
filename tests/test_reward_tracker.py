"""Tests for reward_tracker.py"""
import pytest
import time
from simpledspy.reward_tracker import RewardTracker

def test_add_reward():
    """Test adding rewards to different groups"""
    tracker = RewardTracker()
    
    # Add rewards to group1
    tracker.add_reward("group1", 5.0, time.time())
    tracker.add_reward("group1", 7.0, time.time())
    
    # Add rewards to group2
    tracker.add_reward("group2", 3.0, time.time())
    
    # Verify reward counts
    assert len(tracker.reward_history["group1"]) == 2
    assert len(tracker.reward_history["group2"]) == 1

def test_cumulative_reward_no_episodes():
    """Test cumulative reward without episodes"""
    tracker = RewardTracker(discount_factor=0.9)
    
    # Add rewards with timestamps
    t1 = time.time()
    t2 = t1 + 1
    t3 = t2 + 1
    
    tracker.add_reward("group1", 5.0, t1)
    tracker.add_reward("group1", 7.0, t2)
    tracker.add_reward("group1", 3.0, t3)
    
    # Calculate cumulative reward
    cumulative = tracker.get_cumulative_reward("group1")
    expected = 3.0 + 7.0*0.9 + 5.0*(0.9**2)
    assert cumulative == pytest.approx(expected)

def test_cumulative_reward_with_episodes():
    """Test cumulative reward with episode cutoffs"""
    tracker = RewardTracker(discount_factor=0.9)
    
    # Add rewards to group1
    t1 = time.time()
    t2 = t1 + 1
    tracker.add_reward("group1", 5.0, t1)
    tracker.add_reward("group1", 7.0, t2)
    
    # End episode
    tracker.end_episode("group1")
    
    # Add more rewards
    t3 = t2 + 1
    t4 = t3 + 1
    tracker.add_reward("group1", 3.0, t3)
    tracker.add_reward("group1", 4.0, t4)
    
    # Calculate cumulative reward
    cumulative = tracker.get_cumulative_reward("group1")
    expected = 4.0 + 3.0*0.9
    assert cumulative == pytest.approx(expected)

def test_get_advice_examples():
    """Test getting advice examples"""
    tracker = RewardTracker(discount_factor=0.9)
    
    # Add rewards with timestamps
    t1 = time.time()
    t2 = t1 + 1
    t3 = t2 + 1
    
    tracker.add_reward("group1", 5.0, t1)
    tracker.add_reward("group1", 7.0, t2)
    tracker.add_reward("group1", 3.0, t3)
    
    # Get advice examples
    examples = tracker.get_advice_examples("group1", n_positive=2, n_negative=1)
    
    # Should have 2 positive and 1 negative examples
    assert len(examples) == 3
    assert sum(1 for ex in examples if ex["type"] == "positive") == 2
    assert sum(1 for ex in examples if ex["type"] == "negative") == 1
