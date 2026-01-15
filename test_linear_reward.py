#!/usr/bin/env python3
"""
Test script to verify linear reward calculation in NTL reward functions.
Tests that reward = 0.5 * exact_match + 0.5 * exp(-ntl_loss/tau)
"""

import sys
import math
import numpy as np
import torch

sys.path.append('/home/yeopjin/orcd/pool/workspace/RL_NTL')

from custom_ntl_all import compute_score as compute_score_all
from custom_ntl_final import compute_score as compute_score_final

# Test cases with mock NTL info
test_cases = [
    {
        "name": "Correct answer with high confidence (low NTL loss)",
        "solution": "The answer is #### 42",
        "ground_truth": "42",
        "ntl_loss": 0.1,  # Low loss = high confidence
        "tau": 1.0,
        "expected_exact_match": 1.0,
        "expected_ntl_bonus": math.exp(-0.1/1.0),  # ≈ 0.905
        "expected_reward": 0.5 * 1.0 + 0.5 * math.exp(-0.1/1.0)  # ≈ 0.952
    },
    {
        "name": "Correct answer with low confidence (high NTL loss)",
        "solution": "The answer is #### 42",
        "ground_truth": "42",
        "ntl_loss": 3.0,  # High loss = low confidence
        "tau": 1.0,
        "expected_exact_match": 1.0,
        "expected_ntl_bonus": math.exp(-3.0/1.0),  # ≈ 0.0498
        "expected_reward": 0.5 * 1.0 + 0.5 * math.exp(-3.0/1.0)  # ≈ 0.525
    },
    {
        "name": "Wrong answer with high confidence",
        "solution": "The answer is #### 40",
        "ground_truth": "42",
        "ntl_loss": 0.1,
        "tau": 1.0,
        "expected_exact_match": 0.0,
        "expected_ntl_bonus": math.exp(-0.1/1.0),  # ≈ 0.905
        "expected_reward": 0.5 * 0.0 + 0.5 * math.exp(-0.1/1.0)  # ≈ 0.452
    },
    {
        "name": "Wrong answer with low confidence",
        "solution": "The answer is #### 40",
        "ground_truth": "42",
        "ntl_loss": 3.0,
        "tau": 1.0,
        "expected_exact_match": 0.0,
        "expected_ntl_bonus": math.exp(-3.0/1.0),  # ≈ 0.0498
        "expected_reward": 0.5 * 0.0 + 0.5 * math.exp(-3.0/1.0)  # ≈ 0.025
    },
    {
        "name": "No NTL info available",
        "solution": "The answer is #### 42",
        "ground_truth": "42",
        "ntl_loss": float('inf'),
        "tau": 1.0,
        "expected_exact_match": 1.0,
        "expected_ntl_bonus": 0.0,  # No NTL info
        "expected_reward": 0.5 * 1.0 + 0.5 * 0.0  # = 0.5
    },
    {
        "name": "Different tau value",
        "solution": "The answer is #### 42",
        "ground_truth": "42",
        "ntl_loss": 2.0,
        "tau": 5.0,  # Higher tau = less penalty
        "expected_exact_match": 1.0,
        "expected_ntl_bonus": math.exp(-2.0/5.0),  # ≈ 0.670
        "expected_reward": 0.5 * 1.0 + 0.5 * math.exp(-2.0/5.0)  # ≈ 0.835
    }
]

def create_mock_ntl_info(ntl_loss):
    """Create mock NTL info that will produce the desired loss."""
    if ntl_loss == float('inf'):
        return {}
    
    # Create mock tensors that will produce the desired NTL loss
    seq_len = 10
    digit_log_probs = torch.zeros(seq_len, 10) - 10.0  # Very low log probs
    digit_ground_truth = torch.zeros(seq_len, 10)
    
    # Set some positions as digits
    digit_positions = [2, 5, 8]  # Mock digit positions
    for pos in digit_positions:
        digit_ground_truth[pos, 5] = 1.0  # Digit 5 is the "true" digit
        # Set log prob such that -log(P) = ntl_loss
        digit_log_probs[pos, 5] = -ntl_loss
    
    return {
        'digit_log_probs': digit_log_probs,
        'digit_ground_truth_tensor': digit_ground_truth
    }

def test_linear_reward():
    """Test the linear reward calculation."""
    print("Testing Linear Reward Calculation")
    print("Formula: reward = 0.5 * exact_match + 0.5 * exp(-ntl_loss/tau)")
    print("="*80)
    
    all_passed = True
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"  Solution: {test['solution']}")
        print(f"  Ground truth: {test['ground_truth']}")
        print(f"  NTL loss: {test['ntl_loss']}")
        print(f"  Tau: {test['tau']}")
        
        # Create mock NTL info
        ntl_info = create_mock_ntl_info(test['ntl_loss'])
        
        # Test both functions
        for compute_fn, fn_name in [(compute_score_all, "custom_ntl_all"), 
                                    (compute_score_final, "custom_ntl_final")]:
            
            result = compute_fn(
                data_source="test",
                solution_str=test['solution'],
                ground_truth=test['ground_truth'],
                ntl_info=ntl_info,
                tau=test['tau']
            )
            
            # Check exact match
            exact_match_ok = abs(result['exact_match'] - test['expected_exact_match']) < 0.001
            
            # Check NTL bonus
            ntl_bonus_ok = abs(result['ntl_bonus'] - test['expected_ntl_bonus']) < 0.01
            
            # Check final reward
            reward_ok = abs(result['score'] - test['expected_reward']) < 0.01
            
            # Verify the formula
            calculated_reward = 0.5 * result['exact_match'] + 0.5 * result['ntl_bonus']
            formula_ok = abs(calculated_reward - result['score']) < 0.001
            
            status = "✓" if (exact_match_ok and ntl_bonus_ok and reward_ok and formula_ok) else "✗"
            if status == "✗":
                all_passed = False
            
            print(f"\n  {status} {fn_name}:")
            print(f"    exact_match: {result['exact_match']:.3f} (expected: {test['expected_exact_match']:.3f})")
            print(f"    ntl_bonus: {result['ntl_bonus']:.3f} (expected: {test['expected_ntl_bonus']:.3f})")
            print(f"    score: {result['score']:.3f} (expected: {test['expected_reward']:.3f})")
            print(f"    Formula check: 0.5 * {result['exact_match']:.3f} + 0.5 * {result['ntl_bonus']:.3f} = {calculated_reward:.3f}")
            
            # Verify reward bounds
            if not (0.0 <= result['score'] <= 1.0):
                print(f"    ✗ ERROR: Reward {result['score']} is outside [0, 1] bounds!")
                all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ All tests passed! Linear reward calculation is working correctly.")
        print("  - Rewards are properly bounded between 0 and 1")
        print("  - Formula: reward = 0.5 * exact_match + 0.5 * exp(-ntl_loss/tau)")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    test_linear_reward()