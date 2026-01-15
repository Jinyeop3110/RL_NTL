#!/usr/bin/env python3
"""
Test script to verify Wasserstein distance NTL calculation in reward functions.
"""

import sys
import math
import numpy as np
import torch

sys.path.append('/home/yeopjin/orcd/pool/workspace/RL_NTL')

from custom_ntl_all import compute_score as compute_score_all
from custom_ntl_final import compute_score as compute_score_final

def create_test_ntl_info(confidence_level='low'):
    """Create mock NTL info with different confidence levels."""
    seq_len = 20
    digit_log_probs = torch.zeros(seq_len, 10) - 10.0  # Very low log probs by default
    digit_ground_truth = torch.zeros(seq_len, 10)
    
    # Set some positions as digits (positions 5, 10, 15 have digits 4, 2, 8)
    digit_positions = [5, 10, 15]
    true_digits = [4, 2, 8]
    
    for i, (pos, true_digit) in enumerate(zip(digit_positions, true_digits)):
        digit_ground_truth[pos, true_digit] = 1.0
        
        if confidence_level == 'high':
            # High confidence: true digit gets high log prob
            digit_log_probs[pos, true_digit] = -0.1  # log(0.9) ≈ -0.1
            # Other digits get low log probs
            for d in range(10):
                if d != true_digit:
                    digit_log_probs[pos, d] = -3.0  # log(0.05) ≈ -3.0
                    
        elif confidence_level == 'medium':
            # Medium confidence: true digit gets medium log prob
            digit_log_probs[pos, true_digit] = -1.0  # log(0.37) ≈ -1.0
            # Other digits get lower log probs
            for d in range(10):
                if d != true_digit:
                    digit_log_probs[pos, d] = -2.0  # log(0.14) ≈ -2.0
                    
        else:  # low confidence
            # Low confidence: all digits have similar low log probs
            for d in range(10):
                digit_log_probs[pos, d] = -2.3  # log(0.1) ≈ -2.3
    
    return {
        'digit_log_probs': digit_log_probs,
        'digit_ground_truth_tensor': digit_ground_truth
    }

def test_ntl_methods():
    """Test both MSE and Wasserstein NTL methods."""
    
    print("Testing NTL Methods: MSE vs Wasserstein")
    print("="*60)
    
    test_cases = [
        {
            "name": "Correct answer with high confidence",
            "solution": "The answer is #### 42",
            "ground_truth": "42",
            "confidence": "high"
        },
        {
            "name": "Correct answer with medium confidence", 
            "solution": "The answer is #### 42",
            "ground_truth": "42",
            "confidence": "medium"
        },
        {
            "name": "Correct answer with low confidence",
            "solution": "The answer is #### 42",
            "ground_truth": "42", 
            "confidence": "low"
        },
        {
            "name": "Wrong answer with high confidence",
            "solution": "The answer is #### 40",
            "ground_truth": "42",
            "confidence": "high"
        }
    ]
    
    methods = ['mse', 'wasserstein']
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"  Solution: {test_case['solution']}")
        print(f"  Ground truth: {test_case['ground_truth']}")
        print(f"  Confidence: {test_case['confidence']}")
        
        # Create NTL info for this confidence level
        ntl_info = create_test_ntl_info(test_case['confidence'])
        
        for method in methods:
            # Test custom_ntl_all
            result_all = compute_score_all(
                data_source="test",
                solution_str=test_case['solution'],
                ground_truth=test_case['ground_truth'],
                ntl_info=ntl_info,
                tau=5.0,
                ntl_method=method
            )
            
            # Test custom_ntl_final  
            result_final = compute_score_final(
                data_source="test",
                solution_str=test_case['solution'],
                ground_truth=test_case['ground_truth'],
                ntl_info=ntl_info,
                tau=5.0,
                ntl_method=method
            )
            
            print(f"    {method.upper():12s} - ALL: loss={result_all['ntl_loss']:.4f}, "
                  f"bonus={result_all['ntl_bonus']:.4f}, score={result_all['score']:.4f}")
            print(f"    {' ':12s} - FINAL: loss={result_final['ntl_loss']:.4f}, "
                  f"bonus={result_final['ntl_bonus']:.4f}, score={result_final['score']:.4f}")
    
    print("\n" + "="*60)
    print("Key Differences:")
    print("- MSE method: Uses negative log-likelihood (cross-entropy)")
    print("- Wasserstein method: Uses Wasserstein-1 distance between distributions")
    print("- Wasserstein typically gives more gradual penalties for 'close' wrong predictions")
    print("- MSE gives sharp penalties for any wrong prediction")

def test_wasserstein_calculation():
    """Test Wasserstein distance calculation specifically."""
    print("\nTesting Wasserstein Distance Calculation:")
    print("="*50)
    
    # Create a simple test case
    seq_len = 5
    digit_log_probs = torch.zeros(seq_len, 10) - 100.0  # Very low default
    digit_ground_truth = torch.zeros(seq_len, 10)
    
    # Position 2 has digit 3
    pos, true_digit = 2, 3
    digit_ground_truth[pos, true_digit] = 1.0
    
    # Test case 1: Perfect prediction
    digit_log_probs[pos, :] = torch.log(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    
    ntl_info = {
        'digit_log_probs': digit_log_probs,
        'digit_ground_truth_tensor': digit_ground_truth
    }
    
    result = compute_score_all(
        data_source="test",
        solution_str="#### 42",
        ground_truth="42",
        ntl_info=ntl_info,
        tau=1.0,
        ntl_method='wasserstein'
    )
    
    print(f"Perfect prediction - Wasserstein loss: {result['ntl_loss']:.6f} (should be ~0)")
    
    # Test case 2: Uniform distribution (maximum uncertainty)
    digit_log_probs[pos, :] = torch.log(torch.tensor([0.1] * 10))
    
    ntl_info = {
        'digit_log_probs': digit_log_probs,
        'digit_ground_truth_tensor': digit_ground_truth
    }
    
    result = compute_score_all(
        data_source="test",
        solution_str="#### 42",
        ground_truth="42",
        ntl_info=ntl_info,
        tau=1.0,
        ntl_method='wasserstein'
    )
    
    print(f"Uniform prediction - Wasserstein loss: {result['ntl_loss']:.6f}")
    
    # Test case 3: Close prediction (digit 2 instead of 3)
    probs = torch.tensor([0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    digit_log_probs[pos, :] = torch.log(probs)
    
    ntl_info = {
        'digit_log_probs': digit_log_probs,
        'digit_ground_truth_tensor': digit_ground_truth
    }
    
    result = compute_score_all(
        data_source="test",
        solution_str="#### 42",
        ground_truth="42",
        ntl_info=ntl_info,
        tau=1.0,
        ntl_method='wasserstein'
    )
    
    print(f"Close prediction (2 vs 3) - Wasserstein loss: {result['ntl_loss']:.6f}")

if __name__ == "__main__":
    test_ntl_methods()
    test_wasserstein_calculation()