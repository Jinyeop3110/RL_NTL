#!/usr/bin/env python3
"""
Test script to verify strict mode answer extraction in NTL reward functions.
Tests both custom_ntl_all.py and custom_ntl_final.py to ensure they only
recognize #### format answers, matching the default behavior.
"""

import sys
sys.path.append('/home/yeopjin/orcd/pool/workspace/RL_NTL')

from custom_ntl_all import compute_score as compute_score_all
from custom_ntl_final import compute_score as compute_score_final
from custom_default import compute_score as compute_score_default

# Test cases with various answer formats
test_cases = [
    {
        "name": "Standard #### format",
        "solution": "First, calculate 5 + 3 = 8. Then multiply by 2 to get 16. #### 16",
        "ground_truth": "16",
        "expected_match": True
    },
    {
        "name": "#### format with comma",
        "solution": "The total cost is 1000 + 200 = 1200. #### 1,200",
        "ground_truth": "1200",
        "expected_match": True
    },
    {
        "name": "#### format with dollar sign",
        "solution": "The price after discount is $50. #### $50",
        "ground_truth": "50",
        "expected_match": True
    },
    {
        "name": "Boxed format (should NOT match in strict mode)",
        "solution": "First, calculate 5 + 3 = 8. Then multiply by 2. \\boxed{16}",
        "ground_truth": "16",
        "expected_match": False
    },
    {
        "name": "Bold markdown format (should NOT match)",
        "solution": "The answer is **16**.",
        "ground_truth": "16",
        "expected_match": False
    },
    {
        "name": "Answer indicator format (should NOT match)",
        "solution": "After calculating, the answer is 16.",
        "ground_truth": "16",
        "expected_match": False
    },
    {
        "name": "No answer format",
        "solution": "I calculated something but didn't format the answer properly.",
        "ground_truth": "42",
        "expected_match": False
    },
    {
        "name": "Multiple #### answers (takes first)",
        "solution": "Wrong calculation gives #### 15. Actually, #### 16",
        "ground_truth": "15",
        "expected_match": True
    },
    {
        "name": "Negative number",
        "solution": "The temperature dropped to #### -5",
        "ground_truth": "-5",
        "expected_match": True
    },
    {
        "name": "Decimal number",
        "solution": "The average is #### 3.14",
        "ground_truth": "3.14",
        "expected_match": True
    }
]

def test_reward_function(compute_fn, fn_name):
    """Test a reward function with all test cases."""
    print(f"\n{'='*60}")
    print(f"Testing {fn_name}")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        # Call the compute function
        result = compute_fn(
            data_source="test",
            solution_str=test["solution"],
            ground_truth=test["ground_truth"],
            extra_info=None
        )
        
        # For NTL functions, extract exact_match from dict
        if isinstance(result, dict):
            exact_match = result.get('exact_match', 0.0)
            pred = result.get('pred', '')
        else:
            # For default function, result is the score itself
            exact_match = result
            pred = "N/A"
        
        # Check if match status is as expected
        match_found = exact_match == 1.0
        success = match_found == test["expected_match"]
        
        status = "✓ PASS" if success else "✗ FAIL"
        if success:
            passed += 1
        else:
            failed += 1
            
        print(f"\n{status} - {test['name']}")
        print(f"  Solution: {test['solution'][:60]}...")
        print(f"  Ground truth: {test['ground_truth']}")
        print(f"  Expected match: {test['expected_match']}")
        print(f"  Actual match: {match_found}")
        print(f"  Extracted answer: {pred}")
    
    print(f"\n{fn_name} Summary: {passed} passed, {failed} failed")
    return passed, failed

# Run tests
print("Testing strict mode implementation in NTL reward functions")
print("This ensures NTL versions only recognize #### format like the default")

# Test all three functions
all_passed = 0
all_failed = 0

# Test default function (baseline)
p, f = test_reward_function(compute_score_default, "custom_default.py")
all_passed += p
all_failed += f

# Test NTL all function
p, f = test_reward_function(compute_score_all, "custom_ntl_all.py")
all_passed += p
all_failed += f

# Test NTL final function  
p, f = test_reward_function(compute_score_final, "custom_ntl_final.py")
all_passed += p
all_failed += f

print(f"\n{'='*60}")
print(f"OVERALL SUMMARY: {all_passed} passed, {all_failed} failed")
print(f"{'='*60}")

if all_failed == 0:
    print("\n✓ All tests passed! NTL versions now use strict mode matching default behavior.")
else:
    print("\n✗ Some tests failed. Please check the implementation.")
    sys.exit(1)