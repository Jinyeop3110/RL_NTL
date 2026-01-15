"""
Custom NTL reward function for GSM8K with pure exp(-ntl_loss) scoring.
Simplified to use only NTL loss for reward calculation.
"""

import re
import math
from typing import Dict, Any, Optional


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Pure NTL-based scoring function using exp(-ntl_loss) for reward.
    
    Args:
        data_source: Data source identifier
        solution_str: Generated solution string
        ground_truth: Correct answer
        extra_info: Dictionary containing NTL digit information
        **kwargs: Additional arguments containing ntl_info
    
    Returns:
        Reward score based solely on NTL loss: exp(-ntl_loss)
    """
    # Extract NTL information
    ntl_info = kwargs.get('ntl_info', {})
    
    if not ntl_info:
        return 0.0  # No reward without NTL info
    
    # Pure NTL loss-based reward
    ntl_loss = ntl_info.get('ntl_loss', float('inf'))
    if ntl_loss < float('inf'):
        return math.exp(-ntl_loss)  # Direct exp(-loss) reward
    else:
        return 0.0


def extract_solution(solution_str, method="strict"):
    """
    Extract the final answer from the solution string.
    """
    assert method in ["strict", "flexible"]

    if method == "strict":
        # Look for #### format
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        # Find last number in the response
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            pass
        else:
            invalid_str = ["", "."]
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


# ============================================================
# Example modifications you could make:
# ============================================================

def compute_score_with_partial_credit(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Example: Give partial credit for showing mathematical work.
    """
    base_score = compute_score(data_source, solution_str, ground_truth, extra_info)
    
    # If already correct, return full score
    if base_score >= 1.0:
        return base_score
    
    # Give partial credit for showing work
    partial_credit = 0.0
    
    # Check for mathematical operations
    if any(op in solution_str for op in ['+', '-', '*', '/', '=']):
        partial_credit += 0.2
    
    # Check for step-by-step solution
    if "step" in solution_str.lower() or "first" in solution_str.lower():
        partial_credit += 0.1
    
    # Check for final answer attempt (even if wrong)
    if "#### " in solution_str:
        partial_credit += 0.1
    
    return min(base_score + partial_credit, 1.0)


def compute_score_with_complexity_bonus(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Example: Give bonus for handling complex problems well.
    """
    base_score = compute_score(data_source, solution_str, ground_truth, extra_info)
    
    # If wrong, no bonus
    if base_score < 1.0:
        return base_score
    
    # Bonus for longer, more complex solutions
    complexity_bonus = 0.0
    
    # Count steps or operations
    num_operations = len(re.findall(r'[+\-*/=]', solution_str))
    if num_operations > 5:
        complexity_bonus = 0.1
    
    # Length bonus for detailed explanations
    if len(solution_str) > 200:
        complexity_bonus += 0.1
    
    # Cap at 1.5 maximum
    return min(base_score + complexity_bonus, 1.5)