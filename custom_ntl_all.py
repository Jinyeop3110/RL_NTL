"""
Custom NTL reward function for GSM8K with exact_match × exp(-ntl_loss) scoring.
Uses NTL loss calculated over ALL digit tokens in the sequence.
Now receives full digit log probabilities tensor for advanced NTL computations.
"""

import re
import math
import numpy as np
import torch
from typing import Dict, Any, Optional


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    NTL-based scoring function using exact_match × exp(-ntl_loss/Tau) for reward.
    Calculates NTL loss over ALL digit tokens in the sequence.
    
    Args:
        data_source: Data source identifier
        solution_str: Generated solution string
        ground_truth: Correct answer from dataset
        extra_info: Dictionary containing NTL digit information
        **kwargs: Additional arguments containing ntl_info, tau, and ntl_method
            - ntl_info: Dict with 'digit_log_probs' tensor [seq_len, 10]
            - tau: Temperature parameter for exp(-ntl_loss/tau) (default: 1.0)
            - ntl_method: 'mse' or 'wasserstein' (default: 'mse')
    
    Expected ntl_info structure:
        - 'digit_log_probs': Tensor [seq_len, 10] - log probs for digits 0-9
        - 'digit_ground_truth_tensor': Tensor [seq_len, 10] - one-hot ground truth
        - Additional metadata
    
    Returns:
        Dict containing:
        - 'score': NTL-based reward score: 0.5 * exact_match + 0.5 * exp(-ntl_loss/Tau)
        - 'acc': Binary accuracy/exact match score (1.0 or 0.0)
        - 'exact_match': Binary exact match score (1.0 or 0.0)
        - 'ntl_loss': The NTL loss value computed from ALL digit positions
        - 'ntl_bonus': The exp(-ntl_loss/Tau) value
        - 'pred': Extracted answer string
    """
    # Extract NTL information - now expecting digit_log_probs tensor
    ntl_info = kwargs.get('ntl_info', {})
    tau = kwargs.get('tau', 1.0)  # Temperature parameter
    ntl_method = kwargs.get('ntl_method', 'mse')  # NTL loss method
    
    # Calculate exact match score
    answer = extract_solution(solution_str, "strict")  # Use strict mode for fair comparison
    if answer is None:
        exact_match = 0.0
    else:
        exact_match = 1.0 if answer == ground_truth else 0.0
    
    # Calculate NTL-based reward using the digit_log_probs tensor for ALL digits
    ntl_loss = float('inf')
    ntl_bonus = 0.0
    
    # Process the digit_log_probs tensor [seq_len, 10] for ALL digit positions
    if ntl_info and 'digit_log_probs' in ntl_info and 'digit_ground_truth_tensor' in ntl_info:
        try:
            # The reward function now has access to tensors for this specific sample
            # digit_log_probs: [seq_len, 10] - log probs for digits 0-9 at each position
            # digit_ground_truth_tensor: [seq_len, 10] - one-hot ground truth
            digit_log_probs = ntl_info['digit_log_probs']
            digit_ground_truth_tensor = ntl_info['digit_ground_truth_tensor']
            
            # Convert to tensors if they're numpy arrays
            if isinstance(digit_log_probs, np.ndarray):
                digit_log_probs = torch.from_numpy(digit_log_probs)
            if isinstance(digit_ground_truth_tensor, np.ndarray):
                digit_ground_truth_tensor = torch.from_numpy(digit_ground_truth_tensor)
            
            # Find positions where we have ground truth digits (where any digit is 1)
            digit_mask = digit_ground_truth_tensor.sum(dim=-1) > 0  # [seq_len] - True where digits exist
            
            # Calculate NTL loss over ALL digit positions in the sequence
            if digit_mask.sum() > 0:  # If we have any digits
                # Extract log probs and ground truth only at digit positions
                digit_positions_log_probs = digit_log_probs[digit_mask]  # [num_digits, 10]
                digit_positions_ground_truth = digit_ground_truth_tensor[digit_mask]  # [num_digits, 10]
                
                if ntl_method == 'mse':
                    # MSE method: Calculate NTL loss for each digit position: -log(P(true_digit))
                    # For each position, get the log prob of the true digit (where ground truth = 1)
                    digit_losses = -(digit_positions_log_probs * digit_positions_ground_truth).sum(dim=-1)  # [num_digits]
                    
                    # Mean NTL loss over all digit positions
                    ntl_loss = digit_losses.mean().item()
                    
                elif ntl_method == 'wasserstein':
                    # Wasserstein method: Compute Wasserstein-1 distance
                    # Convert log probs to probs
                    digit_probs = torch.exp(digit_positions_log_probs)  # [num_digits, 10]
                    
                    # Compute CDFs
                    pred_cdfs = torch.cumsum(digit_probs, dim=-1)  # [num_digits, 10]
                    true_cdfs = torch.cumsum(digit_positions_ground_truth, dim=-1)  # [num_digits, 10]
                    
                    # Wasserstein-1 distance for each digit
                    w1_dists = torch.abs(pred_cdfs - true_cdfs).sum(dim=-1)  # [num_digits]
                    
                    # Mean Wasserstein distance over all digit positions
                    ntl_loss = w1_dists.mean().item()
                else:
                    raise ValueError(f"Unknown ntl_method: {ntl_method}")
                
        except Exception as e:
            ntl_loss = float('inf')
    
    if ntl_loss < float('inf'):
        ntl_bonus = math.exp(-ntl_loss / tau)  # exp(-loss/Tau) bonus
    else:
        ntl_bonus = 0.0  # Default to 0.0 if no NTL info (so NTL contributes nothing)
    
    # Linear combination: 0.5 * exact_match + 0.5 * exp(-ntl_loss/Tau)
    # This ensures reward is always between 0 and 1
    final_reward = 0.5 * exact_match + 0.5 * ntl_bonus
    
    # Return multiple metrics
    return {
        'score': final_reward,  # Main reward used for PPO
        'acc': exact_match,     # Binary accuracy based on exact match (for val-core)
        'exact_match': exact_match,  # Keep exact match as separate metric
        'ntl_loss': ntl_loss if ntl_loss < float('inf') else -1.0,
        'ntl_bonus': ntl_bonus,
        'pred': answer if answer is not None else "",
    }


def extract_solution(solution_str, method="flexible"):
    """
    Extract the final answer from the solution string.
    Enhanced to handle both #### format and \\boxed{} format.
    """
    assert method in ["strict", "flexible"]

    if method == "strict":
        # Original strict mode - only looks for #### format
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    else:
        # Flexible mode - handles multiple formats
        final_answer = None
        
        # First try #### format
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution:
            final_answer = solution.group(1).replace(",", "").replace("$", "")
        else:
            # Try \boxed{} format
            boxed_match = re.search(r"\\boxed\{(\\-?[0-9\\.\\,]+)\}", solution_str)
            if boxed_match:
                final_answer = boxed_match.group(1).replace(",", "").replace("$", "")
            else:
                # Try $\boxed{...}$ format
                dollar_boxed = re.search(r"\$\\boxed\{(\\-?[0-9\\.\\,]+)\}\$", solution_str)
                if dollar_boxed:
                    final_answer = dollar_boxed.group(1).replace(",", "").replace("$", "")
                else:
                    # Try \(...\) format
                    paren_boxed = re.search(r"\\\(\\boxed\{(\-?[0-9\.\,]+)\}\\\)", solution_str)
                    if paren_boxed:
                        final_answer = paren_boxed.group(1).replace(",", "").replace("$", "")
                    else:
                        # Try bold markdown format **$X** or **X**
                        bold_match = re.search(r"\*\*\$?([0-9\.\,]+)\*\*", solution_str)
                        if bold_match:
                            final_answer = bold_match.group(1).replace(",", "")
                        else:
                            # Try finding numbers at the end of sentences  
                            end_number = re.search(r"([0-9\.\,]+)\s*(?:dollars?|per week|total)?\s*\.$", solution_str.lower())
                            if end_number:
                                final_answer = end_number.group(1).replace(",", "")
                            else:
                                # Last resort: find any number after common answer indicators
                                indicators = ["answer is", "total is", "result is", "makes", "equals", "=", "is \\$"]
                                for indicator in indicators:
                                    if indicator in solution_str.lower():
                                        # Find the part after the indicator
                                        idx = solution_str.lower().rfind(indicator)
                                        remaining = solution_str[idx + len(indicator):]
                                        # Extract numbers from remaining text
                                        numbers = re.findall(r"(\\-?[0-9\\.\\,]+)", remaining)
                                        if numbers:
                                            # Filter out invalid strings
                                            invalid_str = ["", "."]
                                            for num in numbers:
                                                if num not in invalid_str:
                                                    final_answer = num.replace(",", "")
                                                    break
                                        if final_answer:
                                            break
    
    return final_answer


# ============================================================
# Example modifications you could make:
# ============================================================

def compute_score_with_partial_credit(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Example: Give partial credit for showing mathematical work.
    """
    base_result = compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    
    # If already has good score, return it
    if base_result['score'] >= 0.5:
        return base_result
    
    # Give partial credit for showing work (even if wrong answer)
    partial_credit = 0.0
    
    # Check for mathematical operations
    if any(op in solution_str for op in ['+', '-', '*', '/', '=']):
        partial_credit += 0.1
    
    # Check for step-by-step solution
    if "step" in solution_str.lower() or "first" in solution_str.lower():
        partial_credit += 0.05
    
    # Check for final answer attempt (even if wrong)
    if "#### " in solution_str or "\\boxed{" in solution_str:
        partial_credit += 0.05
    
    base_result['score'] += partial_credit
    return base_result


def compute_score_with_ntl_only_fallback(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Example: Use pure NTL reward when exact match fails.
    """
    # Try standard em_match × exp(-ntl_loss)
    result = compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    
    # If exact match failed but we have good NTL, give some reward
    if result['score'] == 0.0 and result['ntl_bonus'] > 0.0:
        # Give partial reward based on NTL quality alone
        result['score'] = 0.3 * result['ntl_bonus']  # 30% of full reward for NTL quality
    
    return result