"""
Custom NTL reward function for GSM8K with exact_match × exp(-ntl_loss) scoring.
Uses multiplicative scoring to only reward confident digit generation when the final answer is correct.
FINAL ANSWER FOCUS: Only computes NTL loss from digits in the final answer portion, not the entire sequence.
Fixed version: Handles both #### and \boxed{} answer formats.
"""

import re
import math
import numpy as np
import torch
from typing import Dict, Any, Optional


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    NTL-based scoring function using exact_match × exp(-ntl_loss/Tau) for reward.
    Now receives full digit log probabilities tensor for advanced NTL computations.
    
    Args:
        data_source: Data source identifier
        solution_str: Generated solution string
        ground_truth: Correct answer
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
        - 'ntl_loss': The NTL loss value computed from digit_log_probs tensor
        - 'ntl_reward': The 0.5 * exact_match + 0.5 * exp(-ntl_loss/Tau) reward value
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
    
    # Calculate NTL-based reward using the digit_log_probs tensor
    ntl_loss = float('inf')
    ntl_bonus = 0.0
    
    # Debug: Check what NTL info we received
    # Debug removed - was showing 'None' repeatedly
    
    # Process the digit_log_probs tensor [seq_len, 10] for this sample
    if ntl_info and 'digit_log_probs' in ntl_info and 'digit_ground_truth_tensor' in ntl_info:
        try:
            # The reward function now has access to tensors for this specific sample
            # digit_log_probs: [seq_len, 10] - log probs for digits 0-9 at each position
            # digit_ground_truth_tensor: [seq_len, 10] - one-hot ground truth
            digit_log_probs = ntl_info['digit_log_probs']  # [seq_len, 10]
            digit_ground_truth_tensor = ntl_info['digit_ground_truth_tensor']  # [seq_len, 10]
            
            # Convert to tensors if they're numpy arrays
            if isinstance(digit_log_probs, np.ndarray):
                digit_log_probs = torch.from_numpy(digit_log_probs)
            if isinstance(digit_ground_truth_tensor, np.ndarray):
                digit_ground_truth_tensor = torch.from_numpy(digit_ground_truth_tensor)
            
            # FINAL ANSWER FOCUS: Only process digits from the final answer portion
            # Find final answer pattern in the solution string 
            final_answer_start_idx = None
            
            # Look for common final answer indicators
            patterns = [
                (r"#### ", "#### "),
                (r"\\boxed\{", "\\boxed{"),
                (r"answer is ", "answer is "),
                (r"total is ", "total is "),
                (r"makes \$?", "makes")
            ]
            
            for pattern, indicator in patterns:
                match = re.search(pattern, solution_str.lower())
                if match:
                    # Estimate position in token sequence (rough approximation)
                    # This is approximate since we don't have exact token alignment
                    char_pos = match.start()
                    # Rough estimate: average 4 characters per token
                    final_answer_start_idx = max(0, int(char_pos / 4) - 10)  # Start a bit before for safety
                    break
            
            # If no pattern found, use last 20% of sequence (assume answer is at end)
            if final_answer_start_idx is None:
                seq_len = digit_log_probs.shape[0]
                final_answer_start_idx = int(seq_len * 0.8)  # Last 20% of sequence
            
            # Focus only on final answer portion
            final_answer_digit_log_probs = digit_log_probs[final_answer_start_idx:]  # [final_seq_len, 10]
            final_answer_ground_truth = digit_ground_truth_tensor[final_answer_start_idx:]  # [final_seq_len, 10]
            
            # Find positions where we have ground truth digits in final answer portion
            digit_mask = final_answer_ground_truth.sum(dim=-1) > 0  # [final_seq_len] - True where digits exist
            
            if digit_mask.sum() > 0:  # If we have any digits in final answer
                # Extract log probs and ground truth only at digit positions in final answer
                digit_positions_log_probs = final_answer_digit_log_probs[digit_mask]  # [num_final_digits, 10]
                digit_positions_ground_truth = final_answer_ground_truth[digit_mask]  # [num_final_digits, 10]
                
                if ntl_method == 'mse':
                    # MSE method: Convert log probs to probabilities
                    digit_probs = torch.exp(digit_positions_log_probs)  # [num_final_digits, 10]
                    
                    # Calculate confidence as the probability assigned to the true digit
                    # For each digit position, multiply probs with one-hot ground truth and sum
                    confidences = (digit_probs * digit_positions_ground_truth).sum(dim=-1)  # [num_final_digits]
                    
                    # Average confidence across final answer digit positions only
                    avg_confidence = confidences.mean().item()
                    
                    # Convert confidence to NTL loss (higher confidence = lower loss)
                    ntl_loss = -math.log(max(avg_confidence, 1e-8))
                    
                elif ntl_method == 'wasserstein':
                    # Wasserstein method: Compute Wasserstein-1 distance
                    # Convert log probs to probs
                    digit_probs = torch.exp(digit_positions_log_probs)  # [num_final_digits, 10]
                    
                    # Compute CDFs
                    pred_cdfs = torch.cumsum(digit_probs, dim=-1)  # [num_final_digits, 10]
                    true_cdfs = torch.cumsum(digit_positions_ground_truth, dim=-1)  # [num_final_digits, 10]
                    
                    # Wasserstein-1 distance for each digit
                    w1_dists = torch.abs(pred_cdfs - true_cdfs).sum(dim=-1)  # [num_final_digits]
                    
                    # Mean Wasserstein distance over final answer digit positions
                    ntl_loss = w1_dists.mean().item()
                else:
                    raise ValueError(f"Unknown ntl_method: {ntl_method}")
                
        except Exception as e:
            ntl_loss = float('inf')
    
    if ntl_loss < float('inf'):
        ntl_bonus = math.exp(-ntl_loss / tau)  # exp(-loss/Tau) bonus
    else:
        ntl_bonus = 0.0  # No bonus without valid NTL info
    
    # Linear combination: 0.5 * exact_match + 0.5 * exp(-ntl_loss/Tau)
    # This ensures reward is always between 0 and 1
    # Both correct answers and confident digit generation contribute to the reward
    ntl_reward = 0.5 * exact_match + 0.5 * ntl_bonus
    
    # Return multiple metrics
    return {
        'score': ntl_reward,  # Main reward used for PPO
        'acc': exact_match,   # Binary accuracy based on exact match (for val-core)
        'exact_match': exact_match,  # Keep exact match as separate metric
        'ntl_loss': ntl_loss if ntl_loss < float('inf') else -1.0,
        'ntl_bonus': ntl_bonus,
        'ntl_reward': ntl_reward,
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
    
    # If already has high NTL score, return it
    if base_result['score'] >= 0.5:
        return base_result
    
    # Give partial credit for showing work
    partial_credit = 0.0
    
    # Check for mathematical operations
    if any(op in solution_str for op in ['+', '-', '*', '/', '=']):
        partial_credit += 0.2
    
    # Check for step-by-step solution
    if "step" in solution_str.lower() or "first" in solution_str.lower():
        partial_credit += 0.1
    
    # Check for final answer attempt (even if wrong)
    if "#### " in solution_str or "\\boxed{" in solution_str:
        partial_credit += 0.1
    
    base_result['score'] = min(base_result['score'] + partial_credit, 1.0)
    return base_result


def compute_score_with_complexity_bonus(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Example: Give bonus for handling complex problems well.
    """
    base_result = compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    
    # If wrong answer, no bonus
    if base_result['acc'] < 1.0:
        return base_result
    
    # Bonus for longer, more complex solutions
    complexity_bonus = 0.0
    
    # Count steps or operations
    num_operations = len(re.findall(r'[+\-*/=]', solution_str))
    if num_operations > 5:
        complexity_bonus = 0.1
    
    # Length bonus for detailed explanations
    if len(solution_str) > 200:
        complexity_bonus += 0.1
    
    # Apply bonus to NTL reward
    base_result['score'] = min(base_result['score'] + complexity_bonus, 1.5)
    return base_result