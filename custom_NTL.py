"""
Custom NTL reward function for GSM8K with digit-level log probability integration.
Now enhanced to use Number Token Loss information for improved reward signals.
"""

import re
import torch
from typing import Dict, Any, Optional


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Enhanced NTL custom scoring function for GSM8K with digit-level information.
    
    Now incorporates:
    - Standard correctness check (1.0 for correct, 0.0 for wrong)
    - Optional NTL digit-level log probability bonus (final answer or all digits)
    - Digit accuracy rewards for partial credit
    
    Args:
        data_source: Data source identifier
        solution_str: Generated solution string
        ground_truth: Correct answer
        extra_info: Dictionary containing NTL digit information
        **kwargs: Additional arguments
            - use_ntl_bonus: Enable NTL bonus (default: True)
            - ntl_bonus_type: 'final_answer' or 'all' (default: 'all')
            - ntl_bonus_weight: Weight for NTL bonus (default: 0.1)
    """
    # Extract NTL information if available
    ntl_info = kwargs.get('ntl_info', {})
    use_ntl_bonus = kwargs.get('use_ntl_bonus', True)
    ntl_bonus_type = kwargs.get('ntl_bonus_type', 'all')  # 'final_answer' or 'all'
    ntl_bonus_weight = kwargs.get('ntl_bonus_weight', 0.1)
    
    # Configuration
    method = "strict"  # "strict" or "flexible"  
    base_score = 1.0  # Base reward for correct answer
    
    # Extract answer using standard method
    answer = extract_solution(solution_str, method)
    
    # Base correctness score
    if answer is None:
        correctness_score = 0.0
    else:
        correctness_score = base_score if answer == ground_truth else 0.0
    
    # If no NTL info available, return standard score
    if not ntl_info or not use_ntl_bonus:
        return correctness_score
    
    # Calculate NTL-based bonus based on type
    if ntl_bonus_type == 'final_answer':
        ntl_bonus = compute_ntl_bonus_final_answer(ntl_info, solution_str, ground_truth, correctness_score > 0)
    else:  # 'all'
        ntl_bonus = compute_ntl_bonus_all(ntl_info, correctness_score > 0)
    
    # Combine scores
    total_score = correctness_score + ntl_bonus_weight * ntl_bonus
    
    return total_score


def compute_ntl_bonus_all(ntl_info: Dict[str, Any], is_correct: bool) -> float:
    """
    Compute NTL-based bonus from ALL digit-level log probabilities in the sequence.
    
    This is the original NTL approach that analyzes all digit tokens throughout
    the entire generated sequence (intermediate calculations + final answer).
    
    Args:
        ntl_info: Dictionary containing NTL information
        is_correct: Whether the final answer is correct
        
    Returns:
        NTL bonus score (0.0 to 1.0)
    """
    if not ntl_info:
        return 0.0
    
    bonus = 0.0
    
    # Digit accuracy bonus (all digits in sequence)
    digit_accuracy = ntl_info.get('digit_accuracy', 0.0)
    if digit_accuracy > 0:
        bonus += 0.3 * digit_accuracy  # Up to 0.3 points for digit accuracy
    
    # NTL loss bonus (lower loss = better digit predictions)
    ntl_loss = ntl_info.get('ntl_loss', float('inf'))
    if ntl_loss < float('inf'):
        # Convert loss to bonus (exponential decay)
        loss_bonus = min(0.2, 0.2 * torch.exp(-ntl_loss).item())
        bonus += loss_bonus
    
    # High confidence bonus for correct predictions
    if is_correct and ntl_info.get('total_digits', 0) > 0:
        # Reward high confidence in digit predictions
        digit_log_probs = ntl_info.get('digit_log_probs', None)
        if digit_log_probs is not None:
            try:
                # Calculate average confidence for ALL digit tokens
                digit_positions = ntl_info.get('digit_positions', [])
                if digit_positions and any(digit_positions):
                    avg_confidence = 0.0
                    total_positions = 0
                    
                    for batch_positions in digit_positions:
                        for pos in batch_positions:
                            # Get confidence for this digit position
                            digit_probs = torch.softmax(digit_log_probs[0, pos, :], dim=0)
                            max_prob = torch.max(digit_probs).item()
                            avg_confidence += max_prob
                            total_positions += 1
                    
                    if total_positions > 0:
                        avg_confidence /= total_positions
                        confidence_bonus = 0.1 * (avg_confidence - 0.1)  # Bonus for >10% confidence
                        bonus += max(0.0, confidence_bonus)
                        
            except Exception as e:
                # Silently handle errors in confidence calculation
                pass
    
    return min(1.0, bonus)  # Cap bonus at 1.0


def compute_ntl_bonus_final_answer(ntl_info: Dict[str, Any], 
                                 solution_str: str, 
                                 ground_truth: str,
                                 is_correct: bool) -> float:
    """
    Compute NTL-based bonus focusing ONLY on final answer digits.
    
    This approach extracts the final answer from the solution string,
    identifies which digit positions correspond to the final answer,
    and computes NTL metrics only for those positions.
    
    Args:
        ntl_info: Dictionary containing NTL information
        solution_str: Full generated solution string
        ground_truth: Correct answer string
        is_correct: Whether the final answer is correct
        
    Returns:
        NTL bonus score (0.0 to 1.0)
    """
    if not ntl_info:
        return 0.0
    
    try:
        # Extract final answer from solution
        predicted_answer = extract_solution(solution_str, "strict")
        if predicted_answer is None:
            return 0.0
        
        # Get ground truth digits
        try:
            true_digits = [int(d) for d in str(ground_truth) if d.isdigit()]
            pred_digits = [int(d) for d in str(predicted_answer) if d.isdigit()]
        except:
            return 0.0
        
        if not true_digits or not pred_digits:
            return 0.0
        
        # Find final answer digit positions in the sequence
        final_answer_positions = find_final_answer_positions(
            solution_str, predicted_answer, ntl_info
        )
        
        if not final_answer_positions:
            return 0.0
        
        # Calculate NTL metrics only for final answer digits
        bonus = 0.0
        
        # 1. Digit-level accuracy for final answer only
        digit_accuracy = calculate_final_answer_digit_accuracy(
            final_answer_positions, true_digits, pred_digits, ntl_info
        )
        bonus += 0.4 * digit_accuracy  # Higher weight since we're focused
        
        # 2. Confidence bonus for final answer digits
        confidence_bonus = calculate_final_answer_confidence(
            final_answer_positions, ntl_info
        )
        bonus += 0.3 * confidence_bonus
        
        # 3. NTL probability quality for final answer
        if len(true_digits) == len(pred_digits):
            prob_quality = calculate_final_answer_prob_quality(
                final_answer_positions, true_digits, ntl_info
            )
            bonus += 0.3 * prob_quality
        
        return min(1.0, bonus)
        
    except Exception as e:
        # Fallback: if final answer analysis fails, return small general bonus
        return 0.1 * ntl_info.get('digit_accuracy', 0.0)


def find_final_answer_positions(solution_str: str, 
                               predicted_answer: str, 
                               ntl_info: Dict[str, Any]) -> list:
    """
    Find the positions in the sequence that correspond to the final answer digits.
    
    This looks for the "#### X" pattern and identifies which digit positions
    in the ntl_info correspond to the final answer.
    """
    if not predicted_answer or "#### " not in solution_str:
        return []
    
    try:
        # Find the "#### " pattern position in the text
        answer_start_text = solution_str.find("#### ")
        if answer_start_text == -1:
            return []
        
        # Get the text after "#### "
        answer_text = solution_str[answer_start_text + 5:].strip()
        
        # Extract digits from the final answer
        answer_digits = [d for d in predicted_answer if d.isdigit()]
        
        if not answer_digits:
            return []
        
        # Find positions that likely correspond to final answer
        # This is approximate - we look for digit positions near the end
        digit_positions = ntl_info.get('digit_positions', [])
        digit_ground_truth = ntl_info.get('digit_ground_truth', [])
        
        if not digit_positions or not digit_ground_truth:
            return []
        
        # Take the last N digit positions where N = length of final answer
        final_positions = []
        for batch_positions in digit_positions:
            if len(batch_positions) >= len(answer_digits):
                # Take the last len(answer_digits) positions
                final_positions.extend(batch_positions[-len(answer_digits):])
        
        return final_positions
        
    except Exception:
        return []


def calculate_final_answer_digit_accuracy(final_positions: list,
                                        true_digits: list,
                                        pred_digits: list,
                                        ntl_info: Dict[str, Any]) -> float:
    """Calculate digit accuracy for final answer positions only."""
    if not final_positions or not true_digits or not pred_digits:
        return 0.0
    
    if len(true_digits) != len(pred_digits):
        return 0.0
    
    try:
        digit_log_probs = ntl_info.get('digit_log_probs', None)
        if digit_log_probs is None:
            return 0.0
        
        correct = 0
        total = min(len(final_positions), len(true_digits))
        
        for i in range(total):
            if i < len(final_positions):
                pos = final_positions[i]
                # Get predicted digit at this position
                if pos < digit_log_probs.shape[1]:
                    digit_probs = torch.softmax(digit_log_probs[0, pos, :], dim=0)
                    predicted_digit = torch.argmax(digit_probs).item()
                    
                    if i < len(true_digits) and predicted_digit == true_digits[i]:
                        correct += 1
        
        return correct / total if total > 0 else 0.0
        
    except Exception:
        return 0.0


def calculate_final_answer_confidence(final_positions: list, 
                                    ntl_info: Dict[str, Any]) -> float:
    """Calculate average confidence for final answer digit predictions."""
    if not final_positions:
        return 0.0
    
    try:
        digit_log_probs = ntl_info.get('digit_log_probs', None)
        if digit_log_probs is None:
            return 0.0
        
        total_confidence = 0.0
        count = 0
        
        for pos in final_positions:
            if pos < digit_log_probs.shape[1]:
                digit_probs = torch.softmax(digit_log_probs[0, pos, :], dim=0)
                max_confidence = torch.max(digit_probs).item()
                total_confidence += max_confidence
                count += 1
        
        return total_confidence / count if count > 0 else 0.0
        
    except Exception:
        return 0.0


def calculate_final_answer_prob_quality(final_positions: list,
                                      true_digits: list,
                                      ntl_info: Dict[str, Any]) -> float:
    """Calculate probability quality for final answer digits."""
    if not final_positions or not true_digits:
        return 0.0
    
    try:
        digit_log_probs = ntl_info.get('digit_log_probs', None)
        if digit_log_probs is None:
            return 0.0
        
        total_prob = 0.0
        count = 0
        
        for i, pos in enumerate(final_positions):
            if i < len(true_digits) and pos < digit_log_probs.shape[1]:
                true_digit = true_digits[i]
                digit_probs = torch.softmax(digit_log_probs[0, pos, :], dim=0)
                prob_for_true_digit = digit_probs[true_digit].item()
                total_prob += prob_for_true_digit
                count += 1
        
        return total_prob / count if count > 0 else 0.0
        
    except Exception:
        return 0.0


# Legacy function for backward compatibility
def compute_ntl_bonus(ntl_info: Dict[str, Any], is_correct: bool) -> float:
    """Legacy function - now calls compute_ntl_bonus_all for backward compatibility."""
    return compute_ntl_bonus_all(ntl_info, is_correct)


def compute_score_ntl_enhanced(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Fully NTL-enhanced scoring function that uses digit-level information
    to provide more nuanced rewards.
    """
    ntl_info = kwargs.get('ntl_info', {})
    
    # Get base correctness score
    base_score = compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    
    # Extract ground truth digits and predicted digits
    predicted_answer = extract_solution(solution_str, "strict")
    
    if not ntl_info or predicted_answer is None:
        return base_score
    
    try:
        # Convert answers to digit sequences for comparison
        true_digits = [int(d) for d in str(ground_truth) if d.isdigit()]
        pred_digits = [int(d) for d in str(predicted_answer) if d.isdigit()]
        
        if not true_digits or not pred_digits:
            return base_score
        
        # Calculate digit-level accuracy
        digit_score = calculate_digit_sequence_score(true_digits, pred_digits)
        
        # Calculate NTL probability score
        prob_score = calculate_ntl_probability_score(ntl_info, true_digits)
        
        # Weighted combination
        enhanced_score = (
            0.6 * base_score +           # 60% from correctness
            0.25 * digit_score +         # 25% from digit accuracy  
            0.15 * prob_score            # 15% from NTL probability quality
        )
        
        return enhanced_score
        
    except Exception as e:
        # Fallback to base score if anything fails
        return base_score


def calculate_digit_sequence_score(true_digits: list, pred_digits: list) -> float:
    """Calculate similarity score between digit sequences."""
    if not true_digits or not pred_digits:
        return 0.0
    
    # Exact match bonus
    if true_digits == pred_digits:
        return 1.0
    
    # Partial match scoring
    max_len = max(len(true_digits), len(pred_digits))
    min_len = min(len(true_digits), len(pred_digits))
    
    # Alignment score
    correct_positions = sum(1 for i in range(min_len) if true_digits[i] == pred_digits[i])
    alignment_score = correct_positions / max_len
    
    # Length penalty
    length_penalty = min_len / max_len
    
    return 0.7 * alignment_score + 0.3 * length_penalty


def calculate_ntl_probability_score(ntl_info: Dict[str, Any], true_digits: list) -> float:
    """Calculate score based on NTL probability distributions."""
    if not ntl_info or not true_digits:
        return 0.0
    
    try:
        digit_log_probs = ntl_info.get('digit_log_probs', None)
        digit_positions = ntl_info.get('digit_positions', [])
        
        if digit_log_probs is None or not digit_positions:
            return 0.0
        
        # Calculate probability quality for ground truth digits
        total_prob_score = 0.0
        total_positions = 0
        
        for batch_idx, positions in enumerate(digit_positions):
            for pos_idx, pos in enumerate(positions):
                if pos_idx < len(true_digits):
                    true_digit = true_digits[pos_idx]
                    
                    # Get probability distribution for this position
                    probs = torch.softmax(digit_log_probs[batch_idx, pos, :], dim=0)
                    
                    # Score based on probability assigned to correct digit
                    correct_prob = probs[true_digit].item()
                    total_prob_score += correct_prob
                    total_positions += 1
        
        return total_prob_score / total_positions if total_positions > 0 else 0.0
        
    except Exception:
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