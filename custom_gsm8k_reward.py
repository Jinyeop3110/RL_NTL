"""
Custom GSM8K reward function with configurable parameters.
"""

import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Custom GSM8K scoring function with configurable parameters.
    
    You can modify this function to change how rewards are calculated.
    """
    # Configuration options
    method = "strict"  # or "flexible"
    correct_score = 1.0  # Reward for correct answer
    format_score = 0.0   # Reward for correct format but wrong answer
    partial_credit = 0.3  # Optional: partial credit for showing work
    
    # Extract answer based on method
    if method == "strict":
        # Look for #### format
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    else:  # flexible
        # Find all numbers and take the last one
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) > 0:
            invalid_str = ["", "."]
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    
    # Calculate reward
    if final_answer is None:
        # No answer found
        return 0.0
    elif final_answer == ground_truth:
        # Correct answer
        return correct_score
    else:
        # Wrong answer but correct format
        # Optional: Check if solution shows correct reasoning steps
        if "=" in solution_str and "+" in solution_str or "-" in solution_str or "*" in solution_str:
            # Give partial credit for showing mathematical work
            return format_score + partial_credit
        else:
            return format_score


# Alternative scoring functions you can use:

def compute_score_with_steps(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Reward function that gives credit for intermediate steps.
    """
    base_score = compute_score(data_source, solution_str, ground_truth, extra_info)
    
    # Bonus for showing clear steps
    step_bonus = 0.0
    if "step 1:" in solution_str.lower() or "first," in solution_str.lower():
        step_bonus += 0.1
    if "step 2:" in solution_str.lower() or "then," in solution_str.lower():
        step_bonus += 0.1
    if "therefore" in solution_str.lower() or "so the answer is" in solution_str.lower():
        step_bonus += 0.1
    
    return min(1.0, base_score + step_bonus)


def compute_score_lenient(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    More lenient scoring that accepts answers in different formats.
    """
    # Clean the response
    solution_str_clean = solution_str.lower().strip()
    
    # Try multiple extraction patterns
    patterns = [
        r"#### (\\-?[0-9\\.\\,]+)",  # Standard format
        r"answer is (\\-?[0-9\\.\\,]+)",  # "The answer is X"
        r"= (\\-?[0-9\\.\\,]+)$",  # Ends with = X
        r"total: (\\-?[0-9\\.\\,]+)",  # "Total: X"
    ]
    
    final_answer = None
    for pattern in patterns:
        match = re.search(pattern, solution_str_clean)
        if match:
            final_answer = match.group(1).replace(",", "").replace("$", "")
            break
    
    # If no pattern matched, try flexible extraction
    if final_answer is None:
        numbers = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        if numbers:
            final_answer = numbers[-1].replace(",", "").replace("$", "")
    
    # Score calculation
    if final_answer == ground_truth:
        return 1.0
    elif final_answer is not None:
        return 0.1  # Small reward for attempting an answer
    else:
        return 0.0