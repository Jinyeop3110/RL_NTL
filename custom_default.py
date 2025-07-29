"""
VERL's default GSM8K reward function.
This file shows the exact logic used in verl/utils/reward_score/gsm8k.py
"""

import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Default VERL GSM8K scoring function.
    This is the exact implementation from verl/utils/reward_score/gsm8k.py
    
    Returns:
        - 1.0 if the answer is correct
        - 0.0 if the answer is wrong or not found
    """
    # Default parameters from VERL
    method = "strict"  # VERL default
    format_score = 0.0  # No partial credit for format
    score = 1.0  # Full credit for correct answer
    
    # Extract answer using VERL's default method
    answer = extract_solution(solution_str, method)
    
    if answer is None:
        return 0.0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


def extract_solution(solution_str, method="strict"):
    """
    Extract the final answer from the solution string.
    This is the exact implementation from VERL.
    """
    assert method in ["strict", "flexible"]

    if method == "strict":
        # This is VERL's default: look for #### format
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        # Alternative method: find last number
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward if there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer