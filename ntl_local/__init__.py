"""
Local Number Token Loss (NTL) implementation for RL_NTL project.
Based on research from https://github.com/tum-ai/number-token-loss
"""

from .ntl_core import (
    NTLDigitExtractor,
    extract_digit_log_probabilities,
    prepare_digit_logprobs_for_reward,
    compute_ntl_loss_mse,
    compute_ntl_loss_wasserstein,
    compute_ntl_reward_signal
)

from .ntl_utils import (
    get_digit_token_ids,
    is_digit_token,
    token_to_digit_value
)

__version__ = "0.1.0"
__all__ = [
    'NTLDigitExtractor',
    'extract_digit_log_probabilities',
    'prepare_digit_logprobs_for_reward',
    'compute_ntl_loss_mse',
    'compute_ntl_loss_wasserstein',
    'compute_ntl_reward_signal',
    'get_digit_token_ids',
    'is_digit_token',
    'token_to_digit_value'
]