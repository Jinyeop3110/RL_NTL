"""
Extended torch_functional.py with Number Token Loss (NTL) capabilities.
This module extends the original VERL torch_functional.py to include digit-level
log probability extraction for NTL integration.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

# Import original functions
from .torch_functional import (
    logprobs_from_logits,
    logprobs_from_logits_v2,
    logprobs_from_logits_naive,
    gather_from_labels,
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE,
    NPU_CROSS_ENTROPY_LOSS_AVAILABLE
)

# Import our NTL implementation
from ...ntl_local import extract_digit_log_probabilities, compute_ntl_reward_signal


def logprobs_from_logits_with_ntl(logits, labels, tokenizer=None, extract_ntl=False, inplace_backward=True):
    """
    Extended version of logprobs_from_logits that optionally extracts NTL digit information.
    
    Args:
        logits (Tensor): Model outputs of shape (..., vocab_size).
        labels (LongTensor): True class indices of shape matching logits[..., :-1].
        tokenizer: HuggingFace tokenizer (required if extract_ntl=True)
        extract_ntl (bool): Whether to extract digit-level log probabilities for NTL
        inplace_backward (bool): If True and Flash-Attn is available, perform backward in-place.
    
    Returns:
        If extract_ntl=False:
            Tensor: Log-probabilities of the target labels
        If extract_ntl=True:
            Tuple: (log_probabilities, ntl_info_dict)
    """
    # Compute standard log probabilities
    logprobs = logprobs_from_logits(logits, labels, inplace_backward=inplace_backward)
    
    if not extract_ntl or tokenizer is None:
        return logprobs
    
    # Extract NTL digit information
    ntl_info = extract_digit_log_probabilities(
        logits=logits,
        input_ids=labels,
        tokenizer=tokenizer
    )
    
    return logprobs, ntl_info


def compute_ntl_augmented_loss(logits: torch.Tensor,
                             labels: torch.Tensor,
                             tokenizer,
                             ntl_weight: float = 0.1,
                             ntl_method: str = 'mse',
                             base_loss_fn=None) -> Dict[str, torch.Tensor]:
    """
    Compute loss augmented with NTL digit-level loss.
    
    Args:
        logits: Model outputs [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]  
        tokenizer: HuggingFace tokenizer
        ntl_weight: Weight for NTL loss component
        ntl_method: 'mse' or 'wasserstein'
        base_loss_fn: Base loss function (default: cross_entropy)
        
    Returns:
        Dict containing different loss components
    """
    # Compute base loss (e.g., cross-entropy)
    if base_loss_fn is None:
        base_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1),
            reduction='mean'
        )
    else:
        base_loss = base_loss_fn(logits, labels)
    
    # Extract NTL information
    ntl_info = extract_digit_log_probabilities(
        logits=logits,
        input_ids=labels, 
        tokenizer=tokenizer
    )
    
    # Compute NTL loss
    from ...ntl_local.ntl_core import compute_ntl_loss_mse, compute_ntl_loss_wasserstein
    
    if ntl_method == 'mse':
        ntl_loss = compute_ntl_loss_mse(
            ntl_info['digit_log_probs'],
            ntl_info['digit_ground_truth'],
            ntl_info['digit_positions']
        )
    else:
        ntl_loss = compute_ntl_loss_wasserstein(
            ntl_info['digit_log_probs'],
            ntl_info['digit_ground_truth'],
            ntl_info['digit_positions']
        )
    
    # Combined loss
    total_loss = base_loss + ntl_weight * ntl_loss
    
    return {
        'total_loss': total_loss,
        'base_loss': base_loss,
        'ntl_loss': ntl_loss,
        'ntl_info': ntl_info
    }


def extract_digit_logits_and_positions(logits: torch.Tensor,
                                     input_ids: torch.Tensor,
                                     tokenizer) -> Dict[str, Any]:
    """
    Extract logits and positions specifically for digit tokens.
    This is useful for passing digit-level information to reward functions.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Dict containing digit-specific information for reward calculation
    """
    # Get NTL information
    ntl_info = extract_digit_log_probabilities(
        logits=logits,
        input_ids=input_ids,
        tokenizer=tokenizer
    )
    
    # Extract additional useful information
    batch_size, seq_len = input_ids.shape
    device = logits.device
    
    # Create masks for digit positions
    digit_masks = []
    for batch_idx in range(batch_size):
        mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
        for pos in ntl_info['digit_positions'][batch_idx]:
            mask[pos] = True
        digit_masks.append(mask)
    
    digit_masks = torch.stack(digit_masks)  # [batch_size, seq_len]
    
    # Extract logits at digit positions
    digit_logits = []
    for batch_idx in range(batch_size):
        batch_digit_logits = []
        for pos in ntl_info['digit_positions'][batch_idx]:
            batch_digit_logits.append(logits[batch_idx, pos, :])  # [vocab_size]
        if batch_digit_logits:
            digit_logits.append(torch.stack(batch_digit_logits))
        else:
            # No digits in this sequence
            digit_logits.append(torch.empty(0, logits.size(-1), device=device))
    
    return {
        'digit_log_probs': ntl_info['digit_log_probs'],  # [batch_size, seq_len, 10]
        'digit_positions': ntl_info['digit_positions'],  # List[List[int]]
        'digit_ground_truth': ntl_info['digit_ground_truth'],  # List[List[int]]
        'digit_masks': digit_masks,  # [batch_size, seq_len]
        'digit_logits': digit_logits,  # List[Tensor[num_digits, vocab_size]]
        'digit_token_map': ntl_info['digit_token_map'],  # Dict[token_id, digit_value]
        'has_digits': [len(positions) > 0 for positions in ntl_info['digit_positions']]
    }


def prepare_ntl_reward_data(logits: torch.Tensor,
                          input_ids: torch.Tensor,
                          tokenizer,
                          sequence_strings: Optional[list] = None) -> Dict[str, Any]:
    """
    Prepare all NTL-related data needed for reward calculation.
    This function extracts and organizes all digit-level information
    that can be passed to custom reward functions.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: HuggingFace tokenizer
        sequence_strings: Optional list of decoded sequence strings
        
    Returns:
        Dict with comprehensive NTL information for reward calculation
    """
    # Extract base digit information
    digit_info = extract_digit_logits_and_positions(logits, input_ids, tokenizer)
    
    # Compute NTL reward signals
    ntl_reward_info = compute_ntl_reward_signal(
        digit_info['digit_log_probs'],
        digit_info['digit_ground_truth'],
        digit_info['digit_positions'],
        method='mse'
    )
    
    # Organize all information for reward function
    reward_data = {
        # Basic NTL info
        'digit_log_probs': digit_info['digit_log_probs'],
        'digit_positions': digit_info['digit_positions'],
        'digit_ground_truth': digit_info['digit_ground_truth'],
        'digit_token_map': digit_info['digit_token_map'],
        
        # NTL metrics
        'ntl_loss': ntl_reward_info['ntl_loss'],
        'ntl_reward': ntl_reward_info['ntl_reward'],
        'digit_accuracy': ntl_reward_info['digit_accuracy'],
        'total_digits': ntl_reward_info['total_digits'],
        'correct_digits': ntl_reward_info['correct_digits'],
        
        # Additional context
        'has_digits': digit_info['has_digits'],
        'sequence_strings': sequence_strings,
        'batch_size': logits.shape[0],
        'seq_len': logits.shape[1]
    }
    
    return reward_data