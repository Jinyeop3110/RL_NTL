"""
Core Number Token Loss (NTL) implementation.
Based on research from "Regress, Don't Guess – A Regression-like Loss on Number Tokens for Language Models"
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from .ntl_utils import get_cached_digit_mapping, is_digit_token, token_to_digit_value


class NTLDigitExtractor:
    """
    Extracts digit-level log probabilities from model outputs for NTL computation.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.digit_token_map = get_cached_digit_mapping(tokenizer)
        self.digit_token_ids = list(self.digit_token_map.keys())
        
        # Create reverse mapping: digit_value -> token_id
        self.digit_to_token = {v: k for k, v in self.digit_token_map.items()}
        
    
    def extract_digit_log_probs(self, 
                               logits: torch.Tensor, 
                               input_ids: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None) -> Dict:
        """
        Extract digit-level log probabilities from model logits - OPTIMIZED VERSION.
        
        Instead of computing softmax over all 151K tokens, this extracts only the 10 digit tokens.
        This reduces computation from O(vocab_size) to O(10) per position.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dict containing:
            - 'digit_log_probs': Tensor [batch_size, seq_len, 10] - log probs for digits 0-9
            - 'digit_positions': List of positions where digits occur
            - 'digit_ground_truth': Ground truth digit values at digit positions
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Initialize output tensor for digit log probabilities (0-9)
        digit_log_probs = torch.full((batch_size, seq_len, 10), 
                                   fill_value=-float('inf'), 
                                   device=device, dtype=logits.dtype)
        
        # OPTIMIZATION: Extract logits for only the 10 digit tokens instead of computing full vocab softmax
        digit_token_ids = []
        digit_value_map = {}  # Maps index in extracted tensor to digit value
        
        for digit_value in range(10):
            if digit_value in self.digit_to_token:
                token_id = self.digit_to_token[digit_value] 
                digit_token_ids.append(token_id)
                digit_value_map[len(digit_token_ids) - 1] = digit_value
        
        if digit_token_ids:
            # Extract logits for only digit tokens [batch_size, seq_len, num_digit_tokens]
            digit_token_ids_tensor = torch.tensor(digit_token_ids, device=device)
            digit_logits = torch.index_select(logits, dim=-1, index=digit_token_ids_tensor)
            
            # Compute log softmax over only the digit tokens - MUCH faster than full vocab (151K -> ~10)
            digit_log_probs_extracted = F.log_softmax(digit_logits, dim=-1)
            
            # Map the extracted probabilities to correct digit positions
            for extracted_idx, digit_value in digit_value_map.items():
                digit_log_probs[:, :, digit_value] = digit_log_probs_extracted[:, :, extracted_idx]
        
        # Find positions and ground truth for digit tokens
        digit_positions = []
        digit_ground_truth = []
        
        for batch_idx in range(batch_size):
            batch_positions = []
            batch_ground_truth = []
            
            for pos in range(seq_len):
                token_id = input_ids[batch_idx, pos].item()
                if token_id in self.digit_token_map:
                    digit_value = self.digit_token_map[token_id]
                    batch_positions.append(pos)
                    batch_ground_truth.append(digit_value)
            
            digit_positions.append(batch_positions)
            digit_ground_truth.append(batch_ground_truth)
        
        return {
            'digit_log_probs': digit_log_probs,
            'digit_positions': digit_positions,
            'digit_ground_truth': digit_ground_truth,
            'digit_token_map': self.digit_token_map
        }


def extract_digit_log_probabilities(logits: torch.Tensor,
                                  input_ids: torch.Tensor,
                                  tokenizer,
                                  attention_mask: Optional[torch.Tensor] = None) -> Dict:
    """
    Extract digit log probabilities for reward function usage.
    
    This function extracts the full digit log probabilities tensor that will be 
    passed to the reward function. The reward function can then compute its own
    NTL-based metrics using this data.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: HuggingFace tokenizer
        attention_mask: Optional attention mask
        
    Returns:
        Dict with:
        - 'digit_log_probs': [batch_size, seq_len, 10] tensor of digit log probabilities 
        - 'digit_positions': List[List[int]] positions where digits occur
        - 'digit_ground_truth': List[List[int]] ground truth digit values
        - Additional metadata for reward function
    """
    extractor = NTLDigitExtractor(tokenizer)
    return extractor.extract_digit_log_probs(logits, input_ids, attention_mask)


def compute_ntl_loss_mse(digit_log_probs: torch.Tensor,
                        ground_truth_digits: List[List[int]],
                        digit_positions: List[List[int]],
                        reduction: str = 'mean') -> torch.Tensor:
    """
    Compute NTL loss using MSE variant.
    
    NTL-MSE formula: L = (y_true - E[predicted_digit])^2
    where E[predicted_digit] = sum(digit * P(digit)) for digits 0-9
    
    Args:
        digit_log_probs: Digit log probabilities [batch_size, seq_len, 10]
        ground_truth_digits: List of ground truth digit values per batch
        digit_positions: List of digit positions per batch
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        NTL loss tensor
    """
    device = digit_log_probs.device
    losses = []
    
    # Convert log probabilities to probabilities
    digit_probs = torch.exp(digit_log_probs)
    
    # Create digit value tensor [0, 1, 2, ..., 9]
    digit_values = torch.arange(10, device=device, dtype=digit_probs.dtype)
    
    for batch_idx in range(len(ground_truth_digits)):
        batch_losses = []
        
        for i, pos in enumerate(digit_positions[batch_idx]):
            # Get probability distribution for this digit position
            prob_dist = digit_probs[batch_idx, pos, :]  # [10]
            
            # Compute expected digit value: sum(digit * P(digit))
            expected_digit = torch.sum(prob_dist * digit_values)
            
            # Ground truth digit
            true_digit = ground_truth_digits[batch_idx][i]
            
            # MSE loss
            loss = (true_digit - expected_digit) ** 2
            batch_losses.append(loss)
        
        if batch_losses:
            losses.extend(batch_losses)
    
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    losses = torch.stack(losses)
    
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:
        return losses


def compute_ntl_loss_wasserstein(digit_log_probs: torch.Tensor,
                               ground_truth_digits: List[List[int]],
                               digit_positions: List[List[int]],
                               reduction: str = 'mean') -> torch.Tensor:
    """
    Compute NTL loss using Wasserstein-1 distance variant.
    
    For discrete distributions on {0,1,...,9}, W1 distance can be computed as:
    W1(P,Q) = sum_i |CDF_P(i) - CDF_Q(i)|
    
    Args:
        digit_log_probs: Digit log probabilities [batch_size, seq_len, 10]
        ground_truth_digits: List of ground truth digit values per batch
        digit_positions: List of digit positions per batch
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        NTL loss tensor
    """
    device = digit_log_probs.device
    losses = []
    
    # Convert log probabilities to probabilities
    digit_probs = torch.exp(digit_log_probs)
    
    for batch_idx in range(len(ground_truth_digits)):
        batch_losses = []
        
        for i, pos in enumerate(digit_positions[batch_idx]):
            # Get probability distribution for this digit position
            pred_probs = digit_probs[batch_idx, pos, :]  # [10]
            
            # Create ground truth one-hot distribution
            true_digit = ground_truth_digits[batch_idx][i]
            true_probs = torch.zeros(10, device=device)
            true_probs[true_digit] = 1.0
            
            # Compute CDFs
            pred_cdf = torch.cumsum(pred_probs, dim=0)
            true_cdf = torch.cumsum(true_probs, dim=0)
            
            # Wasserstein-1 distance
            w1_dist = torch.sum(torch.abs(pred_cdf - true_cdf))
            batch_losses.append(w1_dist)
        
        if batch_losses:
            losses.extend(batch_losses)
    
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    losses = torch.stack(losses)
    
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:
        return losses


def prepare_digit_logprobs_for_reward(logits: torch.Tensor,
                                    input_ids: torch.Tensor,
                                    tokenizer,
                                    attention_mask: Optional[torch.Tensor] = None) -> Dict:
    """
    Prepare digit log probabilities tensor for reward function.
    
    This is the main function that should be called to extract NTL information
    for passing to the reward function. It returns the full digit log probabilities
    tensor that the reward function can use for its computations.
    
    IMPORTANT: Always returns tensors for ALL samples in the batch, even if some samples
    don't contain digits (they get zero matrices).
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: HuggingFace tokenizer
        attention_mask: Optional attention mask
        
    Returns:
        Dict containing:
        - 'digit_log_probs': Tensor [batch_size, seq_len, 10] - THE KEY DATA FOR REWARD FUNCTION
        - 'digit_ground_truth_tensor': Tensor [batch_size, seq_len, 10] - One-hot encoded ground truth
        - Additional metadata
    """
    # Extract the base digit information
    digit_info = extract_digit_log_probabilities(logits, input_ids, tokenizer, attention_mask)
    
    # Always create tensors for ALL samples in the batch
    batch_size, seq_len = input_ids.shape
    device = logits.device
    
    # CRITICAL: Always return [batch_size, seq_len, 10] tensors, even if some samples have no digits
    # Create digit_log_probs tensor for ALL samples (default to zeros)
    digit_log_probs_full = torch.zeros(batch_size, seq_len, 10, device=device, dtype=torch.float32)
    
    # Create one-hot encoded ground truth tensor for ALL samples (default to zeros)
    digit_ground_truth_tensor = torch.zeros(batch_size, seq_len, 10, device=device, dtype=torch.float32)
    
    # Fill in actual data for samples that have digit information
    # digit_info['digit_log_probs'] might be smaller than [batch_size, seq_len, 10] if some samples have no digits
    if 'digit_log_probs' in digit_info and digit_info['digit_log_probs'] is not None:
        actual_digit_probs = digit_info['digit_log_probs']
        # Copy the actual data, but only for the samples that have it
        if actual_digit_probs.shape[0] <= batch_size:
            digit_log_probs_full[:actual_digit_probs.shape[0]] = actual_digit_probs
    
    # Fill in the ground truth values for samples that have digits
    for batch_idx in range(batch_size):
        if batch_idx < len(digit_info['digit_positions']):  # Safety check
            positions = digit_info['digit_positions'][batch_idx]
            ground_truth = digit_info['digit_ground_truth'][batch_idx]
            
            for i, pos in enumerate(positions):
                if i < len(ground_truth) and pos < seq_len:  # Safety check
                    digit_value = ground_truth[i]
                    if 0 <= digit_value <= 9:  # Ensure valid digit
                        digit_ground_truth_tensor[batch_idx, pos, digit_value] = 1.0
    
    # The key output is the digit_log_probs tensor [batch_size, seq_len, 10]
    # This contains log probabilities for digits 0-9 at every position
    # ALWAYS returns data for ALL samples - samples without digits get zero matrices
    return {
        'digit_log_probs': digit_log_probs_full,  # [batch_size, seq_len, 10] - ALWAYS full batch size
        'digit_ground_truth_tensor': digit_ground_truth_tensor,  # [batch_size, seq_len, 10] - ALWAYS full batch size
        # Removed metadata: digit_token_map, has_digit_info, method (unnecessary for protocol)
    }


def compute_ntl_reward_signal(digit_log_probs: torch.Tensor,
                            ground_truth_digits: List[List[int]],
                            digit_positions: List[List[int]],
                            method: str = 'mse') -> Dict:
    """
    Compute NTL-based reward signals for reinforcement learning.
    
    This function computes various NTL metrics that can be used as reward signals
    or additional information for the reward function.
    
    Args:
        digit_log_probs: Digit log probabilities [batch_size, seq_len, 10]
        ground_truth_digits: List of ground truth digit values per batch
        digit_positions: List of digit positions per batch
        method: 'mse' or 'wasserstein'
        
    Returns:
        Dict containing various NTL metrics and reward signals
    """
    # Compute NTL loss
    if method == 'mse':
        ntl_loss = compute_ntl_loss_mse(digit_log_probs, ground_truth_digits, digit_positions)
    else:
        ntl_loss = compute_ntl_loss_wasserstein(digit_log_probs, ground_truth_digits, digit_positions)
    
    # Convert loss to reward (lower loss = higher reward)
    ntl_reward = torch.exp(-ntl_loss).item()
    
    # Compute digit-level accuracy
    digit_probs = torch.exp(digit_log_probs)
    digit_predictions = torch.argmax(digit_probs, dim=-1)
    
    total_digits = 0
    correct_digits = 0
    
    for batch_idx in range(len(ground_truth_digits)):
        for i, pos in enumerate(digit_positions[batch_idx]):
            predicted_digit = digit_predictions[batch_idx, pos].item()
            true_digit = ground_truth_digits[batch_idx][i]
            
            total_digits += 1
            if predicted_digit == true_digit:
                correct_digits += 1
    
    digit_accuracy = correct_digits / total_digits if total_digits > 0 else 0.0
    
    return {
        'ntl_loss': ntl_loss.item(),
        'ntl_reward': ntl_reward,
        'digit_accuracy': digit_accuracy,
        'total_digits': total_digits,
        'correct_digits': correct_digits,
        'digit_log_probs': digit_log_probs,
        'method': method
    }