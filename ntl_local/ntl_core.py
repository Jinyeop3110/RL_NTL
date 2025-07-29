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
        
        print(f"[NTL] Found {len(self.digit_token_map)} digit tokens: {self.digit_token_map}")
    
    def extract_digit_log_probs(self, 
                               logits: torch.Tensor, 
                               input_ids: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None) -> Dict:
        """
        Extract digit-level log probabilities from model logits.
        
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
        
        # Compute log probabilities for all tokens
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for digit tokens (0-9)
        for digit_value in range(10):
            if digit_value in self.digit_to_token:
                token_id = self.digit_to_token[digit_value]
                digit_log_probs[:, :, digit_value] = log_probs[:, :, token_id]
        
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
    Convenience function to extract digit log probabilities.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: HuggingFace tokenizer
        attention_mask: Optional attention mask
        
    Returns:
        Dict with digit log probabilities and metadata
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