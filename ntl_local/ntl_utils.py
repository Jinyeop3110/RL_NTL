"""
Utility functions for Number Token Loss (NTL) implementation.
"""

import torch
from typing import Dict, List, Optional, Set
import re


def get_digit_token_ids(tokenizer) -> Dict[int, int]:
    """
    Get mapping from digit token IDs to their numerical values (0-9).
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Dict mapping token_id -> digit_value (0-9)
    """
    digit_token_map = {}
    
    # Check for digit tokens 0-9
    for digit in range(10):
        digit_str = str(digit)
        
        # Try different encodings
        for test_str in [digit_str, f" {digit_str}", f"{digit_str} "]:
            try:
                tokens = tokenizer.encode(test_str, add_special_tokens=False)
                if len(tokens) == 1:  # Single token for this digit
                    token_id = tokens[0]
                    digit_token_map[token_id] = digit
                    break
            except:
                continue
    
    return digit_token_map


def is_digit_token(token_id: int, digit_token_map: Dict[int, int]) -> bool:
    """Check if a token ID represents a digit (0-9)."""
    return token_id in digit_token_map


def token_to_digit_value(token_id: int, digit_token_map: Dict[int, int]) -> Optional[int]:
    """Convert token ID to digit value if it's a digit token."""
    return digit_token_map.get(token_id, None)


def find_digit_positions(input_ids: torch.Tensor, digit_token_map: Dict[int, int]) -> List[int]:
    """
    Find positions of digit tokens in a sequence.
    
    Args:
        input_ids: Token IDs tensor [seq_len]
        digit_token_map: Mapping from token_id to digit_value
        
    Returns:
        List of positions where digit tokens occur
    """
    positions = []
    for i, token_id in enumerate(input_ids):
        if token_id.item() in digit_token_map:
            positions.append(i)
    return positions


def extract_number_sequences(input_ids: torch.Tensor, 
                           digit_token_map: Dict[int, int]) -> List[Dict]:
    """
    Extract sequences of consecutive digit tokens to form numbers.
    
    Args:
        input_ids: Token IDs tensor [seq_len]
        digit_token_map: Mapping from token_id to digit_value
        
    Returns:
        List of dicts with keys: 'start_pos', 'end_pos', 'digits', 'value'
    """
    digit_positions = find_digit_positions(input_ids, digit_token_map)
    if not digit_positions:
        return []
    
    number_sequences = []
    current_seq = {
        'start_pos': digit_positions[0],
        'digits': [digit_token_map[input_ids[digit_positions[0]].item()]],
        'positions': [digit_positions[0]]
    }
    
    for i in range(1, len(digit_positions)):
        pos = digit_positions[i]
        prev_pos = digit_positions[i-1]
        
        # If consecutive positions, continue current sequence
        if pos == prev_pos + 1:
            current_seq['digits'].append(digit_token_map[input_ids[pos].item()])
            current_seq['positions'].append(pos)
        else:
            # Finish current sequence and start new one
            current_seq['end_pos'] = current_seq['positions'][-1]
            current_seq['value'] = int(''.join(map(str, current_seq['digits'])))
            number_sequences.append(current_seq)
            
            current_seq = {
                'start_pos': pos,
                'digits': [digit_token_map[input_ids[pos].item()]],
                'positions': [pos]
            }
    
    # Don't forget the last sequence
    if current_seq['digits']:
        current_seq['end_pos'] = current_seq['positions'][-1]
        current_seq['value'] = int(''.join(map(str, current_seq['digits'])))
        number_sequences.append(current_seq)
    
    return number_sequences


class TokenizerDigitCache:
    """Cache for tokenizer digit mappings to avoid recomputation."""
    
    def __init__(self):
        self._cache = {}
    
    def get_digit_mapping(self, tokenizer) -> Dict[int, int]:
        """Get cached digit mapping for tokenizer."""
        tokenizer_name = getattr(tokenizer, 'name_or_path', str(tokenizer))
        
        if tokenizer_name not in self._cache:
            self._cache[tokenizer_name] = get_digit_token_ids(tokenizer)
        
        return self._cache[tokenizer_name]


# Global cache instance
_digit_cache = TokenizerDigitCache()


def get_cached_digit_mapping(tokenizer) -> Dict[int, int]:
    """Get digit token mapping with caching."""
    return _digit_cache.get_digit_mapping(tokenizer)