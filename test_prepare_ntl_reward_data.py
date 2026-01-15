#!/usr/bin/env python3
"""
Test script for prepare_ntl_reward_data function
"""

import torch
import numpy as np
from transformers import AutoTokenizer
import sys
import os

# Add the project root to path
sys.path.append('/home/yeopjin/orcd/pool/workspace/RL_NTL')

from verl.utils.torch_functional_ntl import prepare_ntl_reward_data

def test_prepare_ntl_reward_data():
    """Test prepare_ntl_reward_data with a realistic math problem"""
    
    print("=" * 60)
    print("Testing prepare_ntl_reward_data function")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test sequence with digits (similar to the log example)
    test_sequence = "She meditates for 15 minutes in the morning and 15 minutes at night for a total of 15+15 = 30 minutes. Answer: #### 34"
    
    print(f"Test sequence: {test_sequence}")
    print()
    
    # Tokenize
    inputs = tokenizer(test_sequence, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    batch_size, seq_len = input_ids.shape
    vocab_size = len(tokenizer)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Vocabulary size: {vocab_size}")
    print()
    
    # Create fake logits (random but realistic)
    torch.manual_seed(42)  # For reproducible results
    logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
    
    # Make logits slightly favor the actual tokens (to simulate realistic model output)
    for i in range(seq_len):
        actual_token = input_ids[0, i].item()
        if actual_token < vocab_size:
            logits[0, i, actual_token] += 2.0  # Boost actual token probability
    
    print("Created fake logits with shape:", logits.shape)
    print()
    
    # Test the function
    print("Calling prepare_ntl_reward_data...")
    print("-" * 40)
    
    try:
        result = prepare_ntl_reward_data(
            logits=logits,
            input_ids=input_ids,
            tokenizer=tokenizer,
            sequence_strings=[test_sequence]
        )
        
        print("SUCCESS: prepare_ntl_reward_data returned!")
        print()
        print("Result keys:", list(result.keys()))
        print()
        
        # Print detailed results
        for key, value in result.items():
            print(f"{key}:")
            if isinstance(value, torch.Tensor):
                print(f"  Type: torch.Tensor")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                if value.numel() < 20:  # Only print small tensors
                    print(f"  Values: {value}")
            elif isinstance(value, np.ndarray):
                print(f"  Type: numpy.ndarray")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                if value.size < 20:
                    print(f"  Values: {value}")
            elif isinstance(value, list):
                print(f"  Type: list")
                print(f"  Length: {len(value)}")
                if len(value) < 10:
                    print(f"  Values: {value}")
            else:
                print(f"  Type: {type(value)}")
                print(f"  Value: {value}")
            print()
            
    except Exception as e:
        print(f"ERROR: prepare_ntl_reward_data failed!")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    return True

def test_digit_detection():
    """Test digit detection specifically"""
    print("\n" + "=" * 60)
    print("Testing digit detection")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Test with simple numbers
    test_texts = [
        "The answer is 15",
        "15 + 15 = 30",
        "<<15+15=30>>",
        "#### 34",
        "Numbers: 0 1 2 3 4 5 6 7 8 9"
    ]
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        
        # Check which tokens are digits
        digit_tokens = []
        for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
            if token.isdigit() or (token.startswith('Ġ') and token[1:].isdigit()):
                digit_tokens.append((i, token, token_id))
        
        print(f"Digit tokens found: {digit_tokens}")

if __name__ == "__main__":
    success = test_prepare_ntl_reward_data()
    test_digit_detection()
    
    if success:
        print("\nAll tests passed!")
    else:
        print("\nTests failed!")
        sys.exit(1)