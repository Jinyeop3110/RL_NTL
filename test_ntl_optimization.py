#!/usr/bin/env python3
"""
Test script to verify NTL optimization produces identical results.
"""

import torch
import torch.nn.functional as F
import time

def old_method(logits, digit_token_ids):
    """Original method: compute softmax over full vocabulary"""
    log_probs = F.log_softmax(logits, dim=-1)
    digit_log_probs = []
    for token_id in digit_token_ids:
        digit_log_probs.append(log_probs[:, :, token_id])
    return torch.stack(digit_log_probs, dim=-1)

def new_method(logits, digit_token_ids):
    """Optimized method: extract only digit tokens then softmax"""
    digit_token_ids_tensor = torch.tensor(digit_token_ids, device=logits.device)
    digit_logits = torch.index_select(logits, dim=-1, index=digit_token_ids_tensor)
    return F.log_softmax(digit_logits, dim=-1)

def test_optimization():
    """Test that both methods produce identical results"""
    # Create test data
    batch_size, seq_len, vocab_size = 4, 1536, 151936
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random logits
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    
    # Simulate digit token IDs (Qwen tokenizer digits: "0"->220, "1"->16, etc)  
    digit_token_ids = [220, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 0-9
    
    print(f"Testing on {device}")
    print(f"Logits shape: {logits.shape}")
    print(f"Digit tokens: {len(digit_token_ids)}")
    
    # Test correctness
    print("\n=== Correctness Test ===")
    result_old = old_method(logits, digit_token_ids)
    result_new = new_method(logits, digit_token_ids)
    
    print(f"Old method result shape: {result_old.shape}")
    print(f"New method result shape: {result_new.shape}")
    
    # Check if results are identical
    max_diff = torch.max(torch.abs(result_old - result_new))
    print(f"Maximum difference: {max_diff.item()}")
    
    if max_diff < 1e-6:
        print("✅ Results are identical!")
    else:
        print("❌ Results differ significantly!")
        return False
    
    # Test performance
    print("\n=== Performance Test ===")
    
    # Warm up
    for _ in range(3):
        old_method(logits, digit_token_ids)
        new_method(logits, digit_token_ids)
    
    # Time old method
    start = time.time()
    for _ in range(10):
        result_old = old_method(logits, digit_token_ids)
    old_time = (time.time() - start) / 10
    
    # Time new method  
    start = time.time()
    for _ in range(10):
        result_new = new_method(logits, digit_token_ids)
    new_time = (time.time() - start) / 10
    
    print(f"Old method (full vocab): {old_time:.4f}s")
    print(f"New method (digit only): {new_time:.4f}s") 
    print(f"Speedup: {old_time/new_time:.1f}x")
    
    return True

if __name__ == "__main__":
    print("Testing NTL Optimization...")
    success = test_optimization()
    if success:
        print("\n🎉 Optimization verified successfully!")
    else:
        print("\n💥 Optimization failed verification!")