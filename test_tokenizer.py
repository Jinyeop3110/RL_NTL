#!/usr/bin/env python3
"""
Test script to examine how Qwen2.5 tokenizer handles numbers 0-100.
"""

from transformers import AutoTokenizer

def test_qwen_tokenization():
    """Test how Qwen2.5 tokenizes numbers from 0 to 100."""
    
    # Load Qwen2.5 tokenizer
    print("Loading Qwen2.5-0.5B-Instruct tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print()
    
    # Test numbers 0-100
    print("=" * 80)
    print("TOKENIZATION ANALYSIS: Numbers 0-100")
    print("=" * 80)
    print(f"{'Number':<8} {'Tokens':<15} {'Token IDs':<20} {'Decoded':<15} {'Single Token?'}")
    print("-" * 80)
    
    single_token_numbers = []
    multi_token_numbers = []
    
    for num in range(101):
        num_str = str(num)
        
        # Test raw number
        tokens = tokenizer.encode(num_str, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        is_single = len(tokens) == 1
        
        if is_single:
            single_token_numbers.append(num)
        else:
            multi_token_numbers.append(num)
        
        print(f"{num_str:<8} {str(tokens):<15} {str(tokens):<20} {repr(decoded):<15} {is_single}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Single-token numbers: {len(single_token_numbers)}")
    print(f"Multi-token numbers: {len(multi_token_numbers)}")
    print()
    print(f"Single-token numbers: {single_token_numbers}")
    print(f"Multi-token numbers: {multi_token_numbers}")
    
    # Test with spaces
    print()
    print("=" * 80)
    print("TOKENIZATION WITH SPACES")
    print("=" * 80)
    print(f"{'Number':<8} {'Raw Tokens':<15} {'Space Before':<15} {'Space After':<15}")
    print("-" * 80)
    
    for num in [0, 1, 2, 5, 10, 18, 25, 42, 100]:
        num_str = str(num)
        
        raw_tokens = tokenizer.encode(num_str, add_special_tokens=False)
        space_before_tokens = tokenizer.encode(f" {num_str}", add_special_tokens=False)
        space_after_tokens = tokenizer.encode(f"{num_str} ", add_special_tokens=False)
        
        print(f"{num_str:<8} {str(raw_tokens):<15} {str(space_before_tokens):<15} {str(space_after_tokens):<15}")
    
    # Test digit-by-digit for multi-token numbers
    print()
    print("=" * 80)
    print("DIGIT-BY-DIGIT ANALYSIS FOR MULTI-TOKEN NUMBERS")
    print("=" * 80)
    
    for num in multi_token_numbers[:10]:  # Test first 10 multi-token numbers
        num_str = str(num)
        whole_tokens = tokenizer.encode(num_str, add_special_tokens=False)
        
        print(f"Number: {num} ('{num_str}')")
        print(f"  Whole number tokens: {whole_tokens}")
        
        digit_tokens = []
        for digit_char in num_str:
            digit_token = tokenizer.encode(digit_char, add_special_tokens=False)
            digit_tokens.append(digit_token)
        
        print(f"  Individual digit tokens: {digit_tokens}")
        print(f"  Can reconstruct from digits? {len(digit_tokens) == len(num_str) and all(len(dt) == 1 for dt in digit_tokens)}")
        print()
    
    # Build digit token mapping
    print("=" * 80)
    print("DIGIT TOKEN MAPPING (0-9)")
    print("=" * 80)
    
    digit_token_map = {}
    for digit in range(10):
        digit_str = str(digit)
        
        # Try different contexts
        contexts = [digit_str, f" {digit_str}", f"{digit_str} ", f" {digit_str} "]
        
        print(f"Digit {digit}:")
        for i, context in enumerate(contexts):
            tokens = tokenizer.encode(context, add_special_tokens=False)
            decoded = tokenizer.decode(tokens)
            print(f"  Context {i} ('{context}'): tokens={tokens}, decoded={repr(decoded)}")
            
            if len(tokens) == 1:
                digit_token_map[tokens[0]] = digit
                print(f"    → Single token found: {tokens[0]} -> {digit}")
        print()
    
    print(f"Final digit token mapping: {digit_token_map}")
    print(f"Found {len(digit_token_map)} digit tokens out of 10 possible")
    
    return {
        'single_token_numbers': single_token_numbers,
        'multi_token_numbers': multi_token_numbers,
        'digit_token_map': digit_token_map,
        'tokenizer': tokenizer
    }

if __name__ == "__main__":
    results = test_qwen_tokenization()
    
    print()
    print("=" * 80)
    print("IMPLICATIONS FOR NTL")
    print("=" * 80)
    
    if len(results['multi_token_numbers']) > 0:
        print("⚠️  WARNING: Multi-digit numbers are NOT tokenized as individual digits!")
        print("   This means current NTL implementation will miss most multi-digit answers.")
        print()
        print("   Solutions needed:")
        print("   1. Map multi-digit tokens back to constituent digits")
        print("   2. Use post-processing on decoded text instead of tokens")
        print("   3. Modify NTL to work with subword tokens")
    else:
        print("✅ All numbers 0-100 are single tokens - NTL should work fine!")
    
    print(f"✅ Found digit mappings for: {len(results['digit_token_map'])}/10 individual digits")