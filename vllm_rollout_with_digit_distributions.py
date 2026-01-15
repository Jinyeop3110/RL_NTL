"""
Example modification to VLLM rollout to extract digit distributions during generation.
This would modify vllm_rollout_spmd.py to capture [batch_size, response_length, 10] digit distributions.
"""

import torch
import torch.nn.functional as F
from ntl_local.ntl_utils import get_cached_digit_mapping

def extract_digit_distributions_from_vllm_output(outputs, tokenizer):
    """
    Extract digit probability distributions from VLLM generation outputs.
    
    This function would be called during the rollout phase to capture
    digit distributions alongside the normal rollout_log_probs.
    
    Args:
        outputs: VLLM generation outputs with logprobs
        tokenizer: HuggingFace tokenizer
        
    Returns:
        digit_distributions: [batch_size, response_length, 10] tensor
    """
    # Get digit token mapping
    digit_token_map = get_cached_digit_mapping(tokenizer)
    digit_to_token = {v: k for k, v in digit_token_map.items()}
    
    if not digit_to_token:
        return None
    
    batch_digit_distributions = []
    
    for output in outputs:
        for sample_id in range(len(output.outputs)):
            response_digit_distributions = []
            
            # Iterate through each position in the generated sequence
            for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                # logprob is a dict: {token_id: TokenLogprob}
                # TokenLogprob has .logprob, .rank, .decoded_token attributes
                
                # Extract logits for digits 0-9 at this position
                digit_logits = []
                for digit in range(10):
                    if digit in digit_to_token:
                        token_id = digit_to_token[digit]
                        if token_id in logprob:
                            digit_logits.append(logprob[token_id].logprob)
                        else:
                            # Token not in top-k, approximate with very low probability
                            digit_logits.append(-20.0)  # Very low log probability
                    else:
                        digit_logits.append(-float('inf'))  # Invalid digit token
                
                # Convert to probability distribution
                digit_logits_tensor = torch.tensor(digit_logits, dtype=torch.float32)
                digit_probs = F.softmax(digit_logits_tensor, dim=0)  # [10]
                response_digit_distributions.append(digit_probs)
            
            # Stack into [response_length, 10]
            if response_digit_distributions:
                response_tensor = torch.stack(response_digit_distributions, dim=0)
                batch_digit_distributions.append(response_tensor)
    
    if batch_digit_distributions:
        # Stack into [batch_size, response_length, 10]
        return torch.stack(batch_digit_distributions, dim=0)
    
    return None


# Modified version of the VLLM rollout generation method
def modified_generate_sequences_with_digit_distributions(self, data: DataProto) -> DataProto:
    """
    Modified version of generate_sequences that also extracts digit distributions.
    
    This would replace or extend the existing generate_sequences method in vllm_rollout_spmd.py
    """
    # ... existing generation code ...
    
    # After generation, extract both rollout_log_probs and digit_distributions
    response = []
    rollout_log_probs = []
    digit_distributions_list = []
    
    for output in outputs:
        for sample_id in range(len(output.outputs)):
            response_ids = output.outputs[sample_id].token_ids
            response.append(response_ids)
            
            if self.config.calculate_log_probs:
                curr_log_prob = []
                for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                    curr_log_prob.append(logprob[response_ids[i]].logprob)
                rollout_log_probs.append(curr_log_prob)
    
    # Extract digit distributions
    if self.config.calculate_log_probs:  # Only if we're calculating log probs
        digit_distributions = extract_digit_distributions_from_vllm_output(outputs, self.tokenizer)
        if digit_distributions is not None:
            digit_distributions_list.append(digit_distributions)
    
    # ... existing padding and tensor creation code ...
    
    batch = {
        "input_ids": seq,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": response,
    }
    
    if self.config.calculate_log_probs:
        batch["rollout_log_probs"] = rollout_log_probs
        
        # Add digit distributions if available
        if digit_distributions_list:
            combined_digit_distributions = torch.cat(digit_distributions_list, dim=0)
            batch["digit_distributions"] = combined_digit_distributions
            print(f"[VLLM Rollout] Extracted digit distributions: {combined_digit_distributions.shape}")
    
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# Usage in training configuration
"""
To use this approach, you would:

1. Modify vllm_rollout_spmd.py to include the digit distribution extraction
2. The trainer would then have access to batch["digit_distributions"] 
3. Pass this to the reward function for full NTL computation

Advantages:
- Captures digit distributions at generation time
- No additional forward passes needed
- Complete information available for NTL

Disadvantages:  
- Requires modifying VLLM rollout code
- Increases memory usage during generation
- May slow down generation slightly
"""