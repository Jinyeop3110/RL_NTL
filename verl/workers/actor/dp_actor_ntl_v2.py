"""
Enhanced NTL DataParallel Actor that extracts full digit distributions during compute_log_prob.
This provides [batch_size, response_length, 10] digit probability distributions for complete NTL computation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

# Import the original actor
from .dp_actor import DataParallelPPOActor
from ...utils import torch_functional as verl_F
from ...protocol import DataProto

# Import NTL utilities
from ntl_local.ntl_utils import get_cached_digit_mapping


class DataParallelPPOActorNTLv2(DataParallelPPOActor):
    """
    Enhanced DataParallel PPO Actor with full digit distribution extraction.
    
    This version extracts complete digit probability distributions [batch_size, response_length, 10]
    during compute_log_prob, enabling full NTL regression-based loss computation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # NTL configuration
        self.ntl_enabled = getattr(self.config, 'ntl_enabled', True)
        self.extract_digit_distributions = getattr(self.config, 'extract_digit_distributions', True)
        
        # Cache digit token mapping
        self._digit_token_map = None
        self._digit_to_token = None
        
        print(f"[NTL Actor v2] Initialized with ntl_enabled={self.ntl_enabled}, "
              f"extract_digit_distributions={self.extract_digit_distributions}")
    
    def _initialize_digit_mapping(self, tokenizer):
        """Initialize digit token mapping on first use."""
        if self._digit_token_map is None:
            self._digit_token_map = get_cached_digit_mapping(tokenizer)
            self._digit_to_token = {v: k for k, v in self._digit_token_map.items()}
            print(f"[NTL Actor v2] Initialized digit mapping: {self._digit_token_map}")
    
    def extract_digit_distributions_from_logits(self, 
                                              logits: torch.Tensor, 
                                              tokenizer) -> torch.Tensor:
        """
        Extract digit probability distributions from full logits.
        
        Args:
            logits: Full model logits [batch_size, seq_len, vocab_size]
            tokenizer: HuggingFace tokenizer
            
        Returns:
            digit_distributions: [batch_size, seq_len, 10] probability distributions over digits 0-9
        """
        if not self.extract_digit_distributions:
            return None
            
        # Initialize digit mapping
        self._initialize_digit_mapping(tokenizer)
        
        if not self._digit_to_token:
            return None
            
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Initialize digit distributions tensor
        digit_distributions = torch.zeros(batch_size, seq_len, 10, device=device, dtype=logits.dtype)
        
        # Extract logits for digit tokens (0-9)
        digit_logits = torch.full((batch_size, seq_len, 10), fill_value=-float('inf'), 
                                 device=device, dtype=logits.dtype)
        
        for digit_value in range(10):
            if digit_value in self._digit_to_token:
                token_id = self._digit_to_token[digit_value]
                if token_id < vocab_size:  # Ensure token exists in vocab
                    digit_logits[:, :, digit_value] = logits[:, :, token_id]
        
        # Convert to probabilities using softmax
        digit_distributions = F.softmax(digit_logits, dim=-1)  # [batch_size, seq_len, 10]
        
        return digit_distributions
    
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> DataProto:
        """
        Enhanced compute_log_prob that also extracts digit distributions.
        
        Returns both log probabilities and digit distributions in the DataProto.
        """
        # Set to eval
        self.actor_module.eval()
        
        with torch.no_grad():
            micro_batches = self.split_data_into_micro_batch(data)
            all_log_probs = []
            all_entropy = []
            all_digit_distributions = []
            
            for micro_batch in micro_batches:
                entropy, log_probs = self._forward_micro_batch(micro_batch, calculate_entropy=calculate_entropy)
                
                # Extract digit distributions if enabled
                if self.ntl_enabled and self.extract_digit_distributions:
                    # Get the logits again (we need them for digit extraction)
                    input_ids = micro_batch['input_ids']
                    attention_mask = micro_batch['attention_mask']
                    position_ids = micro_batch.get('position_ids', None)
                    
                    model_inputs = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                    }
                    if position_ids is not None:
                        model_inputs['position_ids'] = position_ids
                    
                    outputs = self.actor_module(**model_inputs)
                    logits = outputs.logits
                    
                    # Extract response portion
                    response_length = micro_batch["responses"].shape[1]
                    response_logits = logits[:, -response_length - 1 : -1, :]  # [batch_size, response_length, vocab_size]
                    
                    # Extract digit distributions
                    digit_distributions = self.extract_digit_distributions_from_logits(
                        response_logits, self.tokenizer
                    )
                    all_digit_distributions.append(digit_distributions)
                else:
                    all_digit_distributions.append(None)
                
                all_log_probs.append(log_probs)
                if calculate_entropy:
                    all_entropy.append(entropy)
        
        # Combine results
        log_probs = torch.cat(all_log_probs, dim=0)
        
        result = DataProto(batch={"log_probs": log_probs})
        
        if calculate_entropy:
            entropy = torch.cat(all_entropy, dim=0)
            result.batch["entropys"] = entropy
        
        # Add digit distributions if available
        if self.ntl_enabled and any(dd is not None for dd in all_digit_distributions):
            # Combine digit distributions
            combined_digit_distributions = torch.cat([dd for dd in all_digit_distributions if dd is not None], dim=0)
            result.batch["digit_distributions"] = combined_digit_distributions
            
            # Store metadata for NTL computation
            result.meta_info["has_digit_distributions"] = True
            result.meta_info["digit_token_map"] = self._digit_token_map
            
            print(f"[NTL Actor v2] Extracted digit distributions: {combined_digit_distributions.shape}")
        
        return result
    
    def _forward_micro_batch(self, micro_batch, temperature=1.0, calculate_entropy=False):
        """Override to ensure we have access to logits when needed."""
        # This is mostly the same as parent class, but we may need logits for digit extraction
        return super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)