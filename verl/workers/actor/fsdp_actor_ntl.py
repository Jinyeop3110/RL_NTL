"""
NTL-enabled FSDP Actor that extends the base actor to extract digit-level information
during forward pass for Number Token Loss integration.
"""

import torch
from typing import Dict, Any, Optional, Tuple

# Import the base DataParallel implementation (used for FSDP as well)
from .dp_actor import DataParallelPPOActor

# Import core utilities
from verl import DataProto

# NTL utilities will be imported dynamically when needed


class FSDPPPOActorNTL(DataParallelPPOActor):
    """
    Extended FSDP PPO Actor with Number Token Loss (NTL) capabilities.
    
    This class extends the DataParallelPPOActor (used for FSDP as well) to extract digit-level
    log probabilities during forward pass, which can then be used for:
    1. NTL-augmented loss computation
    2. Enhanced reward signals in the reward function
    3. Digit-level analysis and metrics
    """
    
    def __init__(self, *args, **kwargs):
        # Extract tokenizer if passed
        self.tokenizer = kwargs.pop('tokenizer', None)
        
        super().__init__(*args, **kwargs)
        
        # NTL configuration
        self.ntl_enabled = getattr(self.config, 'ntl_enabled', True)
        self.ntl_method = getattr(self.config, 'ntl_method', 'mse')
        self.ntl_weight = getattr(self.config, 'ntl_weight', 0.1)
        self.extract_digit_info = getattr(self.config, 'extract_digit_info', True)
    
    def compute_log_prob(self, data: DataProto, calculate_entropy=False):
        """
        Enhanced compute_log_prob that extracts NTL information.
        
        Returns:
            tuple: (log_probs, entropys) matching parent class interface
        """
        print(f"[NTL Actor] compute_log_prob called, ntl_enabled: {self.ntl_enabled}")
        
        # Try to import NTL functions, fall back to standard behavior if not available
        try:
            from ntl_local import prepare_digit_logprobs_for_reward
            ntl_available = True
            print("[NTL Actor] NTL functions imported successfully")
        except ImportError as e:
            ntl_available = False
            print(f"[NTL Actor] Warning: NTL functions not available: {e}, using standard compute_log_prob")
        
        import torch
        
        # Store original _forward_micro_batch method
        original_forward = self._forward_micro_batch
        ntl_infos = []
        
        def ntl_enhanced_forward(micro_batch, temperature, calculate_entropy=False):
            """Enhanced forward that captures logits for NTL extraction"""
            # Call original forward to get entropy and log_probs
            entropy, log_probs = original_forward(micro_batch, temperature, calculate_entropy)
            
            # Extract NTL information if enabled and available
            if self.ntl_enabled and self.tokenizer is not None and ntl_available:
                print(f"[NTL Actor] Attempting NTL extraction for micro-batch")
                try:
                    # We need to capture logits during the forward pass
                    # This is a simplified approach - get logits from the last forward pass
                    input_ids = micro_batch["input_ids"]
                    print(f"[NTL Actor] input_ids shape: {input_ids.shape}")
                    
                    # Re-run forward pass to get logits (inefficient but necessary for NTL)
                    with torch.no_grad():
                        model_output = self.actor_module(
                            input_ids=input_ids,
                            attention_mask=micro_batch.get("attention_mask"),
                            position_ids=micro_batch.get("position_ids"),
                            use_cache=False
                        )
                        logits = model_output.logits
                        print(f"[NTL Actor] logits shape: {logits.shape}")
                        
                        # Extract NTL information
                        ntl_info = prepare_digit_logprobs_for_reward(
                            logits=logits,
                            input_ids=input_ids,
                            tokenizer=self.tokenizer
                        )
                        print(f"[NTL Actor] NTL extraction successful, keys: {list(ntl_info.keys()) if ntl_info else 'None'}")
                        ntl_infos.append(ntl_info)
                except Exception as e:
                    # If NTL extraction fails, continue without it
                    print(f"[NTL Actor] WARNING: NTL extraction failed: {e}")
                    ntl_infos.append({})
            else:
                print(f"[NTL Actor] Skipping NTL extraction - ntl_enabled: {self.ntl_enabled}, has_tokenizer: {self.tokenizer is not None}, ntl_available: {ntl_available}")
                ntl_infos.append({})
            
            return entropy, log_probs
        
        # If NTL is not available, just use the parent method
        if not ntl_available:
            self._last_ntl_info = {}
            return super().compute_log_prob(data, calculate_entropy)
        
        # Temporarily replace the forward method
        self._forward_micro_batch = ntl_enhanced_forward
        
        try:
            # Call parent's compute_log_prob which will use our enhanced forward
            log_probs, entropys = super().compute_log_prob(data, calculate_entropy)
            
            # Combine NTL information from all micro-batches
            print(f"[NTL Actor] Processing {len(ntl_infos)} micro-batches for combination")
            if ntl_infos and any(ntl_infos):
                combined_ntl = self._combine_ntl_info(ntl_infos)
                print(f"[NTL Actor] Combined NTL info keys: {list(combined_ntl.keys()) if combined_ntl else 'None'}")
                if combined_ntl and 'digit_log_probs' in combined_ntl:
                    print(f"[NTL Actor] Combined digit_log_probs shape: {combined_ntl['digit_log_probs'].shape}")
                self._last_ntl_info = combined_ntl
                
                # CRITICAL FIX: Directly inject NTL tensors into DataProto batch
                # This bypasses worker dependency issues and ensures validation works
                try:
                    if 'digit_log_probs' in combined_ntl and combined_ntl['digit_log_probs'] is not None:
                        # Ensure tensor is on CPU and proper dtype for serialization
                        tensor = combined_ntl['digit_log_probs'].cpu().float()
                        data.batch['ntl_digit_log_probs'] = tensor
                        print(f"[NTL Actor] ✓ INJECTED ntl_digit_log_probs: {tensor.shape}")
                        
                    if 'digit_ground_truth_tensor' in combined_ntl and combined_ntl['digit_ground_truth_tensor'] is not None:
                        tensor = combined_ntl['digit_ground_truth_tensor'].cpu().float()
                        data.batch['ntl_digit_ground_truth_tensor'] = tensor
                        print(f"[NTL Actor] ✓ INJECTED ntl_digit_ground_truth_tensor: {tensor.shape}")
                        
                except Exception as e:
                    print(f"[NTL Actor] WARNING: Failed to inject tensors: {e}")
                    # Continue without NTL - don't break the pipeline
            else:
                print("[NTL Actor] No valid NTL info to combine")
                self._last_ntl_info = {}
                
        finally:
            # Restore original forward method
            self._forward_micro_batch = original_forward
        
        return log_probs, entropys
    
    def _combine_ntl_info(self, ntl_info_list: list) -> Dict:
        """
        Combine NTL information from multiple micro-batches.
        
        The key output is 'digit_log_probs' tensor [batch_size, seq_len, 10]
        that will be passed to the reward function.
        """
        if not ntl_info_list or not any(ntl_info_list):
            return {}
        
        import torch
        combined = {}
        
        # Handle tensors that need to be concatenated along batch dimension
        tensor_keys = ['digit_log_probs', 'digit_ground_truth_tensor']
        for key in tensor_keys:
            valid_tensors = []
            for info in ntl_info_list:
                if key in info and info[key] is not None:
                    tensor = info[key]
                    if isinstance(tensor, torch.Tensor):
                        valid_tensors.append(tensor)
                    elif hasattr(tensor, 'shape'):  # numpy array
                        valid_tensors.append(torch.from_numpy(tensor))
            
            if valid_tensors:
                try:
                    # Concatenate along batch dimension
                    concatenated_tensor = torch.cat(valid_tensors, dim=0)
                    
                    # Ensure proper dtype and device
                    if concatenated_tensor.dtype == torch.bfloat16:
                        concatenated_tensor = concatenated_tensor.float()
                    
                    # Keep as tensor (don't convert to numpy) for protocol handling
                    combined[key] = concatenated_tensor.cpu()
                    
                except Exception as e:
                    print(f"Warning: Failed to concatenate NTL tensor {key}: {e}")
                    # Fallback: use first valid tensor with padding
                    if valid_tensors:
                        combined[key] = valid_tensors[0].cpu()
        
        return combined
    
    def get_last_ntl_info(self) -> Dict:
        """Get the NTL information from the last forward pass."""
        return getattr(self, '_last_ntl_info', {})