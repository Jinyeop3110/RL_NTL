"""
NTL-enabled DataParallel Actor that extends dp_actor.py to extract digit-level information
during forward pass for Number Token Loss integration.
"""

import torch
from typing import Dict, Any, Optional, Tuple

# Import the original actor
from .dp_actor import DataParallelPPOActor

# Import NTL utilities
from ...utils.torch_functional_ntl import (
    prepare_ntl_reward_data,
    extract_digit_logits_and_positions,
    logprobs_from_logits_with_ntl
)
from ...utils import torch_functional as verl_F
from ...protocol import DataProto


class DataParallelPPOActorNTL(DataParallelPPOActor):
    """
    Extended DataParallel PPO Actor with Number Token Loss (NTL) capabilities.
    
    This class extends the original DataParallelPPOActor to extract digit-level
    log probabilities during forward pass, which can then be used for:
    1. NTL-augmented loss computation
    2. Enhanced reward signals in the reward function
    3. Digit-level analysis and metrics
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # NTL configuration
        self.ntl_enabled = getattr(self.config, 'ntl_enabled', True)
        self.ntl_method = getattr(self.config, 'ntl_method', 'mse')  # 'mse' or 'wasserstein'
        self.ntl_weight = getattr(self.config, 'ntl_weight', 0.1)
        self.extract_digit_info = getattr(self.config, 'extract_digit_info', True)
        
        print(f"[NTL Actor] Initialized with ntl_enabled={self.ntl_enabled}, "
              f"method={self.ntl_method}, weight={self.ntl_weight}")
    
    def _forward_micro_batch_with_ntl(self, 
                                    micro_batch: Dict[str, torch.Tensor],
                                    temperature: float = 1.0,
                                    calculate_entropy: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Extended forward pass that extracts NTL digit information.
        
        Returns:
            Tuple: (entropy, log_probs, ntl_info)
        """
        # Call original forward pass method
        entropy, log_probs = self._forward_micro_batch(
            micro_batch, temperature, calculate_entropy
        )
        
        ntl_info = {}
        
        if self.ntl_enabled and self.extract_digit_info:
            # We need to get the logits again for NTL extraction
            # This is a bit inefficient but maintains compatibility
            
            with torch.no_grad():
                # Reconstruct the forward pass to get logits
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
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Extract response portion
                response_length = micro_batch["responses"].shape[1]
                response_logits = logits[:, -response_length - 1 : -1, :]
                response_ids = micro_batch["responses"]
                
                # Extract NTL information
                try:
                    ntl_reward_data = prepare_ntl_reward_data(
                        logits=response_logits,
                        input_ids=response_ids,
                        tokenizer=self.tokenizer  # Assuming tokenizer is available
                    )
                    ntl_info = ntl_reward_data
                    
                except Exception as e:
                    print(f"[NTL Actor] Warning: Failed to extract NTL info: {e}")
                    ntl_info = {'error': str(e)}
        
        return entropy, log_probs, ntl_info
    
    def compute_log_prob_with_ntl(self, 
                                data: DataProto, 
                                calculate_entropy: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Compute log probabilities with NTL digit information extraction.
        
        Args:
            data: DataProto containing input sequences
            calculate_entropy: Whether to calculate entropy
            
        Returns:
            Tuple: (log_probs, ntl_info)
        """
        # Set to eval mode
        self.actor_module.eval()
        
        with torch.no_grad():
            micro_batches = self.split_data_into_micro_batch(data)
            all_log_probs = []
            all_ntl_info = []
            
            for micro_batch in micro_batches:
                if self.ntl_enabled:
                    entropy, log_probs, ntl_info = self._forward_micro_batch_with_ntl(
                        micro_batch, calculate_entropy=calculate_entropy
                    )
                    all_ntl_info.append(ntl_info)
                else:
                    entropy, log_probs = self._forward_micro_batch(
                        micro_batch, calculate_entropy=calculate_entropy
                    )
                    all_ntl_info.append({})
                
                all_log_probs.append(log_probs)
            
            # Concatenate results
            log_probs = torch.cat(all_log_probs, dim=0)
            
            # Combine NTL info from all micro-batches
            combined_ntl_info = self._combine_ntl_info(all_ntl_info)
        
        return log_probs, combined_ntl_info
    
    def _combine_ntl_info(self, ntl_info_list: list) -> Dict:
        """Combine NTL information from multiple micro-batches."""
        if not ntl_info_list or not any(ntl_info_list):
            return {}
        
        combined = {}
        
        # Lists that need to be concatenated
        list_keys = ['digit_positions', 'digit_ground_truth', 'has_digits', 'sequence_strings']
        # Tensors that need to be concatenated
        tensor_keys = ['digit_log_probs']
        # Scalars that need to be summed
        sum_keys = ['total_digits', 'correct_digits']
        # Scalars that need to be averaged
        avg_keys = ['ntl_loss', 'ntl_reward', 'digit_accuracy']
        
        # Handle list concatenation
        for key in list_keys:
            if key in ntl_info_list[0]:
                combined[key] = []
                for info in ntl_info_list:
                    if key in info and info[key] is not None:
                        combined[key].extend(info[key])
        
        # Handle tensor concatenation
        for key in tensor_keys:
            if key in ntl_info_list[0]:
                tensors = [info[key] for info in ntl_info_list if key in info and info[key] is not None]
                if tensors:
                    combined[key] = torch.cat(tensors, dim=0)
        
        # Handle sum aggregation
        for key in sum_keys:
            if key in ntl_info_list[0]:
                combined[key] = sum(info.get(key, 0) for info in ntl_info_list)
        
        # Handle average aggregation
        for key in avg_keys:
            if key in ntl_info_list[0]:
                values = [info.get(key, 0) for info in ntl_info_list if key in info]
                combined[key] = sum(values) / len(values) if values else 0.0
        
        # Copy other keys from first batch
        if ntl_info_list:
            first_info = ntl_info_list[0]
            for key, value in first_info.items():
                if key not in combined:
                    combined[key] = value
        
        return combined
    
    def update_policy_with_ntl(self, data: DataProto) -> Dict[str, Any]:
        """
        Extended policy update that incorporates NTL information.
        
        This method extends the original update_policy to:
        1. Extract digit-level information during forward pass
        2. Optionally modify the loss with NTL components
        3. Return NTL metrics for logging
        """
        # Call original update_policy but capture additional info
        original_metrics = self.update_policy(data)
        
        if not self.ntl_enabled:
            return original_metrics
        
        # Extract NTL information for this batch
        try:
            with torch.no_grad():
                _, ntl_info = self.compute_log_prob_with_ntl(data, calculate_entropy=False)
                
                # Add NTL metrics to the returned metrics
                ntl_metrics = {
                    'ntl/digit_accuracy': ntl_info.get('digit_accuracy', 0.0),
                    'ntl/total_digits': ntl_info.get('total_digits', 0),
                    'ntl/correct_digits': ntl_info.get('correct_digits', 0),
                    'ntl/ntl_loss': ntl_info.get('ntl_loss', 0.0),
                    'ntl/ntl_reward': ntl_info.get('ntl_reward', 0.0),
                }
                
                # Store NTL info for potential use by reward function
                if hasattr(self, '_last_ntl_info'):
                    self._last_ntl_info = ntl_info
                
                # Merge with original metrics
                original_metrics.update(ntl_metrics)
                
        except Exception as e:
            print(f"[NTL Actor] Warning: Failed to compute NTL metrics: {e}")
        
        return original_metrics
    
    def get_last_ntl_info(self) -> Dict:
        """Get the NTL information from the last forward pass."""
        return getattr(self, '_last_ntl_info', {})