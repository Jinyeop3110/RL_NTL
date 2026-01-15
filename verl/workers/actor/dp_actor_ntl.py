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
        # Extract tokenizer if passed
        self.tokenizer = kwargs.pop('tokenizer', None)
        
        super().__init__(*args, **kwargs)
        
        # NTL configuration
        self.ntl_enabled = getattr(self.config, 'ntl_enabled', True)
        self.ntl_method = getattr(self.config, 'ntl_method', 'mse')  # 'mse' or 'wasserstein'
        self.ntl_weight = getattr(self.config, 'ntl_weight', 0.1)
        self.extract_digit_info = getattr(self.config, 'extract_digit_info', True)
        
    
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
                
                # Extract NTL information - now returns full digit logprobs tensor
                try:
                    from ntl_local import prepare_digit_logprobs_for_reward
                    ntl_reward_data = prepare_digit_logprobs_for_reward(
                        logits=response_logits,
                        input_ids=response_ids,
                        tokenizer=self.tokenizer  # Assuming tokenizer is available
                    )
                    ntl_info = ntl_reward_data
                    
                except Exception as e:
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
        """
        Combine NTL information from multiple micro-batches.
        
        The key output is 'digit_log_probs' tensor [batch_size, seq_len, 10]
        that will be passed to the reward function.
        """
        if not ntl_info_list or not any(ntl_info_list):
            return {}
        
        combined = {}
        
        # Handle tensors that need to be concatenated along batch dimension
        tensor_keys = ['digit_log_probs', 'digit_ground_truth_tensor']
        for key in tensor_keys:
            if key in ntl_info_list[0]:
                tensors = [info[key] for info in ntl_info_list 
                          if key in info and info[key] is not None]
                if tensors:
                    # Concatenate along batch dimension and convert to numpy
                    concatenated_tensor = torch.cat(tensors, dim=0)
                    # Convert to numpy for protocol compatibility
                    if concatenated_tensor.dtype == torch.bfloat16:
                        combined[key] = concatenated_tensor.float().cpu().numpy()
                    else:
                        combined[key] = concatenated_tensor.cpu().numpy()
        
        # Skip metadata copying - metadata fields were removed to simplify protocol handling
        # metadata_keys = ['digit_token_map', 'has_digit_info', 'method']
        # if ntl_info_list:
        #     first_info = ntl_info_list[0]
        #     for key in metadata_keys:
        #         if key in first_info:
        #             combined[key] = first_info[key]
        pass  # No metadata to copy
        
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
            pass
        
        return original_metrics
    
    def get_last_ntl_info(self) -> Dict:
        """Get the NTL information from the last forward pass."""
        return getattr(self, '_last_ntl_info', {})
    
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> DataProto:
        """
        Override the standard compute_log_prob to automatically extract NTL information.
        
        This ensures that when the trainer calls the standard compute_log_prob method,
        we automatically extract and store NTL information if enabled.
        
        Returns:
            DataProto with log_probs, entropys, and NTL info in meta_info
        """
        if self.ntl_enabled:
            # Use the NTL-enabled method to get both log_probs and ntl_info
            log_probs, entropys = super().compute_log_prob(data, calculate_entropy)
            
            # Now extract NTL info using the same approach as compute_log_prob_with_ntl
            # but without duplicating the forward pass
            ntl_info = {}
            
            if self.extract_digit_info:
                # Validate tokenizer before proceeding
                if self.tokenizer is None:
                    raise ValueError("NTL extraction requires a tokenizer, but tokenizer is None. "
                                   "Pass tokenizer during actor initialization.")
                try:
                    # We need to do another forward pass to get logits for NTL extraction
                    # This is the price we pay for integrating with the existing interface
                    micro_batches = self.split_data_into_micro_batch(data)
                    all_ntl_info = []
                    
                    with torch.no_grad():
                        for micro_batch in micro_batches:
                            # Get logits for NTL extraction
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
                            response_logits = logits[:, -response_length - 1 : -1, :]
                            response_ids = micro_batch["responses"]
                            
                            # Extract NTL information - now returns full digit logprobs tensor
                            from ntl_local import prepare_digit_logprobs_for_reward
                            ntl_reward_data = prepare_digit_logprobs_for_reward(
                                logits=response_logits,
                                input_ids=response_ids,
                                tokenizer=self.tokenizer
                            )
                            all_ntl_info.append(ntl_reward_data)
                    
                    # Combine NTL info from all micro batches
                    if all_ntl_info:
                        ntl_info = self._combine_ntl_info(all_ntl_info)
                        self._last_ntl_info = ntl_info  # Cache for get_last_ntl_info()
                        
                except ImportError as e:
                    raise ImportError(f"Failed to import NTL utilities: {e}. "
                                    "Ensure ntl_local module is available.") from e
                except (KeyError, IndexError, ValueError) as e:
                    raise RuntimeError(f"NTL extraction failed due to data format issue: {e}") from e
                except Exception as e:
                    # Log the error but don't crash training
                    import logging
                    logging.warning(f"NTL extraction failed: {e}")
                    ntl_info = {'error': str(e)}
            
            # Create result DataProto matching the expected format
            result = DataProto(batch={"log_probs": log_probs})
            if calculate_entropy and entropys is not None:
                result.batch["entropys"] = entropys
            
            # TEMPORARY: Completely disable NTL info in meta_info to isolate the issue
            if ntl_info:
                # Cache the full NTL info for potential future access
                self._last_ntl_info = ntl_info
                # DO NOT store ANYTHING in meta_info to avoid protocol concatenation issues
                pass  # Explicitly do nothing with meta_info
            
            return result
        else:
            # Use the parent's standard method
            log_probs, entropys = super().compute_log_prob(data, calculate_entropy)
            result = DataProto(batch={"log_probs": log_probs})
            if calculate_entropy and entropys is not None:
                result.batch["entropys"] = entropys
            return result