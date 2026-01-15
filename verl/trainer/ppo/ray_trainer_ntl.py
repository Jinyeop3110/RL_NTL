# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with NTL (Number Token Loss) support.
Extends ray_trainer.py to extract and pass digit-level information for NTL computation.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.debug import marked_timer
from verl.workers.actor.dp_actor_ntl import DataParallelPPOActorNTL
from verl.workers.actor.fsdp_actor_ntl import FSDPPPOActorNTL

# Import NTL utilities
from verl.utils.torch_functional_ntl import prepare_ntl_reward_data
from ntl_local.ntl_core import compute_ntl_reward_signal


class RayPPOTrainerNTL(RayPPOTrainer):
    """
    Extended PPO Trainer with Number Token Loss (NTL) support.
    
    This trainer extracts digit-level information during training and passes it
    to the reward function for NTL-augmented reward computation.
    """
    
    def __init__(self, *args, **kwargs):
        # Add NTL configuration
        self.ntl_enabled = kwargs.pop('ntl_enabled', True)
        self.ntl_method = kwargs.pop('ntl_method', 'mse')  # 'mse' or 'wasserstein'
        self.ntl_extract_during_generation = kwargs.pop('ntl_extract_during_generation', False)
        
        super().__init__(*args, **kwargs)
        
    
    def _create_worker_cls_fn(self):
        """Override to use NTL-enabled actor classes."""
        worker_cls_fn = super()._create_worker_cls_fn()
        
        if self.ntl_enabled:
            # Replace actor classes with NTL-enabled versions
            original_actor_cls = worker_cls_fn[Role.Actor]
            original_actor_rollout_cls = worker_cls_fn.get(Role.ActorRollout, None)
            original_actor_rollout_ref_cls = worker_cls_fn.get(Role.ActorRolloutRef, None)
            
            # Determine which NTL actor to use based on strategy
            if self.config.actor_rollout_ref.actor.strategy == "fsdp":
                ntl_actor_cls = FSDPPPOActorNTL
            else:
                ntl_actor_cls = DataParallelPPOActorNTL
            
            # Replace actor classes
            worker_cls_fn[Role.Actor] = self._wrap_actor_class(original_actor_cls, ntl_actor_cls)
            
            if original_actor_rollout_cls:
                worker_cls_fn[Role.ActorRollout] = self._wrap_actor_class(
                    original_actor_rollout_cls, ntl_actor_cls
                )
            
            if original_actor_rollout_ref_cls:
                worker_cls_fn[Role.ActorRolloutRef] = self._wrap_actor_class(
                    original_actor_rollout_ref_cls, ntl_actor_cls
                )
            
        
        return worker_cls_fn
    
    def _wrap_actor_class(self, original_cls, ntl_cls):
        """Wrap the original actor class to use NTL-enabled implementation."""
        # If it's a colocated worker class, we need to handle it specially
        if hasattr(original_cls, '__bases__') and any('Colocated' in str(base) for base in original_cls.__bases__):
            # For colocated workers, we need to replace the actor component
            class NTLColocatedWorker(original_cls):
                def __init__(self, *args, **kwargs):
                    # Inject NTL configuration
                    if 'actor_config' in kwargs:
                        kwargs['actor_config'].ntl_enabled = True
                        kwargs['actor_config'].ntl_method = self.ntl_method
                    super().__init__(*args, **kwargs)
                    # Replace actor with NTL version if needed
                    if hasattr(self, 'actor') and not isinstance(self.actor, ntl_cls):
                        actor_config = getattr(self, 'actor_config', self.config.actor)
                        # Pass tokenizer to NTL actor
                        tokenizer = getattr(self, 'tokenizer', None)
                        self.actor = ntl_cls(actor_config, tokenizer=tokenizer)
            
            return NTLColocatedWorker
        else:
            # For standalone actors, just return the NTL class
            return ntl_cls
    
    def compute_log_prob_with_ntl(self, batch: DataProto) -> DataProto:
        """
        Compute log probabilities and extract NTL tensors to be stored in batch.
        
        This method ensures that NTL tensor data (digit_log_probs, digit_ground_truth_tensor)
        are properly extracted and stored in the batch for the reward function.
        """
        # Call the standard compute_log_prob method - this should trigger NTL extraction in actors
        result_batch = self.actor_rollout_wg.compute_log_prob(batch)
        
        # Check if NTL tensors were extracted by the actor and stored in batch
        if self.ntl_enabled:
            ntl_digit_log_probs = result_batch.batch.get('ntl_digit_log_probs', None)
            ntl_digit_ground_truth = result_batch.batch.get('ntl_digit_ground_truth_tensor', None)
            
            if ntl_digit_log_probs is not None and ntl_digit_ground_truth is not None:
                print(f"[NTL Trainer] SUCCESS: NTL tensors extracted - "
                      f"digit_log_probs: {ntl_digit_log_probs.shape}, "
                      f"ground_truth: {ntl_digit_ground_truth.shape}")
            else:
                print("[NTL Trainer] WARNING: NTL enabled but tensors not found in batch")
                print(f"Available batch keys: {list(result_batch.batch.keys())}")
        
        return result_batch
    
    # Removed: extract_ntl_from_log_probs - no longer needed with tensor-based approach
    
    def _unused_extract_ntl_from_log_probs(self, rollout_log_probs: torch.Tensor, 
                                 responses: torch.Tensor, 
                                 attention_mask: torch.Tensor) -> Dict[str, Any]:
        """
        Extract digit confidence information from rollout log probabilities.
        
        IMPORTANT: This is NOT the full NTL loss computation since rollout_log_probs 
        only contains log probabilities of selected tokens, not the full vocabulary distribution.
        
        Instead, we compute a confidence-based metric:
        - Identifies digit tokens in the responses
        - Uses their log probabilities as a confidence measure
        - Creates a simplified "NTL loss" proxy based on digit confidence
        
        For full NTL computation, we'd need access to the complete logits/probabilities
        over the vocabulary during forward pass.
        
        Args:
            rollout_log_probs: Log probabilities of selected tokens [batch_size, response_len]
            responses: Generated response tokens [batch_size, response_len]
            attention_mask: Attention mask [batch_size, full_seq_len]
            
        Returns:
            Dict with digit confidence information formatted like NTL info
        """
        from ntl_local.ntl_utils import get_cached_digit_mapping, find_digit_positions
        
        try:
            # Get digit token mapping
            digit_token_map = get_cached_digit_mapping(self.tokenizer)
            if not digit_token_map:
                return {}
            
            batch_size, response_len = responses.shape
            device = responses.device
            
            # Find digit positions and ground truth
            digit_positions = []
            digit_ground_truth = []
            digit_log_probs_selected = []
            
            for batch_idx in range(batch_size):
                batch_positions = []
                batch_ground_truth = []
                batch_selected_log_probs = []
                
                for pos in range(response_len):
                    token_id = responses[batch_idx, pos].item()
                    if token_id in digit_token_map:
                        digit_value = digit_token_map[token_id]
                        log_prob = rollout_log_probs[batch_idx, pos].item()
                        
                        batch_positions.append(pos)
                        batch_ground_truth.append(digit_value)
                        batch_selected_log_probs.append(log_prob)
                
                digit_positions.append(batch_positions)
                digit_ground_truth.append(batch_ground_truth)
                digit_log_probs_selected.append(batch_selected_log_probs)
            
            # Compute simplified NTL metrics
            total_digits = sum(len(positions) for positions in digit_positions)
            if total_digits == 0:
                return {}
            
            # Compute average log probability of digit tokens (confidence metric)
            all_digit_log_probs = []
            for batch_log_probs in digit_log_probs_selected:
                all_digit_log_probs.extend(batch_log_probs)
            
            avg_digit_log_prob = sum(all_digit_log_probs) / len(all_digit_log_probs) if all_digit_log_probs else 0.0
            
            # Simplified NTL loss: use negative log probability as a proxy
            # This isn't the full NTL loss but gives us a signal about digit prediction confidence
            simplified_ntl_loss = -avg_digit_log_prob
            
            # Compute digit accuracy (all digits are "correct" since they were selected by the model)
            # For actual accuracy, we'd need ground truth from the dataset
            digit_accuracy = 1.0  # This is a placeholder - would need ground truth comparison
            
            return {
                'digit_positions': digit_positions,
                'digit_ground_truth': digit_ground_truth,
                'digit_log_probs_selected': digit_log_probs_selected,
                'ntl_loss': simplified_ntl_loss,
                'digit_accuracy': digit_accuracy,
                'total_digits': total_digits,
                'correct_digits': total_digits,  # Placeholder
                'method': 'simplified_from_rollout_log_probs',
                'avg_digit_log_prob': avg_digit_log_prob
            }
            
        except Exception as e:
            print(f"[NTL Trainer] Error in extract_ntl_from_log_probs: {e}")
            return {}
    
    def compute_reward_with_ntl(self, batch: DataProto, reward_fn) -> tuple[torch.Tensor, dict]:
        """
        Compute rewards with NTL information passed to the reward function.
        
        The reward function will receive NTL information in kwargs if available.
        """
        # Extract NTL info from batch if available
        ntl_info = batch.meta_info.get('ntl_info', None)
        
        if self.ntl_enabled and ntl_info is None:
            # Try to get NTL info from actor if not already available
            try:
                ntl_info = self.actor_rollout_wg.get_last_ntl_info() if hasattr(self.actor_rollout_wg, 'get_last_ntl_info') else {}
                if ntl_info:
                    batch.meta_info['ntl_info'] = ntl_info
            except Exception as e:
                pass
        
        # Use standard reward computation
        return compute_reward(batch, reward_fn)
    
    def step(self, batch: DataProto) -> Dict[str, Any]:
        """Override step to inject NTL tensors BEFORE reward computation."""
        print(f"[NTL Trainer] step() called, ntl_enabled: {self.ntl_enabled}")
        
        if self.ntl_enabled:
            # CORRECT FIX: Override the reward computation to inject NTL tensors first
            from verl.trainer.ppo.reward import compute_reward
            
            original_reward_fn = self.reward_fn
            
            def ntl_enhanced_reward_fn(data, return_dict=False):
                """Enhanced reward function that ensures NTL tensors are available."""
                print("[NTL Trainer] ntl_enhanced_reward_fn called - injecting NTL tensors before reward computation")
                
                # Pre-inject NTL tensors by calling compute_log_prob
                enhanced_batch = self.compute_log_prob_with_ntl(data)
                
                # Verify tensors were injected
                ntl_digit_log_probs = enhanced_batch.batch.get('ntl_digit_log_probs', None)
                print(f"[NTL Trainer] Pre-reward injection - Found ntl_digit_log_probs: {ntl_digit_log_probs is not None}")
                if ntl_digit_log_probs is not None:
                    print(f"[NTL Trainer] Pre-reward injection - ntl_digit_log_probs shape: {ntl_digit_log_probs.shape}")
                
                # Now call the original reward function with the enhanced batch
                return original_reward_fn(enhanced_batch, return_dict=return_dict)
            
            # Temporarily replace reward function
            self.reward_fn = ntl_enhanced_reward_fn
            
            try:
                # Call parent step - when it calls reward computation, NTL tensors will be available
                return super().step(batch)
            finally:
                # Restore original reward function
                self.reward_fn = original_reward_fn
        else:
            # Use standard step if NTL is disabled
            return super().step(batch)
    
    def _validate(self):
        """Override validation to ensure NTL tensors are injected during validation."""
        print(f"[NTL Trainer] _validate() called, ntl_enabled: {self.ntl_enabled}")
        
        if self.ntl_enabled:
            # Store original generate_sequences method
            original_generate_sequences = self.actor_rollout_wg.generate_sequences
            
            def ntl_generate_sequences(batch):
                """Enhanced generate_sequences that injects NTL tensors after generation."""
                print("[NTL Trainer] ntl_generate_sequences called for validation")
                
                # Call original generate_sequences
                result_batch = original_generate_sequences(batch)
                
                # For validation, we need to inject NTL tensors while preserving responses
                try:
                    # Get NTL tensors by calling compute_log_prob_with_ntl
                    ntl_enhanced = self.compute_log_prob_with_ntl(result_batch)
                    
                    # Extract just the NTL tensors from the enhanced batch
                    ntl_digit_log_probs = ntl_enhanced.batch.get('ntl_digit_log_probs', None)
                    ntl_digit_ground_truth = ntl_enhanced.batch.get('ntl_digit_ground_truth_tensor', None)
                    
                    # Inject NTL tensors into the original result_batch (preserving responses)
                    if ntl_digit_log_probs is not None:
                        result_batch.batch['ntl_digit_log_probs'] = ntl_digit_log_probs
                    if ntl_digit_ground_truth is not None:
                        result_batch.batch['ntl_digit_ground_truth_tensor'] = ntl_digit_ground_truth
                    
                    # Verify tensors were injected and responses preserved
                    print(f"[NTL Trainer] Validation - Found ntl_digit_log_probs: {ntl_digit_log_probs is not None}")
                    print(f"[NTL Trainer] Validation - Found responses: {'responses' in result_batch.batch}")
                    if ntl_digit_log_probs is not None:
                        print(f"[NTL Trainer] Validation - ntl_digit_log_probs shape: {ntl_digit_log_probs.shape}")
                    
                    return result_batch
                except Exception as e:
                    print(f"[NTL Trainer] WARNING: Failed to inject NTL tensors during validation: {e}")
                    import traceback
                    traceback.print_exc()
                    return result_batch
            
            # Temporarily replace generate_sequences
            self.actor_rollout_wg.generate_sequences = ntl_generate_sequences
            
            # Handle async rollout manager if present
            original_async_generate = None
            if hasattr(self, 'async_rollout_manager') and self.async_rollout_manager is not None:
                original_async_generate = self.async_rollout_manager.generate_sequences
                self.async_rollout_manager.generate_sequences = ntl_generate_sequences
            
            try:
                # Call parent validation
                return super()._validate()
            finally:
                # Restore original methods
                self.actor_rollout_wg.generate_sequences = original_generate_sequences
                if original_async_generate is not None:
                    self.async_rollout_manager.generate_sequences = original_async_generate
        else:
            # Use standard validation if NTL is disabled
            return super()._validate()


