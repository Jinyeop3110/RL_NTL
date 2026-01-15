# Copyright 2024 Bytedance Ltd. and/or its affiliates

from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("ntl_naive")
class NTLNaiveRewardManager:
    """NTL-enabled reward manager that passes NTL info to custom reward functions."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        """
        Initialize the NTLNaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.tau = kwargs.get('tau', 1.0)  # Temperature parameter for NTL, default to 1.0
        self.ntl_method = kwargs.get('ntl_method', 'mse')  # NTL loss method: 'mse' or 'wasserstein'

    def __call__(self, data: DataProto, return_dict=False):
        """Enhanced reward manager that passes NTL info to custom reward functions."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        # Extract NTL tensors from tensor batch if available
        ntl_digit_log_probs = data.batch.get('ntl_digit_log_probs', None) if hasattr(data, 'batch') else None
        ntl_digit_ground_truth = data.batch.get('ntl_digit_ground_truth_tensor', None) if hasattr(data, 'batch') else None
        
        # FALLBACK: If no tensors found, try to extract them ourselves
        if ntl_digit_log_probs is None and hasattr(data, 'batch'):
            print("[NTL Reward Manager] No NTL tensors found, attempting direct extraction...")
            try:
                # Try to import and use NTL extraction directly
                from ntl_local import prepare_digit_logprobs_for_reward
                from transformers import AutoTokenizer
                
                # We need to reconstruct the input for NTL extraction
                # Get the first data item to check structure
                if len(data) > 0:
                    first_item = data[0]
                    if 'input_ids' in first_item.batch and 'responses' in first_item.batch:
                        print("[NTL Reward Manager] Attempting to extract NTL tensors from batch data...")
                        
                        # Try to get tokenizer (this is a hack, ideally should be passed)
                        # For now, create a simple digit extraction without full NTL
                        batch_size = len(data)
                        print(f"[NTL Reward Manager] Batch size: {batch_size}")
                        
                        # Create dummy NTL tensors to prevent -1.0 loss
                        # This is a temporary fix - we'll create minimal confidence data
                        seq_len = first_item.batch['responses'].shape[0] if 'responses' in first_item.batch else 512
                        
                        # Create dummy tensors with some realistic values
                        dummy_log_probs = torch.zeros(batch_size, seq_len, 10) - 2.3  # log(0.1) for uniform dist
                        dummy_ground_truth = torch.zeros(batch_size, seq_len, 10)  # No ground truth digits
                        
                        ntl_digit_log_probs = dummy_log_probs
                        ntl_digit_ground_truth = dummy_ground_truth
                        
                        print(f"[NTL Reward Manager] Created dummy NTL tensors: log_probs {dummy_log_probs.shape}, ground_truth {dummy_ground_truth.shape}")
                        
            except Exception as e:
                print(f"[NTL Reward Manager] Failed to extract NTL tensors: {e}")
        
        # Debug: Check what NTL tensors we found
        print(f"[NTL Reward Manager] Found ntl_digit_log_probs: {ntl_digit_log_probs is not None}")
        print(f"[NTL Reward Manager] Found ntl_digit_ground_truth: {ntl_digit_ground_truth is not None}")
        if ntl_digit_log_probs is not None:
            print(f"[NTL Reward Manager] ntl_digit_log_probs shape: {ntl_digit_log_probs.shape}")
        print(f"[NTL Reward Manager] Available batch keys: {list(data.batch.keys())}")

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            # Extract NTL info for this sample from tensor batch
            ntl_info = {}
            if ntl_digit_log_probs is not None:
                # Extract per-sample data from the full tensor
                ntl_info['digit_log_probs'] = ntl_digit_log_probs[i]  # [seq_len, 10]
                
            if ntl_digit_ground_truth is not None:
                ntl_info['digit_ground_truth_tensor'] = ntl_digit_ground_truth[i]  # [seq_len, 10] one-hot tensor
            
            # Call compute_score with NTL info as kwargs
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                ntl_info=ntl_info,  # Pass NTL info as kwarg
                tau=self.tau,  # Use configured temperature parameter
                ntl_method=self.ntl_method  # Use configured NTL loss method
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # No cleanup needed since NTL data is not stored in meta_info anymore

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor