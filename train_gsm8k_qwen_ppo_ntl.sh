#!/bin/bash

# VERL GSM8K PPO Training Script with Number Token Loss (NTL) Integration
# Enhanced version that extracts digit-level log probabilities for NTL reward signals
# Prerequisites: Run ./prepare_gsm8k_data.sh first

set -e

# Environment should be sourced before running this script:
# source /home/yeopjin/orcd/pool/init_NTL.sh

# Set CUDA devices for 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Add VERL to Python path
export PYTHONPATH=/home/yeopjin/orcd/pool/workspace/RL_NTL:$PYTHONPATH

# Wandb configuration
export WANDB_PROJECT='RL-NTL'
export EXPERIMENT_NAME="gsm8k-ppo-qwen2.5-0.5b-ntl-$(date +%b%d)"

# Configuration
DATA_DIR="/home/yeopjin/orcd/pool/workspace/RL_NTL/data"
export BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Clean triton cache
rm -rf ~/.cache/torch/triton/

echo "=== VERL GSM8K PPO Training with NTL ==="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${BASE_MODEL}"
echo "Dataset: GSM8K"
echo "Data directory: ${DATA_DIR}"
echo "Wandb project: ${WANDB_PROJECT}"
echo "NTL Integration: ENABLED"
echo ""

# Verify prerequisites
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
    echo "ERROR: Dataset not found. Please run ./prepare_gsm8k_data.sh first"
    exit 1
fi

echo "Prerequisites verified. Starting NTL-enhanced PPO training..."
LOG_FILE="${EXPERIMENT_NAME}.log"
echo "Logging to: ${LOG_FILE}"
echo ""

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.shuffle=true \
    actor_rollout_ref.actor.optim.lr=1e-06 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.ntl_enabled=true \
    actor_rollout_ref.actor.ntl_method=mse \
    actor_rollout_ref.actor.ntl_weight=0.1 \
    actor_rollout_ref.actor.extract_digit_info=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.ignore_eos=false \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=false \
    critic.model.fsdp_config.param_offload=false \
    critic.model.fsdp_config.optimizer_offload=false \
    critic.strategy=fsdp \
    critic.ppo_mini_batch_size=64 \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.ppo_epochs=1 \
    critic.cliprange_value=0.5 \
    critic.grad_clip=1.0 \
    critic.shuffle=true \
    critic.optim.lr=1e-05 \
    critic.optim.lr_warmup_steps_ratio=0.0 \
    critic.optim.warmup_style=constant \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_penalty=kl \
    reward_model.enable=false \
    reward_model.micro_batch_size=64 \
    reward_model.reward_manager=naive \
    custom_reward_function.path=/home/yeopjin/orcd/pool/workspace/RL_NTL/custom_NTL.py \
    custom_reward_function.name=compute_score \
    custom_reward_function.use_ntl_bonus=true \
    custom_reward_function.ntl_bonus_type=all \
    custom_reward_function.ntl_bonus_weight=0.1 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.test_freq=10 \
    trainer.critic_warmup=0 \
    trainer.save_freq=-1 \
    trainer.logger="['console', 'wandb']" \
    +trainer.mode=standard \
    2>&1 | tee $LOG_FILE

echo ""
echo "NTL-enhanced training complete!"
echo "Checkpoints saved in: checkpoints/$WANDB_PROJECT/$EXPERIMENT_NAME"
echo "Log file: $LOG_FILE"
echo ""
echo "NTL Features enabled:"
echo "- Digit-level log probability extraction"
echo "- NTL-enhanced reward calculation"
echo "- Digit accuracy metrics"
echo "- NTL loss monitoring"