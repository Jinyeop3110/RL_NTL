#!/bin/bash

# Quick test to verify NTL info extraction is working
set -e

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/yeopjin/orcd/pool/workspace/RL_NTL:$PYTHONPATH
export WANDB_PROJECT='RL-NTL-TEST'
export EXPERIMENT_NAME="test-ntl-fix-$(date +%H%M)"

DATA_DIR="/home/yeopjin/orcd/pool/workspace/RL_NTL/data"
export BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

echo "=== Quick NTL Test ==="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${BASE_MODEL}"
echo "Testing NTL info extraction with tokenizer fix"
echo ""

# Run just 1 epoch with small batch size for quick test
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.strategy=fsdp \
    +actor_rollout_ref.actor.ntl_enabled=true \
    +actor_rollout_ref.actor.ntl_method=mse \
    +actor_rollout_ref.actor.ntl_weight=0.1 \
    +actor_rollout_ref.actor.extract_digit_info=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enforce_eager=true \
    critic.model.path=$BASE_MODEL \
    critic.strategy=fsdp \
    critic.ppo_mini_batch_size=8 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    reward_model.enable=false \
    reward_model.reward_manager=ntl_naive \
    custom_reward_function.path=/home/yeopjin/orcd/pool/workspace/RL_NTL/custom_ntl_final.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.tau=1.0 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.logger="['console']" \
    +trainer.mode=standard \
    2>&1 | tee quick_test_ntl.log

echo ""
echo "Test complete! Check quick_test_ntl.log for NTL info extraction details."